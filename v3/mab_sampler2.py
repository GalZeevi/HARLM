import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import re

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
# from data_access_v3 import DataAccess
from score_calculator import get_score
from pybandits.core.smab import Smab


class MabSampler:

    # SETTINGS #
    STEPS = 400
    REWARD_THRESHOLD = 0.5

    @staticmethod
    def get_best_cached_mab(max_iters, epsilon):
        version = CheckpointManager.get_max_version()
        path = CheckpointManager.get_checkpoint_path(version)
        checkpoints_files = [f for f in listdir(path) if isfile(join(path, f))]
        reg = re.compile(f'[0-9]+_{epsilon}_smab')
        smab_checkpoints = list(filter(reg.search, checkpoints_files))
        if len(smab_checkpoints) == 0:
            return None

        smab_checkpoints.sort(key=lambda name: int(name.split('_', 1)[0]))
        smab_checkpoints = [name for name in smab_checkpoints if int(name.split('_', 1)[0]) <= max_iters]
        return smab_checkpoints[-1].split('.pkl', 1)[0]

    @staticmethod
    def run(num_rows: int,
            k: int,
            evaluate_score_function,
            max_iter: int = STEPS,
            epsilon: float = REWARD_THRESHOLD):

        mab = Smab(action_ids=[str(a) for a in np.arange(num_rows)])

        for i in tqdm(range(max_iter)):
            # TODO predict only from a random subset of arms and not all of them
            # TODO don't allow the k samples to choose the same arm twice
            predicted_actions, _ = mab.predict(n_samples=k)
            predicted_sample = np.unique([int(a) for a in predicted_actions])
            # TODO evaluate using the train-queries
            sample_score = evaluate_score_function(predicted_sample)
            n_success, n_failures = 0., 0.
            if sample_score >= epsilon:
                n_success = 1.
            else:
                n_failures = 1.
            for a in predicted_sample:
                mab.update(action_id=a, n_successes=n_success, n_failures=n_failures)

            if i > 0 and i % 100 == 0:
                CheckpointManager.save(f'{i}_{epsilon}_smab', mab)

        BATCH_SIZE = 100
        pbar = tqdm(total=k)
        sample = np.array([])
        while len(sample) <= k:
            predicted_actions, predicted_probs = mab.predict(n_samples=BATCH_SIZE, forbidden_actions=sample.tolist())
            agg_probs = np.zeros(num_rows)
            for d in predicted_probs:
                agg_probs[[int(a) for a in d.keys()]] = [*d.values()]
                nonzero_probs = agg_probs[agg_probs > 0]
                num_to_take = min(k - len(sample), len(nonzero_probs))
                topk_ind = np.argpartition(agg_probs, -num_to_take)[-num_to_take:]
                sample = np.append(sample, topk_ind)
                pbar.update(len(topk_ind))

        return sample, evaluate_score_function(sample)


def get_sample(k, dist=False, max_iter=MabSampler.STEPS, threshold=MabSampler.REWARD_THRESHOLD):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    # table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
    table_size = 24589952

    sample, score = MabSampler.run(table_size, k, lambda s: get_score(s, dist), max_iter)
    CheckpointManager.save(f'{k}-{view_size}-{max_iter}_{threshold}_smab_sample', [sample, score])

    return sample, score


if __name__ == '__main__':
    get_sample(200_000, max_iter=100_000, threshold=0.5)

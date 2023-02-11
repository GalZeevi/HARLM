# library imports
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import re

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score


class MabSampler:
    """
    This class is a Multi arm Bandit type algorithm for dataset summarization
    """

    # SETTINGS #
    steps = 400
    initial_exploration_steps = 0.2
    epsilon = 0.3

    @staticmethod
    def get_best_cached_wins_pulls(max_iters, epsilon):
        version = CheckpointManager.get_max_version()
        path = CheckpointManager.get_checkpoint_path(version)
        checkpoints_files = [f for f in listdir(path) if isfile(join(path, f))]
        reg = re.compile(f'[0-9]+_{epsilon}_mab_wins_pulls')
        wins_pulls_checkpoints = list(filter(reg.search, checkpoints_files))
        if len(wins_pulls_checkpoints) == 0:
            return None

        wins_pulls_checkpoints.sort(key=lambda name: int(name.split('_', 1)[0]))
        wins_pulls_checkpoints = [name for name in wins_pulls_checkpoints if int(name.split('_', 1)[0]) <= max_iters]
        return wins_pulls_checkpoints[-1].split('.npy', 1)[0]

    @staticmethod
    def run(num_rows: int,
            k: int,
            evaluate_score_function,
            max_iter: int = steps,
            epsilon: float = epsilon):

        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at MultiArmBanditSummary.run: the max_iter argument must be larger than 1")

        checkpoint_name = MabSampler.get_best_cached_wins_pulls(max_iter, epsilon)
        if checkpoint_name is None:
            starting_iteration = 0
            wins = np.zeros(num_rows)
            pulls = np.zeros(num_rows)
        else:
            starting_iteration = int(checkpoint_name.split('_', 1)[0]) + 1
            wins_pulls = CheckpointManager.load(checkpoint_name, numpy=True)
            wins = wins_pulls[0]
            pulls = wins_pulls[1]

        if starting_iteration < max_iter:
            pbar = tqdm(total=max_iter)
            pbar.update(starting_iteration)

            choices = []
            for n in range(starting_iteration, max_iter):
                if n > starting_iteration and n % 10000 == 0:
                    CheckpointManager.save(f'{n}_{epsilon}_mab_wins_pulls', np.array([wins, pulls]), numpy=True)
                if n >= starting_iteration + int(MabSampler.initial_exploration_steps * (max_iter - starting_iteration)) \
                        and np.random.uniform(size=1) < epsilon:
                    choice = np.argmax(wins / (pulls + 0.1))
                else:
                    choice = np.random.choice(list(set(range(len(wins))) - {np.argmax(wins / (pulls + 0.1))}))
                choices.append(choice)

                payout = evaluate_score_function(choice)
                wins[choice] += payout
                pulls[choice] += 1
                pbar.update(1)

            CheckpointManager.save(f'{max_iter}_{epsilon}_mab_wins_pulls', np.array([wins, pulls]), numpy=True)

        # get best
        best_rows = []
        added_count = 0
        pbar = tqdm(total=k)

        while added_count < k:
            next_index = np.argmax(wins / (pulls + 0.1))
            if next_index < num_rows and len(best_rows) < k:
                best_rows.append(next_index)
                added_count += 1
            wins[next_index] = np.NINF
            pulls[next_index] = 1000.
            pbar.update(1)

        return best_rows, evaluate_score_function(best_rows)


def get_sample(k, dist=False, max_iter=MabSampler.steps, epsilon=MabSampler.epsilon):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    sample, score = MabSampler.run(table_size, k, lambda s: get_score(s, dist), max_iter, epsilon)
    CheckpointManager.save(f'{k}-{view_size}-{max_iter}_{epsilon}_mab_sample', [sample, score])

    return sample, score


if __name__ == '__main__':
    get_sample(200_000, max_iter=100_000, epsilon=0.5)

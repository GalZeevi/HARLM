import numpy as np
from pathos.pools import _ProcessPool

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score
from tqdm import tqdm

num_of_trials = 20


def get_sample(k, dist=False):
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    scores = np.zeros(num_of_trials)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')

    def __random_sample_task(trial_id):
        trial_sample = np.random.choice(table_size, k, replace=False)
        return trial_id, get_score(trial_sample, dist)

    with _ProcessPool(num_of_trials) as pool:
        for i, score in pool.map(__random_sample_task, [*range(num_of_trials)]):
            scores[i] = score

    score = np.average(scores)

    sample = np.random.choice(table_size, k, replace=False)
    CheckpointManager.save(f'{k}-{view_size}-random_sample', [sample, score])
    return sample, score


if __name__ == '__main__':
    # k_list = [10 * 10**3, 50 * 10**3, 100 * 10**3, 200 * 10**3, 250 * 10**3]
    # for k in tqdm(k_list):
    #     get_sample(k)
    get_sample(3000)

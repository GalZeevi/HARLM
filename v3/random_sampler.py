from data_access_v3 import DataAccess
from config_manager_v3 import ConfigManager
from checkpoint_manager_v3 import CheckpointManager
from score_calculator import get_score
import numpy as np
from tqdm import tqdm


def get_sample(k, dist=False):
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    num_of_trials = 10
    sample = None
    scores = np.zeros(num_of_trials)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')

    for trial in tqdm(range(num_of_trials)):
        sample = np.random.choice(table_size, k, replace=False)
        scores[trial] = get_score(sample, view_size, dist)

    score = np.average(scores)

    CheckpointManager.save(f'{k}-{view_size}-random_sample', [sample, score])
    return sample, score


if __name__ == '__main__':
    get_sample(3000)

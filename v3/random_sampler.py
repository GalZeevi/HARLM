import numpy as np
from tqdm import trange

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score2

num_of_trials = 50
schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')


def get_sample(k, checkpoint_version=CheckpointManager.get_max_version()):
    scores = np.zeros(num_of_trials)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')

    for trial_num in trange(num_of_trials):
        scores[trial_num] = get_score2(sample=np.random.choice(table_size, k, replace=False),
                                       queries='test',
                                       checkpoint_version=checkpoint_version)

    score = np.average(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    print(f'Checkpoint: {checkpoint_version}, Min: {min_score}, Max: {max_score}, Avg: {score}')
    sample = np.random.choice(table_size, k, replace=False)
    CheckpointManager.save(f'{k}-{view_size}-random_sample', [sample, score], version=checkpoint_version)
    return sample, score


if __name__ == '__main__':
    print(get_sample(100)[1])

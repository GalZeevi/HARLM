from data_access_v3 import DataAccess
from config_manager_v3 import ConfigManager
from checkpoint_manager_v3 import CheckpointManager
from score_calculator import get_score
import numpy as np
from tqdm import tqdm


def prepare_weights_for_sample():
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    weights = np.zeros(table_size)
    results = CheckpointManager.load('results')
    test_len = int(ConfigManager.get_config('samplerConfig.testSize') * len(results))
    train_results = results[test_len:]

    for query_result in tqdm(train_results):
        weights[query_result] += 1.

    CheckpointManager.save('greedy_sampler_weights', weights, numpy=True)
    return weights


def get_sample(k, dist=False):
    weights = CheckpointManager.load('greedy_sampler_weights', numpy=True)
    if weights is None:
        weights = prepare_weights_for_sample()

    sample = np.argpartition(weights, -k)[-k:]
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    score = get_score(sample, view_size, dist)

    CheckpointManager.save(f'{k}-{view_size}-greedy_sample', [sample, score])
    return sample, score


if __name__ == '__main__':
    get_sample(2500)

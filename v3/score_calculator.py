import numpy as np
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from tuple_distance_calculator_v3 import TupleDistanceCalculator
from data_access_v3 import DataAccess
from tqdm import tqdm

tupleDistanceCalculator = TupleDistanceCalculator()


def get_score(sample, view_size, dist=False):
    results = CheckpointManager.load('results')
    test_len = int(ConfigManager.get_config('samplerConfig.testSize') * len(results))
    test_results = results[:test_len]

    return get_score_for_test_queries(sample, test_results, view_size) if dist is False \
        else get_dist_score_for_test_queries(sample, test_results)


def get_score_for_test_queries(sample, test_results, view_size):
    target_view_sizes = np.array([min(view_size, len(result)) for result in test_results])
    sample_result_sizes = np.array([len(np.intersect1d(result, sample)) for result in test_results])
    attained_result_fraction = np.divide(sample_result_sizes, target_view_sizes)
    score = np.average(np.minimum(attained_result_fraction, 1))
    return score


def get_dist_score_for_test_queries(sample, test_results):
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    score = 0.0
    sample_results = [np.intersect1d(result, sample) for result in test_results]

    for i in tqdm(range(len(test_results))):
        full_result = DataAccess.select(
            f'SELECT * FROM {schema}.{table} where {pivot} IN ({",".join([str(idx) for idx in test_results[i]])})')
        if len(sample_results[i]) > 0:
            sample_result = DataAccess.select(
                f'SELECT * FROM {schema}.{table} where {pivot} IN ({",".join([str(idx) for idx in sample_results[i]])})')
            distances = tupleDistanceCalculator.calculate_dist_matrix(full_result, sample_result)
            score += np.sum(1 - np.min(distances, axis=1))

    return score / len(test_results)

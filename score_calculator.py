import numpy as np
from tqdm import tqdm
from config_manager_v3 import ConfigManager
from checkpoint_manager_v3 import CheckpointManager
from data_access_v3 import DataAccess
# from tuple_distance_calculator_v3 import TupleDistanceCalculator
from train_test_utils import get_test_queries, get_train_queries
from preprocessing import Preprocessing, Column, NULL_VALUE
from typing import Dict


# tupleDistanceCalculator = TupleDistanceCalculator()


def __get_results(queries, checkpoint_version):
    if isinstance(queries, str):
        assert queries in ['test', 'train']
        return get_test_queries(checkpoint_version=checkpoint_version) if queries == 'test' \
            else get_train_queries(checkpoint_version=checkpoint_version)
    elif isinstance(queries, list) or isinstance(queries, np.ndarray):
        return queries
    else:
        raise Exception('queries should be either a string from ["train", "test"] or a list/array of results!')


def get_score2(sample,
               queries='train',
               checkpoint_version=CheckpointManager.get_max_version(),
               average=True):
    Preprocessing.init(checkpoint_version)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    results = __get_results(queries, checkpoint_version)
    target_sizes = np.array([min(view_size, len(result)) for result in results], dtype='int64')
    sampled_sizes = np.array([len(np.intersect1d(result, sample)) for result in results], dtype='int64')
    attained_result_fraction = np.divide(sampled_sizes, target_sizes, out=np.ones_like(target_sizes, dtype='float64'),
                                         where=target_sizes != 0.)
    score = np.minimum(attained_result_fraction, 1.)
    return np.average(score) if average else score


def get_combined_score(sample_tuples, alpha=0.5, queries='train',
                       checkpoint_version=CheckpointManager.get_max_version()):
    Preprocessing.init(checkpoint_version)
    pivot = ConfigManager.get_config('queryConfig.pivot')
    tuple_ids = [tup[pivot] for tup in sample_tuples]
    if alpha == 1.0:
        return get_score2(tuple_ids, queries, checkpoint_version)
    elif alpha == 0.0:
        return get_diversity_score(sample_tuples, checkpoint_version)
    return alpha * get_score2(tuple_ids, queries, checkpoint_version) + \
           (1 - alpha) * get_diversity_score(sample_tuples, checkpoint_version)


def get_threshold_score(sample,
                        queries='train',
                        threshold=0.25,
                        checkpoint_version=CheckpointManager.get_max_version()):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    results = __get_results(queries, checkpoint_version)
    target_sizes = np.array([min(view_size, len(result)) for result in results])
    sampled_sizes = np.array([len(np.intersect1d(result, sample)) for result in results])
    attained_result_fraction = np.divide(sampled_sizes, target_sizes)
    score = np.minimum(attained_result_fraction, 1.)
    score = np.where(score >= threshold, 1, 0)
    return np.average(score)


def get_diversity_score(sample_tuples, checkpoint_version=CheckpointManager.get_max_version()):
    Preprocessing.init(checkpoint_version)
    columns: Dict[str, Column] = Preprocessing.columns_repo.get_all_columns()
    score = 0.
    for col_name, col_data in columns.items():
        if col_data.is_pivot:
            continue
        if col_data.is_categorical:
            # diversity ratio
            col_values = np.array([tup[col_name] if tup[col_name] is not None else NULL_VALUE
                                   for tup in sample_tuples])
            num_uniq_vals = len(np.unique(col_values))
            score += num_uniq_vals / len(col_data.encodings.keys())
        else:
            # variance
            col_values = np.array([tup[col_name] for tup in sample_tuples]).astype(float)
            if col_data.min_val != col_data.max_val:  # column is not fixed
                col_values = (col_values - col_data.min_val) / (col_data.max_val - col_data.min_val)
                sample_col_std = 2.0 * np.std(col_values)
                full_col_std = 2.0 * col_data.std
                score += (1 - abs(sample_col_std - full_col_std))
            else:
                score += 1
    return score / (len(columns.items()) - 1)


def get_score(sample, dist=False):
    raise DeprecationWarning()
    test_results = get_test_queries()
    return get_score_for_test_queries(sample, test_results) if dist is False \
        else get_dist_score_for_test_queries(sample, test_results)


def get_score_for_test_queries(sample, test_results):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
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

    sample_results = [np.intersect1d(result, sample) for result in tqdm(test_results)]

    # for i in tqdm(range(len(test_results))):
    #     if len(sample_results[i]) > 0:
    #         full_result = DataAccess.select(
    #             f'SELECT * FROM {schema}.{table} where {pivot} IN ({",".join([str(idx) for idx in test_results[i]])})')
    #         sample_result = DataAccess.select(
    #             f'SELECT * FROM {schema}.{table} where {pivot} IN ({",".join([str(idx) for idx in sample_results[i]])})')
    #         distances = tupleDistanceCalculator.calculate_dist_matrix(full_result, sample_result)
    #         score += np.average(1 - np.min(distances, axis=1))

    return score / len(test_results)


if __name__ == '__main__':
    Preprocessing.init()
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    tuples = DataAccess.select(f'SELECT * FROM {schema}.{table} LIMIT 5;')
    a = [tuples[0]] * 10
    b = [tuples[0], tuples[1]] * 5
    c = tuples * 2
    d = DataAccess.select(f'SELECT * FROM {schema}.{table} ORDER BY RAND() LIMIT 1000;')
    print('x')
    print('a: ', get_diversity_score(a, 17))
    print('b: ', get_diversity_score(b, 17))
    print('c: ', get_diversity_score(c, 17))
    print('d: ', get_diversity_score(d, 17))

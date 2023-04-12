import numpy as np
from tqdm import tqdm
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
# from tuple_distance_calculator_v3 import TupleDistanceCalculator
from train_test_utils import get_test_queries, get_train_queries
from preprocessing import Preprocessing

# tupleDistanceCalculator = TupleDistanceCalculator()


def get_score(sample, dist=False):  # TODO replace dist with metric
    test_results = get_test_queries()
    return get_score_for_test_queries(sample, test_results) if dist is False \
        else get_dist_score_for_test_queries(sample, test_results)


def get_score2(sample, queries='train', view_size=ConfigManager.get_config('samplerConfig.viewSize')):
    if isinstance(queries, str):
        results = get_test_queries() if queries == 'test' else get_train_queries()
    elif isinstance(queries, list):
        results = queries
    else:
        raise Exception('queries should be either a string from ["train", "test"] or a list of results!')

    view_size = 500 if view_size is None else view_size
    target_view_sizes = np.array([min(view_size, len(result)) for result in results])
    sample_result_sizes = np.array([len(np.intersect1d(result, sample)) for result in results])
    attained_result_fraction = np.divide(sample_result_sizes, target_view_sizes)
    score = np.minimum(attained_result_fraction, 1.)
    return np.average(score)


def get_diversity(tuples):
    diversity = 0.
    Preprocessing.init()
    columns = Preprocessing.columns_repo.get_all_columns()
    for col_name, col in columns.items():
        if col.is_pivot:
            continue
        col_values = np.array([tup[col_name] for tup in tuples])
        if col.is_categorical:
            diversity += (len(np.unique(col_values)) - 1) / len(col_values)
        else:
            col_values = col_values.astype(float)
            min_val = np.min(col_values)
            max_val = np.max(col_values)
            if min_val != max_val:
                # diversity += 4 * np.var((col_values - min_val) / (max_val - min_val))
                continue

    diversity = diversity / (len(columns.keys()) - 1)
    raise NotImplementedError
    # return diversity


def get_score_for_test_queries(sample, test_results):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    target_view_sizes = np.array([min(view_size, len(result)) for result in test_results])
    sample_result_sizes = np.array([len(np.intersect1d(result, sample)) for result in test_results])
    attained_result_fraction = np.divide(sample_result_sizes, target_view_sizes)
    score = np.average(np.minimum(attained_result_fraction, 1))
    return score


def __shifted_sigmoid(x):
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return 2 * sigmoid - 1


def get_differentiable_score_for_test_queries(sample, test_results, view_size):
    target_view_sizes = np.array([view_size * __shifted_sigmoid(len(result) / view_size) for result in test_results])
    sample_result_sizes = np.array([len(np.intersect1d(result, sample)) for result in test_results])
    attained_result_fraction = np.divide(sample_result_sizes, target_view_sizes)
    score = 2 * np.average(__shifted_sigmoid(attained_result_fraction))
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

import argparse
from tqdm import trange
import numpy as np

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score2, get_threshold_score, get_diversity_score
from train_test_utils import get_train_queries


def get_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="The Checkpoint to use."
    )

    parser.add_argument(
        "--k", type=int, default=100, help="Sample size."
    )

    cli_args = parser.parse_args()
    print(f"Running with following args: {cli_args}")
    return cli_args


args = get_args()

schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
pivot = ConfigManager.get_config('queryConfig.pivot')


def select_tuples(ids):
    ids_db_fmt = ','.join([str(idx) for idx in ids])
    return DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({ids_db_fmt})')


def get_cache():
    cache = np.array([])
    cache_capacity = args.k
    train_queries = get_train_queries(args.checkpoint)
    for results in train_queries:
        np.random.shuffle(results)
        if len(cache) + len(results) < cache_capacity:  # We have room to fit all the tuples in cache
            cache = np.concatenate([cache, results])
        elif len(results) > cache_capacity:  # Can't fit entire query in cache
            cache = np.array(results[:cache_capacity])  # flush cache and put query in
        else:  # Can fit entire query in cache
            # need to calculate how many tuples to evacuate
            num_to_evacuate = len(results) - (cache_capacity - len(cache))
            cache = np.concatenate([cache[num_to_evacuate:], results])
        assert len(cache) <= cache_capacity
    # print('Cache creation completed!')
    return cache


if __name__ == '__main__':
    test_scores = []
    test_threshold_scores = []
    # test_diversities = []
    # TODO save sample

    for i in trange(25):
        sample_ids = get_cache()

        test_scores += [get_score2(sample_ids, queries='test', checkpoint_version=args.checkpoint)]
        test_threshold_scores += [get_threshold_score(sample_ids, queries='test', checkpoint_version=args.checkpoint)]
        # test_diversitys += [get_diversity_score(select_tuples(sample_ids), args.checkpoint)]

    print(
        f'############### Sample score: [min: {np.min(test_scores)}, avg: {np.average(test_scores)}, max: {np.max(test_scores)}] ###############')
    print(
        f'############### Sample threshold score: [min: {np.min(test_threshold_scores)}, avg: {np.average(test_threshold_scores)}, max: {np.max(test_threshold_scores)}] ###############')
    # print(
        # f'############### Sample diversity: [min: {np.min(test_diversities)}, avg: {np.average(test_diversities)}, max: {np.max(test_diversities)}] ###############')

import argparse

import numpy as np
from tqdm import tqdm

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score2, get_threshold_score
from train_test_utils import get_train_queries


def prepare_weights_for_sample(verbose=True, checkpoint_version=CheckpointManager.get_max_version()):
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    weights = np.zeros(table_size)
    validation_size = ConfigManager.get_config('samplerConfig.validationSize')
    if validation_size > 0:
        train_results, _ = get_train_queries(validation_size=validation_size, checkpoint_version=checkpoint_version)
    else:
        train_results = get_train_queries(validation_size=validation_size, checkpoint_version=checkpoint_version)
    for query_result in (tqdm(train_results) if verbose else train_results):
        weights[query_result] += 1.

    verbose and CheckpointManager.save('top_queried_sampler_weights', weights, numpy=True, version=checkpoint_version)
    return weights


def prepare_sample(k, verbose=True, checkpoint_version=CheckpointManager.get_max_version()):
    weights = CheckpointManager.load('top_queried_sampler_weights', numpy=True, version=checkpoint_version)
    if weights is None:
        weights = prepare_weights_for_sample(verbose, checkpoint_version=checkpoint_version)

    # print(np.sort(weights)[::-1][:k])

    return np.argpartition(weights, -k)[-k:]


def get_sample(k, verbose=True, checkpoint_version=CheckpointManager.get_max_version()):
    sample = prepare_sample(k, verbose, checkpoint_version)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    score = get_score2(sample, queries='test', checkpoint_version=checkpoint_version)
    threshold_score = get_threshold_score(sample, queries='test', checkpoint_version=checkpoint_version)

    verbose and CheckpointManager.save(f'{k}-{view_size}-top_queried_sample',
                                       [sample, score, threshold_score], version=checkpoint_version)
    return sample, score, threshold_score


if __name__ == '__main__':
    # prepare_weights_for_sample()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="The Checkpoint to use."
    )

    parser.add_argument(
        "--k", type=int, default=1000, help="Sample size."
    )

    args = parser.parse_args()
    print(get_sample(args.k, checkpoint_version=args.checkpoint)[1])
    # prepare_sample(100)
    # k_list = [10 * 10**3, 50 * 10**3, 100 * 10**3, 200 * 10**3, 250 * 10**3]
    # for k in tqdm(k_list):
    #     get_sample(k)

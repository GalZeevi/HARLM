import argparse
import itertools
import os

import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from scipy.spatial.distance import hamming, cityblock as l1

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing, ColumnsRepo, Column
from score_calculator import get_score2, get_threshold_score, get_diversity_score
from train_test_utils import get_train_queries
from tqdm import tqdm
import multiprocessing
import time

MAX_TUPLES_TO_SELECT = 10000


def get_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="The size of the sample to create."
    )

    parser.add_argument(
        "--k", type=int, default=1000, help="Which checkpoint version to use."
    )

    parser.add_argument(
        "--lib", type=str, default='pyclustering', help="Which clustering library to use."
    )

    parser.add_argument(
        "--config_path", type=str, default=None, help="Which config file to use."
    )

    parser.add_argument("--pool_size", type=int, default=6, help="how many processes to use")

    cli_args = parser.parse_args()
    print(f"Running with following CLI args: {cli_args}")
    return cli_args


args = get_args()

args.config_path is not None and ConfigManager._init_config(args.config_path)

schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
pivot = ConfigManager.get_config('queryConfig.pivot')


def distance(x, y):
    if isinstance(x, list):
        x = np.array(x)

    if isinstance(y, list):
        y = np.array(y)

    num_cols_sorted = sorted(Preprocessing.columns_repo.get_numerical_columns(), key=lambda col: col.dim)
    num_dims = [col.dim for col in num_cols_sorted]
    num_normalizers = [col.max_val - col.min_val for col in num_cols_sorted]

    cat_dims = [col.dim for col in Preprocessing.columns_repo.get_categorical_columns()]
    cat_dist = hamming(x[cat_dims], y[cat_dims]) * len(cat_dims)

    num_dist = l1(np.divide(x[num_dims], num_normalizers), np.divide(y[num_dims], num_normalizers))

    return (num_dist + cat_dist) / len(x)


def __get_clusters__(k, result_ids, result_tuples, lib='pyclustering'):
    assert lib in ['pyclustering', 'nltk']
    result_tuples_numpy = Preprocessing.tuples2numpy(result_tuples)

    # print('start clustering...')
    if lib == 'pyclustering':
        initial_centers = random_center_initializer(result_tuples_numpy, k, random_state=5).initialize()
        instanceKm = kmeans(result_tuples_numpy,
                            initial_centers=initial_centers,
                            metric=distance_metric(metric_type=type_metric.USER_DEFINED, func=distance))
        instanceKm.process()
        pyClusters = instanceKm.get_clusters()
        means = instanceKm.get_centers()

        clusters = []
        for pycluster in pyClusters:
            cluster = np.array(result_ids)[pycluster]
            clusters.append(cluster)

        return clusters, means

    elif lib == 'nltk':
        kclusterer = KMeansClusterer(k, distance=distance, repeats=10)
        clusters_mask = kclusterer.cluster(result_tuples_numpy, assign_clusters=True)
        clusters_mask = np.array(clusters_mask)
        means = kclusterer.means()
        clusters = []
        for cluster_id in range(k):
            cluster = np.array(result_ids)[np.where(clusters_mask == cluster_id)]
            clusters.append(cluster)

        return clusters, means


def select_tuples(ids):
    tuples = []
    chunks = [chunk for chunk in np.array_split(ids, np.ceil(len(ids) / MAX_TUPLES_TO_SELECT)) if
              len(chunk) > 0]
    for chunk in chunks:
        ids_db_format = ','.join([str(idx) for idx in chunk])
        tuples += DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({ids_db_format})')

    return tuples


def partition_dataset(ids, target_size=35000):
    numerical_columns = [col_name for col_name, col_data in Preprocessing.columns_repo.get_all_columns().items()
                         if not col_data.is_categorical and not col_data.is_pivot]
    partition = [ids]
    for col_name in numerical_columns:
        print(f'About to split by column: {col_name}')

        good_part = [p for p in partition if len(p) <= target_size]
        bad_part = [p for p in partition if len(p) > target_size]
        bad_part_joined = [*itertools.chain(*bad_part)]
        np.random.shuffle(bad_part_joined)

        bad_part_chunks = [chunk for chunk in
                           np.array_split(bad_part_joined, np.ceil(len(bad_part_joined) / MAX_TUPLES_TO_SELECT)) if
                           len(chunk) > 0]
        partition = good_part
        for chunk in bad_part_chunks:
            new_bad_part = split_by_column(chunk, col_name)
            partition = partition + new_bad_part

        print(f'Partition sizes after split: {[len(p) for p in partition]}')

        CheckpointManager.save(name=f'{args.k}_kmeans_partition', content=partition, version=args.checkpoint)
        done = len([p for p in partition if len(p) > target_size]) == 0

        if done:
            break

    return partition


def split_by_column(ids, col_name):
    NUM_SPLITS = 10
    col_data = Preprocessing.columns_repo.column_by_name(col_name)
    partition = []
    split_points = [col_data.min_val + i * (col_data.max_val - col_data.min_val) / NUM_SPLITS for i in
                    range(NUM_SPLITS + 1)]
    ids_db_format = ','.join([str(idx) for idx in ids])

    for i in range(len(split_points) - 1):
        min_pt = split_points[i]
        max_pt = split_points[i + 1]
        partition.append(
            DataAccess.select(f'SELECT {pivot} FROM {schema}.{table} WHERE {pivot} IN ({ids_db_format}) AND '
                              f'{col_name} >= {min_pt} AND {col_name} < {max_pt}'))

    partition = [p for p in partition if len(p) > 0]
    return partition


def get_kmeans_sample(k, ids, index, lib='pyclustering'):
    Preprocessing.init(args.checkpoint)
    DataAccess()

    if CheckpointManager.load(name=f'{OUTPUT_DIR}/{lib}_{index}_kmeans_clusters', version=args.checkpoint) is not None:
        clusters_means = CheckpointManager.load(name=f'{OUTPUT_DIR}/{lib}_{index}_kmeans_clusters',
                                                version=args.checkpoint)
        clusters, means = clusters_means[0], clusters_means[1]
    else:
        result_tuples = select_tuples(ids)
        clusters, means = __get_clusters__(k, ids, result_tuples, lib=lib)
        CheckpointManager.save(name=f'{OUTPUT_DIR}/{lib}_{index}_kmeans_clusters', content=[clusters, means],
                               version=args.checkpoint)

    sample = []
    result_tuples = select_tuples(ids)
    for cluster_id, cluster in enumerate(clusters):
        cluster_tuples = [tup for tup in result_tuples if tup[pivot] in cluster]
        cluster_tuples_numpy = Preprocessing.tuples2numpy(cluster_tuples)
        cluster_mean = means[cluster_id]
        cluster_distances = [distance(x, cluster_mean) for x in cluster_tuples_numpy]
        cluster_rep_id = cluster_tuples[np.argmin(cluster_distances)][pivot]
        sample.append(cluster_rep_id)

    return sample


if __name__ == '__main__':
    print('Starting k_means_sampler.main()!')
    OUTPUT_DIR = f'{args.k}_{args.lib}_kmeans'
    COMPLETE_OUTPUT_DIR = f'{CheckpointManager.basePath}/{args.checkpoint}/{OUTPUT_DIR}'
    if not os.path.exists(COMPLETE_OUTPUT_DIR):
        os.makedirs(COMPLETE_OUTPUT_DIR)

    Preprocessing.init(args.checkpoint)
    print('Preparing data...')
    train_results = get_train_queries(checkpoint_version=args.checkpoint)
    result_ids = [*set().union(*train_results)]
    print(f'There are {len(result_ids)} tuples in results')

    if CheckpointManager.load(name=f'kmeans_partition', version=args.checkpoint) is None:
        partition = partition_dataset(result_ids)
        CheckpointManager.save(name=f'kmeans_partition', content=partition, version=args.checkpoint)
    else:
        partition = CheckpointManager.load(name=f'kmeans_partition', version=args.checkpoint)
        print(f'Saved partition sizes: {[len(p) for p in partition]}')
    print(f'Partition length: {len(partition)}')

    worker_args = []
    for i, p in tqdm(enumerate(partition)):
        sample_size = min(max(int(np.floor(1.1 * args.k * (len(p) / len(result_ids)))), 3), len(p))
        worker_args.append((sample_size, p, i, args.lib))

    print('Initialising pool')
    start = time.time()
    pool = multiprocessing.Pool(args.pool_size)
    print(f'Initialising pool took: {round(time.time() - start, 2)} sec')

    print(f'Starting to work on {len(worker_args)} tasks with pool of {args.pool_size} workers', flush=True)
    pool_res = pool.starmap_async(get_kmeans_sample, worker_args)
    sample = []
    for k_means_sample in pool_res.get():
        sample += k_means_sample
        CheckpointManager.save(name=f'{args.k}_{args.lib}_kmeans_sample',
                               content=[sample, get_score2(sample, queries='test', checkpoint_version=args.checkpoint)],
                               version=args.checkpoint)
        print(f'Current sample size={len(sample)}', flush=True)

    if len(sample) < args.k:
        print('Calculating table_size', flush=True)
        table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
        print(f'table_size: {table_size}', flush=True)
        available_idx = np.setdiff1d(np.arange(table_size), sample)
        random_ids = np.random.choice(a=available_idx, size=args.k - len(sample), replace=True)
        sample = np.concatenate((sample, random_ids))
    if len(sample) > args.k:
        np.random.shuffle(sample)
        sample = sample[:args.k]

    test_score = get_score2(sample, queries='test', checkpoint_version=args.checkpoint)
    test_threshold_score = get_threshold_score(sample, queries='test', checkpoint_version=args.checkpoint)
    sample_ids_db_format = ','.join([str(idx) for idx in sample])
    sample_tuples = DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({sample_ids_db_format})')
    test_diversity = get_diversity_score(sample_tuples, args.checkpoint)
    print(f'############### Sample score: [{test_score}] ###############')
    print(f'############### Sample threshold score: [{test_threshold_score}] ###############')
    print(f'############### Sample diversity: [{test_diversity}] ###############')
    CheckpointManager.save(name=f'{args.k}_{args.lib}_kmeans_sample', content=[sample, test_score],
                           version=args.checkpoint)

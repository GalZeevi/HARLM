import argparse
import multiprocessing
import os
import time
from functools import reduce

import numpy as np
from tqdm import tqdm, trange

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score2, get_threshold_score, get_combined_score
from train_test_utils import get_test_queries

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=100, help="sample size")
parser.add_argument("--queries", type=str, default='test', help="use test queries or train queries")
parser.add_argument("--pool_size", type=int, default=6, help="how many processes to use")
parser.add_argument("--checkpoint", type=int, default=CheckpointManager.get_max_version(),
                    help="which checkpoint to use")
parser.add_argument("--diversity_coeff", type=float, default=0.5, help="Weight to give diversity in score function.")
args = parser.parse_args()
print(f"Running with following CLI args: {args}")

assert args.queries in ['test', 'train']
schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
pivot = ConfigManager.get_config('queryConfig.pivot')
DataAccess()
table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
results = get_test_queries(args.checkpoint) if args.queries == 'test' else [np.arange(table_size)]
TARGET_CHUNK_SIZE = 1000


def get_tuples_from_ids(ids):
    ids_db_format = ','.join([str(i) for i in ids])
    return DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({ids_db_format})')


def worker_task(tuples, sample, score):
    added_once = False
    new_tuple = None
    sample_ids = [tup[pivot] for tup in sample]
    for t in tuples:
        t_id = t[pivot]
        if t_id in sample_ids:
            continue
        if not added_once or \
                get_combined_score(sample[:-1] + [t], queries=args.queries, alpha=(1 - args.diversity_coeff),
                                   checkpoint_version=args.checkpoint) > score:
            if not added_once:
                sample += [t]
                sample_ids += [t_id]
                added_once = True
            else:
                sample = sample[:-1] + [t]
                sample_ids = sample_ids[:-1] + [t_id]

            new_tuple = t
            # score = get_score2(sample, queries=args.queries, checkpoint_version=args.checkpoint)
            score = get_combined_score(sample, queries=args.queries,
                                       alpha=(1 - args.diversity_coeff),
                                       checkpoint_version=args.checkpoint)
    return new_tuple, score, os.getpid()


def get_sample():
    sample = []
    score = 0.
    print(f'Preparing tuples to process...')
    tuple_ids = reduce(np.union1d, tuple(results))
    print(f'Starting greedy sampler, {len(tuple_ids)} tuples to process')
    print('Initialising pool')
    start = time.time()
    pool = multiprocessing.Pool(args.pool_size)
    print(f'Initialising pool took: {round(time.time() - start, 2)} sec')

    for _ in trange(args.k):
        new_tuple = None
        chunks = [get_tuples_from_ids(chunk) for chunk in
                  np.array_split(tuple_ids, np.ceil(len(tuple_ids) / TARGET_CHUNK_SIZE)) if
                  len(chunk) > 0]
        print(f'Starting to work on {len(chunks)} tasks with {args.pool_size} workers', flush=True)
        worker_args = [(chunk, sample, score) for chunk in chunks]
        pool_res = pool.starmap(worker_task, worker_args)
        for worker_new_tuple, worker_score, worker_pid in pool_res:
            if score < worker_score:
                tqdm.write(
                    f'Sample size: {len(sample)}/{args.k}, \n' +
                    f'current test score: {get_score2([t[pivot] for t in sample], queries="test", checkpoint_version=args.checkpoint)}\n' +
                    f'current test_threshold score: '
                    f'{get_threshold_score([t[pivot] for t in sample], queries="test", checkpoint_version=args.checkpoint)}\n' +
                    f', worker: {worker_pid}'
                )
                sample_ids = [tup[pivot] for tup in sample]
                CheckpointManager.save(f'{args.k}-{args.queries}_greedy_sample', [sample_ids, score],
                                       version=args.checkpoint, verbose=False)
                score = worker_score
                new_tuple = worker_new_tuple
        sample += [new_tuple]

    pool.close()

    return sample, score


if __name__ == '__main__':
    get_sample()

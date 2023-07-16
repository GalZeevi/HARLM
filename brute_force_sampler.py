import itertools
import multiprocessing as mp
import argparse

import numpy as np
from functools import reduce
from tqdm import tqdm

from score_calculator import get_score2
from train_test_utils import get_test_queries
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pool", type=int, default=os.cpu_count() - 1, help="how many workers to use for the pool")
parser.add_argument("--k", type=int, default=100, help="sample size")
parser.add_argument("--checkpoint", type=int, default=CheckpointManager.get_max_version(),
                    help="which checkpoint to use")
args = parser.parse_args()

test_queries = get_test_queries(checkpoint_version=args.checkpoint)
schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
DataAccess()
table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
test_results = get_test_queries(args.checkpoint)
union_results = reduce(np.union1d, tuple(test_results))
print(f'Got n={len(union_results)}, k={args.k}')


def get_sample():
    # Set up the Pool using a context manager.
    # This relieves you of the hassle of join/close.
    max_score = 0.
    best_sample = None
    pbar = tqdm()
    with mp.Pool(args.pool) as p:
        # Just iterate over the results as they come in.
        # No need to check for empty queues, etc.
        for sample, score in p.imap(worker, itertools.combinations(union_results, args.k)):
            pbar.update(1)
            if max_score < score:
                max_score = score
                best_sample = sample
                tqdm.write(f'max score: {max_score}')
                CheckpointManager.save(f'{args.k}-brute_force_sample', [best_sample, max_score],
                                       version=args.checkpoint, verbose=False)
    pbar.close()
    return best_sample, max_score


def worker(sample):
    # A single-argument worker function.
    # If you need multiple args, bundle them in tuple/dict/etc.
    return sample, get_score2(sample, queries=test_queries)


if __name__ == "__main__":
    get_sample()

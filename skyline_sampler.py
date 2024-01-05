import argparse
import multiprocessing
import os
import time
from tqdm import tqdm, trange
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing
from score_calculator import get_score2, get_threshold_score, get_diversity_score

MAX_TUPLES_TO_SELECT = 10000


def get_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="The Checkpoint to use."
    )

    parser.add_argument(
        "--k", type=int, default=100, help="Sample size."
    )

    parser.add_argument(
        "--pool_size", type=int, default=6, help="how many processes to use"
    )

    cli_args = parser.parse_args()
    print(f"Running with following args: {cli_args}")
    return cli_args


args = get_args()

schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
pivot = ConfigManager.get_config('queryConfig.pivot')


def select_tuples(ids):
    tuples = []
    chunks = [chunk for chunk in np.array_split(ids, np.ceil(len(ids) / MAX_TUPLES_TO_SELECT)) if
              len(chunk) > 0]
    for chunk in chunks:
        ids_db_format = ','.join([str(idx) for idx in chunk])
        tuples += DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({ids_db_format})')

    return tuples


def get_single_skyline(df, column_names, take_max=True):
    column_names = [name for name in column_names if name != pivot]  # remove pivot
    Preprocessing.init(args.checkpoint)

    start = time.time()
    for i, col_name in enumerate(column_names):
        column = Preprocessing.columns_repo.column_by_name(col_name)
        if column.is_pivot:
            continue
        elif column.is_categorical is False:  # numerical column
            if take_max:
                sky_val = df[col_name].max()
            else:
                sky_val = df[col_name].min()
        else:  # categorical column
            if take_max:
                sky_val = df[col_name].value_counts().idxmax()
            else:
                sky_val = df[col_name].value_counts().idxmin()
        df = df[df[col_name] == sky_val]
        # print(f'Worker {os.getpid()} finish column no. {i}/{len(column_names)}', flush=True)

    skyline_row = df.to_dict(orient='records')[0]
    print(f'Worker {os.getpid()} finished finding skyline values in {round(time.time() - start, 2)} seconds',
          flush=True)
    return skyline_row


def get_skyline(ids, column_names):
    return [get_single_skyline(ids, column_names, True),
            get_single_skyline(ids, column_names, False)]


def get_scores(sample_ids, sample_tuples):
    test_score = get_score2(sample_ids, queries='test', checkpoint_version=args.checkpoint)
    test_threshold_score = get_threshold_score(sample_ids, queries='test', checkpoint_version=args.checkpoint)
    test_diversity = get_diversity_score(sample_tuples, args.checkpoint)
    return test_score, test_threshold_score, test_diversity


def get_train_queries(checkpoint_version=CheckpointManager.get_max_version(), validation_size=0):
    results = []
    path = f'{CheckpointManager.basePath}/{checkpoint_version}'
    results_files = [f for f in listdir(path) if isfile(join(path, f)) and 'queries_' in f]
    results_files.sort(key=lambda name: int(re.findall(r'\d+', name)[0]))

    test_size = ConfigManager.get_config('samplerConfig.testSize')
    for file in results_files:
        interval = [int(r) for r in re.findall(r'\d+', file)]
        if interval[1] < test_size:
            # file belongs entirely to test queries
            continue
        elif interval[0] <= test_size <= interval[1]:
            # file belongs to both train and test
            # results += CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)[(test_size - interval[0]):]
            data = CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)[(test_size - interval[0]):]
            results = np.concatenate((data, results))
        else:
            # results += CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)
            data = CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)
            results = np.concatenate((data, results))

    if validation_size > 0:
        return results[validation_size:], results[:validation_size]  # return train_set, validation_set
    return results  # return train_set


def get_sample():
    print('Starting skyline_sampler.main()!')
    OUTPUT_DIR = 'skyline'
    COMPLETE_OUTPUT_DIR = f'{CheckpointManager.basePath}/{args.checkpoint}/{OUTPUT_DIR}'
    if not os.path.exists(COMPLETE_OUTPUT_DIR):
        os.makedirs(COMPLETE_OUTPUT_DIR)

    Preprocessing.init(args.checkpoint)
    print('Preparing data...')
    train_results = get_train_queries(checkpoint_version=args.checkpoint)
    # result_ids = [*set().union(*train_results)]
    # print(f'There are {len(result_ids)} tuples in results')

    print('Initialising pool')
    start = time.time()
    pool = multiprocessing.Pool(args.pool_size)
    print(f'Initialising pool took: {round(time.time() - start, 2)} sec')

    sample_tuples = []
    sample_ids = []
    columns = [*Preprocessing.columns_repo.get_all_columns().keys()]

    tasks_per_q = int(max(np.ceil(args.k / (len(train_results))), 1))
    worker_args = []
    for i, q in tqdm(enumerate(train_results)):

        ckpt_res = CheckpointManager.load(f'{OUTPUT_DIR}/query_{i}_df', version=args.checkpoint)
        if ckpt_res is not None:
            df = ckpt_res
        else:
            start = time.time()
            tuples = DataAccess.select(q.replace(f'SELECT {pivot}', 'SELECT *'))
            df = pd.DataFrame(tuples)
            CheckpointManager.save(f'{OUTPUT_DIR}/query_{i}_df', df, version=args.checkpoint)
            print(f'Finished loading query to pandas in {round(time.time() - start, 2)} seconds', flush=True)

        for _ in range(tasks_per_q):
            np.random.shuffle(columns)
            worker_args.append((df.copy(), columns))
    print(f'Starting to work on {len(worker_args)} tasks with pool of {args.pool_size} workers', flush=True)

    pool_res = pool.starmap_async(get_skyline, worker_args)
    for res in pool_res.get():
        if len(res) != 2:
            print(f'WARNING! expected 2 skyline rows and got: {len(res)}', flush=True)
        for skyline_tup in res:
            if skyline_tup[pivot] not in sample_ids:
                sample_tuples.append(skyline_tup)
                sample_ids.append(skyline_tup[pivot])
        # print(f'Current sample length: {len(sample_ids)}', flush=True)

        test_score = get_score2(sample_ids, queries="test", checkpoint_version=args.checkpoint)
        print(
            f'Sample size: {len(sample_ids)}/{args.k}, \n' +
            f'current test score: {test_score}\n' +
            f'current test_threshold score: '
            f'{get_threshold_score(sample_ids, queries="test", checkpoint_version=args.checkpoint)}\n',
            flush=True
        )
        CheckpointManager.save(name=f'{args.k}_skyline_sample', content=[sample_ids, test_score],
                               version=args.checkpoint)

    if len(sample_ids) > args.k:
        test_score_avg = 0.
        test_threshold_score_avg = 0.
        diversity_avg = 0.
        for _ in trange(100):
            np.random.shuffle(sample_tuples)
            truncated_sample_tuples = sample_tuples[:args.k]
            truncated_sample_ids = [tup[pivot] for tup in truncated_sample_tuples]
            res = get_scores(truncated_sample_ids, truncated_sample_tuples)
            test_score_avg += res[0]
            test_threshold_score_avg += res[1]
            diversity_avg += res[2]
        test_score_avg /= 100
        test_threshold_score_avg /= 100
        diversity_avg /= 100

        print(f'############### Sample score: [{test_score_avg}] ###############')
        print(f'############### Sample threshold score: [{test_threshold_score_avg}] ###############')
        print(f'############### Sample diversity: [{diversity_avg}] ###############')

        CheckpointManager.save(name=f'{args.k}_skyline_sample', content=[sample_ids, test_score_avg],
                               version=args.checkpoint)
        print()
        return sample_ids

    elif len(sample_ids) < args.k:
        print('Calculating table_size')
        table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
        print(f'table_size: {table_size}')
        available_idx = np.setdiff1d(np.arange(table_size), sample_ids)
        random_ids = np.random.choice(a=available_idx, size=args.k - len(sample_ids), replace=True)
        random_tuples = select_tuples(random_ids)
        sample_ids = np.concatenate((sample_ids, random_ids))
        sample_tuples = np.concatenate((sample_tuples, random_tuples))

    res = get_scores(sample_ids, sample_tuples)
    print(f'############### Sample score: [{res[0]}] ###############')
    print(f'############### Sample threshold score: [{res[1]}] ###############')
    print(f'############### Sample diversity: [{res[2]}] ###############')

    CheckpointManager.save(name=f'{args.k}_skyline_sample', content=[sample_ids, res[0]],
                           version=args.checkpoint)

    return sample_ids


if __name__ == '__main__':
    get_sample()

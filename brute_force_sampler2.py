import argparse
import datetime
import os
import time
from multiprocessing import Process
from os import listdir
from os.path import isfile, join

import numpy as np

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_combined_score, get_score2, get_threshold_score, get_diversity_score

parser = argparse.ArgumentParser()
parser.add_argument("--pool", type=int, default=6, help="how many workers to use for the pool")
parser.add_argument("--k", type=int, default=100, help="sample size")
parser.add_argument("--checkpoint", type=int, default=CheckpointManager.get_max_version(),
                    help="which checkpoint to use")
parser.add_argument("--diversity_coeff", type=float, default=0., help="Weight to give diversity in score function.")
parser.add_argument("--horizon_time", type=int, default=0.1,
                    help="How much many hours to run each process")
args = parser.parse_args()
print(f"Running with following args: {args}")

schema = ConfigManager.get_config('queryConfig.schema')
table = ConfigManager.get_config('queryConfig.table')
pivot = ConfigManager.get_config('queryConfig.pivot')
DataAccess()
print(f'Calculating table size...')
table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
print('Done.')
DataAccess.disconnect()


def process_func(output_dir):
    start = time.time()
    best_score = 0.
    best_sample = []
    elapsed = 0.
    i = 0
    while elapsed < args.horizon_time * 3600:
        if i % 100 == 0:
            # elapsed_str = datetime.datetime.utcfromtimestamp(elapsed).strftime('%H:%M:%S')
            print(f'Pid: {os.getpid()}, Time elapsed: {round(elapsed / 3600, 3)}/{args.horizon_time} Hrs')

        sample = np.random.choice(a=table_size, size=args.k, replace=False)
        # sample_ids_db_format = ','.join([str(idx) for idx in sample])
        # sample_tuples = DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({sample_ids_db_format})')
        # sample_score = get_combined_score(sample_tuples, queries='test', alpha=(1 - args.diversity_coeff),
        #                                   checkpoint_version=args.checkpoint)
        sample_score = get_score2(sample, queries='test', checkpoint_version=args.checkpoint)
        if best_score < sample_score:
            best_sample = sample
            best_score = sample_score
            CheckpointManager.save(f'{output_dir}/{args.k}-{os.getpid()}_brute_force_sample', [best_sample, best_score],
                                   version=args.checkpoint)
        elapsed = time.time() - start
        i += 1

    return best_sample


def read_best_from_folder(folder_name, checkpoint):
    full_path = f'checkpoints/{checkpoint}/{folder_name}'
    ckpt_files = [f.split('.pkl')[0] for f in listdir(full_path) if isfile(join(full_path, f)) and '.pkl' in f]
    ckpt_content = [CheckpointManager.load(f'{folder_name}/{f}', version=checkpoint) for f in ckpt_files]

    if len(ckpt_content) > 0:
        best_result = sorted(ckpt_content, key=lambda x: x[1], reverse=True)[0]
        sample, score = best_result[0], best_result[1]

        test_score = get_score2(sample, queries='test', checkpoint_version=checkpoint)
        test_threshold_score = get_threshold_score(sample, queries='test', checkpoint_version=checkpoint)
        print(f'############### Sample score: [{test_score}] ###############')
        print(f'############### Sample threshold score: [{test_threshold_score}] ###############')
        CheckpointManager.save(name=f'{args.k}_brute_force_sample', content=best_result, version=checkpoint)


if __name__ == "__main__":
    OUTPUT_DIR = f'{args.k}_brute_force_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    COMPLETE_OUTPUT_DIR = f'{CheckpointManager.basePath}/{args.checkpoint}/{OUTPUT_DIR}'
    if not os.path.exists(COMPLETE_OUTPUT_DIR):
        os.makedirs(COMPLETE_OUTPUT_DIR)

    procs = []
    for i in range(args.pool):
        proc = Process(target=process_func, args=(OUTPUT_DIR,))  # instantiating without any argument
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    ckpt_files = [f.split('.pkl')[0] for f in listdir(COMPLETE_OUTPUT_DIR) if
                  isfile(join(COMPLETE_OUTPUT_DIR, f)) and '.pkl' in f]
    ckpt_content = [CheckpointManager.load(f'{OUTPUT_DIR}/{f}', version=args.checkpoint) for f in ckpt_files]

    if len(ckpt_content) > 0:
        best_result = sorted(ckpt_content, key=lambda x: x[1], reverse=True)[0]
        sample, score = best_result[0], best_result[1]

        test_score = get_score2(sample, queries='test', checkpoint_version=args.checkpoint)
        test_threshold_score = get_threshold_score(sample, queries='test', checkpoint_version=args.checkpoint)
        sample_ids_db_format = ','.join([str(idx) for idx in sample])
        # sample_tuples = DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({sample_ids_db_format})')
        # test_diversity = get_diversity_score(sample_tuples, args.checkpoint)
        print(f'############### Sample score: [{test_score}] ###############')
        print(f'############### Sample threshold score: [{test_threshold_score}] ###############')
        # print(f'############### Sample diversity: [{test_diversity}] ###############')
        CheckpointManager.save(name=f'{args.k}_brute_force_sample', content=best_result, version=args.checkpoint)

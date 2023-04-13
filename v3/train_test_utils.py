import random

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from os import listdir
from os.path import isfile, join
import re


def get_test_queries(checkpoint_version=CheckpointManager.get_max_version()):
    results_read = 0
    results = []
    path = f'{CheckpointManager.basePath}/{checkpoint_version}'
    results_files = [f for f in listdir(path) if isfile(join(path, f)) and 'results' in f]
    results_files.sort(key=lambda name: int(re.findall(r'\d+', name)[0]))

    file_num = 0
    test_size = ConfigManager.get_config('samplerConfig.testSize')
    while results_read < test_size:
        results += CheckpointManager.load(results_files[file_num].replace('.pkl', ''), checkpoint_version)
        interval = [int(r) for r in re.findall(r'\d+', results_files[file_num])]
        results_read += (interval[1] - interval[0])
        file_num += 1

    return results[:test_size]


def get_train_queries(checkpoint_version=CheckpointManager.get_max_version(), validation_size=0):
    results = []
    path = f'{CheckpointManager.basePath}/{checkpoint_version}'
    results_files = [f for f in listdir(path) if isfile(join(path, f)) and 'results' in f]
    results_files.sort(key=lambda name: int(re.findall(r'\d+', name)[0]))

    test_size = ConfigManager.get_config('samplerConfig.testSize')
    for file in results_files:
        interval = [int(r) for r in re.findall(r'\d+', file)]
        if interval[1] < test_size:
            # file belongs entirely to test queries
            continue
        elif interval[0] <= test_size <= interval[1]:
            # file belongs to both train and test
            results += CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)[(test_size - interval[0]):]
        else:
            results += CheckpointManager.load(file.replace('.pkl', ''), checkpoint_version)

    # random.shuffle(results)
    if validation_size > 0:
        return results[validation_size:], results[:validation_size]  # return train_set, validation_set
    return results  # return train_set


if __name__ == '__main__':
    print(len(get_train_queries()))

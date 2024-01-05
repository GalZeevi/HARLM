import argparse

from sklearn.metrics import pairwise_distances
import numpy as np

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing


def sample2tuples(sample_file, checkpoint):
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    sample_ids = CheckpointManager.load(sample_file, checkpoint)[0]
    ids_db_format = ','.join([str(i) for i in sample_ids])
    return DataAccess.select(f'SELECT * FROM {schema}.{table} WHERE {pivot} IN ({ids_db_format})')


def diversity(tuples, checkpoint):
    Preprocessing.init(checkpoint)
    X = Preprocessing.tuples2numpy(tuples)
    jaccard = pairwise_distances(X, metric="jaccard", n_jobs=-1)
    diversity = np.max(np.triu(jaccard, 1))
    return diversity


if __name__ == '__main__':
    # local test
    # x = DataAccess.select('SELECT * from mas.mas_full_data2 ORDER BY RAND() LIMIT 10')
    # print(diversity(x, 17))

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default=None, help="sample path")
    parser.add_argument("--checkpoint", type=int, default=CheckpointManager.get_max_version(),
                        help="which checkpoint to use")
    args = parser.parse_args()
    if args.sample and args.checkpoint:
        print(f"Running with following args: {args}")
        tuples = sample2tuples(args.sample, args.checkpoint)
        print(f'Read [{len(tuples)}] tuples from pkl')
        print(diversity(tuples, args.checkpoint))

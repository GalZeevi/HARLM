from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from checkpoint_manager_v3 import CheckpointManager
import numpy as np

MAX_TUPLES_TO_SELECT = 500
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

        print(f'Read {len(tuples)} tuples so far')

    return tuples


if __name__ == "__main__":
    ids = CheckpointManager.load('1000_CHOOSE_K_PPO_2023-06-28-20-06-03/sample', 8)[0]
    CheckpointManager.save('demo_sample2', select_tuples(ids), 8)

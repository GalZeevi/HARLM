from decimal import Decimal
import math
import numpy as np

from pathos.multiprocessing import _ProcessPool
from config_manager_v2 import ConfigManager
from data_access_v2 import DataAccess
from codetiming import Timer

from checkpoint_manager_v2 import CheckpointManager

BATCH_SIZE = 100
TIMER_NAME = 'distance_matrix_timer'
CHECKPOINT_NAME = 'distance_matrix'


def _null_safe_subtraction(x, y):
    x_safe = 0 if (x is None or math.isnan(x)) else x
    y_safe = 0 if (y is None or math.isnan(y)) else y
    return x_safe - y_safe


class TupleDistanceCalculator:

    def __init__(self):
        self.table = ConfigManager.get_config('queriesConfig.table')
        self.schema = ConfigManager.get_config('queriesConfig.schema')
        self.pivot = ConfigManager.get_config('queriesConfig.pivot')
        DataAccess()
        self.numerical_cols_dict = TupleDistanceCalculator._build_numerical_cols_dict(self.table, self.schema,
                                                                                      self.pivot)
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.chunksize = ConfigManager.get_config('cpuConfig.chunk_size')

    @staticmethod
    def _build_numerical_cols_dict(table, schema, pivot):
        numeric_data_types: list[str] = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        numerical_cols = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                           f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                           f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                           f"AND column_name <> '{pivot}'")

        # build a dict mapping col -> (max - min)
        min_max_diff = {}
        for col in numerical_cols:
            min_val = DataAccess.select_one(f'SELECT MIN({col}) AS val FROM {schema}.{table}')
            max_val = DataAccess.select_one(f'SELECT MAX({col}) AS val FROM {schema}.{table}')
            min_max_diff[col] = max_val - min_val
        return min_max_diff

    def _dist(self, t, s):
        dist = Decimal(0)
        for col in t.keys():
            if col == self.pivot:
                continue
            elif col in self.numerical_cols_dict.keys():  # numerical column
                if self.numerical_cols_dict[col] == 0:  # column is constant
                    dist += Decimal(1)
                else:
                    dist += Decimal(abs(_null_safe_subtraction(t[col], s[col])) / self.numerical_cols_dict[col])
            else:
                dist += (0 if t[col] == s[col] else 1)
        return t[self.pivot], s[self.pivot], dist / (len(t.keys()) - 1)

    def calculate_distance_matrix(self, start=0):
        table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {self.schema}.{self.table}')
        max_tuple_id = DataAccess.select_one(f'SELECT MAX({self.pivot}) as max_id FROM {self.schema}.{self.table}')
        dist_matrix = np.zeros((table_size, table_size))
        timer = Timer(name=TIMER_NAME, initial_text='============= start iteration =============')

        with _ProcessPool(self.num_workers) as pool:
            for tuple_id in range(start, table_size):
                print(f'Start tuple with id: {tuple_id}')
                timer.start()

                dist_matrix[tuple_id, tuple_id] = 0.
                first_tuple = DataAccess.select_one(
                    f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot} = {tuple_id}')

                next_id = tuple_id + 1
                while next_id <= max_tuple_id:
                    tuples = DataAccess.select(
                        f'SELECT * FROM {self.schema}.{self.table} '
                        f'WHERE {self.pivot} > {next_id} ORDER BY {self.pivot} ASC '
                        f'LIMIT {BATCH_SIZE}')

                    items = [(first_tuple, second_tuple) for second_tuple in tuples]
                    for res in pool.starmap(self._dist, items, chunksize=self.chunksize):
                        dist_matrix[res[0], res[1]] = res[2]

                    next_id += BATCH_SIZE

                print(f'Finish tuple with id: {tuple_id}')
                timer.stop()
                CheckpointManager.save(name=CHECKPOINT_NAME,
                                       content=dist_matrix)

        print(f'total time elapsed: {Timer.timers.total(TIMER_NAME):.5f} seconds')
        return dist_matrix

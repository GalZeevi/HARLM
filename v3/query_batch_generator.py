import random
from random import randint

import numpy as np
from tqdm import tqdm

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DBTypes, DataAccess


class QueryGenerator:
    def __init__(self):
        self.dbType = str.lower(ConfigManager.get_config('dbConfig.type'))
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.table = ConfigManager.get_config('queryConfig.table')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.numerical_cols = self.init_numerical_cols()
        self.numerical_vals = self.init_numerical_vals(self.numerical_cols)
        self.categorical_cols = self.init_categorical_cols()
        self.categorical_vals = self.init_categorical_vals(self.categorical_cols)

    def get_random_func(self):
        if DBTypes.IS_POSTGRESQL(self.dbType):
            return 'RANDOM'
        elif DBTypes.IS_MYSQL(self.dbType):
            return 'RAND'
        else:
            raise Exception('Unsupported db type! supported types are "postgresql" or "mysql"')

    def init_numerical_cols(self):
        numeric_data_types = ['smallint', 'integer', 'bigint', 'decimal', 'numeric', 'real', 'double precision',
                              'smallserial', 'serial', 'bigserial', 'float']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                    f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                    f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                    f"AND column_name <> '{self.pivot}'")
        return columns

    def init_categorical_cols(self):
        textual_data_types = ['character varying%%', 'varchar%%', '%%text', 'char%%', 'character%%']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in textual_data_types]
        columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                    f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                    f"AND ({' OR '.join([f'data_type LIKE {data_type}' for data_type in db_formatted_data_types])})")
        return columns

    def init_numerical_vals(self, numerical_cols):
        # build a dict mapping col -> [min, max]
        min_max_vals = {}
        for col in numerical_cols:
            min_val = DataAccess.select_one(f'SELECT MIN({col}) AS val FROM {self.schema}.{self.table}')
            max_val = DataAccess.select_one(f'SELECT MAX({col}) AS val FROM {self.schema}.{self.table}')
            min_max_vals[col] = [min_val, max_val]
        return min_max_vals

    def init_categorical_vals(self, categorical_columns, limit=None):
        # build a dict mapping col -> list of values
        vals = {}
        for col in categorical_columns:
            column_values = DataAccess.select(f"SELECT distinct_values.val AS val FROM (" +
                                              f"SELECT DISTINCT {col} AS val FROM {self.schema}.{self.table}"
                                              f") as distinct_values ORDER BY {self.get_random_func()}() "
                                              f"{'' if not limit else f'LIMIT {limit}'}")
            vals[col] = column_values
        return vals

    def get_query(self, num_of_columns):
        #  making sure no more columns than available are selected
        num_of_columns = min(num_of_columns, len(self.categorical_cols) + len(self.numerical_cols))
        chosen_columns = random.sample(self.categorical_cols + self.numerical_cols, num_of_columns)

        # table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {self.schema}.{self.table}')
        view_size = ConfigManager.get_config('samplerConfig.viewSize')
        max_result_size = int(100 * view_size)

        where_clause = []

        for col in chosen_columns:
            if col in self.categorical_cols:
                # categorical column
                frequency = random.uniform(0, 0.3) * len(self.categorical_vals[col])
                values = random.sample(self.categorical_vals[col], max(int(frequency), 1))
                values = [val.replace("'", "''") for val in values]
                db_formatted_values = " , ".join([f"\'{value}\'" for value in values])

                where_clause.append(f'{col} IN ({db_formatted_values})')

            elif col in self.numerical_cols:
                # numerical column
                choice1 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))
                choice2 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))

                lb = min(choice1, choice2)
                ub = max(choice1, choice2)

                where_clause.append(f'{col} BETWEEN {lb} AND {ub}')

        where_clause_str = ''
        if len(where_clause) > 0:
            where_clause_str = f'WHERE {" AND ".join(where_clause)}'

        return f"SELECT {self.pivot} FROM {self.schema}.{self.table} {where_clause_str} " \
               f"ORDER BY {self.get_random_func()}() LIMIT {max_result_size}"


def generate_batch():
    queries_to_generate = ConfigManager.get_config('queryConfig.numToGenerate')
    batch_size = ConfigManager.get_config('queryConfig.batchSize')
    CheckpointManager.start_new_version()
    pbar = tqdm(total=queries_to_generate)
    query_generator = QueryGenerator()

    first_query_id = 0
    while queries_to_generate > 0:
        queries = []
        results = []
        batch = [query_generator.get_query(randint(1, 3))
                 for _ in range(min(batch_size, queries_to_generate))]

        queries_generated = 0
        for query in batch:
            query_result = np.array(DataAccess.select(query))

            if len(query_result) > 0:
                queries.append(query)
                results.append(query_result)
                queries_generated += 1
                pbar.update(1)

        queries_to_generate -= queries_generated
        CheckpointManager.save(f'queries_{first_query_id}-{first_query_id + queries_generated}', queries,
                               should_print=False)
        CheckpointManager.save(f'results_{first_query_id}-{first_query_id + queries_generated}', results,
                               should_print=False)
        first_query_id += queries_generated


if __name__ == '__main__':
    generate_batch()

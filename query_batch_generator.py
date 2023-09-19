import random
from random import randint

import numpy as np
from tqdm import tqdm
import os

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
        numeric_data_types = ['smallint', 'integer', 'int', 'bigint', 'decimal', 'numeric', 'real', 'double precision',
                              'smallserial', 'serial', 'bigserial', 'float']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                    f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                    f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                    f"AND column_name <> '{self.pivot}'")
        # return columns
        return ['dep_delay', 'distance', 'arr_delay', 'year_date', 'air_time']

    def init_categorical_cols(self):
        textual_data_types = ['character varying%%', 'varchar%%', '%%text', 'char%%', 'character%%']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in textual_data_types]
        columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                    f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                    f"AND ({' OR '.join([f'data_type LIKE {data_type}' for data_type in db_formatted_data_types])})")
        # return columns
        return ['unique_carrier', 'origin', 'dest']

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

    def get_query(self, num_of_columns, agg=False):
        use_group_by = False
        if agg:
            use_group_by = np.random.binomial(n=1, p=0.7) > 0

        #  making sure no more columns than available are selected
        num_of_columns = min(num_of_columns, len(self.categorical_cols) + len(self.numerical_cols))

        chosen_columns = np.array([])
        if agg:
            chosen_columns = np.concatenate([chosen_columns, random.sample(self.numerical_cols, 1)])
            agg_col = chosen_columns[-1]
            agg_func = random.sample(['AVG', 'SUM', 'COUNT'], 1)[0]
        if use_group_by:
            chosen_columns = np.concatenate([chosen_columns, random.sample(self.categorical_cols, 1)])
            group_by_col = chosen_columns[-1]

        free_columns = [col for col in (self.categorical_cols + self.numerical_cols) if col not in list(chosen_columns)]

        if num_of_columns > 0:
            chosen_columns = np.concatenate([chosen_columns,
                                             random.sample(free_columns, num_of_columns)])

        # table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {self.schema}.{self.table}')
        view_size = ConfigManager.get_config('samplerConfig.viewSize')
        max_result_size = int(100 * view_size)

        where_clause = []

        for col in chosen_columns:
            if col in self.categorical_cols:
                # categorical column
                frequency = random.uniform(0, 0.05) * len(self.categorical_vals[col])
                values = random.sample(self.categorical_vals[col], max(int(frequency), 1))
                values = [val.replace("'", "''") for val in values]
                db_formatted_values = ", ".join([f"\'{value}\'" for value in values])

                where_clause.append(f'({col} IN ({db_formatted_values}))')

            elif col in self.numerical_cols:
                # numerical column
                choice1 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))
                choice2 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))

                lb = min(choice1, choice2)
                ub = max(choice1, choice2)

                coin = np.random.choice([0, 1, 2], size=1).item()
                if coin == 0:
                    where_clause.append(f'({col} > {lb})')
                elif coin == 1:
                    where_clause.append(f'({col} < {ub})')
                else:
                    where_clause.append(f'({col} BETWEEN {lb} AND {ub})')

        where_clause_str = ''
        if len(where_clause) > 0:
            where_clause_str = f'WHERE {" AND ".join(where_clause)}'
        if agg is False:
            # return f"SELECT {self.pivot} FROM {self.schema}.{self.table} {where_clause_str} " \
            #        f"ORDER BY {self.get_random_func()}() LIMIT {max_result_size};"
            return f"SELECT {self.pivot} FROM {self.schema}.{self.table} {where_clause_str};"
        elif use_group_by is True:
            return f"SELECT {group_by_col} AS col, {agg_func}({agg_col}) AS agg FROM {self.schema}.{self.table} " \
                   f"{where_clause_str} GROUP BY {group_by_col};"
        else:
            return f"SELECT {agg_func}({agg_col}) AS agg FROM {self.schema}.{self.table} {where_clause_str};"


def generate_batch(num_queries_to_generate, checkpoint, agg, batch_size=100, outdir='queries'):
    pbar = tqdm(total=num_queries_to_generate)
    query_generator = QueryGenerator()

    first_query_id = 0
    while num_queries_to_generate > 0:
        queries = []
        results = []
        batch = [query_generator.get_query(np.random.binomial(n=1, p=0.3) + 1, agg=agg)
                 for _ in range(min(batch_size, num_queries_to_generate))]

        queries_generated = 0
        for query in batch:
            query_result = np.array(DataAccess.select(query))

            if (query_result is not None) and (len(query_result) > 0) and (all(x is not None for x in query_result)):
                queries.append(query)
                results.append(query_result)
                queries_generated += 1
                pbar.update(1)

        num_queries_to_generate -= queries_generated
        CheckpointManager.save(f'{outdir}/queries_{first_query_id}-{first_query_id + queries_generated}',
                               queries,
                               version=checkpoint, verbose=False)
        CheckpointManager.save(f'{outdir}/results_{first_query_id}-{first_query_id + queries_generated}',
                               results,
                               version=checkpoint, verbose=False)
        first_query_id += queries_generated


def write_queries_to_file(ver, outdir, outfile_name):
    filenames = os.listdir(f'checkpoints/{ver}/{outdir}')
    with open(f'checkpoints/{ver}/{outfile_name}.sql', 'w') as outfile:
        for fname in filenames:
            if 'queries' not in fname:
                continue
            queries = CheckpointManager.load(name=f'{outdir}/' + fname.replace('.pkl', ''), version=ver)
            for q in queries:
                outfile.write(f'{q}\n')


if __name__ == '__main__':
    vers = 21
    generate_batch(100, vers, False, outdir='workload')
    # write_queries_to_file(vers, outdir='queries', outfile_name='flights_aqp_queries')
    # q = QueryGenerator().get_query(0, True)
    # print(q)

import random
from random import randint
from enum import Enum

import numpy as np
from tqdm import tqdm
import os

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DBTypes, DataAccess


class Modes(Enum):
    RANDOM = 'random'
    SCIENTIFIC = 'scientific'


class QueryGenerator:
    def __init__(self, numerical_cols=None, categorical_cols=None):
        print('Start initialising QueryGenerator')
        self.dbType = str.lower(ConfigManager.get_config('dbConfig.type'))
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.table = ConfigManager.get_config('queryConfig.table')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.numerical_cols = self.init_numerical_cols() if numerical_cols is None else numerical_cols
        print(f'Numerical cols are: {self.numerical_cols}')
        self.numerical_vals = self.init_numerical_vals(self.numerical_cols)
        self.categorical_cols = self.init_categorical_cols() if categorical_cols is None else categorical_cols
        print(f'Categorical cols are: {self.categorical_cols}')
        self.categorical_vals = self.init_categorical_vals(self.categorical_cols)
        print('Finish initialising QueryGenerator')

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
        print('Initialising numerical columns data')
        min_max_vals = {}
        for col in numerical_cols:
            print(f'Initialising column {col}')
            values = DataAccess.select_one(f'SELECT MIN({col}) AS min_val, MAX({col}) AS max_val, '
                                           f'AVG({col}) AS avg_val, STDDEV({col}) as std_val '
                                           f'FROM {self.schema}.{self.table}')
            min_max_vals[col] = {'min': float(values['min_val']), 'max': float(values['max_val']),
                                 'avg': float(values['avg_val']), 'std': float(values['std_val'])}
        return min_max_vals

    def init_categorical_vals(self, categorical_columns, limit=None):
        # build a dict mapping col -> list of values
        print('Initialising categorical columns data')
        vals = {}
        for col in categorical_columns:
            print(f'Initialising column {col}')
            column_values = DataAccess.select(f"SELECT distinct_values.val AS val FROM (" +
                                              f"SELECT DISTINCT {col} AS val FROM {self.schema}.{self.table}"
                                              f") as distinct_values ORDER BY {self.get_random_func()}() "
                                              f"{'' if not limit else f'LIMIT {limit}'}")
            vals[col] = column_values
        return vals

    def get_numeric_filter(self, col_name, mode):
        class NumericOperator(Enum):
            LESS_THEN = 'LESS THEN'
            GREATER_THEN = 'GREATER THEN'
            EQUALS = 'EQUALS'

        if mode == Modes.RANDOM:
            choice1 = random.uniform(float(self.numerical_vals[col_name]['min']),
                                     float(self.numerical_vals[col_name]['max']))
            choice2 = random.uniform(float(self.numerical_vals[col_name]['min']),
                                     float(self.numerical_vals[col_name]['max']))

            lb = min(choice1, choice2)
            ub = max(choice1, choice2)

        elif mode == Modes.SCIENTIFIC:
            avg = self.numerical_vals[col_name]['avg']
            std = self.numerical_vals[col_name]['std']
            percentage = np.random.choice([0.5, 1.0, 1.5, 2], size=1).item()
            lb = avg - (percentage * std)
            ub = avg + (percentage * std)

        else:
            raise Exception(f'Unrecognized mode! {mode} is not in {[e.value for e in Modes]}')

        operator = np.random.choice([e.value for e in NumericOperator], size=1).item()
        if operator == NumericOperator.GREATER_THEN:
            return f'{col_name} > {lb}'
        elif operator == NumericOperator.LESS_THEN:
            return f'{col_name} < {ub}'
        else:
            return f'{col_name} BETWEEN {lb} AND {ub}'

    def get_query(self, num_of_columns, agg=False, mode=Modes.SCIENTIFIC):
        use_group_by = False
        group_by_col = None
        agg_func = None
        agg_col = None

        #  making sure no more columns than available are selected
        num_of_columns = min(num_of_columns, len(self.categorical_cols) + len(self.numerical_cols))

        chosen_columns = np.array([])
        if agg:
            use_group_by = np.random.binomial(n=1, p=0.7) > 0
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
                # frequency = random.uniform(0, 0.00025) * len(self.categorical_vals[col])
                frequency = random.randint(1, 5)
                values = random.sample(self.categorical_vals[col], max(int(frequency), 1))
                values = [val.replace("'", "''") for val in values]
                db_formatted_values = ", ".join([f"\'{value}\'" for value in values])

                where_clause.append(f'{col} IN ({db_formatted_values})')

            elif col in self.numerical_cols:
                # numerical column
                where_clause.append(self.get_numeric_filter(col, mode))

            else:
                raise Exception('Column not categorical and not numerical!')

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


def generate_batch(num_queries_to_generate,
                   checkpoint,
                   num_cols=None,
                   categ_cols=None,
                   agg=False,
                   batch_size=100,
                   mode=Modes.SCIENTIFIC,
                   outdir='workload'):
    query_generator = QueryGenerator(numerical_cols=num_cols, categorical_cols=categ_cols)

    first_query_id = 0
    pbar = tqdm(total=num_queries_to_generate)
    while num_queries_to_generate > 0:
        queries = []
        results = []
        batch = [query_generator.get_query(max(np.random.binomial(n=1, p=0.3) + 1, 2), agg=agg, mode=mode)
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
        out_path = outdir + "/" if len(outdir) > 0 else ''
        CheckpointManager.save(f'{out_path}queries_{first_query_id}-{first_query_id + queries_generated}',
                               queries,
                               version=checkpoint, verbose=False)
        CheckpointManager.save(f'{out_path}results_{first_query_id}-{first_query_id + queries_generated}',
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
    ### flights ###
    checkpoint = 44
    generate_batch(200,
                   checkpoint,
                   num_cols=['YEAR_DATE'],
                   categ_cols=['UNIQUE_CARRIER'],
                   agg=False,
                   outdir='workload')
    write_queries_to_file(checkpoint, outdir='workload', outfile_name='flights_queries')

    ### instacart ###
    # checkpoint = 37
    # generate_batch(40,
    #                checkpoint,
    #                num_cols=[],
    #                categ_cols=['product_name', 'aisle', 'department'],
    #                agg=False,
    #                outdir='workload')
    # write_queries_to_file(checkpoint, outdir='workload', outfile_name='instacart_queries')

    # g = QueryGenerator(numerical_cols=['publication$citation_count', 'publication$importance'],
    #                    categorical_cols=['domain$name', 'CAST(publication$year AS CHAR(50))'])
    # q = g.get_query(3, False, Modes.SCIENTIFIC)
    # print(q)

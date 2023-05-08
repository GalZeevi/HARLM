import numpy as np
import functools
from typing import Dict
from collections import namedtuple
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from tqdm import tqdm

Column = namedtuple('Column', ['is_pivot', 'col_num', 'dim', 'is_categorical', 'encodings',
                               'max_val', 'min_val', 'none_substitute'])


# TODO: we only support MYSQL for now
class ColumnsRepo:

    def __init__(self, columns: Dict[str, Column]):
        self.columns: Dict[str, Column] = columns

    def column_by_name(self, col_name):
        return self.columns[col_name]

    def column_by_num(self, col_num):
        col = [column for column in self.columns.values() if column.col_num == col_num]
        if len(col) == 0:
            raise Exception(f'Column with num: {col_num} not found!')
        return col[0]

    def get_pivot_column(self):
        col = [column for column in self.columns.values() if column.is_pivot]
        if len(col) == 0:
            raise Exception(f'Pivot column not found!')
        return col[0]

    def get_all_columns(self):
        return self.columns

    def __str__(self):
        return str(self.columns)


NULL_VALUE = 'NIL'


class Preprocessing:
    columns_repo = None

    def __init__(self):
        pass

    @staticmethod
    def validate_init(f):
        @functools.wraps(f)
        def decor(*args, **kwargs):
            if Preprocessing.columns_repo is None:
                raise Exception('Preprocessing has not been initialized! call Preprocessing.init() first')
            res = f(*args, **kwargs)
            return res

        return decor

    @staticmethod
    def init(version=CheckpointManager.get_max_version()):
        if Preprocessing.columns_repo is not None:
            return  # already initialised

        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        pivot = ConfigManager.get_config('queryConfig.pivot')

        cached_column_repo = CheckpointManager.load(f'{schema}.{table}_preprocessing_columns', version)
        if cached_column_repo is not None:
            Preprocessing.columns_repo = cached_column_repo
            return

        print('Start initialising Preprocessing column data')
        columns = {}
        # needs to init ourselves
        columns_data = DataAccess.select(
            f"SELECT column_name AS col_name,  "
            f"data_type NOT IN "
            f"(\'smallint\', \'integer\', \'bigint\', \'decimal\', "
            f"\'numeric\', \'real\', \'double precision\', \'smallserial\', "
            f"\'serial\', \'bigserial\', \'int\') AS is_categorical, "
            f"data_type AS type "
            f"FROM information_schema.columns " +
            f"WHERE table_schema='{schema}' AND table_name='{table}'")

        column_names = sorted([data['col_name'] for data in columns_data])
        column_names_without_pivot = [name for name in column_names if name != pivot]

        for column_data in tqdm(columns_data):
            column_name = column_data['col_name']
            is_categorical = True if column_data['is_categorical'] > 0 else False
            data_type = column_data['type']
            col_num = column_names.index(column_name)
            is_pivot = (column_name == pivot)
            dim = -1 if is_pivot else column_names_without_pivot.index(column_name)

            column_name_db_format = f'`{column_name}`' if ' ' in column_name else column_name
            if is_categorical:
                if data_type in ['date', 'datetime', 'timestamp']:
                    uniq_values = DataAccess.select(
                        f'SELECT DISTINCT {column_name_db_format} as val FROM {schema}.{table} ORDER BY val')
                    uniq_values = [NULL_VALUE if val is None else val for val in uniq_values]
                elif data_type in ['varchar', 'char', 'text', 'binary', 'varbinary']:
                    uniq_values = DataAccess.select(
                        f'SELECT DISTINCT BINARY {column_name_db_format} as val FROM {schema}.{table} ORDER BY val')
                    uniq_values = [NULL_VALUE.encode() if val is None else val for val in uniq_values]
                    uniq_values = [b.decode() for b in uniq_values]
                else:
                    raise Exception(f'Unsupported column type! column: {column_name}, type: {data_type}')
                encoding = Preprocessing.__create_encoding_dict__(uniq_values)
                min_value = np.min([*encoding.values()])
                max_value = np.max([*encoding.values()])
            else:
                encoding = None
                min_value = float(DataAccess.select_one(
                    f'SELECT COALESCE(MIN({column_name_db_format}), 0) as val FROM {schema}.{table}'))
                max_value = float(DataAccess.select_one(
                    f'SELECT COALESCE(MAX({column_name_db_format}), 0) as val FROM {schema}.{table}'))

            columns[column_name] = Column(is_pivot, col_num, dim, is_categorical, encoding, max_value, min_value,
                                          NULL_VALUE if is_categorical else min_value - 1)

        Preprocessing.columns_repo = ColumnsRepo(columns)
        CheckpointManager.save(f'{schema}.{table}_preprocessing_columns', Preprocessing.columns_repo, version)

    @staticmethod
    def __create_encoding_dict__(string_values):
        values, codes = np.unique(string_values, return_inverse=True)
        return {value: code for code, value in zip(codes, values)}

    @staticmethod
    def encode_column(tuples, col_num):
        column = Preprocessing.columns_repo.column_by_num(col_num)
        np_col = tuples[:, col_num]

        if column.is_categorical:  # Categorical column
            # in order to not search the entire thing
            reduced_mapping = {key: val for key, val in column.encodings.items() if key in np_col}

            for key in reduced_mapping.keys():
                np_col[np.where(np_col == key)] = reduced_mapping[key]

        np_col = np_col.astype(float)
        tuples[:, col_num] = np_col
        return tuples

    @staticmethod
    def replace_none(tup):
        for col_name, value in tup.items():
            if value is None:
                column: Column = Preprocessing.columns_repo.column_by_name(col_name)
                tup[col_name] = column.none_substitute
        return tup

    @staticmethod
    def remove_pivot_from_numpy(a):
        return np.delete(a, Preprocessing.columns_repo.get_pivot_column().col_num, 1)

    @staticmethod
    def tuples2numpy(tuples_list):
        tuples_list = [Preprocessing.replace_none(tup) for tup in tuples_list]
        tuples_sorted_by_cols = [sorted([*tup.items()], key=lambda pair: pair[0]) for tup in tuples_list]
        tuples_values_only = [[col_and_value[1] for col_and_value in tup] for tup in
                              tuples_sorted_by_cols]
        tuples_as_numpy_not_encoded = np.array(tuples_values_only)
        encoded_tuples = np.apply_over_axes(Preprocessing.encode_column, tuples_as_numpy_not_encoded,
                                            range(tuples_as_numpy_not_encoded.shape[1]))
        encoded_tuples = Preprocessing.remove_pivot_from_numpy(encoded_tuples)
        try:
            return encoded_tuples.astype(float)
        except Exception as e:
            raise Exception(f'problem with tuples:[{tuples_list}]')


if __name__ == '__main__':
    print('Testing preprocessing')
    try:
        Preprocessing.remove_pivot_from_numpy(np.ones(25))
    except:
        print('Exception was thrown intentionally, that\'s good!')
    Preprocessing.init()
    print(Preprocessing.columns_repo.column_by_name('_id'))
    print(Preprocessing.columns_repo.column_by_name('title$title'))
    imdb_tuples = DataAccess.select('SELECT * FROM new_imdb.join_title_companies_keyword LIMIT 10')
    print(Preprocessing.tuples2numpy(imdb_tuples))

import numpy as np
from sklearn.metrics import pairwise_distances

from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess


def create_encoding_dict(string_values):
    values, codes = np.unique(string_values, return_inverse=True)
    return {value: code for code, value in zip(codes, values)}


def project_on_axis(pairs, axis=1):
    return [pair[axis] for pair in pairs]


def tuples_to_numpy(tuples):
    return np.array([project_on_axis(sorted([*tup.items()], key=lambda pair: pair[0])) for tup in tuples])


class TupleDistanceCalculator:

    def __init__(self):
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.table = ConfigManager.get_config('queryConfig.table')
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.numerical_cols_dict = TupleDistanceCalculator._build_numerical_cols_dict(self.table,
                                                                                      self.schema,
                                                                                      self.pivot)
        self.encodings = TupleDistanceCalculator.get_encodings(self.schema,
                                                               self.table,
                                                               self.numerical_cols_dict.keys(),
                                                               self.pivot)

    @staticmethod
    def _build_numerical_cols_dict(table, schema, pivot):
        numeric_data_types: list[str] = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial', 'int']
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

    def get_cat_num_parts_from_tuple_array(self, tuple_array):
        cat_tuples_part = [{k: v for k, v in tup.items()
                            if k not in self.numerical_cols_dict.keys() and k != self.pivot} for tup in tuple_array]
        cat_np_part = tuples_to_numpy(cat_tuples_part)
        encoded_cat_np_part = np.apply_over_axes(self.encode_cat_column, cat_np_part,
                                                 range(len(cat_tuples_part[0].keys())))

        num_tuples_part = [{k: v for k, v in tup.items()
                            if k in self.numerical_cols_dict.keys()} for tup in tuple_array]
        num_np_part = tuples_to_numpy(num_tuples_part)
        normalized_num_np_part = np.apply_over_axes(self.normalize_num_column, num_np_part,
                                                    range(len(num_tuples_part[0].keys())))

        return encoded_cat_np_part, normalized_num_np_part

    def calculate_dist_matrix(self, tuple_array1, tuple_array2):
        encoded_cat_np_part1, normalized_num_np_part1 = self.get_cat_num_parts_from_tuple_array(tuple_array1)
        encoded_cat_np_part2, normalized_num_np_part2 = self.get_cat_num_parts_from_tuple_array(tuple_array2)
        num_cat_cols = len(encoded_cat_np_part1[0])

        # calculate categorical with Hamming
        hamming_pairwise = pairwise_distances(encoded_cat_np_part1, encoded_cat_np_part2, metric='hamming',
                                              n_jobs=self.num_workers) * num_cat_cols

        # calculate numerical by normalizing + l1
        l1_pairwise = pairwise_distances(normalized_num_np_part1, normalized_num_np_part2, metric='l1',
                                         n_jobs=self.num_workers)

        # return the sum
        num_cols = len(tuple_array1[0])
        distances = (hamming_pairwise + l1_pairwise) / num_cols
        return distances

    @staticmethod
    def get_encodings(schema, table, numeric_cols, pivot):
        db_formatted_numeric_cols = [f"\'{col}\'" for col in numeric_cols]
        categorical_columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                                f"WHERE table_schema='{schema}' AND table_name='{table}' "
                                                f"AND column_name NOT IN ({' , '.join(db_formatted_numeric_cols)}) "
                                                f"AND column_name <> '{pivot}'")
        return {col: create_encoding_dict(DataAccess.select(f'SELECT DISTINCT {col} as val FROM {schema}.{table}'))
                for col in categorical_columns}

    def encode_cat_column(self, tuples, col_num_when_sorted):
        col = tuples[:, col_num_when_sorted]

        col_name = sorted(self.encodings.keys())[col_num_when_sorted]
        mapping = self.encodings[col_name]

        reduced_mapping = {k: v for k, v in mapping.items() if
                           k in col}  # in order to not search the entire thing

        for key in reduced_mapping.keys():
            col[np.where(col == key)] = reduced_mapping[key]
        tuples[:, col_num_when_sorted] = col

        return tuples

    def normalize_num_column(self, tuples, col_num_when_sorted):
        col = tuples[:, col_num_when_sorted]
        col_name = sorted(self.numerical_cols_dict.keys())[col_num_when_sorted]
        if self.numerical_cols_dict[col_name] != 0:
            tuples[:, col_num_when_sorted] = col / (self.numerical_cols_dict[col_name])

        return tuples

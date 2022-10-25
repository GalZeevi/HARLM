from typing import Dict
from copy import deepcopy
import numpy as np
import math
from decimal import *
from data_access import DataAccess


def _one_over(num):
    return np.Infinity if (num is None or math.isnan(num) or num == 0) else 1 / num


def _null_safe_subtraction(x, y):
    x_safe = 0 if (x is None or math.isnan(x)) else x
    y_safe = 0 if (y is None or math.isnan(y)) else y
    return x_safe - y_safe


class SaqpParAdapter:
    def __init__(self, schema, table, index_col, queries_results, queries_weights):
        self.data_access = DataAccess()
        self.schema = schema
        self.table = table
        self.index_col = index_col
        self.numerical_cols = self.init_numerical_cols()
        self.numerical_vals = self.init_numerical_vals(self.numerical_cols)
        self.queries_results = queries_results
        self.queries_weights = queries_weights
        self.tuples = self.data_access.select(f"SELECT * FROM {self.schema}.{self.table}")

    def init_numerical_cols(self):
        numeric_data_types: list[str] = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        columns = self.data_access.select(f"SELECT column_name FROM information_schema.columns " +
                                          f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                          f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                          f"AND column_name <> '{self.index_col}'")
        return [col_obj['column_name'] for col_obj in columns]

    def init_numerical_vals(self, numerical_cols):
        # build a dict mapping col -> [min, max]
        min_max_vals = {}
        for col in numerical_cols:
            min_val = self.data_access.select_one(f'SELECT MIN({col}) AS val FROM {self.schema}.{self.table}')
            max_val = self.data_access.select_one(f'SELECT MAX({col}) AS val FROM {self.schema}.{self.table}')
            min_max_vals[col] = [min_val['val'], max_val['val']]
        return min_max_vals

    def get_cost_function(self):
        return lambda S: len(S)

    def _dist(self, t: Dict, s: Dict):
        dist = Decimal(0)
        for col in t.keys():
            if col == self.index_col:
                continue
            elif col in self.numerical_cols:  # numerical column
                dist += Decimal(abs(_null_safe_subtraction(t[col], s[col]))
                                / (self.numerical_vals[col][1] - self.numerical_vals[col][0]))
            else:
                dist += (0 if t[col] == s[col] else 1)
        return dist / (len(t.keys()) - 1)

    def _set_dist(self, t, S):  # TODO use LSH
        return min([self._dist(t, s) for s in S])

    def _tuple_weight(self, t):
        result_sets = [set(result) for result in self.queries_results]
        return sum([self.queries_weights[i] for i in range(len(self.queries_results))
                    if t[self.index_col] in result_sets[i]])

    def _tuple_loss(self, t, S):
        return self._tuple_weight(t) * self._set_dist(t, S)

    def query_result_score(self, query_over_sample, ground_truth):
        query_over_sample_score = sum([self._tuple_weight(tup)
                                       for tup in query_over_sample])
        ground_truth_score = sum([self._tuple_weight(tup)
                                  for tup in ground_truth])
        return query_over_sample_score / ground_truth_score

    def get_gain_function(self):
        # NOTE: I am implementing the gain function as stated in SAQP problem formulation
        # NOTE: This means summing over tuples not queries as is done in PAR

        # TODO I don't like selecting all the tuples - can we do this better? db function?
        return lambda S: 0 if len(S) == 0 else _one_over(sum([
            self._tuple_loss(tup, S) for tup in self.tuples
        ]))

    def get_population(self):
        return self.tuples

    def get_par_config(self):
        return [self.get_cost_function(), self.get_gain_function(), self.get_population()]

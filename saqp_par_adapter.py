import math
from decimal import *
from typing import Dict
import numpy as np
from data_access import DataAccess
from checkpoint_manager import CheckpointManager
import multiprocessing as mp
from tqdm import tqdm


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
        self.queries_results = [np.array(result) for result in queries_results]
        self.queries_weights = queries_weights
        self.weights_cache = {}
        # TODO I can't really keep selecting all the tuples - can we do this better? db function?
        self.tuples = self.data_access.select(f"SELECT * FROM {self.schema}.{self.table}")

    def init_numerical_cols(self):  # TODO duplicate code, move to dataset-utils
        numeric_data_types: list[str] = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        columns = self.data_access.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                          f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                          f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                          f"AND column_name <> '{self.index_col}'")
        return columns

    def init_numerical_vals(self, numerical_cols):  # TODO duplicate code, move to dataset-utils
        # build a dict mapping col -> [min, max]
        min_max_vals = {}
        for col in numerical_cols:
            min_val = self.data_access.select_one(f'SELECT MIN({col}) AS val FROM {self.schema}.{self.table}')
            max_val = self.data_access.select_one(f'SELECT MAX({col}) AS val FROM {self.schema}.{self.table}')
            min_max_vals[col] = [min_val, max_val]
        return min_max_vals

    def get_cost_function(self):
        # return lambda S: len(S)
        return 1  # unit cost

    def _dist(self, t: Dict, s: Dict):
        dist = Decimal(0)
        for col in t.keys():
            if col == self.index_col:
                continue
            elif col in self.numerical_cols:  # numerical column
                if self.numerical_vals[col][1] == self.numerical_vals[col][0]:  # column is constant
                    dist += Decimal(1)
                else:
                    dist += Decimal(abs(_null_safe_subtraction(t[col], s[col]))
                                    / (self.numerical_vals[col][1] - self.numerical_vals[col][0]))
            else:
                dist += (0 if t[col] == s[col] else 1)
        return dist / (len(t.keys()) - 1)

    def _set_dist(self, t, S):  # TODO use LSH
        # if S is empty we return the maximal distance which is 1
        return Decimal(1) if len(S) == 0 else min([self._dist(t, s) for s in S])

    def _tuple_weight(self, t):
        if t[self.index_col] in self.weights_cache.keys():
            return self.weights_cache[t[self.index_col]]

        # membership_mask = np.array(
        #     [np.isin(t[self.index_col], sub_array) for sub_array in self.queries_results])
        # weight = np.sum(np.array(self.queries_weights)[membership_mask])
        weight = sum([self.queries_weights[i] for i in range(len(self.queries_results))
                      if np.isin(t[self.index_col], np.array(self.queries_results[i])).item()])

        self.weights_cache[t[self.index_col]] = weight
        return weight

    def _tuple_weight_without_cache(self, t):
        # membership_mask = np.array(
        #     [np.isin(t[self.index_col], sub_array) for sub_array in self.queries_results])
        # weight = np.sum(np.array(self.queries_weights)[membership_mask])

        weight = sum([self.queries_weights[i] for i in range(len(self.queries_results))
                      if np.isin(t[self.index_col], np.array(self.queries_results[i])).item()])
        return t[self.index_col], weight

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

        # weights_sum = sum([self._tuple_weight(t) for t in self.tuples])

        self.calculate_weights_parallel(num_workers=4, chunk_size=75)  # TODO MAGIC NUMBERS!! at least use config...
        weights_sum = sum(self.weights_cache.values())

        def gain_v2(S):
            if len(S) == 0:
                return 0
            # gain = sum(pool(
            #     delayed(lambda r: self._tuple_weight(r) * (1 - self._set_dist(r, S)))(t) for t in
            #     self.tuples)) / weights_sum
            gain = sum([self.weights_cache[t[self.index_col]] * (1 - self._set_dist(t, S)) for t in
                        self.tuples]) / weights_sum
            return gain

        return gain_v2
        # return lambda S: 0 if len(S) == 0 else \
        #     sum([self._tuple_weight(t) * (1 - self._set_dist(t, S)) for t in self.tuples]) \
        #     / weights_sum
        # return lambda S: 0 if len(S) == 0 else _one_over(sum([
        #     self._tuple_loss(tup, S) for tup in self.tuples
        # ]))

    def get_population(self):  # TODO: not wise to keep all the population in memory, change this when scaling up
        return self.tuples

    def get_par_config(self):
        return [self.get_cost_function(), self.get_gain_function(), self.get_population()]

    def calculate_weights_parallel(self, num_workers, chunk_size):
        weights = []
        with mp.Pool(num_workers) as pool:
            weights_iterator = tqdm(pool.imap(self._tuple_weight_without_cache, self.tuples, chunk_size))
            for res in weights_iterator:
                weights.append(res)
        print('finished weights calculation!')
        self.weights_cache = dict(weights)
        CheckpointManager.save('weights', self.weights_cache)

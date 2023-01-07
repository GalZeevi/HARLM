from config_manager_v2 import ConfigManager
from data_access_v2 import DataAccess
from graphs_manager_v2 import GraphsManager
from query_generator_v2 import QueryGenerator
from checkpoint_manager_v2 import CheckpointManager
import numpy as np
from random import randint
import gc
from codetiming import Timer

TIMER_NAME = 'weights_timer'
GRAPH_NAME = 'weights_error'
WEIGHTS_CHECKPOINT_NAME = 'weights'
QUERIES_CHECKPOINT_NAME = 'queries'
MAX = 'max'


def print_total_time():
    print(f'total time elapsed: {Timer.timers.total(TIMER_NAME):.5f} seconds')


def normalize(a):
    return a / np.sum(a)


class TupleWeightCalculator:

    def __init__(self, print_warn=False, convergence_metric=MAX):
        self.table = ConfigManager.get_config('queriesConfig.table')
        self.schema = ConfigManager.get_config('queriesConfig.schema')
        self.pivot = ConfigManager.get_config('queriesConfig.pivot')
        self.query_generator = QueryGenerator(self.schema, self.table, self.pivot)
        self.batch_size = ConfigManager.get_config('queriesConfig.batchSize')
        self.print_warn_log = print_warn
        self.convergence_metric = convergence_metric

    def get_weights(self, max_iter, epsilon, start_iter=0, checkpoint_version=None):
        GraphsManager.clear()

        table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {self.schema}.{self.table}')
        executed_queries = [] if checkpoint_version is None else \
            CheckpointManager.load(QUERIES_CHECKPOINT_NAME, checkpoint_version)
        prev_weights = np.zeros(table_size) if checkpoint_version is None \
            else CheckpointManager.load(WEIGHTS_CHECKPOINT_NAME, checkpoint_version, numpy=True)
        weights = prev_weights

        timer = Timer(name=TIMER_NAME, initial_text='============= start iteration =============')
        diff_to_prev = -1

        for i in range(start_iter, max_iter):

            timer.start()
            returned_results = 0
            while returned_results < self.batch_size:
                query_batch = self.get_batch(self.batch_size - returned_results)

                for query in query_batch:
                    # Query should return only a single field - the pivot that numbers tuples consecutively
                    query_result = np.array(DataAccess.select(query))

                    if len(query_result) == 0:
                        self.print_warn_log and print('query returned no results! ignoring query')
                    elif len(query_result) >= .6 * table_size:
                        self.print_warn_log and print('query returned too many results! ignoring query')
                    else:
                        executed_queries.append(query)
                        weights[query_result] += 1.
                        returned_results += 1

            if i > 1:
                diff_to_prev = self.diff_to_prev_weights(prev_weights, weights, i, self.convergence_metric)
                print(f'iteration: {i + 1} has reached diff to previous of: {diff_to_prev:.5f}!')

                if diff_to_prev <= epsilon:
                    print(f'reached error threshold: {epsilon}! stopping after {i + 1} iterations')
                    return TupleWeightCalculator.handle_stop(weights, executed_queries)
            if i > 0:
                del prev_weights
                gc.collect()
                prev_weights = np.copy(weights)
            timer.stop()
            CheckpointManager.save(name=WEIGHTS_CHECKPOINT_NAME,
                                   content=normalize(weights),
                                   append_to_last=i > start_iter,
                                   numpy=True)
            GraphsManager.add_point(GRAPH_NAME, (i, max(diff_to_prev, 0)))

        print(f'reached error threshold: {diff_to_prev}! stopping after {max_iter} iterations')
        return TupleWeightCalculator.handle_stop(weights, executed_queries)

    def get_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.query_generator.get_query(randint(1, 3)))
        return batch

    def diff_to_prev_weights(self, prev_weights, weights, i, metric='max'):
        # TODO should I normalize with sum(weights) instead?
        diffs = np.absolute(np.subtract(prev_weights / (i * self.batch_size), weights / ((i + 1) * self.batch_size)))
        return np.max(diffs) if metric == 'max' else np.average(diffs)

    @staticmethod
    def handle_stop(weights, executed_queries):
        print_total_time()
        result = normalize(weights)
        CheckpointManager.save(name=WEIGHTS_CHECKPOINT_NAME, content=result, numpy=True)
        CheckpointManager.save(name=QUERIES_CHECKPOINT_NAME, content=executed_queries)
        return result

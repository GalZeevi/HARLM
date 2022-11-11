import time
import copy
from abc import ABC, abstractmethod
from data_access import DataAccess
from config_manager import ConfigManager
from checkpoint_manager import CheckpointManager


def union(list1, list2):
    return list(set(list1).union(set(list2)))


class WorkloadApproximator(ABC):

    def __init__(self):
        self.data_access = DataAccess()
        self.similarity_threshold = ConfigManager.get_config('workloadConfig.similarity_threshold')
        self.index_col = ConfigManager.get_config('workloadConfig.index_col')
        self.results = []

    def run(self, max_iter, size=None):
        approx = []
        for i in range(max_iter):
            start = time.time()
            print(f'============ starting iteration {i + 1}   ============')
            result_batch = []

            isThereAnyResults = False
            while not isThereAnyResults:
                query_batch = self.get_batch()

                for query in query_batch:
                    # Query should return only a single field - id
                    query_result = self.data_access.select(query)
                    print(f"Processing query:\n{query}")
                    result_batch.append({'sql': query, 'result': query_result})

                isThereAnyResults = \
                    not all(len(res) == 0 for res in [result_batch[i]['result'] for i in range(len(result_batch))])

            for sql_and_result in result_batch:
                result = sql_and_result['result']
                query = sql_and_result['sql']
                if len(result) > 0:
                    sim_to_approx, twin_in_approx_index = self.get_similarity_to_approx(result,
                                                                                        [d['result'] for d in approx])
                    if sim_to_approx >= self.similarity_threshold:
                        print(f'Adding new query to cluster with num: {twin_in_approx_index}')
                        approx[twin_in_approx_index] = \
                            {
                                'sql': approx[twin_in_approx_index]['sql'] + [query],
                                'result': union(approx[twin_in_approx_index]['result'], result),
                                'frequency': (approx[twin_in_approx_index]['frequency'] + 1)
                            }
                    else:
                        print(f'Creating new cluster with num: {len(approx) + 1}')
                        approx.append({'result': result, 'frequency': 1, 'sql': [query]})

            self.save_to_results(approx)
            print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))

        approx.sort(key=lambda d: d['frequency'], reverse=True)
        approx = approx if size is None else approx[:size]
        CheckpointManager.save(name='workload', content=approx)
        return approx

    @abstractmethod
    def get_batch(self):
        pass

    @abstractmethod
    def similarity(self, list1, list2):
        pass

    def get_similarity_to_approx(self, arr1, approx):
        sim = 0
        twin_index = 0
        for i, arr2 in enumerate(approx):
            similarity_to_arr2 = self.similarity(arr1, arr2)
            if sim <= similarity_to_arr2:
                sim = similarity_to_arr2
                twin_index = i
        return sim, twin_index

    def save_to_results(self, approx):
        self.results.append(copy.deepcopy(approx))

import time
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

    def get_workload_approximation(self, max_iter, size=None):
        approx = []
        for i in range(max_iter):
            start = time.time()
            print(f'============ starting iteration {i + 1}   ============')
            query_batch = self.get_batch()
            result_batch = []

            for query in query_batch:
                query_result = [res[self.index_col] for res in self.data_access.select(query)]
                if len(query_result) > 0:
                    result_batch.append(query_result)

            for result in result_batch:
                sim_to_approx, twin_in_approx_index = self.get_similarity_to_approx(result,
                                                                                    [tup[0] for tup in approx])
                if sim_to_approx >= self.similarity_threshold:
                    approx[twin_in_approx_index] = (union(approx[twin_in_approx_index][0], result), approx[twin_in_approx_index][1])
                else:
                    approx.append((result, 1))
            print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))

        approx.sort(key=lambda tup: tup[1], reverse=True)
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

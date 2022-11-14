from random import randint

from abstract_workload_approximator import WorkloadApproximator
from config_manager import ConfigManager
from query_generator import QueryGenerator


class JaccardWorkloadApproximator(WorkloadApproximator):

    def __init__(self):
        super().__init__()
        clustersConfig = ConfigManager.get_config('clustersConfig')
        queryConfig = [clustersConfig['schema'], clustersConfig['table'], clustersConfig['index_col']]
        self.query_generator = QueryGenerator(*queryConfig)
        self.batch_size = clustersConfig['batchSize']

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(self.query_generator.get_query(randint(1, 3)))
        return batch

    def similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        sim = float(intersection) / union
        return sim

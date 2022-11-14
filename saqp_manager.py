from saqp_par_adapter import SaqpParAdapter
from par_algorithm import PARAlgorithm
from config_manager import ConfigManager
from checkpoint_manager import CheckpointManager


class SaqpManager:

    def __init__(self, queries_results, queries_weights):
        clustersConfig = ConfigManager.get_config('clustersConfig')
        queryConfig = [clustersConfig['schema'], clustersConfig['table'], clustersConfig['index_col']]
        self.saqpParAdapter = SaqpParAdapter(*queryConfig, queries_results, queries_weights)
        self.parAlgorithm = PARAlgorithm(*self.saqpParAdapter.get_par_config())

    def get_sample(self, k, print_debug_logs=True):
        sample = self.parAlgorithm.run(k, print_debug_logs)
        CheckpointManager.save(name='sample', content=sample)
        return sample

    @staticmethod
    def get_test_score_func(test_queries_results, test_queries_weights):
        clustersConfig = ConfigManager.get_config('clustersConfig')
        queryConfig = [clustersConfig['schema'], clustersConfig['table'], clustersConfig['index_col']]
        testSaqpParAdapter = SaqpParAdapter(*queryConfig, test_queries_results, test_queries_weights)
        return lambda sample: testSaqpParAdapter.get_gain_function()(sample)

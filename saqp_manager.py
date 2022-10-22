from saqp_par_adapter import SaqpParAdapter
from par_algorithm import PARAlgorithm
from config_manager import ConfigManager
from checkpoint_manager import CheckpointManager


class SaqpManager:

    def __init__(self, queries_results, queries_weights, k):
        workloadConfig = ConfigManager.get_config('workloadConfig')
        queryConfig = [workloadConfig['schema'], workloadConfig['table'], workloadConfig['index_col']]
        saqpParAdapter = SaqpParAdapter(*queryConfig, queries_results, queries_weights)
        self.parAlgorithm = PARAlgorithm(*saqpParAdapter.get_par_config())
        self.k = k

    def get_sample(self):
        sample = self.parAlgorithm.run(self.k)
        CheckpointManager.save(name='sample', content=sample)
        return sample

    def get_score(self, sample):
        return 0  # TODO

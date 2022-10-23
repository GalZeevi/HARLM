from jaccard_workload_approximator import JaccardWorkloadApproximator
from saqp_manager import SaqpManager
from checkpoint_manager import CheckpointManager

if __name__ == '__main__':
    workload_approximator = JaccardWorkloadApproximator()
    workload = workload_approximator.get_workload_approximation(3)
    saqp_manager = SaqpManager([query['result'] for query in workload], [query['frequency'] for query in workload], 10)
    sample = saqp_manager.get_sample()
    print(sample)
    checkpoint = CheckpointManager.load('sample')
    print(checkpoint)

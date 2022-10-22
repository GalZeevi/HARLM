from jaccard_workload_approximator import JaccardWorkloadApproximator
from saqp_manager import SaqpManager

if __name__ == '__main__':
    workload_approximator = JaccardWorkloadApproximator()
    workload = workload_approximator.get_workload_approximation(1)
    saqp_manager = SaqpManager([query[0] for query in workload], [query[1] for query in workload], 10)
    sample = saqp_manager.get_sample()
    print(sample)

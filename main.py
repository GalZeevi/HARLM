from jaccard_workload_approximator import JaccardWorkloadApproximator
from saqp_manager import SaqpManager

if __name__ == '__main__':
    workload_approximator = JaccardWorkloadApproximator()
    workload = workload_approximator.run(3)
    saqp_manager = SaqpManager([query['result'] for query in workload], [query['frequency'] for query in workload])
    sample = saqp_manager.get_sample(5)
    print(sample)

import numpy as np
from codetiming import Timer
from checkpoint_manager_v2 import CheckpointManager
from config_manager_v2 import ConfigManager
from pathos.multiprocessing import _ProcessPool
import os

TIMER_NAME = 'lazy_greedy_timer'
INNER_TIMER_NAME = 'inner_lazy_greedy_timer'
CHECKPOINT_NAME = 'lazy_greedy'


class LazyGreedySampler:

    def __init__(self,
                 weights,
                 distance_matrix):

        self.weights = weights
        self.W = np.sum(weights)
        self.thresholded_tuples = np.arange(len(weights))[np.where(weights > 0.1)]
        self.distance_matrix = distance_matrix
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.chunk_size = ConfigManager.get_config('cpuConfig.chunk_size')

    def _similarities(self, sample):  # sample is a list of indices
        return 1 - np.min(self.distance_matrix[:, sample],
                          axis=1)  # returns 1 - dist(t,S) for each t in a single matrix

    def gain(self, sample):
        if len(sample) == 0:
            return 0.
        return np.dot(self.weights, self._similarities(sample)) / self.W

    def run(self, budget: float):
        timer = Timer(name=TIMER_NAME, initial_text='============= start iteration =============')

        sample = np.array([], dtype=np.int64)
        current_gain: float = 0
        remaining_budget: float = budget
        deltas = np.ones(len(self.weights)) * np.Inf
        curr = np.zeros(len(self.weights))

        iteration: int = 0
        while remaining_budget > 0:
            timer.start()

            if iteration > 0:
                curr[np.where(deltas > -10)] = 0.

            # return a list of candidates
            max_p_candidates, deltas, curr = self.get_max_p_candidates(deltas, curr, sample, current_gain)
            # choose maximal one from them using their deltas
            p = max_p_candidates[np.argmax(deltas[max_p_candidates])]
            # update the sample
            sample, current_gain, remaining_budget = self.update_sample(sample, remaining_budget, p)
            # remove p
            deltas[p] = -10

            CheckpointManager.save(f'{CHECKPOINT_NAME}_metadata',
                                   [sample, current_gain, remaining_budget, deltas, curr])
            CheckpointManager.save(f'{CHECKPOINT_NAME}_sample', sample)

            timer.stop()
            iteration += 1

        print(f"LazyGreedy finished after {iteration} iterations")
        return sample

    def update_sample(self, sample, remaining_budget, p):
        sample = np.append(sample, p)
        current_gain = self.gain(sample)
        remaining_budget -= 1
        print(f'Added: [{p}] to sample, new gain: [%.4f], remaining budget: [{remaining_budget}]' % current_gain)
        return sample, current_gain, remaining_budget

    def get_max_p_from_subarray(self, p_subarray, deltas, curr, sample, current_gain):
        timer = Timer(name=INNER_TIMER_NAME,
                      initial_text=f'Process {os.getpid()} start calculating chunk of size: {len(p_subarray)}',
                      logger=lambda s: print(s, flush=True))
        timer.start()
        max_iters = len(p_subarray) + 1
        for i in range(max_iters):
            max_delta_ind = np.argmax(deltas)
            p = p_subarray[max_delta_ind]
            if p < 0:
                raise Exception(f'Unexpected Error, could not find candidate p from ${p_subarray}')
            elif curr[max_delta_ind] > 0:
                print(f'Process {os.getpid()} finished calculating chunk of size: {len(p_subarray)}, p={p}', flush=True)
                timer.stop()
                return p, p_subarray, deltas, curr
            else:
                deltas[max_delta_ind] = self.gain(np.append(sample, p)) - current_gain
                curr[max_delta_ind] = 1

    def get_max_p_candidates(self, deltas, curr, sample, current_gain):
        # TODO replace 0.00001 with a percentage of table size
        p_array = np.arange(len(self.weights))[np.where((deltas >= 0) & (self.weights > 0.00001))]
        p_candidates = np.array([], dtype=np.int64)
        items = [(p_subarray, deltas[p_subarray], curr[p_subarray], sample, current_gain) for
                 p_subarray in np.array_split(p_array, self.chunk_size)]

        with _ProcessPool(self.num_workers) as pool:  # TODO reuse map
            result = pool.starmap(self.get_max_p_from_subarray, items)
            for p, p_subarray, updated_deltas, updated_curr in result:
                p_candidates = np.append(p_candidates, p)
                deltas[p_subarray] = updated_deltas
                curr[p_subarray] = updated_curr

        return p_candidates, deltas, curr

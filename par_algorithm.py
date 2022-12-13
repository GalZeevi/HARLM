import time
from collections.abc import Callable
from typing import Literal, Dict, Union, List

import numpy as np
from pathos.pools import _ProcessPool
import os

from config_manager import ConfigManager
from checkpoint_manager import CheckpointManager
from consts import CheckpointNames


class LazyGreedy:  # TODO: change everything to numpy arrays
    runTypes = ['UC', 'CB']

    def __init__(self,
                 cost_func: Callable[[List], float],
                 gain_func: Callable[[List], Union[float, List[float]]],
                 population: List):

        self.cost_func = cost_func
        self.gain_func = gain_func
        self.population = population  # TODO change this when running on large scale examples
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.chunk_size = ConfigManager.get_config('cpuConfig.chunk_size')

    def population_by_idx(self, idx):
        return [self.population[i] for i in idx]

    def run(self, budget: float, run_type: Literal['UC', 'CB'], print_debug_logs=True):
        assert run_type in LazyGreedy.runTypes, f"Unrecognized type! Got {run_type} but should be 'UC' or 'CB'"

        print_debug_logs and print('Starting algorithm initialization')
        start = time.time()
        S: [int] = []
        S_gain: float = 0
        remaining_budget: float = budget
        model: Dict[int, Dict[str, float]] = \
            {p: {
                'delta': np.Inf,
                'cost': self.cost_func([self.population[p]]),
                'curr': False
            } for p in range(len(self.population))}

        print_debug_logs and print(f'Algorithm initialization done after: %.2f ms' % ((time.time() - start) * 1000))

        iter_num: int = 0
        while remaining_budget > 0:
            start = time.time()
            print_debug_logs and print(f'============ starting iteration {iter_num + 1}   ============')

            if iter_num > 0:
                for p in model.keys():
                    model[p]['curr'] = False

            max_p_candidates = self.get_max_p_candidates(model, S, S_gain, run_type, print_debug_logs)
            p = max(max_p_candidates, key=lambda p: model[p]['delta'], default=-1)
            S, S_gain, remaining_budget = self.update_sample(S, remaining_budget, p)
            CheckpointManager.save(f'{run_type}-{CheckpointNames.LAZY_GREEDY_METADATA}', [S, S_gain, remaining_budget, run_type])

            # remove elements that break the budget constraint or already in S
            model = {r: model[r] for r in model.keys() if r != p and model[r]['cost'] <= remaining_budget}
            CheckpointManager.save(f'{run_type}-{CheckpointNames.LAZY_GREEDY_MODEL}', model)

            print_debug_logs and print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))
            iter_num += 1

        print_debug_logs and print(f"LazyGreedy with type {run_type} finished after {iter_num} iterations")
        return self.population_by_idx(S)

    def update_sample(self, S, remaining_budget, p):
        S.append(p)
        S_gain = self.gain_func(self.population_by_idx(S))
        remaining_budget -= self.cost_func(self.population_by_idx([p]))
        print(f'Added: [{p}] to sample, new gain: [%.4f], remaining budget: [{remaining_budget}]' % S_gain)
        return S, S_gain, remaining_budget

    def get_max_p_from_subarray(self, p_subarray, model, S, S_gain, run_type, print_debug_logs):
        # model is assumed to be a partial of outer model
        print_debug_logs and print(f'Process {os.getpid()} starting work on candidates array: \n'
                                   f'{p_subarray}', flush=True)
        max_iters = len(p_subarray) + 1
        for i in range(max_iters):
            p = max(model, key=lambda t: model[t]['delta'], default=-1)
            if p < 0:
                raise Exception(f'Unexpected Error, could not find candidate p from ${p_subarray}')
            elif model[p]['curr']:
                print_debug_logs and print(f'Finished working on a subarray of size: {len(p_subarray)}, p={p}',
                                           flush=True)
                return p, model
            else:
                delta_start = time.time()

                model[p]['delta'] = self.gain_func(self.population_by_idx(S + [p])) - S_gain
                if run_type == LazyGreedy.runTypes[1]:
                    model[p]['delta'] = model[p]['delta'] / model[p]['cost']

                print_debug_logs and print(f'Calculating delta[{p}] took: %.2f ms'
                                           % ((time.time() - delta_start) * 1000), flush=True)
                model[p]['curr'] = True

    def get_max_p_candidates(self, model, S, S_gain, run_type, print_debug_logs):
        with _ProcessPool(self.num_workers) as pool:
            p_array = np.array([*model.keys()])
            p_candidates = []
            items = [(p_subarray, {p: model[p] for p in p_subarray}, S, S_gain, run_type, print_debug_logs) for
                     p_subarray in np.array_split(p_array, self.chunk_size)]
            result = pool.starmap(self.get_max_p_from_subarray, items)
            for p, updated_model in result:
                p_candidates.append(p)
                model.update(updated_model)
        return p_candidates


class PARAlgorithm:

    def __init__(self,
                 cost_func: Union[float, Callable[[list], float]],
                 gain_func: Callable[[list], float],
                 population: list):
        self.gain_func = gain_func

        if type(cost_func) == int or type(cost_func) == float:  # fixed cost
            self.run_only_once = True  # with fixed cost types 'UC' and 'CB' are identical
            self.lazyGreedy = LazyGreedy(lambda S: len(S) * cost_func, gain_func, population)
        else:
            self.lazyGreedy = LazyGreedy(cost_func, gain_func, population)

    def run(self, budget: float, print_debug_logs=True):
        print(f'Start running algorithm with budget: {budget}')

        print_debug_logs and print(f'Start running LazyGreedy with type UC')
        start = time.time()
        res1 = self.lazyGreedy.run(budget, 'UC', print_debug_logs)
        runtime = (time.time() - start) * 1000
        print_debug_logs and print(f'algorithm took: %.2f ms' % runtime)

        res2 = []
        if not self.run_only_once:
            print_debug_logs and print(f'Start running LazyGreedy with type CB')
            start = time.time()
            res2 = self.lazyGreedy.run(budget, 'CB', print_debug_logs)
            cbRuntime = (time.time() - start) * 1000
            print_debug_logs and print(f'algorithm took: %.2f ms' % cbRuntime)
            runtime += cbRuntime

        print(f'PAR algorithm took: %.2f ms' % runtime)

        if self.run_only_once or self.gain_func(res1) > self.gain_func(res2):
            return res1
        else:
            return res2

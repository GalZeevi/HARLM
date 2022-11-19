import multiprocessing as mp
import time
from collections.abc import Callable
from typing import Literal, Dict, Union, List

from tqdm import tqdm

from config_manager import ConfigManager


class LazyGreedy:
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
        B: float = budget
        initial_deltas = self.gain_func(self.population, True)
        model: Dict[int, Dict[str, float]] = \
            {p: {
                'delta': initial_deltas[p],
                'cost': self.cost_func([self.population[p]]),
                'curr': True
            } for p in range(len(self.population))}

        print_debug_logs and print(f'Algorithm initialization done after: %.2f ms' % ((time.time() - start) * 1000))

        isDone: bool = False
        iter_num: int = 0
        while not isDone:
            start = time.time()
            print_debug_logs and print(f'============ starting iteration {iter_num + 1}   ============')

            if iter_num > 0:
                curr_start = time.time()
                for p in model.keys():
                    model[p]['curr'] = False
                print_debug_logs and print(
                    f'Setting all p.isCurrent to False took: %.2f ms' % ((time.time() - curr_start) * 1000))

            updatedSet: bool = False
            p_iter: int = 0
            while not updatedSet:
                p_iter += 1
                p_start = time.time()
                p = max(model, key=lambda p: model[p]['delta'], default=-1)
                print_debug_logs and print(
                    f'Calculating maximal p took: %.2f ms, covered: {p_iter}' % ((time.time() - p_start) * 1000))
                if model[p]['curr']:
                    update_start = time.time()
                    S.append(p)
                    S_gain = self.gain_func(self.population_by_idx(S))
                    B -= self.cost_func(self.population_by_idx([p]))
                    if B == 0:
                        isDone = True
                        break
                    del model[p]

                    # remove elements that break the budget constraint
                    for r in model.keys():
                        if model[r]['cost'] > B:
                            del model[r]

                    print_debug_logs and print(f'Added: [{p}] to S, new gain: [%.4f], remaining budget: [{B}]' % S_gain)
                    updatedSet = True
                    print_debug_logs and print(
                        f'Finished updating S, update took: %.2f ms' % ((time.time() - update_start) * 1000))
                else:
                    delta_start = time.time()
                    model[p]['delta'] = self.gain_func(self.population_by_idx(S + [p])) - S_gain
                    if run_type == LazyGreedy.runTypes[1]:
                        model[p]['delta'] = model[p]['delta'] / model[p]['cost']
                    print_debug_logs and print(f'Calculating delta[{p}] took: %.2f ms'
                                               % ((time.time() - delta_start) * 1000))
                    model[p]['curr'] = True

            print_debug_logs and print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))
            iter_num += 1

        print_debug_logs and print(f"LazyGreedy with type {run_type} finished after {iter_num} iterations")
        return self.population_by_idx(S)


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

        if self.gain_func(res1) > self.gain_func(res2):
            return res1
        else:
            return res2

from collections.abc import Callable
import numpy as np
from typing import Literal
import time
import multiprocessing as mp


class LazyGreedy:
    runTypes = ['UC', 'CB']

    def __init__(self,
                 cost_func: Callable[[[int]], float],
                 gain_func: Callable[[[int]], float],
                 population: list):

        self.cost_func = cost_func
        self.gain_func = gain_func
        self.population = population  # TODO change this when running on large scale examples

    def population_by_idx(self, idx):
        return [self.population[i] for i in idx]

    def run(self, budget: float, run_type: Literal['UC', 'CB'], print_debug_logs=True):
        assert run_type in LazyGreedy.runTypes, f"Unrecognized type! Got {run_type} but should be 'UC' or 'CB'"

        S: [int] = []
        B: float = budget
        delta: [float] = [np.Inf] * len(self.population)
        costs = [self.cost_func([elm]) for elm in self.population]

        isDone: bool = False
        iter_num = 0
        while not isDone:
            start = time.time()
            print_debug_logs and print(f'============ starting iteration {iter_num + 1}   ============')

            curr: [bool] = [False] * len(self.population)
            updatedSet: bool = False
            while not updatedSet:
                p = LazyGreedy.get_maximal_p([*delta], costs, S, B, run_type, mp.cpu_count() - 1)
                if p < 0:
                    isDone = True
                    break
                elif curr[p]:
                    S.append(p)
                    B -= self.cost_func(self.population_by_idx([p]))
                    print_debug_logs and print(
                        f'Updated S, new gain: {self.gain_func(self.population_by_idx(S))}, remaining budget: {B}')
                    updatedSet = True
                    if B == 0:
                        isDone = True
                else:
                    delta[p] = self.gain_func(self.population_by_idx(S + [p])) - \
                               self.gain_func(self.population_by_idx(S))
                    curr[p] = True

            print_debug_logs and print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))
            iter_num += 1

        print_debug_logs and print(f"LazyGreedy with type {run_type} finished after {iter_num} iterations")
        return self.population_by_idx(S)

    @staticmethod
    def calculate_deltas(deltas, S, remaining_budget, run_type, costs, results):
        for i in range(len(deltas)):
            # should disregard what is already in S or what breaks budget constraint
            if (i in set(S)) or (costs[i] > remaining_budget):
                deltas[i] = np.NINF
            if run_type == 'CB' and deltas[i] != np.NINF:
                deltas[i] = deltas[i] / costs[i]

        p = np.argmax(deltas)
        deltas[p] != np.NINF and results.append(p)  # if NINF is reached then no more candidates are left

    @staticmethod
    def get_maximal_p(deltas, costs, S, B, run_type, num_workers):
        results = mp.Manager().list()
        tasks = np.array_split([*zip(deltas, costs)], num_workers)
        procs = []

        for task in tasks:
            proc = mp.Process(target=LazyGreedy.calculate_deltas,
                              args=(
                                  [tup[0] for tup in task],
                                  S,
                                  B,
                                  run_type,
                                  [tup[1] for tup in task],
                                  results))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        return -1 if len(results) == 0 else np.argmax(results)


class PARAlgorithm:

    def __init__(self,
                 cost_func: Callable[[list], float],
                 gain_func: Callable[[list], float],
                 population: list):
        self.gain_func = gain_func
        self.lazyGreedy = LazyGreedy(cost_func, gain_func, population)

    def run(self, budget: float, print_debug_logs=True):
        print(f'Start running algorithm with budget: {budget}')

        print_debug_logs and print(f'Start running LazyGreedy with type UC')
        start = time.time()
        res1 = self.lazyGreedy.run(budget, 'UC', print_debug_logs)
        ucRuntime = (time.time() - start) * 1000
        print_debug_logs and print(f'algorithm took: %.2f ms' % ucRuntime)

        print_debug_logs and print(f'Start running LazyGreedy with type CB')
        start = time.time()
        res2 = self.lazyGreedy.run(budget, 'CB', print_debug_logs)
        cbRuntime = (time.time() - start) * 1000
        print_debug_logs and print(f'algorithm took: %.2f ms' % cbRuntime)

        print(f'PAR algorithm took: %.2f ms' % (ucRuntime + cbRuntime))

        if self.gain_func(res1) > self.gain_func(res2):
            return res1
        else:
            return res2

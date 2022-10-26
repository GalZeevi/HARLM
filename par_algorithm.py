from collections.abc import Callable
import numpy as np
from typing import Literal
import time


class LazyGreedy:

    def __init__(self,
                 cost_func: Callable[[[int]], float],
                 gain_func: Callable[[[int]], float],
                 population: list):

        self.cost_func = cost_func
        self.gain_func = gain_func
        self.population = population  # TODO change this when running on large scale examples

    def population_by_idx(self, idx):
        return [self.population[i] for i in idx]

    def run(self, budget: float, run_type: Literal['UC', 'CB']):
        assert run_type in ['UC', 'CB'], f"Unrecognized type! Got {run_type} but should be 'UC' or 'CB'"

        S: [int] = []
        B: float = budget
        delta: [float] = [np.Inf] * len(self.population)

        isDone: bool = False
        iter_num = 0
        while not isDone:
            start = time.time()
            print(f'============ starting iteration {iter_num + 1}   ============')

            curr: [bool] = [False] * len(self.population)
            updatedSet: bool = False
            while not updatedSet:
                deltaCopy: [float] = [*delta]
                for i in range(len(deltaCopy)):
                    # should disregard what is already in S or what breaks budget constraint
                    if (i in set(S)) or (self.cost_func(self.population_by_idx(S + [i])) > B):
                        deltaCopy[i] = np.NINF
                    if run_type == 'CB' and deltaCopy[i] != np.NINF:
                        deltaCopy[i] = deltaCopy[i] / self.cost_func(self.population_by_idx([i]))

                if all(elm == np.NINF for elm in deltaCopy):  # no candidates are left for p
                    isDone = True
                    break
                p = np.argmax(deltaCopy)

                if curr[p]:
                    S.append(p)
                    print(f'Updated S, new gain: {self.gain_func(self.population_by_idx(S))}')
                    updatedSet = True
                else:
                    delta[p] = self.gain_func(self.population_by_idx(S + [p])) - \
                               self.gain_func(self.population_by_idx(S))
                    # print(f"Updated delta[{p}]")
                    curr[p] = True

            print(f'iteration took: %.2f ms' % ((time.time() - start) * 1000))
            iter_num += 1

        print(f"LazyGreedy with type {run_type} finished after {iter_num} iterations")
        return self.population_by_idx(S)


class PARAlgorithm:
    # TODO make sure we also look at tuples not result of a query (refrence jupyter)
    def __init__(self,
                 cost_func: Callable[[list], float],
                 gain_func: Callable[[list], float],
                 population: list):
        self.gain_func = gain_func
        self.lazyGreedy = LazyGreedy(cost_func, gain_func, population)

    def run(self, budget: float):
        print(f'Start running LazyGreedy with type UC')
        start = time.time()
        res1 = self.lazyGreedy.run(budget, 'UC')
        print(f'algorithm took: %.2f ms' % ((time.time() - start) * 1000))

        print(f'Start running LazyGreedy with type CB')
        start = time.time()
        res2 = self.lazyGreedy.run(budget, 'CB')
        print(f'algorithm took: %.2f ms' % ((time.time() - start) * 1000))

        if self.gain_func(res1) > self.gain_func(res2):
            return res1
        else:
            return res2

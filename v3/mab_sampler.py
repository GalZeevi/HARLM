# library imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from config_manager_v3 import ConfigManager
from score_calculator import get_score
from data_access_v3 import DataAccess
from checkpoint_manager_v3 import CheckpointManager


class MabSampler:
    """
    This class is a Multi arm Bandit type algorithm for dataset summarization
    """

    # SETTINGS #
    steps = 400
    epsilon = 0.5

    # END - SETTINGS #

    @staticmethod
    def run(num_rows: int,
            k: int,
            evaluate_score_function,
            max_iter: int = steps):

        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at MultiArmBanditSummary.run: the max_iter argument must be larger than 1")

        # init random population
        num_bandits = num_rows  # + dataset.shape[1]
        choices = []
        wins = np.zeros(num_bandits)
        pulls = np.zeros(num_bandits)
        for n in tqdm(range(max_iter)):
            if np.random.uniform(size=1) < MabSampler.epsilon:
                choice = np.argmax(wins / (pulls + 0.1))
            else:
                choice = np.random.choice(list(set(range(len(wins))) - {np.argmax(wins / (pulls + 0.1))}))
            choices.append(choice)
            # payout = evaluate_score_function(dataset,
            #                                  dataset.iloc[choice, :] if choice < dataset.shape[0] else dataset.iloc[:,
            #                                                                                            choice -
            #                                                                                            dataset.shape[
            #                                                                                                0]])

            payout = evaluate_score_function(choice)
            wins[choice] += payout
            pulls[choice] += 1

        # get best
        best_rows = []
        added_count = 0
        pbar = tqdm(total=k)

        while added_count < k:
            next_index = np.argmax(wins / (pulls + 0.1))
            if next_index < num_bandits and len(best_rows) < k:
                best_rows.append(next_index)
                added_count += 1
            # wins = np.delete(wins, next_index)
            # pulls = np.delete(pulls, next_index)
            wins[next_index] = np.NINF
            pulls[next_index] = 1000.
            pbar.update(1)

        return best_rows, evaluate_score_function(best_rows)


def get_sample(k, dist=False):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
    evaluation_score_func = lambda s: get_score(s, view_size, dist)

    sample, score = MabSampler.run(table_size, k, evaluation_score_func)
    CheckpointManager.save(f'{k}-{view_size}-mab_sample', [sample, score])

    return sample, score

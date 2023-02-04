# library imports
import numpy as np
from tqdm import tqdm

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score


class MabSampler:
    """
    This class is a Multi arm Bandit type algorithm for dataset summarization
    """

    # SETTINGS #
    steps = 400
    initial_exploration_steps = 0.2
    epsilon = 0.3

    @staticmethod
    def run(num_rows: int,
            k: int,
            evaluate_score_function,
            max_iter: int = steps,
            epsilon: float = epsilon):

        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at MultiArmBanditSummary.run: the max_iter argument must be larger than 1")

        wins_pulls = CheckpointManager.load(f'{max_iter}_{epsilon}_mab_wins_pulls', numpy=True)

        if wins_pulls is None:
            # init random population
            choices = []
            wins = np.zeros(num_rows)
            pulls = np.zeros(num_rows)
            for n in tqdm(range(max_iter)):
                if n >= int(MabSampler.initial_exploration_steps * max_iter) \
                        and np.random.uniform(size=1) < epsilon:
                    choice = np.argmax(wins / (pulls + 0.1))
                else:
                    choice = np.random.choice(list(set(range(len(wins))) - {np.argmax(wins / (pulls + 0.1))}))
                choices.append(choice)

                payout = evaluate_score_function(choice)
                wins[choice] += payout
                pulls[choice] += 1
            CheckpointManager.save(f'{max_iter}_{epsilon}_mab_wins_pulls', np.array([wins, pulls]), numpy=True)

        wins = wins_pulls[0]
        pulls = wins_pulls[1]

        # get best
        best_rows = []
        added_count = 0
        pbar = tqdm(total=k)

        while added_count < k:
            next_index = np.argmax(wins / (pulls + 0.1))
            if next_index < num_rows and len(best_rows) < k:
                best_rows.append(next_index)
                added_count += 1
            wins[next_index] = np.NINF
            pulls[next_index] = 1000.
            pbar.update(1)

        return best_rows, evaluate_score_function(best_rows)


def get_sample(k, dist=False, max_iter=MabSampler.steps, epsilon=MabSampler.epsilon):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    sample, score = MabSampler.run(table_size, k, lambda s: get_score(s, dist), max_iter, epsilon)
    CheckpointManager.save(f'{k}-{view_size}-{max_iter}_{epsilon}_mab_sample', [sample, score])

    return sample, score

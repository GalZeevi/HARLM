import itertools

from mab_sampler import get_sample as get_mab_sample
from mab_sampler2 import get_sample as get_smab_sample
from random_sampler import get_sample as get_random_sample
from score_calculator import get_score_for_test_queries
from top_queried_sampler import get_sample as get_top_q_sample, prepare_weights_for_sample
from train_test_utils import get_test_queries
import imdb_query_batch_generator as imdb
from tqdm import tqdm
import numpy as np
from checkpoint_manager_v3 import CheckpointManager


def get_score_by_k(k_list, dist=False):
    top_q_scores = [0] * len(k_list)
    random_scores = [0] * len(k_list)
    # mab_scores = [0] * len(k_list)

    for i in range(len(k_list)):
        top_q_scores[i] = get_top_q_sample(k_list[i], dist)[1]
        random_scores[i] = get_random_sample(k_list[i], dist)[1]
        # mab_scores[i] = get_mab_sample(k_list[i], dist)[1]

    return k_list, top_q_scores, random_scores  # , mab_scores


def get_imdb_score_by_k(k_list, num_trials=30):
    top_q_scores = np.array([0.] * len(k_list))
    random_scores = np.array([0.] * len(k_list))

    for t in tqdm(range(num_trials)):
        imdb.shuffle_queries()
        prepare_weights_for_sample()
        for i in range(len(k_list)):
            top_q_scores[i] += get_top_q_sample(k_list[i])[1]
            random_scores[i] += get_random_sample(k_list[i])[1]

    CheckpointManager.save(f'imdb_{num_trials}_top_q_and_random_scores_by_k',
                           [k_list, top_q_scores / num_trials, random_scores / num_trials])
    return k_list, top_q_scores / num_trials, random_scores / num_trials


def get_mab_score_by_params(dist=False):
    k = 200000
    epsilons = [0.1, 0.2, 0.3, 0.4]
    max_iters_list = [500, 1000, 1500]
    configs = [*itertools.product(epsilons, max_iters_list)]
    mab_scores = [0] * len(configs)

    for i, (epsilon, max_iters) in enumerate(configs):
        mab_scores[i] = get_mab_sample(k, dist, max_iters, epsilon)[1]

    return configs, mab_scores


def get_smab_score_by_params(dist=False):
    k = 10000
    thresholds = [0.1, 0.3, 0.5, 0.7]
    max_iters = 1500
    smab_scores = [0] * len(thresholds)

    for i, threshold in enumerate(thresholds):
        smab_scores[i] = get_smab_sample(k, dist, max_iters, threshold)[1]

    return smab_scores


def get_score_by_query(sample, test_queries_idx):
    results = get_test_queries()
    scores = []
    for query_id in test_queries_idx:
        scores.append(get_score_for_test_queries(sample, [results[query_id]]))
    return test_queries_idx, scores


def get_score_by_view_size(sample, view_sizes):
    return view_sizes, []  # TODO


def get_score_by_train_size():
    return None  # TODO


if __name__ == '__main__':
    k_list = [10_000, 50_000, 100_000, 150_000, 200_000]
    get_imdb_score_by_k(k_list)
    # get_smab_score_by_params()

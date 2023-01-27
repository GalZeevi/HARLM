from greedy_sampler import get_sample as get_greedy_sample
from random_sampler import get_sample as get_random_sample
from mab_sampler import get_sample as get_mab_sample
from score_calculator import get_score_for_test_queries
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager


def get_score_by_k(k_list, dist=False):
    greedy_scores = [0] * len(k_list)
    random_scores = [0] * len(k_list)
    mab_scores = [0] * len(k_list)

    for i in range(len(k_list)):
        greedy_scores[i] = get_greedy_sample(k_list[i], dist)[1]
        random_scores[i] = get_random_sample(k_list[i], dist)[1]
        mab_scores[i] = get_mab_sample(k_list[i], dist)[1]

    return k_list, greedy_scores, random_scores, mab_scores


def get_score_by_query(sample, test_queries_idx):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    results = CheckpointManager.load('results')
    scores = []
    for query_id in test_queries_idx:
        scores.append(get_score_for_test_queries(sample, [results[query_id]], view_size))
    return test_queries_idx, scores


def get_score_by_view_size(sample, view_sizes):
    return view_sizes, []  # TODO


def get_score_by_train_size():
    return None  # TODO

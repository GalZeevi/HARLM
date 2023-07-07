import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from top_queried_sampler import get_sample

CHECKPOINT_VER = 11
RL_ENV_VER = 2
k = 100
view_size = ConfigManager.get_config('samplerConfig.viewSize')


def main():
    random_score = CheckpointManager.load(f'{k}-{view_size}-random_sample', CHECKPOINT_VER)[1]
    top_q_score = CheckpointManager.load(f'{k}-{view_size}-top_queried_sample', CHECKPOINT_VER)[1]
    policy_net_score = CheckpointManager.load(f'{k}-{view_size}-{k}_5000_1001_policy_net_sample', CHECKPOINT_VER)[1]
    ddqn_score = CheckpointManager.load(f'{k}-{view_size}-{k}_3000_1000_ddqn_sample', CHECKPOINT_VER)[1]

    x = np.array([0, 1, 2, 3])
    my_xticks = ['random', 'top_queried', 'dqn', 'ddqn']
    plt.xticks(x, my_xticks)
    y = np.array([random_score, top_q_score, policy_net_score, ddqn_score])
    plt.bar(x, y)

    plt.xlabel("algorithm")
    plt.ylabel("score")
    plt.title(f"Score by algorithm, k={k}, viewSize={view_size}")
    # plt.legend()
    plt.show()


DDQN = f'{k}_{30000}_{1020}_ddqn'


def plot_losses():
    ep_losses = CheckpointManager.load(f'{DDQN}_losses', CHECKPOINT_VER)
    episodes = [pair[0] for pair in ep_losses]
    losses = [pair[1] for pair in ep_losses]

    plt.plot(episodes, losses)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.title(f"ver{RL_ENV_VER} {DDQN} losses by episode")
    plt.show()


def plot_rewards():
    ep_rewards = CheckpointManager.load(f'{DDQN}_rewards', CHECKPOINT_VER)
    episodes = [pair[0] for pair in ep_rewards]
    rewards = [pair[1] for pair in ep_rewards]

    plt.plot(episodes, rewards)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title(f"ver{RL_ENV_VER} {DDQN} rewards by episode")
    plt.show()


def plot_ddqn_scores():
    all_scores = np.array(CheckpointManager.load(f'{k}-{view_size}-{DDQN}_scores', CHECKPOINT_VER))
    mean = np.mean(all_scores)
    std = np.std(all_scores)
    top_q_score = get_sample(k)[1]
    num_above_top_q = len(all_scores[np.where(all_scores > top_q_score)])
    plt.bar([1, 2], [mean, top_q_score], yerr=[std, 0], align='center', alpha=0.5, capsize=10)
    my_xticks = ['ddqn', 'top_queried']
    plt.xticks([1, 2], my_xticks)
    plt.title(f"ver{RL_ENV_VER} ddqn vs. top_queried mean and std after 9000 episodes")
    plt.show()


def plot_ddqn_scores2():
    all_scores = np.array(CheckpointManager.load(f'{k}-{view_size}-{DDQN}_scores', CHECKPOINT_VER))
    mean = np.mean(all_scores)
    std = np.std(all_scores)
    top_q_score = get_sample(k)[1]
    num_above_top_q = len(all_scores[np.where(all_scores > top_q_score)])
    plt.bar([*range(len(all_scores))][:len(all_scores) - num_above_top_q],
            sorted(all_scores)[:len(all_scores) - num_above_top_q])
    plt.bar([*range(len(all_scores))][len(all_scores) - num_above_top_q:],
            sorted(all_scores)[len(all_scores) - num_above_top_q:])
    plt.title(f"ver{RL_ENV_VER} distribution of 100 ddqn samples after 9000 episodes")
    plt.show()


def plot_ray_results(checkpoint):
    if checkpoint == 3:
        random_scores = [0.182, 0.212, 0.244]
        topk_scores = [0.172, 0.201, 0.216]
        ppo1_scores = [0.490, 0.492, 0.494]
        ppo2_scores = [0.498, 0.501, 0.505]
    elif checkpoint == 4:
        random_scores = [0.32, 0.353, 0.385]
        topk_scores = [0.28, 0.28, 0.28]
        ppo1_scores = [0.644, 0.649, 0.653]
        ppo2_scores = [0.633, 0.635, 0.638]
    elif checkpoint == 5:
        random_scores = [0.217, 0.258, 0.301]
        topk_scores = [0.225, 0.225, 0.225]
        ppo1_scores = [0.438, 0.445, 0.449]
        ppo2_scores = [0.418, 0.426, 0.432]
    else:
        raise Exception('checkpoint not recognized!')

    x = np.array([0, 1, 2, 3])
    plt.xticks(x, ['random', 'top-k', 'PPO1', 'PPO2'])
    y = np.array([random_scores[1], topk_scores[1], ppo1_scores[1], ppo2_scores[1]])
    yerr = np.array([random_scores[-1] - random_scores[1], 0,
                     ppo1_scores[-1] - ppo1_scores[1],
                     ppo2_scores[-1] - ppo2_scores[1]])
    plt.bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)

    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.title(f"Score by algorithm, k={1000}, viewSize={view_size}, checkpoint={checkpoint}")
    plt.show()


IMDB_ALG_NAMES = ['VAE', 'caching', 'random', 'skyline', 'K-means', 'top-k', 'PPO']
IMDB_ALG_SHORTCUTS = ['VAE', 'CACH', 'RAN', 'SKY', 'KM', 'TOP', 'PPO']
IMDB_COLORS = ['#c93699', '#B233FF', '#36c966', '#56a7a9', '#e9b516', '#9e617e', '#eb144f']

ALG_NAMES = ['VAE', 'caching', 'random', 'skyline', 'brute force', 'K-means', 'top-k', 'greedy', 'PPO']
ALG_SHORTCUTS = ['VAE', 'CACH', 'RAN', 'SKY', 'BRT', 'KM', 'TOP', 'GRE', 'PPO']
COLORS = ['#c93699', '#B233FF', '#36c966', '#56a7a9', '#a95856', '#e9b516', '#9e617e', '#b4e41b', '#eb144f']


def get_ray_imdb_data(checkpoint):
    if checkpoint == 2:
        vae_scores = [0] * 3
        cache_scores = [0, 0.00528, 0.014]
        random_scores = [0.285, 0.333, 0.382]
        skyline_scores = [0.394] * 3
        brute_force = [0.284] * 3
        kmeans_scores = [0.376] * 3
        topk_scores = [0.377, 0.377, 0.377]
        # ppo1_scores = [0.468, 0.484, 0.516]
        apex_scores = [0.562, 0.564, 0.566]
        greedy_scores = [0] * 3
        ppo_scores = [0.472, 0.482, 0.494]
    elif checkpoint == 3:
        vae_scores = [0.01] * 3
        cache_scores = [0.102, 0.11, 0.12]
        random_scores = [0.182, 0.212, 0.244]
        skyline_scores = [0.273] * 3
        brute_force = [0.274] * 3
        kmeans_scores = [0.306] * 3
        topk_scores = [0.172, 0.201, 0.216]
        # ppo1_scores = [0.490, 0.492, 0.494]
        apex_scores = [0.32, 0.322, 0.324]
        greedy_scores = [0] * 3
        ppo_scores = [0.498, 0.501, 0.505]
    elif checkpoint == 4:
        vae_scores = [0] * 3
        cache_scores = [0, 0.001, 0.004]
        random_scores = [0.32, 0.353, 0.385]
        skyline_scores = [0.47] * 3
        brute_force = [0.32] * 3
        kmeans_scores = [0.404] * 3
        topk_scores = [0.28, 0.28, 0.28]
        # ppo1_scores = [0.644, 0.649, 0.653]
        apex_scores = [0.374, 0.374, 0.378]
        greedy_scores = [0] * 3
        ppo_scores = [0.633, 0.635, 0.638]
    elif checkpoint == 5:
        vae_scores = [0] * 3
        cache_scores = [0.208, 0.22, 0.242]
        random_scores = [0.217, 0.258, 0.301]
        skyline_scores = [0.251] * 3
        brute_force = [0.31] * 3
        kmeans_scores = [0.2] * 3
        topk_scores = [0.225, 0.225, 0.225]
        # ppo1_scores = [0.438, 0.445, 0.449]
        apex_scores = [0.202, 0.204, 0.208]
        greedy_scores = [0] * 3
        ppo_scores = [0.418, 0.426, 0.432]
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, skyline_scores, brute_force,
                  kmeans_scores, topk_scores, greedy_scores, ppo_scores]
    x = np.arange(len(all_scores))
    y = np.array([scores[1] for scores in all_scores])
    yerr_max = np.array([scores[-1] - scores[1] for scores in all_scores])
    yerr_min = np.array([scores[1] - scores[0] for scores in all_scores])
    return x, y, [yerr_min, yerr_max]


def get_ray_imdb_threshold_data(checkpoint):
    if checkpoint == 2:
        vae_scores = [0.01] * 3
        cache_scores = [0, 0, 0]
        random_scores = [0.285, 0.333, 0.382]
        skyline_scores = [0.394] * 3
        brute_force = [0.4] * 3
        kmeans_scores = [0.376] * 3
        topk_scores = [0.377, 0.377, 0.377]
        # ppo1_scores = [0.468, 0.484, 0.516]
        # apex_scores = [0.562, 0.564, 0.566]
        greedy_scores = [0] * 3
        ppo_scores = [0.472, 0.482, 0.494]
    elif checkpoint == 3:
        vae_scores = [0.01] * 3
        cache_scores = [0.1, 0.1, 0.1]
        random_scores = [0.182, 0.212, 0.244]
        skyline_scores = [0.273] * 3
        brute_force = [0.4] * 3
        kmeans_scores = [0.306] * 3
        topk_scores = [0.172, 0.201, 0.216]
        # ppo1_scores = [0.490, 0.492, 0.494]
        # apex_scores = [0.32, 0.322, 0.324]
        greedy_scores = [0] * 3
        ppo_scores = [0.498, 0.501, 0.505]
    elif checkpoint == 4:
        vae_scores = [0.01] * 3
        cache_scores = [0, 0, 0]
        random_scores = [0.32, 0.353, 0.385]
        skyline_scores = [0.47] * 3
        brute_force = [0.4] * 3
        kmeans_scores = [0.404] * 3
        topk_scores = [0.28, 0.28, 0.28]
        # ppo1_scores = [0.644, 0.649, 0.653]
        # apex_scores = [0.374, 0.374, 0.378]
        greedy_scores = [0] * 3
        ppo_scores = [0.633, 0.635, 0.638]
    elif checkpoint == 5:
        vae_scores = [0.01] * 3
        cache_scores = [0.2, 0.244, 0.3]
        random_scores = [0.217, 0.258, 0.301]
        skyline_scores = [0.251] * 3
        brute_force = [0.4] * 3
        kmeans_scores = [0.2] * 3
        topk_scores = [0.225, 0.225, 0.225]
        # ppo1_scores = [0.438, 0.445, 0.449]
        # apex_scores = [0.202, 0.204, 0.208]
        greedy_scores = [0] * 3
        ppo_scores = [0.418, 0.426, 0.432]
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, skyline_scores, brute_force,
                  kmeans_scores, topk_scores, greedy_scores, ppo_scores]
    x = np.arange(len(all_scores))
    y = np.array([scores[1] for scores in all_scores])
    yerr_max = np.array([scores[-1] - scores[1] for scores in all_scores])
    yerr_min = np.array([scores[1] - scores[0] for scores in all_scores])
    return x, y, [yerr_min, yerr_max]


def plot_ray_imdb_results():
    fig, axs = plt.subplots(2, 2)

    x, y, yerr = get_ray_imdb_data(2)
    axs[0, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[0, 0].set_title('checkpoint 2')

    x, y, yerr = get_ray_imdb_data(3)
    axs[0, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[0, 1].set_title('checkpoint 3')

    x, y, yerr = get_ray_imdb_data(4)
    axs[1, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[1, 0].set_title('checkpoint 4')

    x, y, yerr = get_ray_imdb_data(5)
    axs[1, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[1, 1].set_title('checkpoint 5')

    plt.setp(axs, xticks=x, xticklabels=['random', 'top-k', 'PPO1', 'PPO2'])

    for ax in axs.flat:
        ax.set(xlabel='Algorithm', ylabel='Score')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # plt.xticks(x, ['random', 'top-k', 'PPO1', 'PPO2'])

    # plt.xlabel("Algorithm")
    # plt.ylabel("Score")
    plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


def plot_ray_avg_imdb_results():
    checkpoints = [2, 3, 4, 5]

    x = []
    y = np.zeros(len(ALG_NAMES))
    y_err = np.zeros((2, len(ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_imdb_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    plt.ylabel("Score(S)", fontsize=18)
    plt.show()


def plot_imdb_runtime_results():
    x = [*range(len(ALG_NAMES))]
    y = [32, 1, 1.2, 25, 48, 30, 1.3, 48, 6]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 50, 5), fontsize=18)

    plt.ylabel("Hours", fontsize=18)
    plt.show()


def plot_ray_avg_imdb_threshold_results():
    checkpoints = [2, 3, 4, 5]

    x = []
    y = np.zeros(len(ALG_NAMES))
    y_err = np.zeros((2, len(ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_imdb_threshold_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10)
    plt.title('')
    plt.xticks(x, ALG_NAMES, rotation=45, fontsize=8)

    plt.xlabel("Algorithm")
    plt.ylabel("0.25-Score")
    # plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


MAS_ALG_NAMES = ['VAE', 'caching', 'random', 'skyline', 'brute force', 'K-means', 'top-k', 'greedy', 'PPO']
MAS_ALG_SHORTCUTS = ['VAE', 'CACH', 'RAN', 'SKY', 'BRT', 'KM', 'TOP', 'GRE', 'PPO']
MAS_COLORS = ['#c93699', '#B233FF', '#36c966', '#56a7a9', '#a95856', '#e9b516', '#9e617e', '#b4e41b', '#eb144f']


def plot_mas_runtime_results():
    x = [*range(len(MAS_ALG_NAMES))]
    y = [12, 0.1, 0.2, 8, 48, 10, 0.3, 48, 4]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, align='center', alpha=0.5, capsize=10, width=0.6, color=MAS_COLORS)
    plt.title('')
    plt.xticks(x, MAS_ALG_SHORTCUTS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 50, 5), fontsize=18)

    plt.ylabel("Hours", fontsize=18)
    plt.show()


def get_ray_mas_data(checkpoint):
    if checkpoint == 7:
        cache_scores = [0.324, 0.341, 0.375]
        topk_scores = [0.625] * 3
        random_scores = [0.182, 0.203, 0.230]
        skyline_scores = [0.265] * 3
        brute_force_scores = [0.27] * 3
        kmeans_scores = [0.3475] * 3
        greedy_train_scores = [0.537] * 3
        greedy_train_plus_diversity_scores = [0.532] * 3
        ppo_scores = [0.64, 0.649, 0.665]
        ppo_plus_diversity_scores = [0.6339, 0.6552, 0.6741]
        # ppo_0dot3_scores = [0.6431, 0.65, 0.6638]
        apex_scores = [0.668, 0.668, 0.672]
        apex_plus_diversity_scores = [0.6641, 0.6642, 0.668]
        vae_scores = [0.04] * 3
    elif checkpoint == 8:
        cache_scores = [0.4, 0.4171, 0.455]
        topk_scores = [0.455] * 3
        random_scores = [0.25, 0.283, 0.324]
        skyline_scores = [0.3125] * 3
        brute_force_scores = [0.405] * 3
        kmeans_scores = [0.402] * 3
        greedy_train_scores = [0.632] * 3
        greedy_train_plus_diversity_scores = [0.632] * 3
        ppo_scores = [0.625, 0.633, 0.647]
        ppo_plus_diversity_scores = [0.7525, 0.7621, 0.77]
        # ppo_0dot3_scores = [0.76, 0.768, 0.782]
        apex_scores = [0.765, 0.7651, 0.7675]
        apex_plus_diversity_scores = [0.7675, 0.7677, 0.775]
        vae_scores = [0.02, 0.02, 0.03]
    elif checkpoint == 9:
        cache_scores = [0.125] * 3
        topk_scores = [0.25] * 3
        random_scores = [0.165, 0.205, 0.24]
        skyline_scores = [0.217] * 3
        brute_force_scores = [0.305] * 3
        kmeans_scores = [0.14] * 3
        greedy_train_scores = [0.395] * 3
        greedy_train_plus_diversity_scores = [0.337] * 3
        ppo_scores = [0.611, 0.634, 0.652]
        ppo_plus_diversity_scores = [0.6637, 0.6893, 0.7147]
        # ppo_0dot3_scores = [0.655, 0.685, 0.718]
        apex_scores = [0.6523, 0.6524, 0.6562]
        apex_plus_diversity_scores = [0.658, 0.6592, 0.6644]
        vae_scores = [0.06] * 3
    elif checkpoint == 10:
        cache_scores = [0] * 3
        topk_scores = [0.507] * 3
        random_scores = [0.075, 0.120, 0.157]
        skyline_scores = [0.14] * 3
        brute_force_scores = [0.21] * 3
        kmeans_scores = [0.3814] * 3
        greedy_train_scores = [0.507] * 3
        greedy_train_plus_diversity_scores = [0.51] * 3
        ppo_scores = [0.675, 0.691, 0.722]
        ppo_plus_diversity_scores = [0.6612, 0.6804, 0.7003]
        # ppo_0dot3_scores = [0.660, 0.676, 0.700]
        apex_scores = [0.6836, 0.6839, 0.6875]
        apex_plus_diversity_scores = [0.6875, 0.6877, 0.6914]
        vae_scores = [0.06] * 3
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, skyline_scores, brute_force_scores, kmeans_scores,
                  topk_scores, greedy_train_scores, ppo_scores]
    x = np.arange(len(all_scores))
    y = np.array([scores[1] for scores in all_scores])
    yerr_max = np.array([scores[-1] - scores[1] for scores in all_scores])
    yerr_min = np.array([scores[1] - scores[0] for scores in all_scores])
    return x, y, [yerr_min, yerr_max]


def get_ray_mas_threshold_data(checkpoint):
    if checkpoint == 7:
        cache_scores = [0.375] * 3
        topk_scores = [0.625] * 3
        random_scores = [0.125, 0.16, 0.25]
        skyline_scores = [0.25] * 3
        brute_force_scores = [0.25] * 3
        kmeans_scores = [0.5] * 3
        greedy_train_scores = [0.625] * 3
        greedy_train_plus_diversity_scores = [0.625] * 3
        ppo_scores = [0.625, 0.702, 0.75]
        # ppo_0dot3_scores = [0.625, 0.635, 0.75]
        ppo_plus_diversity_scores = [0.625, 0.6417, 0.75]
        apex_scores = [0.75, 0.75, 0.75]
        apex_plus_diversity_scores = [0.75] * 3
        vae_scores = [0] * 3
    elif checkpoint == 8:
        cache_scores = [0.5] * 3
        topk_scores = [0.5] * 3
        random_scores = [0.25, 0.342, 0.625]
        skyline_scores = [0.375] * 3
        brute_force_scores = [0.5] * 3
        kmeans_scores = [0.375] * 3
        greedy_train_scores = [0.625] * 3
        greedy_train_plus_diversity_scores = [0.625] * 3
        ppo_scores = [0.625, 0.625, 0.625]
        # ppo_0dot3_scores = [0.75, 0.75, 0.75]
        ppo_plus_diversity_scores = [0.75] * 3
        apex_scores = [0.75, 0.75, 0.75]
        apex_plus_diversity_scores = [0.75] * 3
        vae_scores = [0] * 3
    elif checkpoint == 9:
        cache_scores = [0.125] * 3
        topk_scores = [0.25] * 3
        random_scores = [0.125, 0.242, 0.375]
        skyline_scores = [0.125] * 3
        brute_force_scores = [0.375] * 3
        kmeans_scores = [0.25] * 3
        greedy_train_scores = [0.625] * 3
        greedy_train_plus_diversity_scores = [0.25] * 3
        ppo_scores = [0.625, 0.625, 0.625]
        # ppo_0dot3_scores = [0.625, 0.743, 0.875]
        ppo_plus_diversity_scores = [0.625, 0.76, 0.875]
        apex_scores = [0.625, 0.6275, 0.75]
        apex_plus_diversity_scores = [0.75] * 3
        vae_scores = [0] * 3
    elif checkpoint == 10:
        cache_scores = [0] * 3
        topk_scores = [0.5] * 3
        random_scores = [0.125, 0.13, 0.25]
        skyline_scores = [0.125] * 3
        brute_force_scores = [0.375] * 3
        kmeans_scores = [0.5] * 3
        greedy_train_scores = [0.5] * 3
        greedy_train_plus_diversity_scores = [0.5] * 3
        ppo_scores = [0.625, 0.777, 0.875]
        # ppo_0dot3_scores = [0.625, 0.682, 0.875]
        ppo_plus_diversity_scores = [0.625, 0.7175, 0.875]
        apex_scores = [0.75, 0.7525, 0.875]
        apex_plus_diversity_scores = [0.75] * 3
        vae_scores = [0] * 3
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, skyline_scores, brute_force_scores, kmeans_scores,
                  topk_scores, greedy_train_scores, ppo_scores]
    x = np.arange(len(all_scores))
    y = np.array([scores[1] for scores in all_scores])
    yerr_max = np.array([scores[-1] - scores[1] for scores in all_scores])
    yerr_min = np.array([scores[1] - scores[0] for scores in all_scores])

    return x, y, [yerr_min, yerr_max]


def plot_ray_mas_results():
    fig, axs = plt.subplots(2, 2)

    x, y, yerr = get_ray_mas_data(7)
    axs[0, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[0, 0].set_title('checkpoint 7')

    x, y, yerr = get_ray_mas_data(8)
    axs[0, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[0, 1].set_title('checkpoint 8')

    x, y, yerr = get_ray_mas_data(9)
    axs[1, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[1, 0].set_title('checkpoint 9')

    x, y, yerr = get_ray_mas_data(10)
    axs[1, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    axs[1, 1].set_title('checkpoint 10')

    plt.xticks(rotation=90)
    plt.setp([axs[1, 0], axs[1, 1]], xticks=x)
    for ax in [axs[1, 0], axs[1, 1]]:
        ax.tick_params(labelrotation=45)
        ax.set_xticklabels(MAS_ALG_NAMES, fontsize=8)

    plt.setp([axs[0, 0], axs[0, 1]], xticks=[], xticklabels=[])

    for ax in [axs[1, 0], axs[1, 1]]:
        ax.set(xlabel='Algorithm')

    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set(ylabel='Score')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


def plot_ray_mas_threshold_results():
    fig, axs = plt.subplots(2, 2)

    x, y, yerr = get_ray_mas_threshold_data(7)
    axs[0, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    # axs[0, 0].set_title('checkpoint 7')

    x, y, yerr = get_ray_mas_threshold_data(8)
    axs[0, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    # axs[0, 1].set_title('checkpoint 8')

    x, y, yerr = get_ray_mas_threshold_data(9)
    axs[1, 0].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    # axs[1, 0].set_title('checkpoint 9')

    x, y, yerr = get_ray_mas_threshold_data(10)
    axs[1, 1].bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    # axs[1, 1].set_title('checkpoint 10')

    # plt.xticks(rotation=90)
    plt.setp([axs[1, 0], axs[1, 1]], xticks=x)
    for ax in [axs[1, 0], axs[1, 1]]:
        ax.tick_params(labelrotation=45)
        ax.set_xticklabels(MAS_ALG_NAMES, fontsize=8)

    plt.setp([axs[0, 0], axs[0, 1]], xticks=[], xticklabels=[])

    for ax in [axs[1, 0], axs[1, 1]]:
        ax.set(xlabel='Algorithm')

    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set(ylabel='0.25-Score')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.suptitle(f"0.25-Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


def plot_ray_avg_mas_results():
    checkpoints = [7, 8, 9, 10]

    x = []
    y = np.zeros(len(MAS_ALG_NAMES))
    y_err = np.zeros((2, len(MAS_ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_mas_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=MAS_COLORS)
    plt.title('')
    plt.xticks(x, MAS_ALG_SHORTCUTS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 0.8, 0.1), fontsize=18)

    plt.ylabel("Score(S)", fontsize=18)
    plt.show()


def plot_ray_avg_mas_threshold_results():
    checkpoints = [7, 8, 9, 10]

    x = []
    y = np.zeros(len(MAS_ALG_NAMES))
    y_err = np.zeros((2, len(MAS_ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_mas_threshold_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10)
    plt.title('')
    plt.xticks(x, MAS_ALG_NAMES, rotation=45, fontsize=8)

    plt.xlabel("Algorithm")
    plt.ylabel("0.25-Score")
    # plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


############# experiment: drop1, avg: 0.39698507462686566, std: 0.09682936099328326 #############
############# experiment: drop1+random, avg: 0.26135, std: 0.07448139029314638 #############
############# experiment: drop1+choose k, avg: 0.5315294117647059, std: 0.17535447046316976 #############
############# experiment: choose k, avg: 0.5430476190476191, std: 0.14262387304023733 #############


############# experiment: drop1, avg: 0.3977331088137009, std: 0.017051618117053618 #############
############# experiment: drop1+random, avg: 0.27678466386554623, std: 0.01967992128215682 #############
############# experiment: drop1+choose k, avg: 0.5044978354978356, std: 0.008459471253482065 #############
############# experiment: choose k, avg: 0.49850666666666665, std: 0.0009713488600580323 #############


def plot_drop1_imdb_results():
    experiments = ['DropOne', 'DropOne+Rand', 'ChooseK+DropOne', 'ChooseK']
    experiments_short = ['DRP', 'DRP+R', 'CK+DRP', 'CK']
    colors = ['#FFDB58', '#D966E1', '#17D580', '#FF5733']
    scores = [0.39773, 0.27678, 0.49850, 0.52449]
    stds = [0.01705, 0.01967, 0.00097, 0.00845]
    x = [*range(len(experiments))]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.subplots_adjust(bottom=0.3)

    plt.bar(x, scores, yerr=stds, align='center', alpha=0.6, capsize=10, color=colors)
    plt.title('')
    plt.xticks(x, experiments_short, fontsize=18, rotation=90)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    # plt.xlabel("Algorithm")
    plt.ylabel("Score(S)", fontsize=18)
    # plt.suptitle(f"Imdb Data")
    plt.show()


def plot_drop1_extended_results():
    experiments = ['DropOne', 'DropOne+Rand', 'ChooseK+DropOne', 'ChooseK']
    experiments_short = ['DRP', 'DRP+R', 'CK+DRP', 'CK']
    ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO']
    ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#eb144f']
    colors = ['#FFDB58', '#D966E1', '#17D580', '#FF5733']
    i = 0

    drp_scores = [
        [0.39, 0.40120, 0.416],  # DQN
        [0.3502, 0.365, 0.373],  # Diversity DQN
        [0.36, 0.3642, 0.368],  # A3C
        [0.315, 0.3242, 0.343],  # A3C Diversity
        [0.326, 0.3302, 0.348],  # PPO Diversity
        [0.27, 0.2902, 0.308],  # PPO not tuned
        [0.338, 0.3646, 0.39773],  # PPO
    ]
    drp_x = [i - 3 / 7, i - 2 / 7, i - 1 / 7, i, i + 1 / 7, i + 2 / 7, i + 3 / 7]
    i += 1.25
    drp_y = [drp_scores[i][1] for i in range(len(drp_scores))]
    drp_y_err = [[drp_scores[i][1] - drp_scores[i][0] for i in range(len(drp_scores))],
                 [drp_scores[i][2] - drp_scores[i][1] for i in range(len(drp_scores))]]

    drpr_scores = [
        [0.329, 0.336, 0.352],  # DQN
        [0.252, 0.276, 0.289],  # Diversity DQN
        [0.212, 0.2312, 0.258],  # A3C
        [0.2092, 0.2210, 0.252],  # A3C Diversity
        [0.2102, 0.2230, 0.262],  # PPO Diversity
        [0.192, 0.2203, 0.267],  # PPO not tuned
        [0.202, 0.2243, 0.25],  # PPO
    ]
    drpr_x = [i - 3 / 7, i - 2 / 7, i - 1 / 7, i, i + 1 / 7, i + 2 / 7, i + 3 / 7]
    i += 1.25
    drpr_y = [drpr_scores[i][1] for i in range(len(drpr_scores))]
    drpr_y_err = [[drpr_scores[i][1] - drpr_scores[i][0] for i in range(len(drpr_scores))],
                  [drpr_scores[i][2] - drpr_scores[i][1] for i in range(len(drpr_scores))]]

    ckdrp_scores = [
        [0.324, 0.325, 0.326],  # DQN
        [0.286, 0.306, 0.326],  # Diversity DQN
        [0.398, 0.408, 0.412],  # A3C
        [0.392, 0.405, 0.41],  # A3C Diversity
        [0.324, 0.3309, 0.348],  # PPO Diversity
        [0.272, 0.289, 0.3],  # PPO not tuned
        [0.49753, 0.49850, 0.49947],  # PPO
    ]
    ckdrp_x = [i - 3 / 7, i - 2 / 7, i - 1 / 7, i, i + 1 / 7, i + 2 / 7, i + 3 / 7]
    i += 1.25
    ckdrp_y = [ckdrp_scores[i][1] for i in range(len(ckdrp_scores))]
    ckdrp_y_err = [[ckdrp_scores[i][1] - ckdrp_scores[i][0] for i in range(len(ckdrp_scores))],
                   [ckdrp_scores[i][2] - ckdrp_scores[i][1] for i in range(len(ckdrp_scores))]]

    ck_scores = [
        [0.3645, 0.366, 0.369],  # DQN
        [0.2717, 0.2735, 0.3106],  # Diversity DQN
        [0.412, 0.4136, 0.418],  # A3C
        [0.376, 0.3809, 0.388],  # A3C Diversity
        [0.36025, 0.39295, 0.4165],  # PPO Diversity
        [0.405, 0.428125, 0.442],  # PPO not tuned
        [0.51604, 0.52449, 0.53294],  # PPO
    ]
    ck_x = [i - 3 / 7, i - 2 / 7, i - 1 / 7, i, i + 1 / 7, i + 2 / 7, i + 3 / 7]
    i += 1.25
    ck_y = [ck_scores[i][1] for i in range(len(ck_scores))]
    ck_y_err = [[ck_scores[i][1] - ck_scores[i][0] for i in range(len(ck_scores))],
                [ck_scores[i][2] - ck_scores[i][1] for i in range(len(ck_scores))]]

    x = [0, 1.25, 2.5, 3.75]

    all_x = [drp_x, drpr_x, ckdrp_x, ck_x]
    all_y = [drp_y, drpr_y, ckdrp_y, ck_y]
    all_yerr = [drp_y_err, drpr_y_err, ckdrp_y_err, ck_y_err]

    plt.figure(figsize=(16, 6))
    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    # plt.subplots_adjust(bottom=0.1, left=0.15, top=0.65)
    plt.subplots_adjust(top=0.75)

    for i, name in enumerate(ALGOS):
        plt.bar([x[i] for x in all_x], [y[i] for y in all_y],
                yerr=[[yerr[0][i] for yerr in all_yerr], [yerr[1][i] for yerr in all_yerr]],
                width=0.1, align='center', alpha=0.6, capsize=5, label=name,
                color=[ALGOS_COLORS[i]] * len(experiments))

    plt.title('')
    plt.xticks(x, experiments_short, fontsize=18)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    plt.ylabel("Score(S)", fontsize=18)
    plt.legend(bbox_to_anchor=(0.48, 1.35), loc='upper center', ncol=4, prop={'size': 18})
    plt.show()


# IMDB_ALG_NAMES = ['random', 'skyline', 'K-means', 'top-k', 'DQN', 'PPO']

def plot_imdb_k_array_results():
    x = [*range(len(ALG_NAMES))]

    y_1k = [
        0,  # VAE
        0.11,  # caching
        0.212,  # random
        0.273,  # skyline
        0.29,  # brute
        0.306,  # kmeans
        0.201,  # topk
        0.,  # greedy
        0.501,  # ppo
    ]
    min_max_1k = [
        [0,
         0.102,
         0.182,
         0.273,
         0.29,  # brute
         0.306,
         0.201,
         0.,
         0.498],
        [0.01,
         0.12,
         0.244,
         0.273,
         0.29,  # brute
         0.306,
         0.201,
         0.,
         0.505]
    ]
    yerr_1k = [[abs(a - b) for (a, b) in zip(min_max_1k[0], y_1k)],
               [abs(a - b) for (a, b) in zip(min_max_1k[1], y_1k)]]

    y_5k = [
        0.0,  # VAE
        0.152,  # caching
        0.4354,  # random
        0.451,  # skyline
        0.45,  # brute
        0.316,  # kmeans
        0.346,  # topk
        0,  # greedy
        0.6376,  # ppo
    ]
    min_max_5k = [
        [0,
         0.17,
         0.414,
         0.451,
         0.45,  # brute
         0.316,
         0.346,
         0.,
         0.624],
        [0.01,
         0.24,
         0.456,
         0.451,
         0.45,  # brute
         0.316,
         0.346,
         0.,
         0.648]
    ]
    yerr_5k = [[abs(a - b) for (a, b) in zip(min_max_5k[0], y_5k)],
               [abs(a - b) for (a, b) in zip(min_max_5k[1], y_5k)]]

    y_10k = [
        0.01,  # VAE
        0.20,  # caching
        0.469,  # random
        0.511,  # skyline
        0.526,  # skyline
        0.432,  # kmeans
        0.43,  # topk
        0.,  # greedy
        0.6653,  # ppo
    ]
    min_max_10k = [
        [0,
         0.17,
         0.43,
         0.511,
         0.526,
         0.432,
         0.43,
         0.,
         0.66],
        [0.02,
         0.24,
         0.492,
         0.511,
         0.526,
         0.432,
         0.43,
         0.,
         0.67]
    ]
    yerr_10k = [[abs(a - b) for (a, b) in zip(min_max_10k[0], y_10k)],
                [abs(a - b) for (a, b) in zip(min_max_10k[1], y_10k)]]

    y_15k = [
        0.01,  # VAE
        0.249,  # caching
        0.50164,  # random
        0.534,  # skyline
        0.566,  # skyline
        0.492,  # kmeans
        0.476,  # topk
        0.,  # greedy
        0.714,  # ppo
    ]
    min_max_15k = [
        [0,
         0.284,
         0.466,
         0.534,
         0.566,
         0.492,
         0.476,
         0.,
         0.708],
        [0.02,
         0.208,
         0.522,
         0.534,
         0.566,
         0.492,
         0.476,
         0.,
         0.72]
    ]
    yerr_15k = [[abs(a - b) for (a, b) in zip(min_max_15k[0], y_15k)],
                [abs(a - b) for (a, b) in zip(min_max_15k[1], y_15k)]]

    # plt.xticks(x, IMDB_ALG_SHORTCUTS, rotation=90, fontsize=18)
    # plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    # plt.ylabel("Score(S)", fontsize=18)

    fig, axs = plt.subplots(4)
    axs[0].bar(x, y_1k, yerr=yerr_1k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[0].set_title('k=1000', y=1.0, pad=-14)
    axs[0].set_yticks(np.arange(0, 0.9, 0.2))
    # axs[0].set_xticks([])
    axs[0].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[1].bar(x, y_5k, yerr=yerr_5k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[1].set_title('k=5000', y=1.0, pad=-14)
    # axs[1].set_xticks([])
    axs[1].set_yticks(np.arange(0, 0.9, 0.2))
    axs[1].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[2].bar(x, y_10k, yerr=yerr_10k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[2].set_title('k=10000', y=1.0, pad=-14)
    # axs[2].set_xticks([])
    axs[2].set_yticks(np.arange(0, 0.9, 0.2))
    axs[2].grid(which='major', color='#bfc1c7', linewidth=0.8)

    axs[3].bar(x, y_15k, yerr=yerr_15k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[3].set_title('k=15000', y=1.0, pad=-14)
    axs[3].set_yticks(np.arange(0, 0.9, 0.2))
    axs[3].grid(which='major', color='#bfc1c7', linewidth=0.8)

    # plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    # plt.setp(axs, yticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    # plt.suptitle('Baselines with varying sample size')

    # axs[-1].set_xticklabels(IMDB_ALG_NAMES, fontsize=8)
    plt.subplots_adjust(bottom=0.2)
    plt.minorticks_on()

    plt.setp(axs[:3], xticks=x, xticklabels=[])
    plt.setp([axs[-1]], xticks=x, xticklabels=ALG_SHORTCUTS)
    axs[-1].tick_params(axis='x', labelsize=16, rotation=90)

    plt.show()


def plot_view_size_array_results():
    x = [*range(len(ALG_NAMES))]

    y_25 = [
        0.01,  # VAE
        0.11,  # caching
        0.30152,  # random
        0.446,  # skyline
        0.4,  # brute
        0.348,  # kmeans
        0.261,  # topk
        0.,  # greedy
        0.649,  # ppo
    ]
    min_max_25 = [
        [0.01,
         0.102,
         0.244,
         0.440,
         0.4,
         0.348,
         0.261,
         0.,  # greedy
         0.643],
        [0.02,
         0.115,
         0.344,
         0.49,
         0.4,
         0.348,
         0.261,
         0.,  # greedy
         0.662]
    ]
    yerr_25 = [[abs(a - b) for (a, b) in zip(min_max_25[0], y_25)],
               [abs(a - b) for (a, b) in zip(min_max_25[1], y_25)]]

    y_50 = [
        0,  # VAE
        0.11,  # caching
        0.212,  # random
        0.273,  # skyline
        0.29,  # brute
        0.306,  # kmeans
        0.201,  # topk
        0,  # greedy
        0.501,  # ppo
    ]
    min_max_50 = [
        [0,
         0.102,
         0.182,
         0.273,
         0.29,
         0.306,
         0.201,
         0,  # greedy
         0.498],
        [0.01,
         0.12,
         0.244,
         0.273,
         0.29,
         0.306,
         0.201,
         0,  # greedy
         0.505]
    ]
    yerr_50 = [[abs(a - b) for (a, b) in zip(min_max_50[0], y_50)],
               [abs(a - b) for (a, b) in zip(min_max_50[1], y_50)]]

    y_75 = [
        0.015,  # VAE
        0.249,  # caching
        0.15,  # random
        0.247,  # skyline
        0.25,  # brute
        0.303,  # kmeans
        0.237,  # topk
        0,  # greedy
        0.461,  # ppo
    ]
    min_max_75 = [
        [0,
         0.284,
         0.12,
         0.247,
         0.25,  # brute
         0.292,
         0.237,
         0,  # greedy
         0.458],
        [0.025,
         0.208,
         0.17,
         0.247,
         0.25,  # brute
         0.313,
         0.237,
         0,  # greedy
         0.476]
    ]
    yerr_75 = [[abs(a - b) for (a, b) in zip(min_max_75[0], y_75)],
               [abs(a - b) for (a, b) in zip(min_max_75[1], y_75)]]

    y_100 = [
        0.01,  # VAE
        0.06,  # caching
        0.11,  # random
        0.190,  # skyline
        0.23,  # brute
        0.28,  # kmeans
        0.23,  # topk
        0,  # greedy
        0.4353,  # ppo
    ]
    min_max_100 = [
        [0,
         0.,
         0.095,
         0.190,
         0.23,  # brute
         0.28,
         0.23,
         0,  # greedy
         0.42],
        [0.07,
         0.1,
         0.13,
         0.190,
         0.23,  # brute
         0.28,
         0.23,
         0,  # greedy
         0.45]
    ]
    yerr_100 = [[abs(a - b) for (a, b) in zip(min_max_100[0], y_100)],
                [abs(a - b) for (a, b) in zip(min_max_100[1], y_100)]]

    # plt.xticks(x, IMDB_ALG_SHORTCUTS, rotation=90, fontsize=18)
    # plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)
    #
    # plt.ylabel("Score(S)", fontsize=18)

    fig, axs = plt.subplots(4)
    axs[0].bar(x, y_25, yerr=yerr_25, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[0].set_title('F=25', y=1.0, pad=-14)
    axs[0].set_yticks(np.arange(0, 0.9, 0.2))
    # axs[0].set_xticks([])
    axs[0].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[1].bar(x, y_50, yerr=yerr_50, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[1].set_title('F=50', y=1.0, pad=-14)
    # axs[1].set_xticks([])
    axs[1].set_yticks(np.arange(0, 0.9, 0.2))
    axs[1].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[2].bar(x, y_75, yerr=yerr_75, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[2].set_title('F=75', y=1.0, pad=-14)
    # axs[2].set_xticks([])
    axs[2].set_yticks(np.arange(0, 0.9, 0.2))
    axs[2].grid(which='major', color='#bfc1c7', linewidth=0.8)

    axs[3].bar(x, y_100, yerr=yerr_100, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[3].set_title('F=100', y=1.0, pad=-14)
    axs[3].set_yticks(np.arange(0, 0.9, 0.2))
    axs[3].grid(which='major', color='#bfc1c7', linewidth=0.8)

    # plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    # plt.setp(axs, yticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    # plt.suptitle('Baselines with varying sample size')

    # axs[-1].set_xticklabels(IMDB_ALG_NAMES, fontsize=8)
    plt.subplots_adjust(bottom=0.2)
    plt.minorticks_on()

    plt.setp(axs[:3], xticks=x, xticklabels=[])
    plt.setp([axs[-1]], xticks=x, xticklabels=ALG_SHORTCUTS)
    axs[-1].tick_params(axis='x', labelsize=16, rotation=90)

    plt.show()


def plot_imdb_rl_algos():
    RL_ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO+DropOne', 'PPO']
    RL_ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#a1153b', '#eb144f']
    x = [*range(len(RL_ALGOS))]
    y = np.zeros(len(RL_ALGOS))
    y_err = np.zeros((2, len(RL_ALGOS)))
    num_checkpoints = 4

    # 2
    dqn_v2_scores = [0.562, 0.564, 0.566]
    dqn_diversity_v2_scores = [0.331, 0.334, 0.346]
    a3c_v2_scores = [0.41, 0.4101, 0.412]
    a3c_diversity_v2_scores = [0.414, 0.414, 0.414]
    ppo_v2_scores = [0.477, 0.489, 0.5]
    ppo_untuned_v2_scores = [0.456, 0.4819, 0.51]
    ppo_diversity_v2_scores = [0.394, 0.4682, 0.51]
    ppo_choosek_drop1_v2_scores = [0.474, 0.4914, 0.506]

    v2_scores = [dqn_v2_scores, dqn_diversity_v2_scores, a3c_v2_scores, a3c_diversity_v2_scores,
                 ppo_diversity_v2_scores, ppo_untuned_v2_scores, ppo_choosek_drop1_v2_scores, ppo_v2_scores]

    y += [alg[1] for alg in v2_scores]
    y_err += [np.array([alg[1] for alg in v2_scores]) - np.array([alg[0] for alg in v2_scores]),
              np.array([alg[2] for alg in v2_scores]) - np.array([alg[1] for alg in v2_scores])]

    # 3
    dqn_v3_scores = [0.32, 0.322, 0.324]
    dqn_diversity_v3_scores = [0.284, 0.285, 0.286]
    a3c_v3_scores = [0.412, 0.4136, 0.418]
    a3c_diversity_v3_scores = [0.376, 0.3809, 0.388]
    ppo_v3_scores = [0.498, 0.501, 0.505]
    ppo_untuned_v3_scores = [0.372, 0.4105, 0.432]
    ppo_diversity_v3_scores = [0.313, 0.3306, 0.352]
    ppo_choosek_drop1_v3_scores = [0.528, 0.5329, 0.542]

    v3_scores = [dqn_v3_scores, dqn_diversity_v3_scores, a3c_v3_scores, a3c_diversity_v3_scores,
                 ppo_diversity_v3_scores, ppo_untuned_v3_scores, ppo_choosek_drop1_v3_scores, ppo_v3_scores]
    y += [alg[1] for alg in v3_scores]
    y_err += [np.array([alg[1] for alg in v3_scores]) - np.array([alg[0] for alg in v3_scores]),
              np.array([alg[2] for alg in v3_scores]) - np.array([alg[1] for alg in v3_scores])]

    # 4
    dqn_v4_scores = [0.374, 0.374, 0.378]
    dqn_diversity_v4_scores = [0.294, 0.296, 0.3]
    a3c_v4_scores = [0.54, 0.5436, 0.55]
    a3c_diversity_v4_scores = [0.494, 0.4964, 0.502]
    ppo_v4_scores = [0.633, 0.635, 0.638]
    ppo_untuned_v4_scores = [0.474, 0.52, 0.564]
    ppo_diversity_v4_scores = [0.472, 0.486, 0.5]
    ppo_choosek_drop1_v4_scores = [0.734, 0.7422, 0.76]

    v4_scores = [dqn_v4_scores, dqn_diversity_v4_scores, a3c_v4_scores, a3c_diversity_v4_scores,
                 ppo_diversity_v4_scores, ppo_untuned_v4_scores, ppo_choosek_drop1_v4_scores, ppo_v4_scores]
    y += [alg[1] for alg in v4_scores]
    y_err += [np.array([alg[1] for alg in v4_scores]) - np.array([alg[0] for alg in v4_scores]),
              np.array([alg[2] for alg in v4_scores]) - np.array([alg[1] for alg in v4_scores])]

    # 5
    dqn_v5_scores = [0.202, 0.204, 0.208]
    dqn_diversity_v5_scores = [0.178, 0.179, 0.18]
    a3c_v5_scores = [0.256, 0.2568, 0.26]
    a3c_diversity_v5_scores = [0.272, 0.273, 0.274]
    ppo_v5_scores = [0.418, 0.426, 0.432]
    ppo_untuned_v5_scores = [0.24, 0.2521, 0.262]
    ppo_diversity_v5_scores = [0.262, 0.287, 0.304]
    ppo_choosek_drop1_v5_scores = [0.244, 0.2543, 0.27]

    v5_scores = [dqn_v5_scores, dqn_diversity_v5_scores, a3c_v5_scores, a3c_diversity_v5_scores,
                 ppo_diversity_v5_scores, ppo_untuned_v5_scores, ppo_choosek_drop1_v5_scores, ppo_v5_scores]
    y += [alg[1] for alg in v5_scores]
    y_err += [np.array([alg[1] for alg in v5_scores]) - np.array([alg[0] for alg in v5_scores]),
              np.array([alg[2] for alg in v5_scores]) - np.array([alg[1] for alg in v5_scores])]

    y /= num_checkpoints
    y_err /= num_checkpoints

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=RL_ALGOS_COLORS)
    plt.title('')
    plt.xticks(x, RL_ALGOS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.5)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    # plt.xlabel("Algorithm")
    plt.ylabel("Score(S)", fontsize=18)
    # plt.suptitle(f"RL algorithms on Mas Data")
    plt.show()


def plot_mas_rl_algos():
    RL_ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO+DropOne', 'PPO']
    RL_ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#a1153b', '#eb144f']
    x = [*range(len(RL_ALGOS))]
    y = np.zeros(len(RL_ALGOS))
    y_err = np.zeros((2, len(RL_ALGOS)))
    num_checkpoints = 4

    # 7
    dqn_v7_scores = [0.608, 0.608, 0.618]
    dqn_diversity_v7_scores = [0.6041, 0.6142, 0.618]
    a3c_v7_scores = [0.6525, 0.6525, 0.6525]
    a3c_diversity_v7_scores = [0.5975, 0.5996, 0.6075]
    ppo_untuned_v7_scores = [0.64, 0.649, 0.665]
    ppo_v7_scores = [0.6875, 0.6888, 0.6925]
    ppo_diversity_v7_scores = [0.63, 0.6488, 0.6688]
    ppo_choosek_drop1_v7_scores = [0.6475, 0.6544, 0.6625]

    v7_scores = [dqn_v7_scores, dqn_diversity_v7_scores, a3c_v7_scores, a3c_diversity_v7_scores,
                 ppo_diversity_v7_scores, ppo_untuned_v7_scores, ppo_choosek_drop1_v7_scores, ppo_v7_scores]

    y += [alg[1] for alg in v7_scores]
    y_err += [np.array([alg[1] for alg in v7_scores]) - np.array([alg[0] for alg in v7_scores]),
              np.array([alg[2] for alg in v7_scores]) - np.array([alg[1] for alg in v7_scores])]

    # 8
    dqn_v8_scores = [0.628, 0.628, 0.632]
    dqn_diversity_v8_scores = [0.6141, 0.6142, 0.628]
    a3c_v8_scores = [0.75, 0.75, 0.75]
    a3c_diversity_v8_scores = [0.765, 0.7652, 0.7675]
    ppo_untuned_v8_scores = [0.625, 0.633, 0.647]
    ppo_v8_scores = [0.75, 0.78, 0.79]
    ppo_diversity_v8_scores = [0.6637, 0.6893, 0.7147]
    ppo_choosek_drop1_v8_scores = [0.75, 0.75, 0.75]

    v8_scores = [dqn_v8_scores, dqn_diversity_v8_scores, a3c_v8_scores, a3c_diversity_v8_scores,
                 ppo_diversity_v8_scores, ppo_untuned_v8_scores, ppo_choosek_drop1_v8_scores, ppo_v8_scores]
    y += [alg[1] for alg in v8_scores]
    y_err += [np.array([alg[1] for alg in v8_scores]) - np.array([alg[0] for alg in v8_scores]),
              np.array([alg[2] for alg in v8_scores]) - np.array([alg[1] for alg in v8_scores])]

    # 9
    dqn_v9_scores = [0.6123, 0.6124, 0.6162]
    dqn_diversity_v9_scores = [0.618, 0.6192, 0.6244]
    a3c_v9_scores = [0.6181, 0.6192, 0.622]
    a3c_diversity_v9_scores = [0.6595, 0.6613, 0.6659]
    ppo_v9_scores = [0.6637, 0.6893, 0.7147]
    ppo_untuned_v9_scores = [0.6145, 0.6275, 0.6445]
    ppo_diversity_v9_scores = [0.611, 0.634, 0.652]
    ppo_choosek_drop1_v9_scores = [0.5728, 0.6054, 0.6217]

    v9_scores = [dqn_v9_scores, dqn_diversity_v9_scores, a3c_v9_scores, a3c_diversity_v9_scores,
                 ppo_diversity_v9_scores, ppo_untuned_v9_scores, ppo_choosek_drop1_v9_scores, ppo_v9_scores]
    y += [alg[1] for alg in v9_scores]
    y_err += [np.array([alg[1] for alg in v9_scores]) - np.array([alg[0] for alg in v9_scores]),
              np.array([alg[2] for alg in v9_scores]) - np.array([alg[1] for alg in v9_scores])]

    # 10
    dqn_v10_scores = [0.6436, 0.6439, 0.6475]
    dqn_diversity_v10_scores = [0.6475, 0.6477, 0.6514]
    a3c_v10_scores = [0.6719, 0.672, 0.6758]
    a3c_diversity_v10_scores = [0.6691, 0.6698, 0.673]
    ppo_v10_scores = [0.675, 0.691, 0.722]
    ppo_untuned_v10_scores = [0.6562, 0.6761, 0.6914]
    ppo_diversity_v10_scores = [0.6612, 0.6804, 0.7003]
    ppo_choosek_drop1_v10_scores = [0.6758, 0.6951, 0.7109]

    v10_scores = [dqn_v10_scores, dqn_diversity_v10_scores, a3c_v10_scores, a3c_diversity_v10_scores,
                  ppo_diversity_v10_scores, ppo_untuned_v10_scores, ppo_choosek_drop1_v10_scores, ppo_v10_scores]
    y += [alg[1] for alg in v10_scores]
    y_err += [np.array([alg[1] for alg in v10_scores]) - np.array([alg[0] for alg in v10_scores]),
              np.array([alg[2] for alg in v10_scores]) - np.array([alg[1] for alg in v10_scores])]

    y /= num_checkpoints
    y_err /= num_checkpoints

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=RL_ALGOS_COLORS)
    plt.title('')
    plt.xticks(x, RL_ALGOS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.5)
    plt.yticks(np.arange(0, 0.8, 0.1), fontsize=18)

    # plt.xlabel("Algorithm")
    plt.ylabel("Score(S)", fontsize=18)
    # plt.suptitle(f"RL algorithms on Mas Data")
    plt.show()


def plot_query_answer_time():
    # ALG_NAMES = ['VAE', 'caching', 'random', 'skyline', 'brute force', 'K-means', 'top-k', 'greedy', 'PPO']
    x = [*range(len(ALG_NAMES))]
    y_min = [5.54 + 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006]
    y = [8.47 + 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011]
    y_max = [11.65 + 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017]

    y_err = [[y[i] - y_min[i] for i in range(len(y))],
             [y_max[i] - y[i] for i in range(len(y))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=18)
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.yticks(np.arange(0, 12, 1), fontsize=18)

    plt.ylabel("seconds", fontsize=18)
    plt.show()


def plot_flights_aqp_results2():
    # TODO: change this to show query categories in x-axis
    # TODO: plot quality instead of error
    # TODO: add plot showing miss rate
    xticks = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Avg']

    x_orig = [*range(len(xticks))]
    x_vae = [x - 0.2 for x in x_orig]
    x_ours = [x for x in x_orig]

    y_vae = [0.96, 0.999, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    y_vae.append(np.average(y_vae))
    y_ours = [0.009, 0.068, 0.883, 0.639, 0.675, 0.903, 0.888, 0.714, 0.887, 0.746, 0.666]
    y_ours.append(np.average(y_ours))

    plt.bar(x_vae, y_vae, width=0.2, label='VAE')
    plt.bar(x_ours, y_ours, width=0.2, label='RL')
    plt.xticks(x_orig, xticks)

    plt.xlabel("Query")
    plt.ylabel("Relative Error")
    plt.suptitle(f"AQP queries on Flights data on 1% sample")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_ray_avg_mas_results()
    # plot_ray_avg_imdb_results()
    # plot_imdb_runtime_results()
    # plot_mas_runtime_results()
    # plot_ray_avg_mas_threshold_results()
    # plot_ray_avg_imdb_threshold_results()
    # plot_drop1_imdb_results()
    # plot_imdb_k_array_results()
    # plot_view_size_array_results()
    # plot_imdb_rl_algos()
    # plot_mas_rl_algos()
    # plot_flights_aqp_results2()
    # plot_query_answer_time()
    plot_drop1_extended_results()

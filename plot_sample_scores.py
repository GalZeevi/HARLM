import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib import transforms
import pandas as pd
import seaborn as sns

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


ALG_NAMES = ['VAE', 'caching', 'random', 'quickr', 'verdict', 'skyline', 'brute force', 'K-means', 'top-k', 'greedy',
             'ASQP-RL']
ALG_SHORTCUTS = ['VAE', 'CACH', 'RAN', 'QUIK', 'VERD', 'SKY', 'BRT', 'QRD', 'TOP', 'GRE', 'ASQP-RL']
# TOOD: choose two new colors for verdict and quickr
COLORS = ['#c93699', '#B233FF', '#36c966', '#E46D08', '#E4DB08', '#56a7a9', '#a95856', '#e9b516', '#9e617e', '#b4e41b',
          '#eb144f']


def get_ray_imdb_data(checkpoint):
    if checkpoint == 2:
        vae_scores = [0] * 3
        cache_scores = [0, 0.00528, 0.014]
        random_scores = [0.285, 0.333, 0.382]
        verdict_scores = [0.284, 0.36184, 0.39184]
        quickr_scores = [0.296, 0.31, 0.33]
        skyline_scores = [0.394] * 3
        brute_force = [0.284] * 3
        kmeans_scores = [0.376] * 3
        topk_scores = [0.377, 0.377, 0.377]
        # ppo1_scores = [0.468, 0.484, 0.516]
        apex_scores = [0.562, 0.564, 0.566]
        greedy_scores = [0] * 3
        ppo_scores = [0.472, 0.482, 0.494]
        # ppo2_scores = [0.432, 0.464, 0.483]
    elif checkpoint == 3:
        vae_scores = [0.01] * 3
        cache_scores = [0.102, 0.11, 0.12]
        random_scores = [0.182, 0.212, 0.244]
        verdict_scores = [0.192, 0.22, 0.26]
        quickr_scores = [0.212, 0.24, 0.27]
        skyline_scores = [0.273] * 3
        brute_force = [0.274] * 3
        kmeans_scores = [0.306] * 3
        topk_scores = [0.172, 0.201, 0.216]
        # ppo1_scores = [0.490, 0.492, 0.494]
        apex_scores = [0.32, 0.322, 0.324]
        greedy_scores = [0] * 3
        ppo_scores = [0.498, 0.501, 0.505]
        # ppo2_scores = [0.22, 0.26, 0.316]
    elif checkpoint == 4:
        vae_scores = [0] * 3
        cache_scores = [0, 0.001, 0.004]
        random_scores = [0.32, 0.353, 0.385]
        verdict_scores = [0.335, 0.37, 0.4]
        quickr_scores = [0.34, 0.39, 0.41]
        skyline_scores = [0.47] * 3
        brute_force = [0.32] * 3
        kmeans_scores = [0.404] * 3
        topk_scores = [0.28, 0.28, 0.28]
        # ppo1_scores = [0.644, 0.649, 0.653]
        apex_scores = [0.374, 0.374, 0.378]
        greedy_scores = [0] * 3
        ppo_scores = [0.633, 0.635, 0.638]
        # ppo2_scores = [0.54, 0.61, 0.66]
    elif checkpoint == 5:
        vae_scores = [0] * 3
        cache_scores = [0.208, 0.22, 0.242]
        random_scores = [0.217, 0.258, 0.301]
        verdict_scores = [0.274, 0.3014, 0.33]
        quickr_scores = [0.186, 0.22, 0.262]
        skyline_scores = [0.251] * 3
        brute_force = [0.31] * 3
        kmeans_scores = [0.2] * 3
        topk_scores = [0.225, 0.225, 0.225]
        # ppo1_scores = [0.438, 0.445, 0.449]
        apex_scores = [0.202, 0.204, 0.208]
        greedy_scores = [0] * 3
        ppo_scores = [0.418, 0.426, 0.432]
        # ppo2_scores = [0.182, 0.21, 0.252]
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, quickr_scores,
                  verdict_scores, skyline_scores, brute_force, kmeans_scores,
                  topk_scores, greedy_scores, ppo_scores]
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
        verdict_scores = [0.3, 0.47, 0.5]
        quickr_scores = [0.3, 0.455, 0.5]
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
        verdict_scores = [0.2, 0.37, 0.4]
        quickr_scores = [0.3, 0.39, 0.4]
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
        verdict_scores = [0.5, 0.58, 0.6]
        quickr_scores = [0.5, 0.58, 0.6]
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
        verdict_scores = [0.2, 0.292, 0.4]
        quickr_scores = [0.3, 0.303, 0.4]
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

    all_scores = [vae_scores, cache_scores, random_scores, quickr_scores,
                  verdict_scores, skyline_scores, brute_force, kmeans_scores,
                  topk_scores, greedy_scores, ppo_scores]
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


def plot_imdb_baselines():  # TODO: update with new experiments (like 42)
    checkpoints = [2, 3, 4, 5]
    plt.figure(figsize=(8, 8))
    fontsize = 25

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
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=fontsize)

    plt.ylabel("Score(S)", fontsize=fontsize)
    plt.savefig('plots/imdb_baselines2.pdf')
    plt.savefig('plots/imdb_baselines2.png')
    plt.show()


METHODS = ['UNIVERS', 'VERDICT', 'QUICKR']


def plot_imdb_initial_sample():
    plt.figure(figsize=(8, 8))
    fontsize = 25

    x = [*range(len(METHODS))]
    y = np.zeros(range(len(ALG_NAMES)))
    y_min = np.zeros(range(len(ALG_NAMES)))
    y_max = np.zeros(range(len(ALG_NAMES)))

    # universal table
    # checkpoint 22, 23, 24, 25
    y = np.append(y, np.average([0.489, 0.501, 0.635, 0.426]))
    y_min = np.append(y_min, np.average([0.477, 0.498, 0.633, 0.418]))
    y_max = np.append(y_max, np.average([0.5, 0.505, 0.638, 0.432]))

    # verdict
    # checkpoint 22, 23, 24, 25
    y = np.append(y, np.average([0.464, 0.4745, 0.61, 0.298]))
    y_min = np.append(y_min, np.average([0.432, 0.42, 0.54, 0.28]))
    y_max = np.append(y_max, np.average([0.483, 0.516, 0.66, 0.308]))

    # quickr
    # checkpoint 32, 33, 34, 35
    y = np.append(y, np.average([0.4356, 0.4957, 0.6916, 0.3096]))
    y_min = np.append(y_min, np.average([0.3733, 0.47, 0.656, 0.296]))
    y_max = np.append(y_max, np.average([0.482, 0.55, 0.72, 0.32]))

    y_err = [[y[i] - y_min[i] for i in range(len(y))], [y_max[i] - y[i] for i in range(len(y))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=['#e74c3c', '#1abc9c', '#96FA0B'])
    plt.title('')
    plt.xticks(x, METHODS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=fontsize)

    plt.ylabel("Score(S)", fontsize=fontsize)
    plt.savefig('plots/imdb_initial_sample.pdf')
    plt.savefig('plots/imdb_initial_sample.png')
    plt.show()


def plot_mas_initial_sample():
    plt.figure(figsize=(8, 8))
    fontsize = 25

    x = [*range(len(METHODS))]
    y = np.zeros(range(len(ALG_NAMES)))
    y_min = np.zeros(range(len(ALG_NAMES)))
    y_max = np.zeros(range(len(ALG_NAMES)))

    # universal table
    # checkpoint 27, 28, 29, 30
    y = np.append(y, np.average([0.6544, 0.75, 0.6893, 0.6804]))
    y_min = np.append(y_min, np.average([0.6475, 0.75, 0.6637, 0.6612]))
    y_max = np.append(y_max, np.average([0.6625, 0.75, 0.7147, 0.7003]))

    # verdict
    # checkpoint 27, 28, 29, 30
    y = np.append(y, np.average([0.63, 0.5015, 0.39, 0.555]))
    y_min = np.append(y_min, np.average([0.617, 0.47, 0.375, 0.5128125]))
    y_max = np.append(y_max, np.average([0.66, 0.515, 0.41, 0.585]))

    # quickr # TODO: complete
    # checkpoint 27, 28, 29, 30
    y = np.append(y, np.average([0, 0, 0, 0]))
    y_min = np.append(y_min, np.average([0, 0, 0, 0]))
    y_max = np.append(y_max, np.average([0, 0, 0, 0]))

    y_err = [[y[i] - y_min[i] for i in range(len(y))], [y_max[i] - y[i] for i in range(len(y))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=['#e74c3c', '#1abc9c', '#34495e'])
    plt.title('')
    plt.xticks(x, METHODS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.8, 0.1), fontsize=fontsize)

    plt.ylabel("Score(S)", fontsize=fontsize)
    plt.savefig('plots/mas_initial_sample.pdf')
    plt.savefig('plots/mas_initial_sample.png')
    plt.show()


def plot_imdb_runtime_results():
    x = [*range(len(ALG_NAMES))]
    y = [32, 1, 1.2, 25, 48, 30, 1.3, 48, 6]

    fontsize = 25
    plt.figure(figsize=(8, 8))
    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 50, 5), fontsize=fontsize)

    plt.ylabel("Hours", fontsize=fontsize)
    plt.savefig('plots/imdb_runtime.pdf')
    plt.savefig('plots/imdb_runtime.png')
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


MAS_ALG_NAMES = ['VAE', 'caching', 'random', 'skyline', 'brute force', 'K-means', 'top-k', 'greedy', 'ASQP-RL']
MAS_ALG_SHORTCUTS = ['VAE', 'CACH', 'RAN', 'SKY', 'BRT', 'QRD', 'TOP', 'GRE', 'ASQP-RL']
MAS_COLORS = ['#c93699', '#B233FF', '#36c966', '#56a7a9', '#a95856', '#e9b516', '#9e617e', '#b4e41b', '#eb144f']


def plot_mas_runtime_results():
    x = [*range(len(ALG_NAMES))]
    y = [12, 0.1, 0.2, 8, 48, 10, 0.3, 48, 4]

    fontsize = 25
    plt.figure(figsize=(8, 8))
    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3)
    plt.yticks(np.arange(0, 50, 5), fontsize=fontsize)

    plt.ylabel("Hours", fontsize=fontsize)
    plt.savefig('plots/mas_runtime.pdf')
    plt.savefig('plots/mas_runtime.png')
    plt.show()


def get_ray_mas_data(checkpoint):
    if checkpoint == 7:
        cache_scores = [0.324, 0.341, 0.375]
        topk_scores = [0.625] * 3
        random_scores = [0.182, 0.203, 0.230]
        verdict_scores = [0.177, 0.204, 0.24]
        quickr_scores = [0.15, 0.179, 0.215]
        skyline_scores = [0.265] * 3
        brute_force_scores = [0.27] * 3
        kmeans_scores = [0.3475] * 3
        greedy_train_scores = [0.537] * 3
        greedy_train_plus_diversity_scores = [0.532] * 3
        ppo_scores = [0.69, 0.699, 0.715]
        ppo_plus_diversity_scores = [0.6339, 0.6552, 0.6741]
        # ppo_0dot3_scores = [0.6431, 0.65, 0.6638]
        apex_scores = [0.668, 0.668, 0.672]
        apex_plus_diversity_scores = [0.6641, 0.6642, 0.668]
        vae_scores = [0.04] * 3
    elif checkpoint == 8:
        cache_scores = [0.4, 0.4171, 0.455]
        topk_scores = [0.455] * 3
        random_scores = [0.25, 0.283, 0.324]
        verdict_scores = [0.235, 0.289, 0.3475]
        quickr_scores = [0.152, 0.186, 0.217]
        skyline_scores = [0.3125] * 3
        brute_force_scores = [0.405] * 3
        kmeans_scores = [0.402] * 3
        greedy_train_scores = [0.632] * 3
        greedy_train_plus_diversity_scores = [0.632] * 3
        ppo_scores = [0.675, 0.693, 0.697]
        ppo_plus_diversity_scores = [0.7525, 0.7621, 0.77]
        # ppo_0dot3_scores = [0.76, 0.768, 0.782]
        apex_scores = [0.765, 0.7651, 0.7675]
        apex_plus_diversity_scores = [0.7675, 0.7677, 0.775]
        vae_scores = [0.02, 0.02, 0.03]
    elif checkpoint == 9:
        cache_scores = [0.125] * 3
        topk_scores = [0.25] * 3
        random_scores = [0.165, 0.205, 0.24]
        verdict_scores = [0.139, 0.204, 0.241]
        quickr_scores = [0.13, 0.160, 0.21]
        skyline_scores = [0.217] * 3
        brute_force_scores = [0.305] * 3
        kmeans_scores = [0.14] * 3
        greedy_train_scores = [0.395] * 3
        greedy_train_plus_diversity_scores = [0.337] * 3
        ppo_scores = [0.661, 0.694, 0.702]
        ppo_plus_diversity_scores = [0.6637, 0.6893, 0.7147]
        # ppo_0dot3_scores = [0.655, 0.685, 0.718]
        apex_scores = [0.6523, 0.6524, 0.6562]
        apex_plus_diversity_scores = [0.658, 0.6592, 0.6644]
        vae_scores = [0.06] * 3
    elif checkpoint == 10:
        cache_scores = [0] * 3
        topk_scores = [0.507] * 3
        random_scores = [0.075, 0.120, 0.157]
        verdict_scores = [0.08, 0.121, 0.162]
        quickr_scores = [0.042, 0.076, 0.126]
        skyline_scores = [0.14] * 3
        brute_force_scores = [0.21] * 3
        kmeans_scores = [0.3814] * 3
        greedy_train_scores = [0.507] * 3
        greedy_train_plus_diversity_scores = [0.51] * 3
        ppo_scores = [0.705, 0.731, 0.762]
        ppo_plus_diversity_scores = [0.6612, 0.6804, 0.7003]
        # ppo_0dot3_scores = [0.660, 0.676, 0.700]
        apex_scores = [0.6836, 0.6839, 0.6875]
        apex_plus_diversity_scores = [0.6875, 0.6877, 0.6914]
        vae_scores = [0.06] * 3
    else:
        raise Exception('checkpoint not recognized!')

    all_scores = [vae_scores, cache_scores, random_scores, quickr_scores,
                  verdict_scores, skyline_scores, brute_force_scores, kmeans_scores,
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
        verdict_scores = [0.125, 0.175, 0.25]
        quickr_scores = [0.125, 0.1325, 0.25]
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
        verdict_scores = [0.25, 0.325, 0.5]
        quickr_scores = [0.125, 0.17, 0.25]
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
        verdict_scores = [0.125, 0.2125, 0.5]
        quickr_scores = [0.125, 0.23, 0.25]
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
        verdict_scores = [0.125, 0.1275, 0.25]
        quickr_scores = [0.0, 0.0925, 0.25]
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

    all_scores = [vae_scores, cache_scores, random_scores, quickr_scores,
                  verdict_scores, skyline_scores, brute_force_scores, kmeans_scores,
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
        ax.set_xticklabels(ALG_NAMES, fontsize=8)

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
        ax.set_xticklabels(ALG_NAMES, fontsize=8)

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


def plot_mas_baselines():
    checkpoints = [7, 8, 9, 10]
    fontsize = 25
    plt.figure(figsize=(8, 8))

    x = []
    y = np.zeros(len(ALG_NAMES))
    y_err = np.zeros((2, len(ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_mas_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    plt.title('')
    plt.xticks(x, ALG_SHORTCUTS, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.8, 0.1), fontsize=fontsize)

    plt.ylabel("Score(S)", fontsize=fontsize)
    plt.savefig('plots/mas_baselines2.pdf')
    plt.savefig('plots/mas_baselines2.png')
    plt.show()


def plot_ray_avg_mas_threshold_results():
    checkpoints = [7, 8, 9, 10]

    x = []
    y = np.zeros(len(ALG_NAMES))
    y_err = np.zeros((2, len(ALG_NAMES)))
    for c in checkpoints:
        data = get_ray_mas_threshold_data(c)
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
    experiments_short = ['DRP', 'DRP+R', 'GSL+DRP', 'GSL']
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
        [0.4945, 0.506, 0.519],  # DQN
        [0.4217, 0.4235, 0.4406],  # Diversity DQN
        [0.452, 0.4636, 0.478],  # A3C
        [0.446, 0.4509, 0.468],  # A3C Diversity
        [0.49025, 0.50295, 0.5165],  # PPO Diversity
        [0.485, 0.508125, 0.522],  # PPO not tuned
        [0.58804, 0.59849, 0.60894],  # PPO
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
    plt.yticks(np.arange(0, 0.7, 0.1), fontsize=18)

    plt.ylabel("Score(S)", fontsize=18)
    plt.legend(bbox_to_anchor=(0.48, 1.35), loc='upper center', ncol=4, prop={'size': 18})
    plt.savefig('plots/all_envs_all_rl.pdf')
    plt.savefig('plots/all_envs_all_rl.png')
    plt.show()


# IMDB_ALG_NAMES = ['random', 'skyline', 'K-means', 'top-k', 'DQN', 'PPO']

def plot_imdb_k_array_results():
    x = [*range(len(ALG_NAMES))]

    y_1k = [
        0,  # VAE
        0.15,  # caching
        0.252,  # random
        0.303,  # skyline
        0.29,  # brute
        0.306,  # kmeans
        0.381,  # topk
        0.25,  # greedy
        0.601,  # ppo
    ]
    min_max_1k = [
        [0,
         0.152,
         0.232,
         0.303,
         0.29,  # brute
         0.306,
         0.381,
         0.25,
         0.598],
        [0.01,
         0.16,
         0.274,
         0.303,
         0.29,  # brute
         0.306,
         0.381,
         0.25,
         0.615]
    ]
    yerr_1k = [[abs(a - b) for (a, b) in zip(min_max_1k[0], y_1k)],
               [abs(a - b) for (a, b) in zip(min_max_1k[1], y_1k)]]

    y_5k = [
        0.0,  # VAE
        0.172,  # caching
        0.4354,  # random
        0.471,  # skyline
        0.45,  # brute
        0.316,  # kmeans
        0.406,  # topk
        0.30,  # greedy
        0.6876,  # ppo
    ]
    min_max_5k = [
        [0,
         0.17,
         0.414,
         0.471,
         0.45,  # brute
         0.316,
         0.406,
         0.30,
         0.674],
        [0.01,
         0.24,
         0.456,
         0.471,
         0.45,  # brute
         0.316,
         0.406,
         0.30,
         0.698]
    ]
    yerr_5k = [[abs(a - b) for (a, b) in zip(min_max_5k[0], y_5k)],
               [abs(a - b) for (a, b) in zip(min_max_5k[1], y_5k)]]

    y_10k = [
        0.01,  # VAE
        0.20,  # caching
        0.469,  # random
        0.531,  # skyline
        0.526,  # brute
        0.432,  # kmeans
        0.47,  # topk
        0.33,  # greedy
        0.7253,  # ppo
    ]
    min_max_10k = [
        [0,
         0.17,
         0.43,
         0.531,
         0.526,
         0.432,
         0.47,
         0.33,
         0.71],
        [0.02,
         0.24,
         0.492,
         0.531,
         0.526,
         0.432,
         0.47,
         0.33,
         0.73]
    ]
    yerr_10k = [[abs(a - b) for (a, b) in zip(min_max_10k[0], y_10k)],
                [abs(a - b) for (a, b) in zip(min_max_10k[1], y_10k)]]

    y_15k = [
        0.01,  # VAE
        0.249,  # caching
        0.50164,  # random
        0.574,  # skyline
        0.566,  # brute
        0.492,  # kmeans
        0.526,  # topk
        0.38,  # greedy
        0.774,  # ppo
    ]
    min_max_15k = [
        [0,
         0.284,
         0.466,
         0.574,
         0.566,
         0.492,
         0.526,
         0.38,
         0.768],
        [0.02,
         0.208,
         0.522,
         0.574,
         0.566,
         0.492,
         0.526,
         0.38,
         0.78]
    ]
    yerr_15k = [[abs(a - b) for (a, b) in zip(min_max_15k[0], y_15k)],
                [abs(a - b) for (a, b) in zip(min_max_15k[1], y_15k)]]

    # plt.xticks(x, IMDB_ALG_SHORTCUTS, rotation=90, fontsize=18)
    # plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)

    # plt.ylabel("Score(S)", fontsize=18)

    fontsize = 18
    fig, axs = plt.subplots(4, figsize=(8, 8))
    axs[0].bar(x, y_1k, yerr=yerr_1k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[0].set_title('k=1000', y=1.0, pad=-15, fontsize=fontsize)
    axs[0].set_yticks(np.arange(0, 0.9, 0.2))
    # axs[0].set_xticks([])
    axs[0].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[1].bar(x, y_5k, yerr=yerr_5k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[1].set_title('k=5000', y=1.0, pad=-15, fontsize=fontsize)
    # axs[1].set_xticks([])
    axs[1].set_yticks(np.arange(0, 0.9, 0.2))
    axs[1].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[2].bar(x, y_10k, yerr=yerr_10k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[2].set_title('k=10000', y=1.0, pad=-15, fontsize=fontsize)
    # axs[2].set_xticks([])
    axs[2].set_yticks(np.arange(0, 0.9, 0.2))
    axs[2].grid(which='major', color='#bfc1c7', linewidth=0.8)

    axs[3].bar(x, y_15k, yerr=yerr_15k, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[3].set_title('k=15000', y=1.0, pad=-15, fontsize=fontsize)
    axs[3].set_yticks(np.arange(0, 0.9, 0.2), fontsize=fontsize)
    axs[3].grid(which='major', color='#bfc1c7', linewidth=0.8)

    # plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    # plt.setp(axs, yticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    # plt.suptitle('Baselines with varying sample size')

    # axs[-1].set_xticklabels(IMDB_ALG_NAMES, fontsize=8)
    # plt.subplots_adjust(bottom=0.3)
    plt.minorticks_on()

    plt.setp(axs[:3], xticks=x, xticklabels=[])
    plt.setp([axs[-1]], xticks=x, xticklabels=ALG_SHORTCUTS)
    axs[-1].tick_params(axis='x', labelsize=fontsize, rotation=90)
    for ax in axs:
        ax.set_ylabel("Score(S)", fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.5,
                        bottom=0.15,
                        left=0.15)

    plt.savefig('plots/tune_k.pdf')
    plt.savefig('plots/tune_k.png')
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
        0.401,  # topk
        0.25,  # greedy
        0.659,  # ppo
    ]
    min_max_25 = [
        [0.01,
         0.102,
         0.244,
         0.440,
         0.4,
         0.348,
         0.401,
         0.25,  # greedy
         0.643],
        [0.02,
         0.115,
         0.344,
         0.49,
         0.4,
         0.348,
         0.401,
         0.25,  # greedy
         0.662]
    ]
    yerr_25 = [[abs(a - b) for (a, b) in zip(min_max_25[0], y_25)],
               [abs(a - b) for (a, b) in zip(min_max_25[1], y_25)]]

    y_50 = [
        0,  # VAE
        0.15,  # caching
        0.212,  # random
        0.303,  # skyline
        0.29,  # brute
        0.306,  # kmeans
        0.381,  # topk
        0.22,  # greedy
        0.601,  # ppo
    ]
    min_max_50 = [
        [0,
         0.152,
         0.182,
         0.303,
         0.29,
         0.306,
         0.381,
         0.22,  # greedy
         0.598],
        [0.01,
         0.16,
         0.244,
         0.303,
         0.29,
         0.306,
         0.381,
         0.22,  # greedy
         0.605]
    ]
    yerr_50 = [[abs(a - b) for (a, b) in zip(min_max_50[0], y_50)],
               [abs(a - b) for (a, b) in zip(min_max_50[1], y_50)]]

    y_75 = [
        0.015,  # VAE
        0.139,  # caching
        0.15,  # random
        0.247,  # skyline
        0.25,  # brute
        0.303,  # kmeans
        0.257,  # topk
        0.17,  # greedy
        0.561,  # ppo
    ]
    min_max_75 = [
        [0,
         0.124,
         0.12,
         0.247,
         0.25,  # brute
         0.292,
         0.257,
         0.17,  # greedy
         0.558],
        [0.025,
         0.208,
         0.17,
         0.247,
         0.25,  # brute
         0.313,
         0.257,
         0.17,  # greedy
         0.576]
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
        0.231,  # topk
        0.13,  # greedy
        0.5353,  # ppo
    ]
    min_max_100 = [
        [0,
         0.,
         0.095,
         0.190,
         0.23,  # brute
         0.28,
         0.231,
         0.13,  # greedy
         0.52],
        [0.07,
         0.1,
         0.13,
         0.190,
         0.23,  # brute
         0.28,
         0.231,
         0.13,  # greedy
         0.55]
    ]
    yerr_100 = [[abs(a - b) for (a, b) in zip(min_max_100[0], y_100)],
                [abs(a - b) for (a, b) in zip(min_max_100[1], y_100)]]

    # plt.xticks(x, IMDB_ALG_SHORTCUTS, rotation=90, fontsize=18)
    # plt.yticks(np.arange(0, 0.6, 0.1), fontsize=18)
    #
    # plt.ylabel("Score(S)", fontsize=18)

    fontsize = 18
    fig, axs = plt.subplots(4, figsize=(8, 8))
    axs[0].bar(x, y_25, yerr=yerr_25, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[0].set_title('F=25', y=1.0, pad=-15, fontsize=fontsize)
    axs[0].set_yticks(np.arange(0, 0.9, 0.2))
    # axs[0].set_xticks([])
    axs[0].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[1].bar(x, y_50, yerr=yerr_50, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[1].set_title('F=50', y=1.0, pad=-15, fontsize=fontsize)
    # axs[1].set_xticks([])
    axs[1].set_yticks(np.arange(0, 0.9, 0.2))
    axs[1].grid(which='both', color='#bfc1c7', linewidth=0.8)

    axs[2].bar(x, y_75, yerr=yerr_75, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[2].set_title('F=75', y=1.0, pad=-15, fontsize=fontsize)
    # axs[2].set_xticks([])
    axs[2].set_yticks(np.arange(0, 0.9, 0.2))
    axs[2].grid(which='major', color='#bfc1c7', linewidth=0.8)

    axs[3].bar(x, y_100, yerr=yerr_100, align='center', alpha=0.5, capsize=10, width=0.6, color=COLORS)
    axs[3].set_title('F=100', y=1.0, pad=-15, fontsize=fontsize)
    axs[3].set_yticks(np.arange(0, 0.9, 0.2))
    axs[3].grid(which='major', color='#bfc1c7', linewidth=0.8)

    # plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    # plt.setp(axs, yticks=[0.1, 0.3, 0.5, 0.7, 0.9])
    # plt.suptitle('Baselines with varying sample size')

    # axs[-1].set_xticklabels(IMDB_ALG_NAMES, fontsize=8)
    # plt.subplots_adjust(bottom=0.3)
    plt.minorticks_on()

    plt.setp(axs[:3], xticks=x, xticklabels=[])
    plt.setp([axs[-1]], xticks=x, xticklabels=ALG_SHORTCUTS)
    axs[-1].tick_params(axis='x', labelsize=16, rotation=90)
    for ax in axs:
        ax.set_ylabel("Score(S)", fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.5,
                        bottom=0.15,
                        left=0.15)

    plt.savefig('plots/tune_f.pdf')
    plt.savefig('plots/tune_f.png')
    plt.show()


def plot_imdb_rl_algos():
    RL_ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO']
    xlabels = ['DQN', 'DQN+D', 'A3C', 'A3C+D', 'PPO+D', 'NT+PPO', 'PPO']
    RL_ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#eb144f']
    fontsize = 25
    plt.figure(figsize=(8, 8))

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
                 ppo_diversity_v2_scores, ppo_untuned_v2_scores, ppo_v2_scores]

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
                 ppo_diversity_v3_scores, ppo_untuned_v3_scores, ppo_v3_scores]
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
                 ppo_diversity_v4_scores, ppo_untuned_v4_scores, ppo_v4_scores]
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
                 ppo_diversity_v5_scores, ppo_untuned_v5_scores, ppo_v5_scores]
    y += [alg[1] for alg in v5_scores]
    y_err += [np.array([alg[1] for alg in v5_scores]) - np.array([alg[0] for alg in v5_scores]),
              np.array([alg[2] for alg in v5_scores]) - np.array([alg[1] for alg in v5_scores])]

    y /= num_checkpoints
    y_err /= num_checkpoints

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=RL_ALGOS_COLORS)
    plt.title('')
    plt.xticks(x, xlabels, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=fontsize)

    # plt.xlabel("Algorithm")
    plt.ylabel("Score(S)", fontsize=fontsize)
    # plt.suptitle(f"RL algorithms on Mas Data")

    plt.savefig('plots/imdb_all_rl.pdf')
    plt.savefig('plots/imdb_all_rl.png')
    plt.show()


def plot_mas_rl_algos():
    RL_ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO']
    xlabels = ['DQN', 'DQN+D', 'A3C', 'A3C+D', 'PPO+D', 'NT+PPO', 'PPO']
    RL_ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#eb144f']
    fontsize = 25
    plt.figure(figsize=(8, 8))

    x = [*range(len(RL_ALGOS))]
    y = np.zeros(len(RL_ALGOS))
    y_err = np.zeros((2, len(RL_ALGOS)))
    num_checkpoints = 4

    # 7
    dqn_v7_scores = [0.608, 0.608, 0.618]
    dqn_diversity_v7_scores = [0.5541, 0.5552, 0.558]
    a3c_v7_scores = [0.5325, 0.5325, 0.5325]
    a3c_diversity_v7_scores = [0.5375, 0.5376, 0.5395]
    ppo_untuned_v7_scores = [0.62, 0.629, 0.645]
    ppo_v7_scores = [0.6875, 0.6888, 0.6925]
    ppo_diversity_v7_scores = [0.61, 0.6288, 0.6388]
    ppo_choosek_drop1_v7_scores = [0.6475, 0.6544, 0.6625]

    v7_scores = [dqn_v7_scores, dqn_diversity_v7_scores, a3c_v7_scores, a3c_diversity_v7_scores,
                 ppo_diversity_v7_scores, ppo_untuned_v7_scores, ppo_v7_scores]

    y += [alg[1] for alg in v7_scores]
    y_err += [np.array([alg[1] for alg in v7_scores]) - np.array([alg[0] for alg in v7_scores]),
              np.array([alg[2] for alg in v7_scores]) - np.array([alg[1] for alg in v7_scores])]

    # 8
    dqn_v8_scores = [0.628, 0.628, 0.632]
    dqn_diversity_v8_scores = [0.5641, 0.5642, 0.568]
    a3c_v8_scores = [0.55, 0.55, 0.55]
    a3c_diversity_v8_scores = [0.565, 0.5652, 0.5675]
    ppo_untuned_v8_scores = [0.615, 0.623, 0.637]
    ppo_v8_scores = [0.75, 0.78, 0.79]
    ppo_diversity_v8_scores = [0.6237, 0.6393, 0.6447]
    ppo_choosek_drop1_v8_scores = [0.75, 0.75, 0.75]

    v8_scores = [dqn_v8_scores, dqn_diversity_v8_scores, a3c_v8_scores, a3c_diversity_v8_scores,
                 ppo_diversity_v8_scores, ppo_untuned_v8_scores, ppo_v8_scores]
    y += [alg[1] for alg in v8_scores]
    y_err += [np.array([alg[1] for alg in v8_scores]) - np.array([alg[0] for alg in v8_scores]),
              np.array([alg[2] for alg in v8_scores]) - np.array([alg[1] for alg in v8_scores])]

    # 9
    dqn_v9_scores = [0.6123, 0.6124, 0.6162]
    dqn_diversity_v9_scores = [0.568, 0.5692, 0.5744]
    a3c_v9_scores = [0.5581, 0.5592, 0.562]
    a3c_diversity_v9_scores = [0.5695, 0.5713, 0.5759]
    ppo_v9_scores = [0.6637, 0.6893, 0.7147]
    ppo_untuned_v9_scores = [0.6045, 0.6175, 0.6345]
    ppo_diversity_v9_scores = [0.611, 0.634, 0.642]
    ppo_choosek_drop1_v9_scores = [0.5728, 0.6054, 0.6217]

    v9_scores = [dqn_v9_scores, dqn_diversity_v9_scores, a3c_v9_scores, a3c_diversity_v9_scores,
                 ppo_diversity_v9_scores, ppo_untuned_v9_scores, ppo_v9_scores]
    y += [alg[1] for alg in v9_scores]
    y_err += [np.array([alg[1] for alg in v9_scores]) - np.array([alg[0] for alg in v9_scores]),
              np.array([alg[2] for alg in v9_scores]) - np.array([alg[1] for alg in v9_scores])]

    # 10
    dqn_v10_scores = [0.6436, 0.6439, 0.6475]
    dqn_diversity_v10_scores = [0.5975, 0.6077, 0.6114]
    a3c_v10_scores = [0.5719, 0.572, 0.5758]
    a3c_diversity_v10_scores = [0.5691, 0.5698, 0.573]
    ppo_v10_scores = [0.675, 0.691, 0.722]
    ppo_untuned_v10_scores = [0.6362, 0.6561, 0.6714]
    ppo_diversity_v10_scores = [0.6212, 0.6304, 0.6503]
    ppo_choosek_drop1_v10_scores = [0.6758, 0.6951, 0.7109]

    v10_scores = [dqn_v10_scores, dqn_diversity_v10_scores, a3c_v10_scores, a3c_diversity_v10_scores,
                  ppo_diversity_v10_scores, ppo_untuned_v10_scores, ppo_v10_scores]
    y += [alg[1] for alg in v10_scores]
    y_err += [np.array([alg[1] for alg in v10_scores]) - np.array([alg[0] for alg in v10_scores]),
              np.array([alg[2] for alg in v10_scores]) - np.array([alg[1] for alg in v10_scores])]

    y /= num_checkpoints
    y_err /= num_checkpoints

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=RL_ALGOS_COLORS)
    plt.title('')
    plt.xticks(x, xlabels, rotation=90, fontsize=fontsize)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.yticks(np.arange(0, 0.8, 0.1), fontsize=fontsize)

    # plt.xlabel("Algorithm")
    plt.ylabel("Score(S)", fontsize=fontsize)
    # plt.suptitle(f"RL algorithms on Mas Data")

    plt.savefig('plots/mas_all_rl.pdf')
    plt.savefig('plots/mas_all_rl.png')
    plt.show()


def plot_param_tune_results():
    x_lr = [0.00005, 0.0005, 0.005, 0.05]
    y_lr = [0.521, 0.521, 0.518, 0.516]
    fontsize = 20
    # plt.figure(figsize=(8, 10))

    fig, axs = plt.subplots(3, figsize=(8, 8))
    axs[0].plot(x_lr, y_lr, linestyle='--', marker='o', label='lr', color='#F6B613')
    # axs[0].set_title('Tuning lr', y=1.0, fontsize=fontsize)
    # axs[0].set_yticks(np.arange(0, 0.9, 0.2))
    # axs[0].set_xticks([])
    axs[0].grid(which='both', color='#bfc1c7', linewidth=0.8)
    axs[0].tick_params(axis='x', labelsize=20)
    axs[0].tick_params(axis='y', labelsize=20)
    # axs[0].minorticks_on()

    x_ent = [0, 0.001, 0.0015, 0.01, 0.015, 0.02]
    y_ent = [0.521, 0.60, 0.575, 0.559, 0.5392, 0.535]

    axs[1].plot(x_ent, y_ent, linestyle='--', marker='o', label='entropy_coeff', color='#971CD1')
    # axs[1].set_title('Tuning entropy coefficient', y=1.0, fontsize=fontsize)
    # axs[1].set_xticks([])
    # axs[1].set_yticks(np.arange(0, 0.9, 0.2))
    axs[1].grid(which='both', color='#bfc1c7', linewidth=0.8)
    axs[1].tick_params(axis='x', labelsize=20)
    axs[1].tick_params(axis='y', labelsize=20)
    # axs[1].minorticks_on()

    x_kl = [0.2, 0.3, 0.5, 0.7, 0.9]
    y_kl = [0.521, 0.54, 0.54, 0.538, 0.537]

    axs[2].plot(x_kl, y_kl, linestyle='--', marker='o', label='kl_coeff', color='#138DF6')
    # axs[2].set_title('Tuning kl coefficient', y=1.0, fontsize=fontsize)
    # axs[2].set_xticks([])
    # axs[2].set_yticks(np.arange(0, 0.9, 0.2))
    axs[2].grid(which='major', color='#bfc1c7', linewidth=0.8)
    axs[2].tick_params(axis='x', labelsize=20)
    axs[2].tick_params(axis='y', labelsize=20)
    # axs[2].minorticks_on()

    # axs[-1].set_xticklabels(IMDB_ALG_NAMES, fontsize=8)
    # plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(wspace=0.4,
                        hspace=1.0,
                        bottom=0.1,
                        left=0.2)

    # plt.setp(axs[:2], xticks=x, xticklabels=[])
    # plt.setp([axs[-1]], xticks=x, xticklabels=ALG_SHORTCUTS)
    # axs[-1].tick_params(axis='x', labelsize=16, rotation=90)

    for ax in axs:
        ax.legend(bbox_to_anchor=(0.5, 1.65), loc='upper center', prop={'size': 18})

    # plt.legend(bbox_to_anchor=(0.48, 1.35), loc='upper center', ncol=4, prop={'size': 18})
    for ax in axs:
        ax.set_ylabel("Score(S)", fontsize=18)
        ax.tick_params(axis='y', labelsize=18)

    plt.savefig('plots/rl_tune2.pdf')
    plt.savefig('plots/rl_tune2.png')
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


def plot_flights_aqp_results3():
    xticks = ['SUM', 'AVG', 'CNT', 'G+SUM', 'G+AVG', 'G+CNT']
    fontsize = 25
    plt.figure(figsize=(8, 6))

    x_orig = [*range(len(xticks))]
    x_gaqp = [x - 0.2 for x in x_orig]
    x_deepdb = [x for x in x_orig]
    x_ours = [x + 0.2 for x in x_orig]

    # y_gaqp = [0.002, 0.002, 0.07, 0.0001, 0.05, 0.02]
    # std_gaqp = [0.005, 0.005, 0.09, 0, 0.2, 0.09]
    # y_deepdb = [0.12, 0.03, 0.77, 0.11, 0.06, 0.41]
    # std_deepdb = [0.2, 0.05, 0.29, 0.23, 0.08, 0.47]
    # y_ours = [0.61, 0.91, 0.55, 0.002, 0.45, 0.001]
    # std_ours = [0.41, 0.13, 0.45, 0.005, 0.41, 0.003]

    y_gaqp = [0.994, 0.997, 0.91, 0.99, 0.93, 0.95]
    std_gaqp = [0.005, 0.005, 0.09, 0, 0.2, 0.09]
    y_deepdb = [0.87, 0.96, 0.22, 0.88, 0.93, 0.58]
    std_deepdb = [0.2, 0.05, 0.29, 0.23, 0.08, 0.47]
    y_ours = [0.432, 0.084, 0.44, 0.99, 0.54, 0.99]
    std_ours = [0.42, 0.13, 0.45, 0.005, 0.41, 0.003]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x_gaqp, y_gaqp, yerr=[y_gaqp, std_gaqp], alpha=0.5, capsize=5, width=0.2, label='gAQP')
    plt.bar(x_deepdb, y_deepdb, yerr=[y_deepdb, std_deepdb], alpha=0.5, capsize=5, width=0.2, label='DeepDb')
    plt.bar(x_ours, y_ours, yerr=[y_ours, std_ours], alpha=0.5, capsize=5, width=0.2, label='ASQP-RL')

    # plt.xlabel("Query")
    plt.xticks(x_orig, xticks, fontsize=20, rotation=90)
    plt.subplots_adjust(bottom=0.20, left=0.12, top=0.8)
    plt.ylabel("Average Relative Error", fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)
    plt.legend()
    plt.legend(bbox_to_anchor=(0.48, 1.25), loc='upper center', ncol=3, prop={'size': 23})
    plt.savefig('plots/flights_aqp.pdf')
    plt.savefig('plots/flights_aqp.png')
    plt.show()


def plot_flights_aqp_results4():
    xticks = ['SUM', 'CNT', 'AVG']
    fontsize = 25
    plt.figure(figsize=(8, 6))

    x_orig = [*range(len(xticks))]
    x_asqp_hist = [x - 0.2 for x in x_orig]
    x_asqp_no_hist = [x for x in x_orig]
    x_deepdb = [x + 0.2 for x in x_orig]

    y_asqp_hist = [0.8498179403, 0.3947707937, 0.7934275]
    y_asqp_hist_min = [0.001501, 0, 0.000146]
    y_asqp_hist_max = [1.54, 1, 2.020912]
    y_asqp_hist_err = [[y_asqp_hist[i] - y_asqp_hist_min[i] for i in range(len(y_asqp_hist))],
                       [y_asqp_hist_max[i] - y_asqp_hist[i] for i in range(len(y_asqp_hist))]]

    y_asqp_no_hist = [0.9999982836, 0.9999942857, 0.7934275]
    y_asqp_no_hist_min = [0.999818, 0.999847, 0.000146]
    y_asqp_no_hist_max = [1.000456, 1, 2.020912]
    y_asqp_no_hist_err = [[y_asqp_no_hist[i] - y_asqp_no_hist_min[i] for i in range(len(y_asqp_no_hist))],
                          [y_asqp_no_hist_max[i] - y_asqp_no_hist[i] for i in range(len(y_asqp_no_hist))]]

    y_deepdb = [1.969731448, 4.58461747, 0.97192375]
    y_deepdb_min = [0.178153, 0.000123, 0.2168]
    y_deepdb_max = [9.3082, 9.24396, 2.214728]
    y_deepdb_err = [[y_deepdb[i] - y_deepdb_min[i] for i in range(len(y_deepdb))],
                    [y_deepdb_max[i] - y_deepdb[i] for i in range(len(y_deepdb))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x_asqp_hist, y_asqp_hist, yerr=y_asqp_hist_err, alpha=0.5, capsize=5, width=0.2, label='ASQP-RL+HIST')
    plt.bar(x_asqp_no_hist, y_asqp_no_hist, yerr=y_asqp_no_hist_err, alpha=0.5, capsize=5, width=0.2, label='ASQP-RL')
    plt.bar(x_deepdb, y_deepdb, yerr=y_deepdb_err, alpha=0.5, capsize=5, width=0.2, label='DeepDB')

    plt.xticks(x_orig, xticks, fontsize=20, rotation=90)
    plt.subplots_adjust(bottom=0.20, left=0.12, top=0.8)
    plt.ylabel("Average Relative Error", fontsize=20)
    plt.yticks(np.arange(0, 10, 2), fontsize=20)
    plt.legend()
    plt.legend(bbox_to_anchor=(0.48, 1.25), loc='upper center', ncol=3, prop={'size': 18})
    # plt.savefig('plots/flights_aqp.pdf')
    # plt.savefig('plots/flights_aqp.png')
    plt.show()


def plot_flights_aqp_results5():
    xticks = ['SUM', 'AVG', 'CNT', 'G-SUM', 'G-AVG', 'G-CNT']
    fontsize = 25
    plt.figure(figsize=(8, 6))

    x_orig = [*range(len(xticks))]
    x_asqp_hist = [x - 0.2 for x in x_orig]
    x_asqp_no_hist = [x for x in x_orig]
    x_deepdb = [x + 0.2 for x in x_orig]

    y_asqp_hist = [0.668, 0.465, 0.129, 0.937, 0.006, 0.013]
    y_asqp_hist_min = [0.31, 0.04, 0, 0.71, 0., 0.]
    y_asqp_hist_max = [1, 0.88, 0.4, 1., 0.036, 0.043]
    y_asqp_hist_err = [[y_asqp_hist[i] - y_asqp_hist_min[i] for i in range(len(y_asqp_hist))],
                       [y_asqp_hist_max[i] - y_asqp_hist[i] for i in range(len(y_asqp_hist))]]

    y_asqp_no_hist = [0.89, 0.22, 0.99, 0.99, 0.979, 0.99]
    y_asqp_no_hist_min = [0.44, 0, 0., 0., 0.8, 0.]
    y_asqp_no_hist_max = [1, 0.84, 2.1, 2.3, 1.14, 2.]
    y_asqp_no_hist_err = [[y_asqp_no_hist[i] - y_asqp_no_hist_min[i] for i in range(len(y_asqp_no_hist))],
                          [y_asqp_no_hist_max[i] - y_asqp_no_hist[i] for i in range(len(y_asqp_no_hist))]]

    y_deepdb = [0.919, 0.94, 0.275, 0.9, 0.96, 0.1]
    y_deepdb_min = [0.73, 0.88, 0., 0.77, 0.91, 0.]
    y_deepdb_max = [1.06, 1., 0.65, 1.03, 1.01, 0.33]
    y_deepdb_err = [[y_deepdb[i] - y_deepdb_min[i] for i in range(len(y_deepdb))],
                    [y_deepdb_max[i] - y_deepdb[i] for i in range(len(y_deepdb))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x_asqp_hist, y_asqp_hist, yerr=y_asqp_hist_err, alpha=0.5, capsize=5, width=0.2, label='ASQP-RL+HIST')
    plt.bar(x_asqp_no_hist, y_asqp_no_hist, yerr=y_asqp_no_hist_err, alpha=0.5, capsize=5, width=0.2, label='ASQP-RL')
    plt.bar(x_deepdb, y_deepdb, yerr=y_deepdb_err, alpha=0.5, capsize=5, width=0.2, label='DeepDB')

    plt.xticks(x_orig, xticks, fontsize=20, rotation=90)
    plt.subplots_adjust(bottom=0.20, left=0.12, top=0.8)
    plt.ylabel("Average Relative Error", fontsize=20)
    plt.yticks(np.arange(0, 2.5, 0.5), fontsize=20)
    plt.legend()
    plt.legend(bbox_to_anchor=(0.48, 1.25), loc='upper center', ncol=3, prop={'size': 18})
    # plt.savefig('plots/flights_aqp.pdf')
    # plt.savefig('plots/flights_aqp.png')
    plt.show()


def plot_initial_sample_exp1():
    xticks = ['SAMPL', 'UNI']
    fontsize = 25
    plt.figure(figsize=(8, 6))

    x = [*range(len(xticks))]

    y_sample = [0.26, 0.3]
    y_sample_min = [0.25, 0.3]
    y_sample_max = [0.27, 0.3]
    y_sample_err = [[y_sample[i] - y_sample_min[i] for i in range(len(y_sample))],
                    [y_sample_max[i] - y_sample[i] for i in range(len(y_sample))]]

    y_universal = [0.3]
    y_universal_min = [0.3]
    y_universal_max = [0.3]
    y_universal_err = [[y_universal[i] - y_universal_min[i] for i in range(len(y_universal))],
                       [y_universal_max[i] - y_universal[i] for i in range(len(y_universal))]]

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y_sample, yerr=y_sample_err, alpha=0.5, capsize=5, width=0.2, label='Initial Sample')
    # plt.bar(x, y_universal, yerr=y_universal_err, alpha=0.5, capsize=5, width=0.2, label='Universal Table')

    plt.xticks(x, xticks, fontsize=20, rotation=90)
    plt.subplots_adjust(bottom=0.20, left=0.12, top=0.8)
    plt.ylabel("Score", fontsize=20)
    plt.yticks(np.arange(0, 0.5, 0.2), fontsize=20)
    # plt.legend()
    # plt.legend(bbox_to_anchor=(0.48, 1.25), loc='upper center', ncol=3, prop={'size': 18})
    # plt.savefig('plots/flights_aqp.pdf')
    # plt.savefig('plots/flights_aqp.png')
    plt.show()


def plot_rl_tune():
    RL_ALGOS = ['DQN', 'DQN+diverse', 'A3C', 'A3C+diverse', 'PPO+diverse', 'PPO (not tuned)', 'PPO+DropOne', 'PPO']
    RL_ALGOS_COLORS = ['#21A428', '#17D580', '#615FB1', '#9290CA', '#ed8ad3', '#ed8aa5', '#a1153b', '#eb144f']
    colors = ['#ED93C9',
              '#E778B7',
              '#E35DA5',
              '#DE428F',
              '#D92879',
              '#D40D63',
              '#CF004D',
              '#CA0040',
              '#EB144F']

    # ent_coeff in [0, 0.001, 0.01]
    # lr in [5e-6, 5e-5, 5e-4]

    ent_range = [0, 1e-3, 1e-2]
    lr_range = [5e-6, 5e-5, 5e-4]
    experiments = [*itertools.product(ent_range, lr_range)]

    x = np.array([*range(len(experiments))])
    xlabels = [f'[{p[0]},{p[1]}]' for p in experiments]
    y = \
        [0.41, 0.428, 0.432,  # ent=0
         0.43, 0.454, 0.455,  # ent=1e-3
         0.47, 0.501, 0.49]  # ent=1e-2
    y_err \
        = [0.03, 0.019, 0.022,  # ent=0
           0.019, 0.011, 0.02,  # ent=1e-3
           0.02, 0.009, 0.015]  # ent=1e-2

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10, width=0.6, color=colors)
    plt.title('')
    plt.xticks(x, xlabels, fontsize=16, rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.yticks(np.arange(0, 0.6, 0.1), fontsize=16)

    plt.ylabel("Score(S)", fontsize=16)
    plt.savefig('plots/rl_tune.pdf')
    plt.savefig('plots/rl_tune.png')
    plt.show()


def plot_query_sampling():
    fontsize = 20
    plt.figure(figsize=(8, 6))
    y = [0.12539, 0.03324, 0.01446, 0.01195, 0.01686, 0.00542, 0.00394, 0.00985, 0.00503, 0.00487, 0.00499, 0.00451,
         0.00350, 0.00230, 0.00352, 0.00364, 0.00405, 0.00416, 0.00444, 0.00369, 0.00225, 0.00358, 0.00262, 0.00222,
         0.00243, 0.00326, 0.00150, 0.00219, 0.00257, 0.00137, 0.00127, 0.00123, 0.00155, 0.00183, 0.00155, 0.00106,
         0.00138, 0.00098, 0.00106, 0.00089, 0.00142, 0.00109, 0.00097, 0.00076, 0.00207, 0.00085, 0.00122, 0.00071,
         0.00117, 0.00101, 0.00129, 0.00064, 0.00071, 0.00088]
    x = 5 * np.array([*range(len(y))])

    # plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    # plt.minorticks_on()
    plt.tick_params(axis=u'x', which=u'major', length=0)
    plt.bar(x, y, align='center', alpha=1., capsize=10, width=2.7)
    plt.title('')
    xlabels = [str(xx) if xx % 100 == 0 else '' for xx in x]
    plt.xticks(x, xlabels, fontsize=fontsize)
    plt.subplots_adjust(left=0.16)
    plt.yticks(np.arange(0, 0.13, 0.01), fontsize=fontsize)

    plt.ylabel("difference", fontsize=fontsize)
    plt.savefig('plots/query_sampling.pdf')
    plt.savefig('plots/query_sampling.png')
    plt.show()


def plot_imdb_query_runtimes():
    plt.figure(figsize=(8, 8))
    fontsize = 25

    y_1GB = [0.37290567, 0.73444667, 2.060309, 2.7051647, 3.20728937, 5.1188, 5.94528, 7.275455]
    error_1GB = [2.07983, 3.98570, 4.938157, 5.04440698,
                 6.02186715, 5.82418906, 7.49202439, 8.18376604]
    lower_err_1GB = [error_1GB[i] if y_1GB[i] - error_1GB[i] >= 0 else y_1GB[i] for i in range(len(error_1GB))]
    upper_err_1GB = error_1GB

    y_5GB = [3.39313734, 8.66066888, 12.84205509, 19.19051647,
             22.40728937, 24.41188118, 26.23452928, 28.97544255]
    error_5GB = [10.07983989, 12.98557407, 12.93814557, 17.14440698,
                 16.02186715, 15.82418906, 15.49202439, 14.98376604]
    lower_err_5GB = [error_5GB[i] if y_5GB[i] - error_5GB[i] >= 0 else y_5GB[i] for i in range(len(error_5GB))]
    upper_err_5GB = error_5GB

    y_20GB = [9.12538028, 16.80709085, 21.15697188, 27.96361342,
              31.60289612, 35.34696328, 39.05663365, 45.73312915]
    error_20GB = [4.69315427, 6.79849444, 7.22088862, 9.77672638,
                  9.2807193, 11.5866428, 14.10600016, 8.63300464]
    lower_err_20GB = [error_20GB[i] if y_20GB[i] - error_20GB[i] >= 0 else y_20GB[i] for i in range(len(error_20GB))]
    upper_err_20GB = error_20GB

    x = [*range(len(y_5GB))]

    plt.errorbar(x, y_1GB, yerr=[lower_err_1GB, upper_err_1GB], marker='o', color='#2E8B57', label='1GB', capsize=10)
    plt.errorbar(x, y_5GB, yerr=[lower_err_5GB, upper_err_5GB], marker='o', color='#3498db', label='5GB', capsize=10)
    plt.errorbar(x, y_20GB, yerr=[lower_err_20GB, upper_err_20GB], marker='o', color='#ff7f50', label='20GB',
                 capsize=10)
    plt.xticks(x, fontsize=fontsize)
    plt.yticks(np.arange(0, 60, 5), fontsize=fontsize)
    plt.xlabel('Num queries', fontsize=fontsize)
    plt.ylabel('Hours (total)', fontsize=fontsize)
    # plt.title('Line Plot with Shaded Error Area')
    plt.legend()
    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.savefig(f'plots/imdb_query_runtimes.png')
    plt.savefig(f'plots/imdb_query_runtimes.pdf')
    plt.show()


def imdb_answerable_queries():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fontsize = 25

    PERCENTAGES = ['100%', '60%', '30%']

    x = [*range(len(PERCENTAGES))]
    y_acc = []
    y_acc_min = []
    y_acc_max = []
    y_time = []
    y_time_min = []
    y_time_max = []

    # all training queries
    # checkpoint 42, trial 1000_CHOOSE_K_PPO_2023-12-15-03-10-16
    y_acc = np.append(y_acc, 0.7193)
    y_acc_min = np.append(y_acc_min, 0.6733)
    y_acc_max = np.append(y_acc_max, 0.7522)

    y_time = np.append(y_time, 4.7)
    y_time_min = np.append(y_time_min, 4.4)
    y_time_max = np.append(y_time_max, 5.0)

    # 60% percent of training queries
    # checkpoint 38, trial 1000_CHOOSE_K_PPO_2023-12-15-01-09-40
    y_acc = np.append(y_acc, 0.6084)
    y_acc_min = np.append(y_acc_min, 0.5733)
    y_acc_max = np.append(y_acc_max, 0.68)

    y_time = np.append(y_time, 2.0)
    y_time_min = np.append(y_time_min, 2.2)
    y_time_max = np.append(y_time_max, 1.82)

    # 30% percent of training queries
    # checkpoint 39, trial 1000_CHOOSE_K_PPO_2023-12-15-14-36-49
    y_acc = np.append(y_acc, 0.5133)
    y_acc_min = np.append(y_acc_min, 0.4867)
    y_acc_max = np.append(y_acc_max, 0.54)

    y_time = np.append(y_time, 1.0)
    y_time_min = np.append(y_time_min, 1.14)
    y_time_max = np.append(y_time_max, 0.792)

    y_acc_err = [[y_acc[i] - y_acc_min[i] for i in range(len(y_acc))],
                 [y_acc_max[i] - y_acc[i] for i in range(len(y_acc))]]
    y_time_err = [[y_time[i] - y_time_min[i] for i in range(len(y_time))],
                  [y_time_max[i] - y_time[i] for i in range(len(y_time))]]

    plt.subplots_adjust(bottom=0.2)

    ax1.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax1.minorticks_on()
    ax1.bar(x, y_acc, yerr=y_acc_err, align='center', alpha=0.5, capsize=10, width=0.6,
            color=['#4287f5', '#42f55f', '#f5d142'])
    # plt.title('')
    ax1.set_xticks(x, PERCENTAGES, fontsize=fontsize)
    ax1.set_yticks(np.arange(0, 0.8, 0.1))
    ax1.set_title('(a) Score(S)', y=-0.2, loc='center', fontsize=fontsize)
    # plt.ylabel("Score(S)", fontsize=fontsize)

    # plt.figure()
    ax2.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax2.minorticks_on()
    ax2.bar(x, y_time, yerr=y_time_err, align='center', alpha=0.5, capsize=10, width=0.6,
            color=['#4287f5', '#42f55f', '#f5d142'])
    # ax2.set_title('Time')
    ax2.set_title('(b) Training time (hrs)', y=-0.2, loc='center', fontsize=fontsize)
    ax2.set_xticks(x, PERCENTAGES, fontsize=fontsize)
    # plt.subplots_adjust(bottom=0.3, left=0.15)
    ax2.set_yticks(np.arange(0, 6, 1))
    # plt.ylabel("Hours", fontsize=fontsize)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)

    plt.savefig('plots/imdb_choose_queries.pdf')
    plt.savefig('plots/imdb_choose_queries.png')
    plt.show()


def imdb_answerable_queries2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fontsize = 25

    PERCENTAGES = ['100%+DB', '100%', '60%', '30%', '30%+LR']

    x = [*range(len(PERCENTAGES))]
    y_acc = []
    y_acc_min = []
    y_acc_max = []
    y_time = []
    y_time_min = []
    y_time_max = []

    # all training queries + DB for un-answerable queries
    # checkpoint 42, trial 1000_CHOOSE_K_PPO_2023-12-15-03-10-16
    y_acc = np.append(y_acc, 0.8193)
    y_acc_min = np.append(y_acc_min, 0.7733)
    y_acc_max = np.append(y_acc_max, 0.8522)

    y_time = np.append(y_time, 4.1)
    y_time_min = np.append(y_time_min, 3.6)
    y_time_max = np.append(y_time_max, 4.5)

    # all training queries
    # checkpoint 42, trial 1000_CHOOSE_K_PPO_2023-12-15-03-10-16
    y_acc = np.append(y_acc, 0.7193)
    y_acc_min = np.append(y_acc_min, 0.6733)
    y_acc_max = np.append(y_acc_max, 0.7522)

    y_time = np.append(y_time, 3.1)
    y_time_min = np.append(y_time_min, 2.7)
    y_time_max = np.append(y_time_max, 3.4)

    # 60% percent of training queries
    # checkpoint 38, trial 1000_CHOOSE_K_PPO_2023-12-15-01-09-40
    y_acc = np.append(y_acc, 0.6784)
    y_acc_min = np.append(y_acc_min, 0.6433)
    y_acc_max = np.append(y_acc_max, 0.75)

    y_time = np.append(y_time, 2.0)
    y_time_min = np.append(y_time_min, 2.2)
    y_time_max = np.append(y_time_max, 1.82)

    # 30% percent of training queries
    # checkpoint 39, trial 1000_CHOOSE_K_PPO_2023-12-15-14-36-49
    y_acc = np.append(y_acc, 0.5833)
    y_acc_min = np.append(y_acc_min, 0.5567)
    y_acc_max = np.append(y_acc_max, 0.61)

    y_time = np.append(y_time, 1.0)
    y_time_min = np.append(y_time_min, 1.14)
    y_time_max = np.append(y_time_max, 0.792)

    # 30% percent of training queries + early stop + lr
    # checkpoint 39, trial 1000_CHOOSE_K_PPO_2023-12-15-14-36-49
    y_acc = np.append(y_acc, 0.5433)
    y_acc_min = np.append(y_acc_min, 0.5167)
    y_acc_max = np.append(y_acc_max, 0.57)

    y_time = np.append(y_time, 27 / 60)
    y_time_min = np.append(y_time_min, 41 / 60)
    y_time_max = np.append(y_time_max, 24 / 60)

    y_acc_err = [[y_acc[i] - y_acc_min[i] for i in range(len(y_acc))],
                 [y_acc_max[i] - y_acc[i] for i in range(len(y_acc))]]
    y_time_err = [[y_time[i] - y_time_min[i] for i in range(len(y_time))],
                  [y_time_max[i] - y_time[i] for i in range(len(y_time))]]

    plt.subplots_adjust(bottom=0.35)

    ax1.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax1.minorticks_on()
    ax1.bar(x, y_acc, yerr=y_acc_err, align='center', alpha=0.5, capsize=10, width=0.6,
            color=['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'])
    # ax1.errorbar(x, y_acc, yerr=y_acc_err, marker='o', color='#4287f5', capsize=10)
    # plt.title('')
    ax1.set_xticks(x, PERCENTAGES, fontsize=fontsize)
    # ax1.set_xticks(x, [], fontsize=fontsize)
    ax1.set_yticks(np.arange(0., 1, 0.2))
    ax1.set_title('(a) Accuracy', y=-0.6, loc='center', fontsize=fontsize)
    # plt.ylabel("Score(S)", fontsize=fontsize)

    # plt.figure()
    ax2.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(which='major', color='#bfc1c7', linewidth=0.8)
    ax2.minorticks_on()
    ax2.bar(x, y_time, yerr=y_time_err, align='center', alpha=0.5, capsize=10, width=0.6,
            color=['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087'])
    # ax2.errorbar(x, y_time, yerr=y_time_err, marker='o', color='#4287f5', capsize=10)

    # ax2.set_title('Time')
    ax2.set_title('(b) Training time (hrs)', y=-0.6, loc='center', fontsize=fontsize)
    ax2.set_xticks(x, PERCENTAGES, fontsize=fontsize)
    # ax2.set_xticks(x, [], fontsize=fontsize)
    # plt.subplots_adjust(bottom=0.3, left=0.15)
    ax2.set_yticks(np.arange(0, 5, 1))
    # plt.ylabel("Hours", fontsize=fontsize)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize, rotation=90)

    plt.savefig('plots/imdb_choose_queries+improvements.pdf')
    plt.savefig('plots/imdb_choose_queries+improvements.png')
    plt.show()


def plot_imdb_confidence():
    plt.figure(figsize=(8, 8))
    fontsize = 25

    PERCENTAGES = ['100%', '60%', '30%']

    x = [*range(len(PERCENTAGES))]
    y_pred = []
    y_pred_min = []
    y_pred_max = []
    y_actual = []
    y_actual_min = []
    y_actual_max = []

    # all training queries
    # checkpoint 42, trial 1000_CHOOSE_K_PPO_2023-12-15-03-10-16
    y_pred = np.append(y_pred, 90)
    y_pred_min = np.append(y_pred_min, 87.66)
    y_pred_max = np.append(y_pred_max, 92.22)

    y_actual = np.append(y_actual, 80)
    y_actual_min = np.append(y_actual_min, 78.88)
    y_actual_max = np.append(y_actual_max, 83.33)

    # 60% percent of training queries
    # checkpoint 38, trial 1000_CHOOSE_K_PPO_2023-12-15-01-09-40
    y_pred = np.append(y_pred, 80)
    y_pred_min = np.append(y_pred_min, 79.99)
    y_pred_max = np.append(y_pred_max, 84.44)

    y_actual = np.append(y_actual, 70)
    y_actual_min = np.append(y_actual_min, 67.77)
    y_actual_max = np.append(y_actual_max, 73.33)

    # 30% percent of training queries
    # checkpoint 39, trial 1000_CHOOSE_K_PPO_2023-12-15-14-36-49
    y_pred = np.append(y_pred, 60)
    y_pred_min = np.append(y_pred_min, 54.44)
    y_pred_max = np.append(y_pred_max, 62.22)

    y_actual = np.append(y_actual, 70)
    y_actual_min = np.append(y_actual_min, 68.88)
    y_actual_max = np.append(y_actual_max, 74.44)

    y_pred_err = [[y_pred[i] - y_pred_min[i] for i in range(len(y_pred))],
                  [y_pred_max[i] - y_pred[i] for i in range(len(y_pred))]]
    y_actual_err = [[y_actual[i] - y_actual_min[i] for i in range(len(y_actual))],
                    [y_actual_max[i] - y_actual[i] for i in range(len(y_actual))]]

    plt.subplots_adjust(left=0.2)

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    plt.bar(x, y_pred, yerr=y_pred_err, align='center', alpha=0.5, capsize=10, width=0.22,
            color='#FFA500', label='predicted')
    plt.bar([xx + 0.25 for xx in x], y_actual, yerr=y_actual_err, align='center', alpha=0.5, capsize=10, width=0.22,
            color='#983A00', label='actual')
    # plt.title('')
    plt.xticks([xx + 0.125 for xx in x], PERCENTAGES, fontsize=fontsize)
    plt.yticks(np.arange(0, 110, 10), [f'{n}%' for n in range(0, 110, 10)], fontsize=fontsize)
    # ax1.set_title('(a) Score(S)', y=-0.2, loc='center', fontsize=fontsize)
    plt.ylabel("Answerable queries (%)", fontsize=fontsize)
    plt.legend(bbox_to_anchor=(0.48, 1.11), loc='upper center', ncol=2, prop={'size': 18})
    plt.savefig('plots/imdb_answerable_queries.pdf')
    plt.savefig('plots/imdb_answerable_queries.png')
    plt.show()


def plot_imdb_estimator():
    plt.figure(figsize=(8, 8))
    fontsize = 25

    PERCENTAGES = ['100%', '60%', '30%']

    x = [*range(len(PERCENTAGES))]
    y_precision = []
    y_precision_min = []
    y_precision_max = []
    y_recall = []
    y_recall_min = []
    y_recall_max = []

    # all training queries
    # checkpoint 42, trial 1000_CHOOSE_K_PPO_2023-12-15-03-10-16
    y_precision = np.append(y_precision, 92)
    y_precision_min = np.append(y_precision_min, 90.66)
    y_precision_max = np.append(y_precision_max, 94.22)

    y_recall = np.append(y_recall, 96)
    y_recall_min = np.append(y_recall_min, 94.88)
    y_recall_max = np.append(y_recall_max, 98.00)

    # 60% percent of training queries
    # checkpoint 38, trial 1000_CHOOSE_K_PPO_2023-12-15-01-09-40
    y_precision = np.append(y_precision, 83)
    y_precision_min = np.append(y_precision_min, 80.99)
    y_precision_max = np.append(y_precision_max, 84.44)

    y_recall = np.append(y_recall, 88)
    y_recall_min = np.append(y_recall_min, 84.77)
    y_recall_max = np.append(y_recall_max, 90.33)

    # 30% percent of training queries
    # checkpoint 39, trial 1000_CHOOSE_K_PPO_2023-12-15-14-36-49
    y_precision = np.append(y_precision, 71)
    y_precision_min = np.append(y_precision_min, 66.44)
    y_precision_max = np.append(y_precision_max, 73.22)

    y_recall = np.append(y_recall, 85)
    y_recall_min = np.append(y_recall_min, 82.88)
    y_recall_max = np.append(y_recall_max, 88.44)

    y_precision_err = [[y_precision[i] - y_precision_min[i] for i in range(len(y_precision))],
                  [y_precision_max[i] - y_precision[i] for i in range(len(y_precision))]]
    y_recall_err = [[y_recall[i] - y_recall_min[i] for i in range(len(y_recall))],
                    [y_recall_max[i] - y_recall[i] for i in range(len(y_recall))]]

    plt.subplots_adjust(left=0.2)

    plt.grid(which='major', color='#bfc1c7', linewidth=0.8)
    plt.minorticks_on()
    print(x)
    print(y_precision)
    plt.errorbar(x, y_precision, yerr=y_precision_err, marker='o', alpha=0.5, capsize=10, color='#FFA500', label='precision')
    plt.errorbar(x, y_recall, yerr=y_recall_err, marker='o', alpha=0.5, capsize=10, color='#983A00', label='recall')

    # plt.title('')
    plt.xticks(x, PERCENTAGES, fontsize=fontsize)
    plt.yticks(np.arange(70, 110, 10), [f'{n}%' for n in range(70, 110, 10)], fontsize=fontsize)
    # ax1.set_title('(a) Score(S)', y=-0.2, loc='center', fontsize=fontsize)
    # plt.ylabel("Answerable queries (%)", fontsize=fontsize)
    plt.legend(bbox_to_anchor=(0.48, 1.11), loc='upper center', ncol=2, prop={'size': 18})
    plt.savefig('plots/plot_imdb_estimator.pdf')
    plt.savefig('plots/plot_imdb_estimator.png')
    plt.show()


if __name__ == '__main__':
    # plot_mas_baselines()
    # plot_imdb_baselines()
    # plot_imdb_runtime_results()
    # plot_mas_runtime_results()
    # plot_ray_avg_mas_threshold_results()
    # plot_ray_avg_imdb_threshold_results()
    # plot_drop1_imdb_results()
    # plot_view_size_array_results()
    # plot_imdb_rl_algos()
    # plot_mas_rl_algos()
    # plot_flights_aqp_results3()
    # plot_query_answer_time()
    # plot_drop1_extended_results()
    # plot_rl_tune()
    # plot_param_tune_results()
    # plot_query_sampling()
    # plot_flights_aqp_results5()
    # plot_initial_sample_exp1()
    # plot_imdb_initial_sample()
    # plot_mas_initial_sample()
    # plot_imdb_query_runtimes()
    # plot_imdb_choose_queries()
    # imdb_answerable_queries2()
    # plot_imdb_confidence()
    plot_imdb_estimator()

import matplotlib.pyplot as plt
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


def get_ray_imdb_data(checkpoint):
    if checkpoint == 2:
        random_scores = [0.285, 0.333, 0.382]
        topk_scores = [0.377, 0.377, 0.377]
        ppo1_scores = [0.468, 0.484, 0.516]
        ppo2_scores = [0.472, 0.482, 0.494]
    elif checkpoint == 3:
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
    y = np.array([random_scores[1], topk_scores[1], ppo1_scores[1], ppo2_scores[1]])
    yerr = np.array([random_scores[-1] - random_scores[1], 0,
                     ppo1_scores[-1] - ppo1_scores[1],
                     ppo2_scores[-1] - ppo2_scores[1]])
    return x, y, yerr


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
    algos = ['random', 'top-k', 'PPO']
    x = []
    y = np.zeros(len(algos))
    yerr = np.zeros(len(algos))
    for c in checkpoints:
        data = get_ray_imdb_data(c)
        y += data[1][:-1]
        yerr += data[2][:-1]
        x = data[0][:-1]

    y = y / len(checkpoints)
    yerr = yerr / len(checkpoints)

    plt.bar(x, y, yerr=yerr, align='center', alpha=0.5, capsize=10)
    plt.title('')
    plt.xticks(x, algos)

    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    # plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


# TODO
def get_ray_mas_data(checkpoint):
    if checkpoint == 7:
        topk_scores = [0.625] * 3
        random_scores = [0.182, 0.203, 0.230]
        greedy_train_scores = [0.537] * 3
        ppo_scores = [0.64, 0.649, 0.665]
        ppo_0dot3_scores = [0.6431, 0.65, 0.6638]
        apex_scores = [0.668, 0.668, 0.672]
    elif checkpoint == 8:
        topk_scores = [0.455] * 3
        random_scores = [0.25, 0.283, 0.324]
        greedy_train_scores = [0.632] * 3
        ppo_scores = [0.625, 0.633, 0.647]
        ppo_0dot3_scores = [0.76, 0.768, 0.782]
        apex_scores = [0.765, 0.7651, 0.7675]
    elif checkpoint == 9:
        topk_scores = [0.25] * 3
        random_scores = [0.165, 0.205, 0.24]
        greedy_train_scores = [0.395] * 3
        ppo_scores = [0.611, 0.634, 0.652]
        ppo_0dot3_scores = [0.655, 0.685, 0.718]
        apex_scores = [0.6523, 0.6524, 0.6562]
    elif checkpoint == 10:
        topk_scores = [0.507] * 3
        random_scores = [0.075, 0.120, 0.157]
        greedy_train_scores = [0.507] * 3
        ppo_scores = [0.675, 0.691, 0.722]
        ppo_0dot3_scores = [0.660, 0.676, 0.700]
        apex_scores = [0.6836, 0.6839, 0.6875]
    else:
        raise Exception('checkpoint not recognized!')

    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([random_scores[1], topk_scores[1], greedy_train_scores[1], ppo_scores[1], ppo_0dot3_scores[1],
                  apex_scores[1]])
    yerr_max = np.array([random_scores[-1] - random_scores[1], 0, 0, ppo_scores[-1] - ppo_scores[1],
                         ppo_0dot3_scores[-1] - ppo_0dot3_scores[1], apex_scores[-1] - apex_scores[1]])
    yerr_min = np.array([random_scores[1] - random_scores[0], 0, 0, ppo_scores[1] - ppo_scores[0],
                         ppo_0dot3_scores[1] - ppo_0dot3_scores[0], apex_scores[1] - apex_scores[0]])
    return x, y, [yerr_min, yerr_max]


def get_ray_mas_threshold_data(checkpoint):
    if checkpoint == 7:
        topk_scores = [0.625] * 3
        random_scores = [0.125, 0.16, 0.25]
        greedy_train_scores = [0.625] * 3
        ppo_scores = [0.625, 0.702, 0.75]
        ppo_0dot3_scores = [0.625, 0.635, 0.75]
        apex_scores = [0.75, 0.75, 0.75]
    elif checkpoint == 8:
        topk_scores = [0.5] * 3
        random_scores = [0.25, 0.342, 0.625]
        greedy_train_scores = [0.625] * 3
        ppo_scores = [0.625, 0.625, 0.625]
        ppo_0dot3_scores = [0.75, 0.75, 0.75]
        apex_scores = [0.75, 0.75, 0.75]
    elif checkpoint == 9:
        topk_scores = [0.25] * 3
        random_scores = [0.125, 0.242, 0.375]
        greedy_train_scores = [0.625] * 3
        ppo_scores = [0.625, 0.625, 0.625]
        ppo_0dot3_scores = [0.625, 0.743, 0.875]
        apex_scores = [0.625, 0.6275, 0.75]
    elif checkpoint == 10:
        topk_scores = [0.5] * 3
        random_scores = [0.125, 0.13, 0.25]
        greedy_train_scores = [0.5] * 3
        ppo_scores = [0.625, 0.777, 0.875]
        ppo_0dot3_scores = [0.625, 0.682, 0.875]
        apex_scores = [0.75, 0.7525, 0.875]
    else:
        raise Exception('checkpoint not recognized!')

    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([random_scores[1], topk_scores[1], greedy_train_scores[1], ppo_scores[1], ppo_0dot3_scores[1],
                  apex_scores[1]])
    yerr_max = np.array([random_scores[-1] - random_scores[1], 0, 0, ppo_scores[-1] - ppo_scores[1],
                         ppo_0dot3_scores[-1] - ppo_0dot3_scores[1], apex_scores[-1] - apex_scores[1]])
    yerr_min = np.array([random_scores[1] - random_scores[0], 0, 0, ppo_scores[1] - ppo_scores[0],
                         ppo_0dot3_scores[1] - ppo_0dot3_scores[0], apex_scores[1] - apex_scores[0]])
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
        ax.set_xticklabels(['rand', 'topk', 'greedy', 'PPO', 'PPO_0.3', 'ApexDQN'], fontsize=8)

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
        ax.set_xticklabels(['rand', 'topk', 'greedy', 'PPO', 'PPO_0.3', 'ApexDQN'], fontsize=8)

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

    algos = ['rand', 'topk', 'greedy', 'PPO', 'PPO_0.3', 'ApexDQN']
    x = []
    y = np.zeros(len(algos))
    y_err = np.zeros((2, len(algos)))
    for c in checkpoints:
        data = get_ray_mas_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10)
    plt.title('')
    plt.xticks(x, algos, rotation=45, fontsize=8)

    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    # plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


def plot_ray_avg_mas_threshold_results():
    checkpoints = [7, 8, 9, 10]

    algos = ['rand', 'topk', 'greedy', 'PPO', 'PPO_0.3', 'ApexDQN']
    x = []
    y = np.zeros(len(algos))
    y_err = np.zeros((2, len(algos)))
    for c in checkpoints:
        data = get_ray_mas_threshold_data(c)
        y += data[1]
        y_err += data[2]
        x = data[0]

    y = y / len(checkpoints)
    y_err = y_err / len(checkpoints)

    plt.bar(x, y, yerr=y_err, align='center', alpha=0.5, capsize=10)
    plt.title('')
    plt.xticks(x, algos, rotation=45, fontsize=8)

    plt.xlabel("Algorithm")
    plt.ylabel("0.25-Score")
    # plt.suptitle(f"Score by algorithm, k={1000}, view_size={view_size}")
    plt.show()


if __name__ == '__main__':
    plot_ray_avg_mas_threshold_results()

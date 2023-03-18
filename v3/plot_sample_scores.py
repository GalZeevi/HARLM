import matplotlib.pyplot as plt

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
import numpy as np

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


if __name__ == '__main__':
    plot_rewards()

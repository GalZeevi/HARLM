import collections
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from dqn_sampler import Preprocess
from dqn_sampler import SaqpEnv, SaqpEnv2
from score_calculator import get_score2

"""
Implementation of Double DQN for environments with discrete action space.
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device_ids = [*range(torch.cuda.device_count())] if torch.cuda.is_available() else []


def parallel_model(model):
    if use_cuda:
        return nn.DataParallel(model)
    else:
        return model


"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""

ENV_VER = 2


class QNetwork(nn.Module):
    def __init__(self, action_dim, state_shape, hidden_dim, embed=True):
        super(QNetwork, self).__init__()

        self.embed = embed
        if self.embed:
            self.embedding_dims_and_names = []
            self.embedding_dims = []
            self.numerical_dims = []
            self.embedding_layers = []
            inp_dim = self.init_embedding(state_shape)
        else:
            inp_dim = state_shape[0] * state_shape[1]
        self.fc_1 = parallel_model(nn.Linear(inp_dim, hidden_dim, dtype=torch.float32, device=device))
        self.fc_2 = parallel_model(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32, device=device))
        self.fc_3 = parallel_model(nn.Linear(hidden_dim, action_dim, dtype=torch.float32, device=device))

    def init_embedding(self, state_shape):
        Preprocess.init()
        encodings = Preprocess.get_encodings()
        self.embedding_dims_and_names = [(i, col_name) for i, col_name in
                                         Preprocess.get_categorical_columns_sorted()
                                         if len(encodings[col_name]) > 2]
        self.embedding_dims = [i for i, col_name in self.embedding_dims_and_names]
        self.numerical_dims = [i for i in range(len(Preprocess.get_all_columns(with_pivot=False)))
                               if i not in self.embedding_dims]
        col_and_lens = {col_name: len(d.keys()) for col_name, d in encodings.items() if len(d.keys()) > 2}
        embed_sizes = {col: (col_len, min(50, (col_len + 1) // 2)) for col, col_len in col_and_lens.items()}
        self.embedding_layers = nn.ModuleList([
            parallel_model(nn.Embedding(embed_sizes[col_name][0], embed_sizes[col_name][1], device=device))
            for _, col_name in self.embedding_dims_and_names
        ])
        inp_dim = state_shape[0] * ((state_shape[1] - len(self.embedding_dims)) +
                                    sum([size for (length, size) in embed_sizes.values()]))
        return inp_dim

    def forward(self, inp):
        if len(inp.size()) < 3:
            inp = inp.unsqueeze(0)

        if self.embed:
            inp_cat = inp[:, :, self.embedding_dims].long()
            inp_cat_embed = [self.embedding_layers[i](inp_cat[:, :, i]) for i, _ in enumerate(self.embedding_dims)]
            inp_cat_embed = torch.cat(inp_cat_embed, -1)

            inp_num = inp[:, :, self.numerical_dims]
            inp = torch.cat([inp_cat_embed, inp_num], -1)

        x1 = torch.flatten(inp, start_dim=1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1

    def select_action(self, env, state, eps):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.forward(state.unsqueeze(0))

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, env.num_actions)
            if hasattr(env, 'forbidden_actions'):
                while np.isin([action], env.forbidden_actions)[0]:
                    action = np.random.randint(0, env.num_actions)
        else:
            actions_values = values.cpu().numpy()
            if hasattr(env, 'forbidden_actions') and len(env.forbidden_actions) > 0:
                actions_values[:, env.forbidden_actions] = np.NINF
            action = np.argmax(actions_values)

        return action


"""
memory to save the state, action, reward sequence from the current episode. 
"""


class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.states = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state.
        # Otherwise, we have more states per episode than rewards
        # and actions which leads to a mismatch when we sample from memory.
        if not done:
            self.states.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, next state, reward, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n - 1), batch_size)

        # TODO this can be improved
        states_numpy = np.array(self.states)
        states = torch.Tensor(states_numpy)[idx].to(device)
        actions = torch.LongTensor(self.action)[idx].to(device)
        next_states = torch.Tensor(states_numpy)[1 + np.array(idx)].to(device)
        rewards = torch.Tensor(self.rewards)[idx].to(device)
        is_done = torch.Tensor(self.is_done)[idx].to(device)
        return states, actions, next_states, rewards, is_done

    def reset(self):
        self.rewards.clear()
        self.states.clear()
        self.action.clear()
        self.is_done.clear()


def update(batch_size, current, target, optim, memory, gamma):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(-1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


def evaluate(Qmodel, k, eps, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """

    if ENV_VER == 1:
        env = SaqpEnv(k, max_iters=-1)
    else:
        env = SaqpEnv2(k)
    Qmodel.eval()
    perform = 0.
    for r in range(repeats):
        tqdm.write(f'Starting measure step, repeat num: {r + 1}/{repeats}')
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            action = Qmodel.select_action(env, state, eps)
            state, reward, done = env.step(action)
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


HIDDEN_DIM = 64
NUM_EPISODES = 3000
HORIZON = 1222

rewards_graph = []
losses_graph = []


def train(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.999, eps_min=0.01, update_step=10, batch_size=128,
          update_repeats=25, num_episodes=NUM_EPISODES, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100,
          measure_step=25, measure_repeats=10, hidden_dim=HIDDEN_DIM, horizon=HORIZON, k=100):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to assess performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param k: sample size
    :return: the trained Q-Network and the measured performances
    """
    if ENV_VER == 1:
        env = SaqpEnv(k, max_iters=horizon)
    else:
        env = SaqpEnv2(k)

    Q_1 = QNetwork(action_dim=env.num_actions, state_shape=env.state_shape,
                   hidden_dim=hidden_dim).to(device)
    Q_2 = QNetwork(action_dim=env.num_actions, state_shape=env.state_shape,
                   hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in trange(num_episodes):
        # display the performance
        if episode > 0 and episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, k, eps, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("eps: ", eps)
            rewards_graph.append((episode, performance[-1][1]))
            CheckpointManager.save(f'{k}_{num_episodes}_{horizon}_ddqn_rewards', rewards_graph)

        state = env.reset()
        memory.states.append(state)

        for i in range(int(horizon) + 1):
            action = Q_1.select_action(env, state, eps)
            state, reward, done = env.step(action)

            # render the environment
            env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

            if done:
                break

        if episode >= min_episodes and episode % update_step == 0:
            ep_loss = 0.
            for r in range(update_repeats):
                tqdm.write(f'Starting update step, repeat num: {r + 1}/{update_repeats}')
                ep_loss += update(batch_size, Q_1, Q_2, optimizer, memory, gamma)
            print("Episode: ", episode)
            print("loss: ", ep_loss)
            print("eps: ", eps)
            losses_graph.append((episode, ep_loss / update_repeats))
            CheckpointManager.save(f'{k}_{num_episodes}_{horizon}_ddqn_losses', losses_graph)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

            optimizer.step()
            scheduler.step()

        # save model
        torch.save({
            'q1_state_dict': Q_1.state_dict(),
            'q2_state_dict': Q_2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': eps,
            'episode': episode
        }, f'{CheckpointManager.get_checkpoint_path()}/{k}_{num_episodes}_{horizon}_ddqn.pt')

        # update learning rate and eps
        eps = max(eps * eps_decay, eps_min)

    return Q_1, performance


def get_sample(k=100, n_trials=10, num_episodes=NUM_EPISODES, horizon=HORIZON, verbose=True):
    checkpoint = torch.load(f'{CheckpointManager.get_checkpoint_path()}/{k}_{num_episodes}_{horizon}_ddqn.pt')
    if checkpoint is None:
        train(k=k, num_episodes=num_episodes, horizon=horizon)
        checkpoint = torch.load(
            f'{CheckpointManager.get_checkpoint_path()}/{k}_{num_episodes}_{horizon}_ddqn.pt')

    view_size = ConfigManager.get_config('samplerConfig.viewSize')

    if ENV_VER == 1:
        env = SaqpEnv(k, max_iters=-1)
    else:
        env = SaqpEnv2(k)
    validation_set = env.validation_set
    best_validation_score = -1
    best_sample = None

    # Prepare the model for inference
    ddqn = QNetwork(action_dim=env.num_actions, state_shape=env.state_shape,
                    hidden_dim=HIDDEN_DIM).to(device)
    ddqn.load_state_dict(checkpoint['q1_state_dict'])
    ddqn.eval()

    for _ in (trange(n_trials) if verbose is True else range(n_trials)):
        state = env.reset()  # Resets the env and returns a random initial state
        state = torch.Tensor(state).to(device)
        env.render()  # Visualize for human

        # Infer
        done = False
        while not done:
            # Get the next_state, reward, and are we done
            action = ddqn.select_action(env, state, 0.01)
            next_state, reward, done = env.step(action)
            env.render()  # Visualize for human

            # Move to next state
            state = torch.Tensor(next_state).to(device)

        curr_sample = env.sample()
        curr_validation_score = get_score2(curr_sample, queries=validation_set)
        verbose and tqdm.write(f'current train score: {get_score2(curr_sample, queries="train")}, '
                               f'current validation score: {curr_validation_score}, '
                               f'current test score: {get_score2(curr_sample, queries="test")}')
        if curr_validation_score > best_validation_score:
            best_validation_score = curr_validation_score
            best_sample = curr_sample

    verbose and print(f'best score on validation set: {best_validation_score}')
    test_score = get_score2(best_sample, queries='test')
    verbose and print(f'score on test set: {test_score}')
    verbose and CheckpointManager.save(f'{k}-{view_size}-{k}_{num_episodes}_{horizon}_ddqn_sample',
                                       [best_sample, test_score, best_validation_score])
    return best_sample, test_score


def get_scores(k, n_trials=100):
    min_score = 10.
    max_score = -10.
    avg_score = 0.
    all_scores = []
    for _ in trange(n_trials):
        sample, new_score = get_sample(k=k, verbose=False, n_trials=1)
        all_scores.append(new_score)
        print(f'current: {new_score}')
        avg_score += new_score
        min_score = min(min_score, new_score)
        max_score = max(max_score, new_score)
    avg_score /= n_trials
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    CheckpointManager.save(f'{k}-{view_size}-{k}_{NUM_EPISODES}_{HORIZON}_ddqn_scores', all_scores)
    print(f'avg: {avg_score}')
    print(f'min: {min_score}')
    print(f'max: {max_score}')


if __name__ == '__main__':
    k = 1000  # TODO change to 1000 on server, 500 locally

    train(k=k)
    # get_scores(k=k)

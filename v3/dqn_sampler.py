from itertools import count
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm, trange
from sklearn.utils import shuffle

from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from score_calculator import get_score2
from train_test_utils import get_train_queries
from checkpoint_manager_v3 import CheckpointManager
from top_queried_sampler import prepare_sample, prepare_weights_for_sample


def get_permutation(n):
    original = np.arange(n)
    permuted = np.random.permutation(original)
    return lambda i: permuted[i]

def threshold_positive(value, threshold):
    if 0 <= value < threshold:
        return 0
    else:
        return value


NULL_VALUE = '<NULL>'


class Preprocess:
    _encodings = dict()
    _max_values = dict()
    _columns = list()

    @staticmethod
    def create_encoding_dict(string_values):
        values, codes = np.unique(string_values, return_inverse=True)
        return {value: code for code, value in zip(codes, values)}

    @staticmethod
    def get_all_columns(with_pivot=True):
        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                    f"WHERE table_schema='{schema}' AND table_name='{table}'")
        if not with_pivot:
            pivot = ConfigManager.get_config('queryConfig.pivot')
            columns.remove(pivot)

        return sorted(columns)

    @staticmethod
    def get_categorical_columns_sorted():
        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        pivot = ConfigManager.get_config('queryConfig.pivot')

        numeric_data_types = ['smallint', 'integer', 'bigint',
                              'decimal', 'numeric', 'real', 'double precision',
                              'smallserial', 'serial', 'bigserial', 'int']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        categorical_cols = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                             f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                             f"AND data_type NOT IN ({' , '.join(db_formatted_data_types)}) " +
                                             f"AND column_name <> '{pivot}'")

        all_columns = Preprocess.get_all_columns()
        return [pair for pair in enumerate(all_columns) if pair[1] in categorical_cols]

    @staticmethod
    def get_encodings():
        if len(Preprocess._encodings.keys()) > 0:
            return Preprocess._encodings

        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        pivot = ConfigManager.get_config('queryConfig.pivot')
        numeric_data_types = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial', 'int']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        categorical_columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                                f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                                f"AND data_type NOT IN ({' , '.join(db_formatted_data_types)}) " +
                                                f"AND column_name <> '{pivot}'")
        # TODO: fix support in null fields - currently it converts everything to str
        # TODO: --> can convert time to number or string
        return {col: Preprocess.create_encoding_dict(
            # DataAccess.select(f'SELECT DISTINCT COALESCE({col}, \'{NULL_VALUE}\') as val FROM {schema}.{table}'))
            DataAccess.select(f'SELECT DISTINCT {col} as val FROM {schema}.{table} ORDER BY val'))
            for col in categorical_columns}

    @staticmethod
    def get_max_values():
        if len(Preprocess._max_values.keys()) > 0:
            return Preprocess._max_values

        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        pivot = ConfigManager.get_config('queryConfig.pivot')
        numeric_data_types = \
            ['smallint', 'integer', 'bigint',
             'decimal', 'numeric', 'real', 'double precision',
             'smallserial', 'serial', 'bigserial', 'int']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        numerical_columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                              f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                              f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                              f"AND column_name <> '{pivot}'")
        return {col: DataAccess.select_one(f'SELECT COALESCE(MAX({col}), 0) as val FROM {schema}.{table}')
                for col in numerical_columns}

    @staticmethod
    def encode_column(tuples, col_num_when_sorted):
        if len(Preprocess._encodings.keys()) == 0 or len(Preprocess._columns) == 0 or \
                len(Preprocess._max_values.keys()) == 0:
            Preprocess._encodings = Preprocess.get_encodings()
            Preprocess._columns = Preprocess.get_all_columns()
            Preprocess._max_values = Preprocess.get_max_values()

        col_name = sorted(Preprocess._columns)[col_num_when_sorted]
        col = tuples[:, col_num_when_sorted]
        if col_name in Preprocess._max_values:  # numerical column
            col[np.where(col == None)] = Preprocess._max_values[col_name] + 1  # Handle null-values for numeric columns
            return tuples
        elif col_name in Preprocess._encodings:
            col[np.where(col == None)] = NULL_VALUE  # Handle null-values for categorical columns
            mapping = Preprocess._encodings[col_name]
            reduced_mapping = {k: v for k, v in mapping.items() if k in col}  # in order to not search the entire thing

            for key in reduced_mapping.keys():
                col[np.where(col == key)] = reduced_mapping[key]
            tuples[:, col_num_when_sorted] = col

        return tuples

    @staticmethod
    def tuples2numpy(tuples_list):
        if len(Preprocess._encodings.keys()) == 0 or len(Preprocess._columns) == 0:
            Preprocess._encodings = Preprocess.get_encodings()
            Preprocess._columns = Preprocess.get_all_columns()

        tuples_sorted_by_cols = [sorted([*tup.items()], key=lambda pair: pair[0]) for tup in tuples_list]
        tuples_values_sorted_by_cols = [[col_and_value[1] for col_and_value in tup] for tup in tuples_sorted_by_cols]
        tuples_as_numpy_not_encoded = np.array(tuples_values_sorted_by_cols)
        encoded_tuples = np.apply_over_axes(Preprocess.encode_column, tuples_as_numpy_not_encoded,
                                            range(tuples_as_numpy_not_encoded.shape[1]))
        return Preprocess.remove_pivot_from_numpy(encoded_tuples).astype(float)

    @staticmethod
    def remove_pivot_from_numpy(a):
        pivot = ConfigManager.get_config('queryConfig.pivot')
        pivot_col_location_in_columns = Preprocess._columns.index(pivot)
        return np.delete(a, pivot_col_location_in_columns, 1)


MAX_ITERS = 1005


class SaqpEnv:

    def __init__(self, k, max_iters=MAX_ITERS):
        self.k = k
        self.num_actions = k + 1
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.table = ConfigManager.get_config('queryConfig.table')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {self.schema}.{self.table}')
        num_cols_not_pivot = DataAccess.select_one(f"SELECT COUNT(column_name) FROM information_schema.columns " +
                                                   f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                                   f"AND column_name <> '{self.pivot}'")
        validation_size = 10 if ConfigManager.get_config('samplerConfig.validationSize') is None else \
            ConfigManager.get_config('samplerConfig.validationSize')
        self.train_set, self.validation_set = get_train_queries(validation_size=validation_size)
        self.state_shape = (k + 1, num_cols_not_pivot)
        self.step_count = 0
        self.max_iters = max_iters
        self.states_tuples_idx = []
        self.best_k = None
        self.best_k_numpy = None
        self.current_score = 0.
        self.next_tuple = None
        self.next_tuple_numpy = None
        self.num_queried_tuples_to_include = None if max_iters < 0 else max_iters // 2
        self.forbidden_actions = np.array([])

    def reset(self):
        self.step_count = 0
        first_state_idx, self.states_tuples_idx = self.get_states_tuples2()
        np.random.shuffle(self.states_tuples_idx)
        indices = np.concatenate((first_state_idx, [self.states_tuples_idx[0]]))
        db_format_indices = [str(idx) for idx in indices]
        tuples = DataAccess.select(
            f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot} IN ({",".join(db_format_indices)})')
        self.best_k = tuples[:self.k]
        self.best_k_numpy = Preprocess.tuples2numpy(self.best_k)
        self.current_score = get_score2([tup[self.pivot] for tup in self.best_k], queries=self.train_set)
        self.next_tuple = tuples[-1]
        self.next_tuple_numpy = Preprocess.tuples2numpy([self.next_tuple])
        return np.concatenate((self.best_k_numpy, self.next_tuple_numpy))

    def get_states_tuples(self):
        queried_tuples_idx = prepare_sample(min(int(4 * self.k), self.table_size))
        starting_idx = np.random.choice(queried_tuples_idx, size=self.k, replace=False)
        # starting_idx = np.random.choice(self.table_size, size=self.k, replace=False)
        all_tuple_idx = np.arange(self.table_size)
        other_idx = all_tuple_idx[~np.isin(all_tuple_idx, starting_idx)]
        if self.max_iters <= 0 or (self.k + self.max_iters + 2) >= self.table_size:
            random_indices = other_idx
        else:
            other_queried_tuples_idx = queried_tuples_idx[~np.isin(queried_tuples_idx, starting_idx)]
            other_queried_tuples_idx = other_queried_tuples_idx[:int(self.num_queried_tuples_to_include)]
            self.num_queried_tuples_to_include *= 0.99965
            idx_left_to_choose_from = other_idx[~np.isin(other_idx, other_queried_tuples_idx)]
            random_indices = np.random.choice(idx_left_to_choose_from,
                                              size=max(0, self.max_iters + 2 - len(other_queried_tuples_idx)),
                                              replace=False)
            random_indices = np.concatenate((random_indices, other_queried_tuples_idx))
        return starting_idx, random_indices

    def get_states_tuples2(self):
        available_idx = np.arange(self.table_size)
        weights = prepare_weights_for_sample(False)
        weights[np.where(weights < 0.1 * len(self.train_set))] = 0
        max_seed_size = int(0.7 * self.k)
        # init_state_idx = np.where(weights >= 0.3 * len(self.train_set))[0][:max_seed_size].tolist()
        init_state_idx = np.argpartition(weights, -max_seed_size)[-max_seed_size:].tolist()
        self.forbidden_actions = np.arange(len(init_state_idx))
        available_idx = np.setdiff1d(available_idx, init_state_idx)

        init_state_idx += np.random.choice(available_idx, size=self.k - len(init_state_idx), replace=False).tolist()
        available_idx = np.setdiff1d(available_idx, init_state_idx)

        if self.max_iters <= 0 or (self.k + self.max_iters + 2) >= self.table_size:
            other_states_idx = available_idx
        else:
            other_states_idx = np.random.choice(available_idx, size=self.max_iters + 2, replace=False)

        return init_state_idx, other_states_idx

    def render(self, mode='NONE'):
        if mode == 'FULL':
            headers = [*self.best_k[0].keys()]
            best_k_data = np.array([[*t.values()] for t in self.best_k])
            table = tabulate(best_k_data, headers, tablefmt="fancy_grid")
            print('============= Current sample =============')
            print(table)
            next_tuple_data = np.array([*self.next_tuple.values()])
            table = tabulate(next_tuple_data, headers, tablefmt="fancy_grid")
            print('=============   Next tuple   =============')
            print(table)
        elif mode == 'REDUCED':
            print(
                f'Current sample: [{[tup[self.pivot] for tup in self.best_k]}], next tuple: [{self.next_tuple[self.pivot]}]')
        else:
            pass

    def step(self, action):
        tuple_num_to_replace = action
        old_score = self.current_score
        if tuple_num_to_replace < self.k:  # Agent decided to make a replacement
            self.best_k[tuple_num_to_replace] = self.next_tuple
            self.best_k_numpy[tuple_num_to_replace] = Preprocess.tuples2numpy([self.next_tuple])
            self.current_score = get_score2([tup[self.pivot] for tup in self.best_k], queries=self.train_set)
        self.step_count += 1
        reward = (self.current_score - old_score) * len(self.train_set)  # Calculate reward

        # reshuffle the best_k so the agent does not take positions into account
        perm = get_permutation(self.k)
        permuted_best_k = [*self.best_k]
        for i in range(self.k):
            permuted_best_k[perm(i)] = self.best_k[i]
        for i, a in enumerate(self.forbidden_actions):
            self.forbidden_actions[i] = perm(a)
        self.best_k = permuted_best_k
        self.best_k_numpy = Preprocess.tuples2numpy(self.best_k)

        # Prepare next step
        if self.step_count < len(self.states_tuples_idx):
            next_tuple_ind = self.states_tuples_idx[self.step_count]
            self.next_tuple = DataAccess.select_one(f'SELECT * FROM {self.schema}.{self.table} '
                                                    f'WHERE {self.pivot}={next_tuple_ind}')
            self.next_tuple_numpy = Preprocess.tuples2numpy([self.next_tuple])
            return np.concatenate((self.best_k_numpy, self.next_tuple_numpy)), \
                   reward, \
                   self.step_count == self.max_iters or self.k + 1 + self.step_count == self.table_size
        else:  # no more tuples to give as next_tuple
            return np.concatenate((self.best_k_numpy, [np.zeros_like(self.best_k_numpy[0]) - 1])), \
                   reward, \
                   True


class DQN(nn.Module):
    def __init__(self, k):
        super(DQN, self).__init__()
        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        pivot = ConfigManager.get_config('queryConfig.pivot')
        num_cols_not_pivot = DataAccess.select_one(f"SELECT COUNT(column_name) FROM information_schema.columns " +
                                                   f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                                   f"AND column_name <> '{pivot}'")
        self.k = k
        indim2 = (k + 1) * num_cols_not_pivot
        outdim2 = int((k + 1) * num_cols_not_pivot / sqrt(k + 1))
        self.fc2 = nn.Linear(indim2, outdim2, dtype=torch.float32)
        self.fc3 = nn.Linear(outdim2, k + 1, dtype=torch.float32)  # Prob of whom to throw out

    def forward(self, x):
        x = torch.flatten(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Parameters
total_episodes = 3000
batch_size = 64
learning_rate = 0.01
gamma = 0.99


def train(k=100):
    # Plot duration curve:
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []

    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    env = SaqpEnv(k)
    dqn = DQN(k)
    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    last_100_ep_rewards = []

    for episode_num in tqdm(range(total_episodes)):

        state = env.reset()  # Resets the env and returns a random initial state
        state = torch.from_numpy(state).float()
        state = Variable(state)
        env.render()  # Visualize for human
        ep_reward = 0.

        for t in count():  # Run until done

            # Compute action - here agent output distribution over [0,1,...,k]
            probs = dqn(state)
            m = Categorical(probs=probs)
            tuple_num_to_replace = m.sample()

            # Get the next_state, reward, and are we done
            tuple_num_to_replace = tuple_num_to_replace.item()
            next_state, reward, done = env.step(tuple_num_to_replace)
            env.render()  # Visualize for human
            ep_reward += reward

            # To mark boundaries between episodes
            if done:
                reward = 0

            state_pool.append(state)
            action_pool.append(float(tuple_num_to_replace))
            reward_pool.append(reward)

            # Move to next state
            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)

        # Update policy
        if episode_num > 0 and episode_num % batch_size == 0:  # update every batch_size episodes

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Descent
            optimizer.zero_grad()

            loss = 1.

            for i in range(steps):
                state = state_pool[i]
                tuple_num_to_replace = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = dqn(state)
                m = Categorical(probs)
                # If reward is 0 or very small then the gradients zero out
                loss = -m.log_prob(tuple_num_to_replace) * reward  # Negative score function x reward
                loss.backward()
                print(f'Episode: {episode_num}/{total_episodes}, Loss: {loss.item():.4f}, '
                      f'Return: {np.mean(last_100_ep_rewards):.8f}')

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

            torch.save({
                'epoch': episode_num / batch_size,
                'model_state_dict': dqn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'{CheckpointManager.get_checkpoint_path()}/{k}_{total_episodes}_{MAX_ITERS}_dqn.pt')


def get_sample(k):
    checkpoint = torch.load(f'{CheckpointManager.get_checkpoint_path()}/{k}_{total_episodes}_{MAX_ITERS}_dqn.pt')
    if checkpoint is None:
        train(k)
        checkpoint = torch.load(
            f'{CheckpointManager.get_checkpoint_path()}/{k}_{total_episodes}_{MAX_ITERS}_dqn.pt')

    model = DQN(k)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    env = SaqpEnv(k, max_iters=-1)
    state = env.reset()  # Resets the env and returns a random initial state
    state = torch.from_numpy(state).float()
    env.render()  # Visualize for human

    pbar = tqdm()
    done = False
    while not done:
        probs = model(state)
        m = Categorical(probs=probs)
        tuple_num_to_replace = m.sample()

        # Get the next_state, reward, and are we done
        tuple_num_to_replace = tuple_num_to_replace.item()
        next_state, reward, done = env.step(tuple_num_to_replace)
        env.render()  # Visualize for human

        # Move to next state
        state = next_state
        state = torch.from_numpy(state).float()
        pbar.update(1)

    sample = env.best_k
    pivot = ConfigManager.get_config('queryConfig.pivot')
    score = get_score2([tup[pivot] for tup in sample], queries='test')
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    CheckpointManager.save(f'{k}-{view_size}-{k}_{total_episodes}_{MAX_ITERS}_dqn_sample',
                           [sample, score])
    return sample, score


def get_scores(n_trials=10, k=100):
    min_score = 10.
    max_score = -10.
    avg_score = 0.
    for _ in trange(n_trials):
        sample, new_score = get_sample(k)
        print(f'current: {new_score}')
        avg_score += new_score
        min_score = min(min_score, new_score)
        max_score = max(max_score, new_score)
    avg_score /= n_trials
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    CheckpointManager.save(f'{k}-{view_size}-{k}_{total_episodes}_{MAX_ITERS}_dqn_scores',
                           [avg_score, min_score, max_score])
    print(f'avg: {avg_score}')
    print(f'min: {min_score}')
    print(f'max: {max_score}')


if __name__ == '__main__':
    k = 100
    trials = 25
    train(k)
    # get_scores(trials)

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict
from gymnasium.spaces import Discrete, Box

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing
from score_calculator import get_combined_score, get_score2, get_threshold_score, get_diversity_score
from top_queried_sampler import prepare_sample as get_queried_tuples
from train_test_utils import get_train_queries


# TODO: remove ep_reward_sum from callbacks!


class ChooseKEnv(gym.Env):

    def __init__(self, env_config):
        super(gym.Env, self).__init__()

        self.inference_mode = env_config.get('inference_mode', False)
        self.k = env_config['k']
        self.checkpoint_version = env_config.get('checkpoint_version', CheckpointManager.get_max_version())

        # env config
        self.schema = env_config['table_details']['schema']
        self.table = env_config['table_details']['table']
        self.pivot = env_config['table_details']['pivot']
        self.table_size = env_config['table_details']['table_size']
        self.action_space_size = env_config.get('action_space_size', self.table_size)
        self.random_actions_coeff = env_config.get('random_actions_coeff', 0)
        self.diversity_coeff = env_config.get('diversity_coeff', 0)
        self.trial_name = env_config['trial_name']

        validation_size = ConfigManager.get_config('samplerConfig.validationSize')
        results = get_train_queries(checkpoint_version=self.checkpoint_version,
                                    validation_size=validation_size)
        if validation_size > 0:
            self.train_set, self.validation_set = results
        else:
            self.train_set, self.validation_set = results, None

        # TODO: There seems to be a bug if I don't add --actions_num
        if self.action_space_size < self.table_size:
            self.actions = self.__get_actions__(self.k,
                                                self.random_actions_coeff,
                                                self.checkpoint_version)
            self.num_actions = len(self.actions)
        else:
            self.num_actions = self.table_size
            self.actions = np.arange(self.num_actions)

        Preprocessing.init(self.checkpoint_version)
        num_data_cols = len(Preprocessing.columns_repo.get_all_columns().keys()) - 1
        self.state_shape = (self.k, num_data_cols)
        self.step_count = 0
        self.selected_tuples = []
        self.selected_tuples_numpy = np.array([])
        self.current_score = 0.

        self.action_mask = np.ones(self.num_actions)
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Dict(
            {
                "action_mask": Box(low=0.0, high=1.0, shape=(self.num_actions,)),
                "observations": Box(low=np.NINF, high=np.Inf, shape=self.state_shape),
            }
        )

        self.reward_config = {'marginal_reward': env_config.get('marginal_reward', False),
                              'large_reward': env_config.get('large_reward', False)}
        self.ep_reward = 0.

    def __get_actions__(self, k, random_actions_coeff, checkpoint_ver):
        if self.action_space_size <= 0:
            raise Exception('ACTION_SPACE_SIZE must be positive to use __get_actions__!')

        saved_actions_file_name = \
            f'{self.trial_name}/k={k}_sample-coeff={random_actions_coeff}_size={self.action_space_size}_actions'
        saved_actions = CheckpointManager.load(name=saved_actions_file_name, version=checkpoint_ver)
        if saved_actions is not None:
            return saved_actions

        queried_tuples_actions = self.__get_queries_actions__(
            int((1 - random_actions_coeff) * self.action_space_size))
        if len(queried_tuples_actions) == self.action_space_size:
            actions = queried_tuples_actions
        else:
            sampled_tuples_actions = self.__get_sampled_actions__(
                self.action_space_size - len(queried_tuples_actions), queried_tuples_actions)
            actions = np.concatenate((queried_tuples_actions, sampled_tuples_actions))
        CheckpointManager.save(name=saved_actions_file_name, content=actions, version=checkpoint_ver)
        return actions

    def __get_queries_actions__(self, num_queries_actions):
        return get_queried_tuples(num_queries_actions, verbose=False, checkpoint_version=self.checkpoint_version)

    def __get_sampled_actions__(self, num_actions, excluded_actions, method='random'):
        if method == 'random':
            all_actions = np.arange(self.table_size)
            return np.random.choice(a=all_actions[~np.isin(all_actions, excluded_actions)],
                                    size=num_actions,
                                    replace=False)
        else:
            raise Exception('Only random sampling is currently supported for actions!')

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.selected_tuples = []
        self.selected_tuples_numpy = np.zeros(self.state_shape)
        self.action_mask = np.ones(self.num_actions)
        self.ep_reward = 0.
        return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, {}

    def get_episode_scores(self):
        ep_scores = {
            'test_score': get_score2(self.get_sample_ids(), queries='test',
                                     checkpoint_version=self.checkpoint_version),
            'train_score': get_score2(self.get_sample_ids(), queries=self.train_set,
                                      checkpoint_version=self.checkpoint_version),
            'test_threshold_score': get_threshold_score(self.get_sample_ids(), queries='test',
                                                        checkpoint_version=self.checkpoint_version),
            'train_threshold_score': get_threshold_score(self.get_sample_ids(), queries=self.train_set,
                                                         checkpoint_version=self.checkpoint_version),
            'diversity_score': get_diversity_score(self.get_sample_tuples()),
            'episode_reward_sum': self.ep_reward
        }
        if self.validation_set is not None:
            ep_scores['val_score'] = get_score2(self.get_sample_ids(), queries=self.validation_set,
                                                checkpoint_version=self.checkpoint_version)
            ep_scores['val_threshold_score'] = get_threshold_score(self.get_sample_ids(), queries=self.validation_set,
                                                                   checkpoint_version=self.checkpoint_version)
        return ep_scores

    def step(self, action):
        if self.action_mask[action] == 0:  # state remains unchanged
            print(f'WARNING! invalid action selected! action: {action}')
            return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, 0., False, False, {}

        tuple_num = self.actions[action]
        selected_tuple = DataAccess.select_one(
            f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot}={tuple_num}')
        self.selected_tuples.append(selected_tuple)
        self.selected_tuples_numpy[self.step_count] = Preprocessing.tuples2numpy([selected_tuple])[0]
        self.step_count += 1
        self.action_mask[action] = 0.

        # calculate reward
        if self.inference_mode:
            new_score = get_score2(self.get_sample_ids(), queries='test', checkpoint_version=self.checkpoint_version)
        else:
            # new_score = get_score2(self.get_sample_ids(), queries=self.train_set,
            #                        checkpoint_version=self.checkpoint_version)
            new_score = get_combined_score(self.get_sample_tuples(), queries=self.train_set,
                                           alpha=(1 - self.diversity_coeff),
                                           checkpoint_version=self.checkpoint_version)

        reward = new_score
        if self.reward_config['marginal_reward']:
            reward -= self.current_score
        if self.reward_config['large_reward']:
            reward *= self.k
        self.ep_reward += reward

        self.current_score = new_score
        done = (self.step_count == self.k)
        info = {} if not done else self.get_episode_scores()

        return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, reward, done, False, info

    def render(self, mode='human'):
        pass

    def get_sample_ids(self):
        return [tup[self.pivot] for tup in self.selected_tuples]

    def get_sample_tuples(self):
        return self.selected_tuples


REJECT = 'reject'
FLUSH = 'flush'


class DropOneEnv(gym.Env):

    def __init__(self, env_config):
        super(gym.Env, self).__init__()

        self.inference_mode = env_config.get('inference_mode', False)
        self.k = env_config['k']
        self.checkpoint_version = env_config.get('checkpoint_version', CheckpointManager.get_max_version())

        # env config
        self.schema = env_config['table_details']['schema']
        self.table = env_config['table_details']['table']
        self.pivot = env_config['table_details']['pivot']
        self.table_size = env_config['table_details']['table_size']
        self.action_space_size = env_config.get('action_space_size',
                                                self.table_size) + self.k + 1  # We add k+1 for the reset
        self.random_actions_coeff = env_config.get('random_actions_coeff', 0)
        self.diversity_coeff = env_config.get('diversity_coeff', 0)
        self.trial_name = env_config['trial_name']
        self.horizon = env_config.get('horizon', 100_000)  # in drop-one horizon time is important
        self.reset_state_method = env_config.get('reset_state_method', 'step_tuples')
        if self.reset_state_method == 'custom':
            self.existing_sample = env_config['existing_sample']  # TODO

        # if self.spec is None:
        #     self.spec = EnvSpec()

        validation_size = ConfigManager.get_config('samplerConfig.validationSize')
        results = get_train_queries(checkpoint_version=self.checkpoint_version,
                                    validation_size=validation_size)
        if validation_size > 0:
            self.train_set, self.validation_set = results
        else:
            self.train_set, self.validation_set = results, None

        self.num_actions = self.k + 1  # k + 1 tuples
        self.actions = [*range(self.k)] + [REJECT]

        Preprocessing.init(self.checkpoint_version)
        num_data_cols = len(Preprocessing.columns_repo.get_all_columns().keys()) - 1
        self.state_shape = (self.k + 1, num_data_cols)
        self.step_count = 0

        self.selected_tuples = []
        self.selected_tuples_numpy = np.array([])
        self.next_tuple = None
        self.obs = None
        self.current_score = 0.

        self.action_mask = np.ones(self.num_actions)
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Dict(
            {
                "action_mask": Box(low=0.0, high=1.0, shape=(self.num_actions,)),
                "observations": Box(low=np.NINF, high=np.Inf, shape=self.state_shape),
            }
        )
        self.step_tuples = self.__get_step_tuples__()

        self.reward_config = {'large_reward': env_config.get('large_reward', False)}
        self.ep_reward = 0.

    def __get_step_tuples__(self):
        if self.action_space_size <= 0:
            raise Exception('ACTION_SPACE_SIZE must be positive to use __get_step_tuples__!')

        saved_actions_file_name = \
            f'{self.trial_name}/k={self.k}_sample-coeff={self.random_actions_coeff}_size={self.action_space_size}_actions'
        saved_actions = CheckpointManager.load(name=saved_actions_file_name, version=self.checkpoint_version)
        if saved_actions is not None:
            np.random.shuffle(saved_actions)
            return saved_actions

        queried_tuples_actions = self.__get_queries_actions__(
            int((1 - self.random_actions_coeff) * self.action_space_size))
        if len(queried_tuples_actions) == self.action_space_size:
            actions = queried_tuples_actions
        else:
            sampled_tuples_actions = self.__get_sampled_actions__(
                self.action_space_size - len(queried_tuples_actions), queried_tuples_actions)
            actions = np.concatenate((queried_tuples_actions, sampled_tuples_actions))
        tuple_ids_db_fmt = [str(idx) for idx in actions]
        # TODO: This takes too long
        actions_tuples = DataAccess.select(
            f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot} IN ({",".join(tuple_ids_db_fmt)})')
        CheckpointManager.save(name=saved_actions_file_name, content=actions_tuples, version=self.checkpoint_version)
        np.random.shuffle(actions_tuples)
        return actions_tuples

    def __get_queries_actions__(self, num_queries_actions):
        return get_queried_tuples(num_queries_actions, verbose=False, checkpoint_version=self.checkpoint_version)

    def __get_sampled_actions__(self, num_actions, excluded_actions, method='random'):
        if method == 'random':
            all_actions = np.arange(self.table_size)
            return np.random.choice(a=all_actions[~np.isin(all_actions, excluded_actions)],
                                    size=num_actions,
                                    replace=False)
        else:
            raise Exception('Only random sampling is currently supported for actions!')

    def get_initial_tuples(self, method='step_tuples'):
        # TODO: 1. random set, 2. top-k, 3. output of ppo/dqn in other env
        initial_tuples = None
        if method == 'step_tuples':
            return self.step_tuples[:self.k]
        elif method == 'random':
            initial_k_ids = np.random.choice(self.table_size, self.k, replace=False)
        elif method == 'top-k' or method == 'topk':
            initial_k_ids = get_queried_tuples(self.k, verbose=False, checkpoint_version=self.checkpoint_version)
        elif method == 'custom' and self.existing_sample is not None:
            loaded_sample = CheckpointManager.load(name=self.existing_sample.replace('.pkl', ''),
                                                   version=self.checkpoint_version)[0]
            if isinstance(loaded_sample[0], int):
                initial_k_ids = loaded_sample
                initial_tuples = None
            elif isinstance(loaded_sample[0], dict):
                initial_k_ids = None
                initial_tuples = loaded_sample
            else:
                raise TypeError
        else:
            raise NotImplementedError
        if initial_tuples is None:
            initial_k_ids_db_fmt = ",".join([str(idx) for idx in initial_k_ids])
            initial_tuples = DataAccess.select(
                f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot} IN ({initial_k_ids_db_fmt})')
        return initial_tuples

    # def get_next_tuple(self):
    #     # TODO make this faster somehow? what if we just select a random tuple from the database?
    #     #  need to update the mask accordingly
    #     #  OR
    #     # TODO we should add back the action_space thing and then select tuples from there
    #     taken_ids = [tup[self.pivot] for tup in self.selected_tuples]
    #     available_ids = np.setdiff1d(np.arange(self.table_size), taken_ids)
    #     next_id = np.random.choice(a=available_ids, size=1, replace=True)[0]
    #     next_tuple = DataAccess.select_one(f'SELECT * FROM {self.schema}.{self.table} WHERE {self.pivot}={next_id}')
    #     return next_tuple

    def get_next_tuple(self):
        idx = self.k + 1 + self.step_count
        idx = idx % len(self.step_tuples)
        return self.step_tuples[idx]

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.selected_tuples = self.get_initial_tuples(self.reset_state_method)
        self.selected_tuples_numpy = Preprocessing.tuples2numpy(self.selected_tuples)
        self.action_mask = np.ones(self.num_actions)  # TODO for now this should always be 1
        self.ep_reward = 0.
        self.next_tuple = self.get_next_tuple()
        next_tuple_numpy = Preprocessing.tuples2numpy([self.next_tuple])[0]
        self.obs = np.concatenate([self.selected_tuples_numpy, np.expand_dims(next_tuple_numpy, axis=0)])
        return {'observations': self.obs, 'action_mask': self.action_mask}, {}

    def get_episode_scores(self):
        ep_scores = {
            'test_score': get_score2(self.get_sample_ids(), queries='test',
                                     checkpoint_version=self.checkpoint_version),
            'train_score': get_score2(self.get_sample_ids(), queries=self.train_set,
                                      checkpoint_version=self.checkpoint_version),
            'test_threshold_score': get_threshold_score(self.get_sample_ids(), queries='test',
                                                        checkpoint_version=self.checkpoint_version),
            'train_threshold_score': get_threshold_score(self.get_sample_ids(), queries=self.train_set,
                                                         checkpoint_version=self.checkpoint_version),
            'diversity_score': get_diversity_score(self.get_sample_tuples()),
            'episode_reward_sum': self.ep_reward
        }
        if self.validation_set is not None:
            ep_scores['val_score'] = get_score2(self.get_sample_ids(), queries=self.validation_set,
                                                checkpoint_version=self.checkpoint_version)
            ep_scores['val_threshold_score'] = get_threshold_score(self.get_sample_ids(), queries=self.validation_set,
                                                                   checkpoint_version=self.checkpoint_version)
        return ep_scores

    def step(self, action_num):
        if self.action_mask[action_num] == 0:  # state remains unchanged
            print(f'WARNING! invalid action selected! action: {action_num}')
            return {'observations': self.obs, 'action_mask': self.action_mask}, 0., False, False, {}

        selected_action = self.actions[action_num]
        if selected_action == REJECT:  # no replacement will be made
            self.step_count += 1
            done = (self.step_count == self.horizon)
            info = {} if not done else self.get_episode_scores()
            self.next_tuple = self.get_next_tuple()
            next_tuple_numpy = Preprocessing.tuples2numpy([self.next_tuple])[0]
            self.obs = np.concatenate([self.selected_tuples_numpy, np.expand_dims(next_tuple_numpy, axis=0)])
            return {'observations': self.obs, 'action_mask': self.action_mask}, 0., done, False, info

        assert isinstance(selected_action, int)  # replace a tuple

        tuple_num = int(selected_action)  # which tuple to replace

        self.selected_tuples[tuple_num] = self.next_tuple
        self.selected_tuples_numpy[tuple_num] = Preprocessing.tuples2numpy([self.next_tuple])[0]
        self.step_count += 1

        # calculate reward
        if self.inference_mode:
            new_score = get_score2(self.get_sample_ids(), queries='test', checkpoint_version=self.checkpoint_version)
        else:
            new_score = get_combined_score(self.get_sample_tuples(), queries=self.train_set,
                                           alpha=(1 - self.diversity_coeff),
                                           checkpoint_version=self.checkpoint_version)

        self.next_tuple = self.get_next_tuple()
        next_tuple_numpy = Preprocessing.tuples2numpy([self.next_tuple])[0]

        reward = new_score - self.current_score
        if self.reward_config['large_reward']:
            reward *= self.k
        self.ep_reward += reward

        self.current_score = new_score
        done = (self.step_count >= self.horizon)
        info = {} if not done else self.get_episode_scores()
        self.obs = np.concatenate([self.selected_tuples_numpy, np.expand_dims(next_tuple_numpy, axis=0)])

        return {'observations': self.obs, 'action_mask': self.action_mask}, reward, done, False, info

    def render(self, mode='human'):
        pass

    def get_sample_ids(self):
        return [tup[self.pivot] for tup in self.selected_tuples]

    def get_sample_tuples(self):
        return self.selected_tuples


if __name__ == '__main__':
    print('Creating test env:')
    tst_schema = ConfigManager.get_config('queryConfig.schema')
    tst_table = ConfigManager.get_config('queryConfig.table')
    tst_pivot = ConfigManager.get_config('queryConfig.pivot')
    env = DropOneEnv({'k': 1000,
                      'checkpoint_version': CheckpointManager.get_max_version(),
                      'marginal_reward': False,  # ChooseK specific
                      'large_reward': False,
                      'table_details': {'schema': tst_schema, 'table': tst_table, 'pivot': tst_pivot,
                                        'table_size': 640585},
                      'action_space_size': 100,
                      'random_actions_coeff': 0.,
                      'diversity_coeff': 0.,
                      'trial_name': 'TEST',
                      'horizon': 100  # DropOne specific
                      })
    print('Calling env.reset()')
    observation, info = env.reset()
    obs = observation['observations']
    print(f'State shape: {env.state_shape}')
    print(f'Obs shape: {np.shape(obs)}')
    print(f'Min: {np.min(obs)}, Max: {np.max(obs)}')
    print(f'Env Spec: {env.spec}')

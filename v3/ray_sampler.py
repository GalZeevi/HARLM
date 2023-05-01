import argparse
import datetime
import os
import time

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from gymnasium.spaces import Discrete, Box
from ray.rllib.algorithms import algorithm, ppo, impala, dqn, apex_dqn, a3c
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from tqdm import tqdm, trange

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing
from score_calculator import get_score2, get_threshold_score
from top_queried_sampler import prepare_sample as get_queried_tuples
from train_test_utils import get_train_queries


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--remote", type=bool, default=False, help="Whether to use config for remote server or local server"
    )

    parser.add_argument(
        "--alg", type=str, default=AlgorithmNames.PPO, help="The RLlib-registered algorithm to use."
    )

    parser.add_argument(
        "--cpus", type=int, default=6, help="How many cpus to use for the algorithm."
    )

    parser.add_argument(
        "--rollouts", type=int, default=3, help="How many rollout workers to use."
    )

    parser.add_argument(
        "--gpus", type=int, default=0, help="How many gpus to use for the algorithm."
    )

    parser.add_argument(
        "--steps", type=int, default=300, help="How many train steps to run."
    )

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="Which checkpoint version to use."
    )

    parser.add_argument(
        "--random_actions_coeff", type=float, default=0, help="Percentage of random tuples to add to action space."
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


def __init_table_details__():
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
    return schema, table, pivot, table_size


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space['observations'],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask  # TODO: handle 2-dim logits

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class AlgorithmNames:
    IMPALA = 'IMPALA'
    A3C = 'A3C'
    PPO = 'PPO'
    DQN = 'DQN'
    APEX_DQN = 'APEX_DQN'


class MyCallbacks(DefaultCallbacks):

    @staticmethod
    def add_metric_to_episode(episode, ep_infos, metric='score'):
        train_scores = []
        val_scores = []
        test_scores = []

        for key, agent_info in ep_infos.items():
            if key != '__common__':
                train_scores.append(agent_info.get(f'train_{metric}', 0))
                f'val_{metric}' in agent_info.keys() and val_scores.append(agent_info.get(f'val_{metric}'))
                test_scores.append(agent_info.get(f'test_{metric}', 0))

        train_score = np.mean(train_scores)
        episode.custom_metrics[f'train_{metric}'] = train_score

        test_score = np.mean(test_scores)
        episode.custom_metrics[f'test_{metric}'] = test_score

        if len(val_scores) > 0:
            val_score = np.mean(val_scores)
            episode.custom_metrics[f'val_{metric}'] = val_score

    def on_episode_end(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            env_index=None,
            **kwargs):

        ep_infos = episode._last_infos
        MyCallbacks.add_metric_to_episode(episode, ep_infos, metric='score')
        MyCallbacks.add_metric_to_episode(episode, ep_infos, metric='threshold_score')


class MyEnv(gym.Env):

    def __init__(self, env_config):
        super(gym.Env, self).__init__()

        self.inference_mode = env_config.get('inference_mode', False)
        self.k = env_config['k']
        self.checkpoint_version = env_config.get('checkpoint_version', CheckpointManager.get_max_version())

        validation_size = ConfigManager.get_config('samplerConfig.validationSize')
        results = get_train_queries(checkpoint_version=self.checkpoint_version,
                                    validation_size=validation_size)
        if validation_size > 0:
            self.train_set, self.validation_set = results
        else:
            self.train_set, self.validation_set = results, None

        # reduce_action_dim = env_config.get('reduce_action_dim', True)
        if ACTION_SPACE_SIZE < table_size:
            self.actions = MyEnv.__get_actions__(self.k, table_size, RANDOM_ACTIONS_COEFF, self.checkpoint_version)
            self.num_actions = len(self.actions)
        else:
            self.num_actions = table_size
            self.actions = np.arange(self.num_actions)

        num_data_cols = DataAccess.select_one(f"SELECT COUNT(column_name) FROM information_schema.columns " +
                                              f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                              f"AND column_name <> '{pivot}'")
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
        Preprocessing.init()

    @staticmethod
    def __get_actions__(k, max_action_id, random_actions_coeff, checkpoint_ver):
        if ACTION_SPACE_SIZE <= 0:
            raise Exception('ACTION_SPACE_SIZE must be positive to use __get_actions__!')

        saved_actions_file_name = f'{TRIAL_NAME}/k={k}_sample-coeff={random_actions_coeff}_size={ACTION_SPACE_SIZE}_actions'
        saved_actions = CheckpointManager.load(name=saved_actions_file_name, version=checkpoint_ver)
        if saved_actions is not None:
            return saved_actions

        queried_tuples_actions = MyEnv.__get_queries_actions__(int((1 - random_actions_coeff) * ACTION_SPACE_SIZE))
        if len(queried_tuples_actions) == ACTION_SPACE_SIZE:
            actions = queried_tuples_actions
        else:
            sampled_tuples_actions = MyEnv.__get_sampled_actions__(ACTION_SPACE_SIZE - len(queried_tuples_actions),
                                                                   max_action_id, queried_tuples_actions)
            actions = np.concatenate((queried_tuples_actions, sampled_tuples_actions))
        CheckpointManager.save(name=saved_actions_file_name, content=actions, version=checkpoint_ver)
        return actions

    @staticmethod
    def __get_queries_actions__(num_actions):
        return get_queried_tuples(num_actions, verbose=False)

    @staticmethod
    def __get_sampled_actions__(num_actions, max_action_id, excluded_actions, method='random'):
        if method == 'random':
            all_actions = np.arange(max_action_id)
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
        return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, {}

    def step(self, action):
        if self.action_mask[action] == 0:  # state remains unchanged
            print(f'WARNING! invalid action selected! action: {action}')
            return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, 0., False, False, {}

        tuple_num = self.actions[action]
        selected_tuple = DataAccess.select_one(
            f'SELECT * FROM {schema}.{table} WHERE {pivot}={tuple_num}')
        self.selected_tuples.append(selected_tuple)
        self.selected_tuples_numpy[self.step_count] = Preprocessing.tuples2numpy([selected_tuple])[0]
        self.step_count += 1
        self.action_mask[action] = 0.

        # calculate reward
        if self.inference_mode:
            new_score = get_score2(self.get_tuple_ids(), queries='test', checkpoint_version=self.checkpoint_version)
        else:
            new_score = get_score2(self.get_tuple_ids(), queries=self.train_set,
                                   checkpoint_version=self.checkpoint_version)

        # TODO try with and without smoothing, add all that to env_config
        reward = new_score
        if self.reward_config['marginal_reward']:
            reward -= self.current_score
        if self.reward_config['large_reward']:
            reward *= 100

        self.current_score = new_score
        done = (self.step_count == self.k)
        info = {}
        if done:
            info = {
                'test_score': get_score2(self.get_tuple_ids(), queries='test',
                                         checkpoint_version=self.checkpoint_version),
                'train_score': get_score2(self.get_tuple_ids(), queries=self.train_set,
                                          checkpoint_version=self.checkpoint_version),
                'test_threshold_score': get_threshold_score(self.get_tuple_ids(), queries='test',
                                                            checkpoint_version=self.checkpoint_version),
                'train_threshold_score': get_threshold_score(self.get_tuple_ids(), queries=self.train_set,
                                                             checkpoint_version=self.checkpoint_version)
            }
            if self.validation_set is not None:
                info['val_score'] = get_score2(self.get_tuple_ids(), queries=self.validation_set,
                                               checkpoint_version=self.checkpoint_version)
                info['val_threshold_score'] = get_threshold_score(self.get_tuple_ids(), queries=self.validation_set,
                                                                  checkpoint_version=self.checkpoint_version)

        return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, reward, done, False, info

    def render(self, mode='human'):
        pass

    def get_tuple_ids(self):
        return [tup[pivot] for tup in self.selected_tuples]


cli_args = get_cli_args()
local_config_params = {'K': 100}
remote_config_params = {'K': 1000, 'ACTION_SPACE_SIZE': 5_000}
config = remote_config_params if cli_args.remote else local_config_params

K = config.get('K')
schema, table, pivot, table_size = __init_table_details__()
ACTION_SPACE_SIZE = config.get('ACTION_SPACE_SIZE', table_size)
MAX_ITERS = cli_args.steps
NUM_GPUS = cli_args.gpus
NUM_CPUS = cli_args.cpus
ALG = cli_args.alg
NUM_ROLLOUT_WORKERS = cli_args.rollouts
CHECKPOINT_VER = cli_args.checkpoint
RANDOM_ACTIONS_COEFF = cli_args.random_actions_coeff

TRIAL_NAME = f'{K}_{ALG}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
OUTPUT_DIR = f'{CheckpointManager.basePath}/{CHECKPOINT_VER}/{TRIAL_NAME}'
RECORD_RAW_TRAIN_RESULTS = config.get('RECORD_RAW_TRAIN_RESULTS', True)
SAVE_RESULT_STEP = 5
SAVE_MODEL_STEP = 5


def get_env_config():
    env_config = {'k': K, 'checkpoint_version': CHECKPOINT_VER}
    if not cli_args.remote:
        env_config['reduce_action_dim'] = False
    return env_config


def get_algorithm():
    model_config = {'custom_model': TorchActionMaskModel, 'custom_model_config': {}}
    env_config = get_env_config()
    if ALG == AlgorithmNames.IMPALA:
        # TODO NUM_GPUS // NUM_ROLLOUT_WORKERS or NUM_GPUS
        alg_config = impala.ImpalaConfig() \
            .environment(env=MyEnv, render_env=False, env_config=env_config) \
            .training(lr=0.0003, train_batch_size=512) \
            .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .framework('torch') \
            .callbacks(callbacks_class=MyCallbacks) \
            .rollouts(num_rollout_workers=NUM_ROLLOUT_WORKERS)
        alg = impala.Impala(alg_config)
        raise NotImplementedError
    elif ALG == AlgorithmNames.PPO:
        alg_config = ppo.PPOConfig() \
            .environment(env=MyEnv, render_env=False, env_config=env_config) \
            .training(model=model_config, entropy_coeff=0.01) \
            .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(num_rollout_workers=NUM_ROLLOUT_WORKERS) \
            .framework('torch') \
            .callbacks(callbacks_class=MyCallbacks)

        alg = ppo.PPO(config=alg_config)
    elif ALG == AlgorithmNames.A3C:
        alg_config = a3c.A3CConfig() \
            .environment(env=MyEnv, render_env=False, env_config=env_config) \
            .training(lr=0.01, grad_clip=30.0, model=model_config, entropy_coeff=0.3) \
            .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(num_rollout_workers=NUM_ROLLOUT_WORKERS) \
            .framework('torch') \
            .callbacks(callbacks_class=MyCallbacks) \
            .exploration(explore=True, exploration_config={'epsilon_timesteps': 20000,
                                                           'final_epsilon': 0.01,
                                                           "initial_epsilon": 0.996,
                                                           "type": "EpsilonGreedy"})  # TODO is this doing anything?
        alg = a3c.A3C(config=alg_config)
    elif ALG == AlgorithmNames.DQN:
        alg_config = dqn.DQNConfig() \
            .environment(env=MyEnv, render_env=False, env_config=env_config) \
            .resources(num_gpus=NUM_GPUS // NUM_ROLLOUT_WORKERS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(rollout_fragment_length=4, num_rollout_workers=NUM_ROLLOUT_WORKERS, num_envs_per_worker=1) \
            .callbacks(callbacks_class=MyCallbacks) \
            .training(
            model=model_config,
            double_q=True,
            dueling=True,
            num_atoms=1,
            noisy=False,
            replay_buffer_config={'type': 'MultiAgentReplayBuffer', 'capacity': 100000},
            num_steps_sampled_before_learning_starts=20000,
            n_step=1,
            target_network_update_freq=8000,
            lr=0.0000625,
            optimizer={'adam_epsilon': 0.00015},
            hiddens=[512],
            train_batch_size=32,
        ) \
            .exploration(explore=True, exploration_config={'epsilon_timesteps': 200000, 'final_epsilon': 0.01}) \
            .reporting(min_sample_timesteps_per_iteration=10000)
        alg = dqn.DQN(config=alg_config)
        raise NotImplementedError
    elif ALG == AlgorithmNames.APEX_DQN:
        # model_config['fcnet_hiddens'] = [ACTION_SPACE_SIZE] # TODO
        # model_config['no_final_linear'] = True # TODO
        alg_config = apex_dqn.ApexDQNConfig() \
            .environment(env=MyEnv, render_env=False, env_config=env_config) \
            .resources(num_gpus=min(NUM_GPUS, 1), num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(rollout_fragment_length=32, num_rollout_workers=NUM_ROLLOUT_WORKERS, num_envs_per_worker=15) \
            .callbacks(callbacks_class=MyCallbacks) \
            .training(
            model=model_config,
            double_q=True,
            dueling=False,
            num_atoms=1,  # TODO
            noisy=False,
            replay_buffer_config={'capacity': 1000000,
                                  'replay_buffer_shards_colocated_with_driver': True,
                                  "prioritized_replay_alpha": 0.45,
                                  "prioritized_replay_beta": 0.55,
                                  "prioritized_replay_eps": 3e-6},
            num_steps_sampled_before_learning_starts=2000,
            n_step=1,
            target_network_update_freq=8000,
            lr=0.0000625,
            optimizer={'adam_epsilon': 0.00015},
            train_batch_size=128,
            hiddens=[]
        ) \
            .exploration(explore=True,
                         exploration_config={'epsilon_timesteps': 2000000, 'final_epsilon': 0.01,
                                             "initial_epsilon": 1.0, "type": "EpsilonGreedy"}) \
            .reporting(min_sample_timesteps_per_iteration=1000) \
            .framework('torch')
        alg = apex_dqn.ApexDQN(config=alg_config)
    else:
        raise NotImplementedError
    return alg


def init_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def train_model():
    start = time.time()
    print(f'############### Initialising ray ###############')
    ray.init()
    print(f'Initialising ray took: {round(time.time() - start, 2)} sec')

    start = time.time()
    print(f'############### Building algorithm: {ALG} ###############')
    algo = get_algorithm()
    print(f'Building algorithm: {ALG} took: {round(time.time() - start, 2)} sec')

    i = 0
    pbar = tqdm(total=MAX_ITERS)
    while i < MAX_ITERS:
        res = algo.train()
        i += 1
        if i > 0 and i % SAVE_MODEL_STEP == 0:
            algo.save(checkpoint_dir=OUTPUT_DIR)
        if RECORD_RAW_TRAIN_RESULTS and i > 0 and i % SAVE_RESULT_STEP == 0:
            f = open(f'{OUTPUT_DIR}/iter_{i}_train_results.txt', 'w')
            f.write(str(res))
            f.close()
        pbar.update()
    return algo


def _get_sample_from_model(model):
    env = MyEnv(env_config=get_env_config())
    done = False
    obs, _ = env.reset()
    prev_action = None
    prev_reward = None

    while not done:
        action = model.compute_single_action(observation=obs, prev_action=prev_action, prev_reward=prev_reward)
        next_obs, reward, done, _, _ = env.step(action=action)
        prev_action = action
        prev_reward = reward
        obs = next_obs

    sample = env.get_tuple_ids()
    scores = {'test': get_score2(sample, queries='test', checkpoint_version=CHECKPOINT_VER),
              'train': get_score2(sample, queries=env.train_set, checkpoint_version=CHECKPOINT_VER)}
    if env.validation_set is not None:
        scores['val'] = get_score2(sample, queries=env.validation_set, checkpoint_version=CHECKPOINT_VER)

    return sample, scores


def test_model(ray_checkpoint_path=None, algo=None, num_trials=50, output_trial_name=TRIAL_NAME):
    if ray_checkpoint_path is None and algo is None:
        raise Exception('One of "ray_checkpoint_path" or "model" must not be None!')
    algo = algo if algo is not None else algorithm.Algorithm.from_checkpoint(ray_checkpoint_path)

    best_sample = None
    best_test_score = 0.
    scores = []
    for _ in trange(num_trials):
        sample, scores_dict = _get_sample_from_model(algo)
        scores.append(scores_dict)

        if scores_dict['test'] > best_test_score:
            best_test_score = scores_dict['test']
            best_sample = sample

    CheckpointManager.save(name=f'{output_trial_name}/test_scores', content=scores, version=CHECKPOINT_VER)
    CheckpointManager.save(name=f'{output_trial_name}/sample', content=[best_sample, best_test_score],
                           version=CHECKPOINT_VER)
    return best_sample, best_test_score


if __name__ == '__main__':
    init_output_dir()
    model = train_model()
    sample, score = test_model(algo=model)
    print(f'############### Sample score: {round(score, ndigits=4)} ###############')

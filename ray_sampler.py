import argparse
import datetime
import os
import time
import shutil

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
from ray_environments import ChooseKEnv, DropOneEnv
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing, ColumnsRepo, Column
from score_calculator import get_combined_score, get_score2, get_threshold_score, get_diversity_score
from top_queried_sampler import prepare_sample as get_queried_tuples
from train_test_utils import get_train_queries


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k", type=int, default=100, help="Size of the subset to create"
    )

    parser.add_argument(
        "--alg", type=str, default=AlgorithmNames.PPO, help="The RLlib-registered algorithm to use."
    )

    parser.add_argument(
        "--env", type=str, default=EnvNames.CHOOSE_K, help="The environment to use."
    )

    parser.add_argument(
        "--cpus", type=int, default=1, help="How many cpus to use for the algorithm."
    )

    parser.add_argument(
        "--gpus", type=int, default=0, help="How many gpus to use for the algorithm."
    )

    parser.add_argument(
        "--rollouts", type=int, default=1, help="How many rollout workers to use."
    )

    parser.add_argument(
        "--steps", type=int, default=300, help="How many train steps to run."
    )

    parser.add_argument(
        "--eval_steps", type=int, default=25, help="How many evaluation steps to run."
    )

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="Which checkpoint version to use."
    )

    parser.add_argument(
        "--num_actions", type=int, default=100, help="How many actions to use for the model."
    )

    parser.add_argument(
        "--margin_reward", action='store_true', default=False
    )

    parser.add_argument(
        "--random_actions_coeff", type=float, default=0, help="Percentage of random tuples to add to action space."
    )

    parser.add_argument(
        "--diversity_coeff", type=float, default=0.0, help="Weight to give diversity in score function."
    )

    parser.add_argument(
        "--horizon", type=int, default=50_000, help="Time horizon for each episode (for DropOneEnv)."
    )

    parser.add_argument(
        "--reset_state_method", type=str, default='shuffle', help="How to create initial state (for DropOneEnv)."
    )

    parser.add_argument(
        "--existing_sample", type=str, default=None, help="Path to an existing sample (for DropOneEnv)."
    )

    parser.add_argument(
        "--trial_name", type=str, default=None, help="Name of the trial."
    )

    parser.add_argument(
        "--ray_checkpoint", type=str, default='latest_checkpoint', help="Name of ray's checkpoint directory to use."
    )

    parser.add_argument(
        "--ppo_ent_coeff", type=float, default=0, help="PPO Entropy coefficient to use."
    )

    parser.add_argument(
        "--ppo_kl_coeff", type=float, default=0.2, help="PPO kl coefficient to use."
    )

    parser.add_argument(
        "--ppo_lr", type=float, default=5e-5, help="PPO lr to use."
    )

    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}", flush=True)
    return args


def __init_table_details__():
    start = time.time()
    print(f'############### Initialising table details... ###############')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
    print(f'Initialising table details took: {round(time.time() - start, 2)} sec')
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


class EnvNames:
    CHOOSE_K = 'CHOOSE_K'
    DROP_ONE = 'DROP_ONE'


class MyCallbacks(DefaultCallbacks):  # TODO: remove ep_reward_sum from callbacks!

    @staticmethod
    def add_simple_metric_to_episode(episode, ep_infos, metric):
        scores = []
        for key, agent_info in ep_infos.items():
            if key != '__common__':
                scores.append(agent_info.get(metric, 0))
        score = np.mean(scores)
        episode.custom_metrics[metric] = score

    @staticmethod
    def add_complex_metric_to_episode(episode, ep_infos, metric='score'):
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
        MyCallbacks.add_complex_metric_to_episode(episode, ep_infos, metric='score')
        MyCallbacks.add_complex_metric_to_episode(episode, ep_infos, metric='threshold_score')
        MyCallbacks.add_simple_metric_to_episode(episode, ep_infos, metric='diversity_score')
        MyCallbacks.add_simple_metric_to_episode(episode, ep_infos, metric='episode_reward_sum')


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

        if ACTION_SPACE_SIZE < table_size:
            self.actions = MyEnv.__get_actions__(self.k, table_size, cli_args.random_actions_coeff,
                                                 self.checkpoint_version)
            self.num_actions = len(self.actions)
        else:
            self.num_actions = table_size
            self.actions = np.arange(self.num_actions)

        Preprocessing.init(CHECKPOINT_VER)
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
            print(f'WARNING! invalid action selected! action: {action}', flush=True)
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
            new_score = get_score2(self.get_sample_ids(), queries='test', checkpoint_version=self.checkpoint_version)
        else:
            # new_score = get_score2(self.get_sample_ids(), queries=self.train_set,
            #                        checkpoint_version=self.checkpoint_version)
            new_score = get_combined_score(self.get_sample_tuples(), queries=self.train_set,
                                           alpha=(1 - cli_args.diversity_coeff),
                                           checkpoint_version=self.checkpoint_version)

        reward = new_score
        if self.reward_config['marginal_reward']:
            reward -= self.current_score
        if self.reward_config['large_reward']:
            reward *= K
        self.ep_reward += reward

        self.current_score = new_score
        done = (self.step_count == self.k)
        info = {} if not done else self.get_episode_scores()

        return {'observations': self.selected_tuples_numpy, 'action_mask': self.action_mask}, reward, done, False, info

    def render(self, mode='human'):
        pass

    def get_sample_ids(self):
        return [tup[pivot] for tup in self.selected_tuples]

    def get_sample_tuples(self):
        return self.selected_tuples


cli_args = get_cli_args()

K = cli_args.k
schema, table, pivot, table_size = __init_table_details__()
ACTION_SPACE_SIZE = table_size if cli_args.num_actions < 0 else cli_args.num_actions
MAX_ITERS = cli_args.steps
NUM_GPUS = cli_args.gpus
NUM_CPUS = cli_args.cpus
ALG = cli_args.alg
NUM_ROLLOUT_WORKERS = cli_args.rollouts
CHECKPOINT_VER = cli_args.checkpoint

TRIAL_NAME = f'{K}_{cli_args.env}_{ALG}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}' \
    if cli_args.trial_name is None else cli_args.trial_name
OUTPUT_DIR = f'{CheckpointManager.basePath}/{CHECKPOINT_VER}/{TRIAL_NAME}'
RECORD_RAW_TRAIN_RESULTS = True
SAVE_RESULT_STEP = 10
SAVE_MODEL_STEP = 15


def get_env_config():
    config = {'k': K,
              'checkpoint_version': CHECKPOINT_VER,
              'marginal_reward': cli_args.margin_reward,  # ChooseK specific
              'large_reward': cli_args.margin_reward,
              'table_details': {'schema': schema, 'table': table, 'pivot': pivot, 'table_size': table_size},
              'action_space_size': ACTION_SPACE_SIZE,
              'random_actions_coeff': cli_args.random_actions_coeff,
              'diversity_coeff': cli_args.diversity_coeff,
              'trial_name': TRIAL_NAME,
              'horizon': cli_args.horizon,  # DropOne specific
              'reset_state_method': cli_args.reset_state_method  # DropOne specific
              }
    if cli_args.reset_state_method == 'custom':
        config['existing_sample'] = cli_args.existing_sample  # DropOne specific
    return config


def get_algorithm():
    model_config = {'custom_model': TorchActionMaskModel, 'custom_model_config': {}}
    env_config = get_env_config()
    env_class = get_environment()
    if ALG == AlgorithmNames.IMPALA:
        alg_config = impala.ImpalaConfig() \
            .environment(env=env_class, render_env=False, env_config=env_config) \
            .training(lr=0.0003, train_batch_size=512) \
            .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .framework('torch') \
            .callbacks(callbacks_class=MyCallbacks) \
            .rollouts(num_rollout_workers=NUM_ROLLOUT_WORKERS)
        alg = impala.Impala(alg_config)
        raise NotImplementedError
    elif ALG == AlgorithmNames.PPO:
        alg_config = ppo.PPOConfig() \
            .environment(env=env_class, render_env=False, env_config=env_config) \
            .training(model=model_config, entropy_coeff=cli_args.ppo_ent_coeff, kl_coeff=cli_args.ppo_kl_coeff, lr=cli_args.ppo_lr) \
            .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(num_rollout_workers=NUM_ROLLOUT_WORKERS) \
            .framework('torch') \
            .callbacks(callbacks_class=MyCallbacks)
        alg = ppo.PPO(config=alg_config)
    elif ALG == AlgorithmNames.A3C:
        alg_config = a3c.A3CConfig() \
            .environment(env=env_class, render_env=False, env_config=env_config) \
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
            .environment(env=env_class, render_env=False, env_config=env_config) \
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
            .environment(env=env_class, render_env=False, env_config=env_config) \
            .resources(num_gpus=min(NUM_GPUS, 1), num_cpus_per_worker=NUM_CPUS // NUM_ROLLOUT_WORKERS) \
            .rollouts(rollout_fragment_length=32, num_rollout_workers=NUM_ROLLOUT_WORKERS, num_envs_per_worker=1) \
            .callbacks(callbacks_class=MyCallbacks) \
            .training(
            model=model_config,
            double_q=True,
            dueling=False,
            num_atoms=1,
            noisy=False,
            replay_buffer_config={'capacity': 1000000,
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
                         exploration_config={'epsilon_timesteps': 562_500, 'final_epsilon': 0.01,
                                             "initial_epsilon": 1.0, "type": "EpsilonGreedy"}) \
            .reporting(min_sample_timesteps_per_iteration=1000) \
            .framework('torch')
        alg = apex_dqn.ApexDQN(config=alg_config)
    else:
        raise NotImplementedError
    return alg


def get_environment():
    if cli_args.env == EnvNames.CHOOSE_K:
        return ChooseKEnv
    elif cli_args.env == EnvNames.DROP_ONE:
        return DropOneEnv
    else:
        raise NotImplementedError


def init_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_namespace():
    with open(f'{OUTPUT_DIR}/namespace.txt', "w") as f:
        f.write(str(cli_args))


def train_model():
    start = time.time()
    print(f'############### Initialising ray ###############', flush=True)
    ray.init()
    print(f'Initialising ray took: {round(time.time() - start, 2)} sec', flush=True)

    start = time.time()
    print(f'############### Building algorithm: {ALG} ###############', flush=True)
    algo = get_algorithm()
    print(f'Building algorithm: {ALG} took: {round(time.time() - start, 2)} sec', flush=True)

    i = 0
    pbar = tqdm(total=MAX_ITERS)
    while i < MAX_ITERS:
        res = algo.train()
        i += 1
        if i > 0 and i % SAVE_MODEL_STEP == 0:
            path_to_checkpoint = algo.save(checkpoint_dir=OUTPUT_DIR)
            if os.path.exists(f'{OUTPUT_DIR}/latest_checkpoint'):
                shutil.rmtree(f'{OUTPUT_DIR}/latest_checkpoint')
            shutil.copytree(path_to_checkpoint, f'{OUTPUT_DIR}/latest_checkpoint')

        if RECORD_RAW_TRAIN_RESULTS and i > 0 and i % SAVE_RESULT_STEP == 0:
            f = open(f'{OUTPUT_DIR}/iter_{i}_train_results.txt', 'w')
            f.write(str(res))
            f.close()
        pbar.update()
    return algo


def _get_sample_from_model(model, env):
    done = False
    obs, _ = env.reset()
    prev_action = None
    prev_reward = None

    pbar = tqdm(total=cli_args.k if cli_args.env == EnvNames.CHOOSE_K else cli_args.horizon)
    while not done:
        action = model.compute_single_action(observation=obs, prev_action=prev_action, prev_reward=prev_reward)
        next_obs, reward, done, _, _ = env.step(action)
        prev_action = action
        prev_reward = reward
        obs = next_obs
        pbar.update()

    sample_ids = env.get_sample_ids()
    scores = env.get_episode_scores()
    return sample_ids, scores


def test_model(ray_checkpoint_path=None, algo=None, num_trials=cli_args.eval_steps):
    if ray_checkpoint_path is None and algo is None:
        raise Exception('One of \'ray_checkpoint_path\' or \'model\' must not be None!')
    algo = algo if algo is not None else algorithm.Algorithm.from_checkpoint(ray_checkpoint_path)

    start = time.time()
    print(f'############### Initialising environment for test... ###############', flush=True)
    env_config = get_env_config()
    env_config['inference_mode'] = True
    env = get_environment()(env_config)
    print(f'Initialising environment for test took: {round(time.time() - start, 2)} sec', flush=True)

    best_sample = None
    best_test_score = 0.
    min_test_score = 1.0
    avg_test_score = 0.
    scores = []
    for i in trange(num_trials):
        sample, scores_dict = _get_sample_from_model(algo, env)
        scores.append(scores_dict)
        # print(f'Current scores: {scores_dict}', flush=True)

        curr_test_score = scores_dict['test_score']

        if curr_test_score > best_test_score:
            best_test_score = curr_test_score
            best_sample = sample

        elif curr_test_score < min_test_score:
            min_test_score = curr_test_score

        avg_test_score += curr_test_score
        print(f'Min score: [{min_test_score}], Avg score: [{avg_test_score / (i + 1)}], Max score: [{best_test_score}]',
              flush=True)

        CheckpointManager.save(name=f'{TRIAL_NAME}/trial_scores', content=scores, version=CHECKPOINT_VER)
        CheckpointManager.save(name=f'{TRIAL_NAME}/sample', content=[best_sample, best_test_score],
                               version=CHECKPOINT_VER)

    return best_sample, best_test_score


if __name__ == '__main__':
    DataAccess()
    print(f'############### Starting trial: {TRIAL_NAME} ###############', flush=True)
    if cli_args.test:
        test_model(
            ray_checkpoint_path=f'{OUTPUT_DIR}/{cli_args.ray_checkpoint}')
    else:
        init_output_dir()
        save_namespace()
        model = train_model()
        DataAccess.reconnect()
        sample, score = test_model(algo=model)
    all_scores = CheckpointManager.load(name=f'{TRIAL_NAME}/trial_scores', version=CHECKPOINT_VER)
    test_scores = [scores_dict.get('test_score', 0) for scores_dict in all_scores]
    threshold_scores = [scores_dict.get('test_threshold_score', 0) for scores_dict in all_scores]
    diversity_scores = [scores_dict.get('diversity_score', 0) for scores_dict in all_scores]
    print(f'############### Sample score: '
          f'[min: {round(np.min(test_scores), ndigits=4)}, '
          f'avg: {round(np.average(test_scores), ndigits=4)}, '
          f'max: {round(np.max(test_scores), ndigits=4)}] ###############')
    print(f'############### Sample 0.25-score: '
          f'[min: {round(np.min(threshold_scores), ndigits=4)}, '
          f'avg: {round(np.average(threshold_scores), ndigits=4)}, '
          f'max: {round(np.max(threshold_scores), ndigits=4)}] ###############')
    # print(f'############### Sample diversity score: '
    #       f'[min: {round(np.min(diversity_scores), ndigits=4)}, '
    #       f'avg: {round(np.average(diversity_scores), ndigits=4)}, '
    #       f'max: {round(np.max(diversity_scores), ndigits=4)}] ###############')

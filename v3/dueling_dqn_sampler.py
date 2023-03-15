import random
from collections import namedtuple, deque
from tqdm import tqdm, trange
from checkpoint_manager_v3 import CheckpointManager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_sampler import SaqpEnv
from config_manager_v3 import ConfigManager
from score_calculator import get_score2


# TODO handle seed
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Only stores the last N experience tuples in the replay memory

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # Initialize replay memory
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # set N memory size
        self.batch_size = batch_size
        # build named experience tuples
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        we store the agent's experiences at each time-step, e_t = (s_t,a_t,r_t,s_(t+1))
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Samples uniformly at random from D(D_t = {e_1,...,e_t}) when  performing updates
        """
        # D
        experiences = random.sample(self.memory, k=self.batch_size)
        # store in
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)  # gpu
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        # return D
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc1_units=64, fc2_units=64):
        """
        Builds a feedforward network with two hidden layers
        Initialize parameters

        Params
        =========
        state_size (int): Dimension of each state (input_size)
        action_size (int): dimension of each action (output_size)
        seed (int): Random seed(using 0)
        fc1_units (int): Size of the first hidden layer
        fc2_units (int): Size of the second hidden layer
        """
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        # Add the first layer, input to hidden layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        # Add more hidden layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # State-value V
        self.V = nn.Linear(fc2_units, 1)

        # Advantage function A
        self.A = nn.Linear(fc2_units, action_size)

    def forward(self, inp):
        """
        Forward pass through the network. Build a network that mps state -> action values.

        return Q function
        """
        if len(inp.size()) < 3:
            inp = inp.unsqueeze(0)

        x = torch.flatten(inp, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)

        return V + (A - A.mean(dim=1, keepdim=True))


# hyperparameters
LR = 5e-4  # learning rate
BUFFER_SIZE = int(1e5)  # replay buffer size N
BATCH_SIZE = 64  # minibatch size
UPDATE_EVERY = 24  # how often to update the network
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # for soft update of target parameters
HORIZON = 1015
NUM_EPISODES = 3000
K = 100
# TODO try to change to UPDATE_EVERY param

# Setup Gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Build Agent(): Evaluate our agent on unmodified games (dqn agent)
class Agent:

    def __init__(self, state_size, action_size, k, seed):
        """
        Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state = 8
            action_size (int): dimension of each action = 4
            seed (int): random seed = 0
        """
        self.state_size = state_size
        self.action_size = action_size
        # self.seed = random.seed(seed)
        self.k = k

        # Q-Network: Neural network function approx. with weights theta θ as a Q-Network. A Q-Network can be trained
        # by adjusting the parameters θ_i at iteration i to reduce the mse in the Bellman equation The outputs
        # correspond to the predicted Q-values of the individual action for input state
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # gpu
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # specify optimizer(Adam)
        # optim.Adam(Qnet.parameters(), small learning rate)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # First, use a technique known as experience replay in which we stre the agent's experience at each
        # time-step, e_t= (s_t, a_t, r_t, s_(t_1)), in a data set D_t ={e_1,...,e_t},pooled over many episodes(where
        # the end of an episode occurs when a terminal state is reached) into a replay memory. Initialize replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (update every UPDATE_EVERY steps)
        self.t_step = 0
        self.huber_loss = nn.SmoothL1Loss()
        self.use_learn_v2 = False
        self.losses = []

    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        loss = -55.

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if self.use_learn_v2:
                    loss = self.learn_v2(experiences, GAMMA)
                else:
                    loss = self.learn(experiences, GAMMA)
                # save model
                torch.save({
                    'model_state_dict': self.qnetwork_local.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'{CheckpointManager.get_checkpoint_path()}/{self.k}_{NUM_EPISODES}_{HORIZON}_dueling_dqn.pt')
        return loss

    def act(self, state, eps=0.):
        """
        Choose action A from state S using policy pi <- epsilon-Greedy(q^hat (S,A,w))
        Return actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # It is off-policy: it learns about the greedy policy a = argmax Q(s,a';θ),
        # while following a behaviour distribution is often selected by an eps-greedy policy
        # that follows the greey policy with probability 1-eps and selects a random action
        # with probability eps.
        # Epsilon-greedy action selection
        #
        # with probability epsilon select a random action a_t
        # otherwise select a_t = argmax_a Q (phi(s_t),a; θ)
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn_v2(self, experiences, gamma):
        """Perform a training iteration on a batch of data."""
        states, actions, rewards, next_states, dones = experiences

        next_qs_argmax = self.qnetwork_local(next_states).argmax(dim=-1, keepdim=True)
        masked_next_qs = self.qnetwork_target(next_states).gather(1, next_qs_argmax).squeeze()
        target = rewards + ((1.0 - dones) * gamma * masked_next_qs)
        masked_qs = self.qnetwork_local(states).gather(1, actions.unsqueeze(dim=-1)).squeeze()
        loss = self.huber_loss(masked_qs, target.detach())
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return loss

    def learn(self, experiences,
              gamma):  # ----only use the local and target Q-networks to compute the loss before taking a step
        # towards minimizing the loss
        """
        Update value parameters using given batch of experience tuples

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """
        states, actions, rewards, next_states, dones = experiences
        #####DQN
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states)[:, actions]

        ##### Double DQN
        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #    # fetch max action arguemnt to pass
        #    Q_pred = self.qnetwork_local(next_states)
        #    max_actions = torch.argmax(Q_pred, dim=1).long().unsqueeze(1)
        #    # Q_targets over next statesfrom actions will be taken based on Q_pred's max_action
        #    Q_next = self.qnetwork_target(next_states)
        # self.qnetwork_local.train()
        # Q_targets = rewards + (gamma * Q_next.gather(1, max_actions) * (1.0 - dones))
        ### Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)

        ###############
        # apply loss fucntion
        # calculate the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.losses.append(loss.item())

        # zero the parameter (weight) gradients
        self.optimizer.zero_grad()

        # backward pass to calculate the parameter gradients
        loss.backward()

        # update the parameters
        self.optimizer.step()

        #################
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  ###

        return loss

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def train(k=K, horizon=HORIZON, num_episodes=NUM_EPISODES):
    env = SaqpEnv(k, max_iters=horizon)
    eps = 1.
    last_100_ep_rewards = []
    loss = -66.
    eps_losses_rewards = []
    # env.seed(0)

    agent = Agent(state_size=env.state_shape[0] * env.state_shape[1], action_size=k + 1, k=k, seed=0)

    for episode in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        for iteration in range(horizon + 1):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            # print(reward)
            if eps > 0.01:
                eps -= 1.1e-6
            if done:
                print(f'episode reward sum: {ep_reward}')
                break


        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)

        if 0 < episode and episode % 10 == 0:
            print(f'Episode: {episode}/{num_episodes}, Epsilon: {eps:.3f}, '
                  f'Loss: {agent.losses[-1]:.4f}, Return: {np.mean(last_100_ep_rewards):.8f}')

            eps_losses_rewards.append((episode, agent.losses[-1], ep_reward))
            CheckpointManager.save(f'{k}_{num_episodes}_{horizon}_dueling_dqn_graph', eps_losses_rewards)


def get_sample(k=K, n_trials=10, num_episodes=NUM_EPISODES, horizon=HORIZON):
    checkpoint = torch.load(f'{CheckpointManager.get_checkpoint_path()}/{k}_{num_episodes}_{horizon}_dueling_dqn.pt')
    if checkpoint is None:
        train(k=k, num_episodes=num_episodes, horizon=horizon)
        checkpoint = torch.load(
            f'{CheckpointManager.get_checkpoint_path()}/{k}_{num_episodes}_{horizon}_dueling_dqn.pt')

    pivot = ConfigManager.get_config('queryConfig.pivot')
    view_size = ConfigManager.get_config('samplerConfig.viewSize')

    env = SaqpEnv(k, max_iters=-1)
    agent = Agent(state_size=env.state_shape[0] * env.state_shape[1], action_size=k + 1, k=k, seed=0)
    validation_set = env.validation_set
    best_validation_score = -1
    best_sample = None

    # Prepare the model for inference
    dueling_dqn = QNetwork(agent.state_size, agent.action_size).to(device)
    dueling_dqn.load_state_dict(checkpoint['model_state_dict'])
    dueling_dqn.eval()

    for _ in trange(n_trials):
        state = env.reset()  # Resets the env and returns a random initial state
        state = torch.Tensor(state).to(device)
        env.render()  # Visualize for human

        # Infer
        for iteration in range(horizon + 1):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        curr_sample = [tup[pivot] for tup in env.best_k]
        curr_validation_score = get_score2(curr_sample, queries=validation_set)
        curr_test_score = get_score2(curr_sample, queries="test")
        tqdm.write(f'current validation score: {curr_validation_score}, current test score: {curr_test_score}')
        if curr_validation_score > best_validation_score:
            best_validation_score = curr_validation_score
            best_sample = curr_sample

    print(f'best score on validation set: {best_validation_score}')
    test_score = get_score2(best_sample, queries='test')
    print(f'score on test set: {test_score}')
    CheckpointManager.save(f'{k}-{view_size}-{k}_{num_episodes}_{horizon}_dueling_dqn_sample',
                           [best_sample, test_score, best_validation_score])
    return best_sample, test_score


if __name__ == '__main__':
    trials = 20
    train()

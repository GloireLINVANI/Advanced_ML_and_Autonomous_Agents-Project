#%% md
# # Comparing Tree Search and Reinforcement Learning Approaches for King and Courtesan Game
# ## Gloire LINVANI
# ### CSC-52081-EP Advanced Machine Learning and Autonomous Agents Project
# 
# ### The following contains adapted material from the labs 6 and 7 developed by Jérémie Decock.
# 
# <img src="https://raw.githubusercontent.com/jeremiedecock/polytechnique-csc-52081-ep-2025-students/refs/heads/main/assets/logo.jpg" style="float: left; width: 15%" />
# 
# [CSC-52081-EP-2025](https://moodle.polytechnique.fr/course/view.php?id=19336)
#%% md
# # Requirements
#%% md
# This notebook relies on several libraries including `gymnasium[classic-control]` (v1.0.0), `ipywidgets`, `matplotlib`, `moviepy`, `numpy`, `pandas`, `pygame`, `seaborn`, `torch`, and `tqdm`.
# A complete list of dependencies can be found in the `requirements.txt` file at the root of the repository
#%% md
# # Imports
#%%
import collections
import gymnasium as gym
import math
import itertools
import numpy as np

# from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
import random
import torch
from typing import Union, List, Tuple, Optional, Callable
#%%
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm.notebook import tqdm
#%%
from IPython.display import Video
from ipywidgets import interact
#%%
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
#%%
sns.set_context("talk")
#%%
PLOTS_DIR = Path("figs/") / "RL Agents"  # Where to save plots (.png or .svg files)
MODELS_DIR = Path("models/") / "RL Agents"  # Where to save models (.pth files)
#%%
if not PLOTS_DIR.exists():
    PLOTS_DIR.mkdir(parents=True)
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True)
#%%
DEFAULT_NUMBER_OF_TRAININGS = 3
#%% md
# ## PyTorch setup
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Set the device to CUDA if available, otherwise use CPU
#%%
print("Available GPUs:")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"- Device {i}: {torch.cuda.get_device_name(i)}")
elif torch.backends.mps.is_available():
    print("Using Metal Performance Shaders (MPS)")
    print(f"{torch.mps.device_count()} GPU(s) available.")
else:
    print("- No GPU available.")
#%%
print(f"PyTorch will train and test neural networks on {device}")
#%% md
# ## 1. Python King And Courtesan Environment Wrapper
#%%

#%% md
# Print some information about the environment:
#%% md
# #### Testing the environment with two random policies
#%%
env =

observation = env.reset()

for t in range(50):
    action =
    observation, reward, terminated, truncated = env.step(action)

env.close()
#%% md
# ## 2. Deep value-based Reinforcement Learning with Deep Q-Networks (DQN)
# ## Deep Q-Networks v2 (DQN version 2015) with infrequent weight updates
#%% md
# ## 2.1. The Q-network
#%%
import torch.nn as nn
import torch.nn.functional as F
#%%
class QNetwork(torch.nn.Module):
    """
    A Q-Network implemented with PyTorch.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        First fully connected layer.
    layer2 : torch.nn.Linear
        Second fully connected layer.
    layer3 : torch.nn.Linear
        Third fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Define the forward pass of the QNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        """
        Initialize a new instance of QNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        nn_l1 : int
            The number of neurons on the first layer.
        nn_l2 : int
            The number of neurons on the second layer.
        """
        super(QNetwork, self).__init__()

        # TODO...

        self.layer1 = nn.Linear(n_observations, nn_l1)
        self.layer2 = nn.Linear(nn_l1, nn_l2)
        self.layer3 = nn.Linear(nn_l2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the QNetwork.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (state).

        Returns
        -------
        torch.Tensor
            The output tensor (Q-values).
        """

        # TODO...

        x = F.relu(self.layer1(x))

        x = F.relu(self.layer2(x))

        output_tensor = self.layer3(x)

        return output_tensor

    def get_params(self) -> np.ndarray:
        """
        Get the parameters.

        Returns
        -------
        np.ndarray
            The parameters of the model.
        """
        return self.params.copy()

    def set_params(self, params: np.ndarray) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        params : np.ndarray
            The parameters of the model.
        """
        self.params = params.copy()
#%% md
# ### 2.1.1 Inference Function
#%%
def test_q_network_agent(
        env: gym.Env, q_network: torch.nn.Module, num_episode: int = 1
) -> List[float]:
    """
    Test a naive agent in the given environment using the provided Q-network.

    Parameters
    ----------
    env : gym.Env
        The environment in which to test the agent.
    q_network : torch.nn.Module
        The Q-network to use for decision making.
    q_network_adversary : torch.nn.Module
        The Q-network of the adversary.
    num_episode : int, optional
        The number of episodes to run, by default 1.

    Returns
    -------
    List[float]
        A list of rewards per episode.
    """
    episode_reward_list = []

    for episode_id in range(num_episode):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            q_values = q_network(state_tensor)


            action = q_values.argmax(dim=1).squeeze().item()

            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)

            state = next_state

        episode_reward_list.append(episode_reward)
        print(f"Episode reward: {episode_reward}")

    return episode_reward_list
#%% md
# Testing this function on the untrained agent.
#%%
dqnv2 = QNetwork(observation_dim, num_actions, nn_l1=128, nn_l2=128).to(device)
#%%
NUM_EPISODES = 3

env =

test_q_network_agent(env, dqnv2, num_episode=NUM_EPISODES)

env.close()
#%% md
# Copy and paste the output of the following cell into the first question of the [Lab 6 - Evaluation](https://moodle.polytechnique.fr/course/section.php?id=66534) in Moodle:
# *"What is the total number of parameters in the Q-Network constructed in the second exercise?"*
#%% md
# ### Epsilon Greedy Function
#%%
class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(
            self,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            env: gym.Env,
            q_network: torch.nn.Module,
    ):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """

        if random.random() < self.epsilon:
            action = int(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                q_values = self.q_network(state_tensor)

                action = q_values.argmax(dim=1).squeeze().item()

        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#%% md
# ### Learning Rate Scheduler
#%%
class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            lr_decay: float,
            last_epoch: int = -1,
            min_lr: float = 1e-6,
    ):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]
#%% md
# ### 2.1.2. Training Function
#%% md
# To compute the target value, we can eliminate the need for an if statement to differentiate between terminal and non-terminal states by using the following formula:
# 
# $$
# y = r + \gamma \max_{\mathbf{a}^\star \in \mathcal{A}} \hat{Q}_{\mathbf{\omega}}(\mathbf{s'})_{\mathbf{a}^\star} \times (1 - \text{done})
# $$
# 
# where $\text{done} = 1$ if $s'$ is a terminal state and 0 otherwise.
#%% md
# ### Replay Buffer
# 
# Memory buffer where experiences are stored. We sample a random batch of experiences from this buffer to update the weights.
#%%
class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def add(
            self,
            state: np.ndarray,
            action: np.int64,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
            self, batch_size: int
    ) -> Tuple[np.ndarray, Tuple[int], Tuple[float], np.ndarray, Tuple[bool]]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)
#%% md
# In 2015, DeepMind further advanced the field of reinforcement learning with the publication of the paper "Human-level control through deep reinforcement learning" by Volodymyr Mnih and colleagues (https://www.nature.com/articles/nature14236). This work introduced the second version of Deep Q-Networks (DQN).
# 
# <img src="https://raw.githubusercontent.com/jeremiedecock/polytechnique-csc-52081-ep-2025-students/main/assets/lab6_dqn_nature_journal.jpg" width="200px" />
# 
# The key contribution of this paper was the introduction of a method to stabilize the learning process by infrequently updating the target weights. This technique, known as *infrequent updates of target weights*, significantly improved the stability of the learning process.
#%% md
# #### Infrequent weight updates
# 
# Infrequent weight updates, also known as the use of a target network, is a technique used in Deep Q-Networks (DQN) to address the issue of learning from a moving target.
# 
# In a typical DQN setup, there are two neural networks: the Q-network and the target network. The Q-network is used to predict the Q-values and is updated at every time step. The target network is used to compute the target Q-values for the update, and its weights are updated less frequently, typically every few thousand steps, by copying the weights from the Q-network.
# 
# The idea behind infrequent weight updates is to stabilize the learning process by keeping the target Q-values fixed for a number of steps. This mitigates the issue of learning from a moving target, as the target Q-values remain fixed between updates.
# 
# Without infrequent weight updates, both the predicted and target Q-values would change at every step, which could lead to oscillations and divergence in the learning process. By introducing a delay between updates of the target Q-values, the risk of such oscillations is reduced.
#%% md
# #### DQN v2015 Algorithm
#%% md
# Note: main differences with the previous algorithm are highlighted in red.
#%% md
# <b>Input</b>:<br>
# 	$\quad\quad$ none<br>
# <b>Algorithm parameter</b>:<br>
# 	$\quad\quad$ discount factor $\gamma$<br>
# 	$\quad\quad$ step size $\alpha \in (0,1]$<br>
# 	$\quad\quad$ small $\epsilon > 0$<br>
# 	$\quad\quad$ capacity of the experience replay memory $M$<br>
# 	$\quad\quad$ batch size $m$<br>
# 	$\quad\quad$ target network update frequency $\color{red}{\tau}$<br><br>
# 
# <b>Initialize</b> replay memory $\mathcal{D}$ to capacity $M$<br>
# <b>Initialize</b> action-value function $\hat{Q}_{\mathbf{\omega_1}}$ with random weights $\mathbf{\omega_1}$<br>
# <b>Initialize</b> target action-value function $\hat{Q}_{\mathbf{\omega_2}}$ with weights $\color{red}{\mathbf{\omega_2} = \mathbf{\omega_1}}$<br><br>
# 
# <b>FOR EACH</b> episode<br>
# 	$\quad$ $\mathbf{s} \leftarrow \text{env.reset}()$<br>
# 	$\quad$ <b>DO</b> <br>
# 		$\quad\quad$ $\mathbf{a} \leftarrow \epsilon\text{-greedy}(\mathbf{s}, \hat{Q}_{\mathbf{\omega_1}})$<br>
# 		$\quad\quad$ $r, \mathbf{s'} \leftarrow \text{env.step}(\mathbf{a})$<br>
# 		$\quad\quad$ Store transition $(\mathbf{s}, \mathbf{a}, r, \mathbf{s'})$ in $\mathcal{D}$<br>
# 		$\quad\quad$ If $\mathcal{D}$ contains "enough" transitions<br>
# 			$\quad\quad\quad$ Sample random batch of transitions $(\mathbf{s}_j, \mathbf{a}_j, r_j, \mathbf{s'}_j)$ from $\mathcal{D}$ with $j=1$ to $m$<br>
# 			$\quad\quad\quad$ For each $j$, set $y_j =
# 			\begin{cases}
# 			r_j & \text{for terminal } \mathbf{s'}_j\\
# 			r_j + \gamma \max_{\mathbf{a}^\star} \hat{Q}_{\mathbf{\omega_{\color{red}{2}}}} (\mathbf{s'}_j)_{\mathbf{a}^\star} & \text{for non-terminal } \mathbf{s'}_j
# 			\end{cases}$<br>
# 			$\quad\quad\quad$ Perform a gradient descent step on $\left( y_j - \hat{Q}_{\mathbf{\omega_1}}(\mathbf{s}_j)_{\mathbf{a}_j} \right)^2$ with respect to the weights $\mathbf{\omega_1}$<br>
# 			$\quad\quad\quad$ Every $\color{red}{\tau}$ steps reset $\hat{Q}_{\mathbf{\omega_2}}$ to $\hat{Q}_{\mathbf{\omega_1}}$, i.e., set $\color{red}{\mathbf{\omega_2} \leftarrow \mathbf{\omega_1}}$<br>
# 		$\quad\quad$ $\mathbf{s} \leftarrow \mathbf{s'}$ <br>
# 	$\quad$ <b>UNTIL</b> $\mathbf{s}$ is final<br><br>
# <b>RETURN</b> $\mathbf{\omega_1}$ <br>
# 
#%% md
# Infrequent weight updates in the training function:
# 
# 1. **Update the Target Network Infrequently**: Instead of updating the weights of the target network at every time step, update them less frequently, for example, every few thousand steps. The weights of the target network are updated by copying the weights from the Q-network.
# 
# 2. **Compute Target Q-values with the Target Network**: When computing the target Q-values for the update, use the target network instead of the Q-network. This ensures that the target Q-values remain fixed between updates, which stabilizes the learning process.
#%%
def train_dqn2_agent(
        env: gym.Env,
        q_network: torch.nn.Module,
        q_network_adversary: torch.nn.Module,
        target_q_network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epsilon_greedy: EpsilonGreedy,
        device: torch.device,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_episodes: int,
        gamma: float,
        batch_size: int,
        replay_buffer: ReplayBuffer,
        target_q_network_sync_period: int,
) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    q_network_adversary : torch.nn.Module
        The Q-network of the adversary.
    target_q_network : torch.nn.Module
        The target Q-network to use for estimating the target Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    target_q_network_sync_period : int
        The number of episodes after which the target Q-network should be updated with the weights of the Q-network.

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    iteration = 0
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state = env.reset()
        episode_reward = 0.0

        for t in itertools.count():
            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, float(reward), next_state, done)

            episode_reward += float(reward)

            # Update the q_network weights with a batch of experiences from the buffer

            if len(replay_buffer) > batch_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                    batch_size)

                # Convert to PyTorch tensors
                batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

                # Compute the target Q values for the batch
                with torch.no_grad():
                    next_state_q_values = target_q_network(batch_next_states_tensor)
                    best_next_q_values = next_state_q_values.max(1)[0]

                    targets = batch_rewards_tensor + gamma * best_next_q_values * (1 - batch_dones_tensor)

                current_q_values = q_network(batch_states_tensor)
                current_q_values = current_q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)

                # Compute loss
                loss = loss_fn(current_q_values, targets)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step()

            # Update the target q-network weights

            iteration += 1

            if iteration % target_q_network_sync_period == 0:
                target_q_network.load_state_dict(q_network.state_dict())

            if done:
                break

            state = next_state

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list
#%% md
# ### Training
# 
# We need to instantiate and initialize the two neural networks.
# 
# A target network that has the same architecture as the Q-network. The weights of the target network are initially copied from the Q-network.
#%%
env =

NUMBER_OF_TRAININGS = DEFAULT_NUMBER_OF_TRAININGS
dqn2_trains_result_list: List[List[Union[int, float]]] = [[], [], []]

for train_index in range(NUMBER_OF_TRAININGS):
    # Instantiate required objects

    dqnv2 = QNetwork(observation_dim, num_actions, nn_l1=128, nn_l2=128).to(device)
    q_network_adversary = QNetwork(observation_dim, num_actions, nn_l1=128, nn_l2=128).to(device)

    target_q_network = QNetwork(observation_dim, num_actions, nn_l1=128, nn_l2=128).to(device)

    target_q_network.load_state_dict(dqnv2.state_dict())

    optimizer = torch.optim.AdamW(dqnv2.parameters(), lr=0.004, amsgrad=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    epsilon_greedy = EpsilonGreedy(
        epsilon_start=0.82,
        epsilon_min=0.013,
        epsilon_decay=0.9675,
        env=env,
        q_network=dqnv2,
    )

    replay_buffer = ReplayBuffer(2000)

    # Train the q-network

    episode_reward_list = train_dqn2_agent(
        env,
        dqnv2,
        q_network_adversary,
        target_q_network,
        optimizer,
        loss_fn,
        epsilon_greedy,
        device,
        lr_scheduler,
        num_episodes=150,
        gamma=0.9,
        batch_size=128,
        replay_buffer=replay_buffer,
        target_q_network_sync_period=30,
    )
    dqn2_trains_result_list[0].extend(range(len(episode_reward_list)))
    dqn2_trains_result_list[1].extend(episode_reward_list)
    dqn2_trains_result_list[2].extend([train_index for _ in episode_reward_list])

dqn2_trains_result_df = pd.DataFrame(
    np.array(dqn2_trains_result_list).T,
    columns=["num_episodes", "mean_final_episode_reward", "training_index"],
)
dqn2_trains_result_df["agent"] = "DQN 2015"

# Save the action-value estimation function

torch.save(dqnv2, MODELS_DIR / "dqn2_q_network.pth")

env.close()
#%% md
# #### Plots
#%%
g = sns.relplot(
    x="num_episodes",
    y="mean_final_episode_reward",
    kind="line",
    hue="agent",
    estimator=None,
    units="training_index",
    data=dqn2_trains_result_df,
    height=7,
    aspect=2,
    alpha=0.5,
)
plt.savefig(PLOTS_DIR / "dqnv2_trains_results.png")
plt.show()
#%% md
# ### Testing Against a Random Policy
#%%
NUM_EPISODES = 3

env =

test_q_network_agent(env, dqnv2, q_network_adversary, num_episode=NUM_EPISODES)

env.close()
#%% md
# ### Testing Against our Java ID Alpha Beta Agent
#%%
NUM_EPISODES = 3

env =

test_q_network_agent(env, dqnv2, num_episode=NUM_EPISODES)

env.close()
#%% md
# #### Score
#%%
train_score_dqn2 = dqn2_trains_result_df[["num_episodes", "mean_final_episode_reward"]].groupby("num_episodes").mean().max()
train_score_dqn2
#%% md
# ## 3. Deep policy-based Reinforcement Learning with Monte Carlo Policy Gradient (REINFORCE)
#%% md
# ### The Policy Gradient theorem
#%% md
# This is a policy gradient method that directly searchs in a family of parameterized policies $\pi_\theta$ for the optimal policy.
# 
# This method performs gradient ascent in the policy space so that the total return is maximized.
# We will restrict our work to episodic tasks, *i.e.* tasks that have a starting states and last for a finite and fixed number of steps $T$, called horizon.
# 
# More formally, we define an optimization criterion that we want to maximize:
# 
# $$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T r(s_t,a_t)\right],$$
# 
# where $\mathbb{E}_{\pi_\theta}$ means $a \sim \pi_\theta(\cdot|s)$ and $T$ is the horizon of the episode.
# In other words, we want to maximize the value of the starting state: $V^{\pi_\theta}(s)$.
# The policy gradient theorem tells us that:
# 
# $$
# \nabla_\theta J(\theta) = \nabla_\theta V^{\pi_\theta}(s) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta (a|s) ~ Q^{\pi_\theta}(s,a) \right],
# $$
# 
# where the $Q$-function is defined as:
# 
# $$Q^{\pi_\theta}(a|s) = \mathbb{E}^{\pi_\theta} \left[\sum_{t=1}^T r(s_t,a_t)|s=s_1, a=a_1\right].$$
# 
# The policy gradient theorem is particularly effective because it allows gradient computation without needing to understand the system's dynamics, as long as the $Q$-function for the current policy is computable. By simply applying the policy and observing the one-step transitions, sufficient information is gathered. Implementing a stochastic gradient ascent and substituting $Q^{\pi_\theta}(s_t,a_t)$ with a Monte Carlo estimate $R_t = \sum_{t'=t}^T r(s_{t'},a_{t'})$ for a single trajectory, we derive the REINFORCE algorithm.
#%% md
# The REINFORCE algorithm, introduced by Williams in 1992, is a Monte Carlo policy gradient method. It updates the policy in the direction that maximizes rewards, using full-episode returns as an unbiased estimate of the gradient. Each step involves generating an episode using the current policy, computing the gradient estimate, and updating the policy parameters. This algorithm is simple yet powerful, and it's particularly effective in environments where the policy gradient is noisy or the dynamics are complex.
# 
# For further reading and a deeper understanding, refer to Williams' seminal paper (https://link.springer.com/article/10.1007/BF00992696) and the comprehensive text on reinforcement learning by Richard S. Sutton and Andrew G. Barto: "Reinforcement Learning: An Introduction", chap.13 (http://incompleteideas.net/book/RLbook2020.pdf).
#%% md
# Here is the REINFORCE algorithm.
#%% md
# ### Monte Carlo policy gradient (REINFORCE)
# 
# <b>REQUIRE</b> <br>
# $\quad$ A differentiable policy $\pi_{\boldsymbol{\theta}}$ <br>
# $\quad$ A learning rate $\alpha \in \mathbb{R}^+$ <br>
# <b>INITIALIZATION</b> <br>
# $\quad$ Initialize parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ <br>
# <br>
# <b>FOR EACH</b> episode <br>
# $\quad$ Generate full trace $\tau = \{ \boldsymbol{s}_0, \boldsymbol{a}_0, r_1, \boldsymbol{s}_1, \boldsymbol{a}_1, \dots, r_T, \boldsymbol{s}_T \}$ following $\pi_{\boldsymbol{\theta}}$ <br>
# $\quad$ <b>FOR</b> $~ t=0,\dots,T-1$ <br>
# $\quad\quad$ $G \leftarrow \sum_{k=t}^{T-1} r_k$ <br>
# $\quad\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha ~ \underbrace{G ~ \nabla_{\boldsymbol{\theta}} \ln \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t)}_{\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})}$ <br>
# <br>
# <b>RETURN</b> $\boldsymbol{\theta}$
#%% md
# ### 3.1 Policy Implementation
# 
# We will implement a stochastic policy to control the agent's actions.
#%% md
# The network takes an input tensor representing the state of the environment and outputs a tensor of action probabilities.
#%%
class PolicyNetwork(torch.nn.Module):
    """
    A neural network used as a policy for the REINFORCE algorithm.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        A fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor) -> torch.Tensor
        Define the forward pass of the PolicyNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize a new instance of PolicyNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        """
        super(PolicyNetwork, self).__init__()

        # TODO...

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of each action for the given state.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor (state).
            The shape of the tensor should be (N, dim),
            where N is the number of states vectors in the batch
            and dim is the dimension of state vectors.

        Returns
        -------
        torch.Tensor
            The output tensor (the probability of each action for the given state).
        """

        # TODO...

        return out

    def get_params(self) -> np.ndarray:
        """
        Get the parameters.

        Returns
        -------
        np.ndarray
            The parameters of the model.
        """
        return self.params.copy()

    def set_params(self, params: np.ndarray) -> None:
        """
        Set the parameters.

        Parameters
        ----------
        params : np.ndarray
            The parameters of the model.
        """
        self.params = params.copy()
#%% md
# `sample_discrete_action` function: This function is used to sample a discrete action based on a given state and a policy network. It first converts the state into a tensor if not already one and passes it through the policy network to get the parameters of the action probability distribution. Then, it creates a categorical distribution from these parameters and samples an action from this distribution. It also calculates the log probability of the sampled action according to the distribution. The function returns the sampled action and its log probability.
#%%
def sample_discrete_action(
        policy_nn: PolicyNetwork, state: np.ndarray
) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : np.ndarray
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """

    # Convert the state into a tensor, specify its data type as float32, and send it to the device (CPU or GPU).
    # The unsqueeze(0) function is used to add an extra dimension to the tensor to match the input shape required by the policy network.
    # state_tensor = TODO...

    # Pass the state tensor through the policy network to get the parameters of the action probability distribution.
    # actions_probability_distribution_params = TODO...

    # Create the categorical distribution used to sample an action from the parameters obtained from the policy network.
    # See https://pytorch.org/docs/stable/distributions.html#categorical
    # actions_probability_distribution = TODO...

    # Sample an action from the categorical distribution.
    # sampled_action_tensor = TODO...

    # Convert the tensor containing the sampled action into a Python integer.
    # sampled_action = TODO...

    # Calculate the log probability of the sampled action according to the categorical distribution.
    # sampled_action_log_probability = TODO...

    # Return the sampled action and its log probability.
    return sampled_action, sampled_action_log_probability
#%% md
# Testing the `sample_discrete_action` function on a random state using an untrained policy network.
#%%
env =

# policy_nn = TODO...

# state = TODO...
# theta = TODO...
# action, action_log_probability = TODO...

print("state:", state)
print("theta:", theta)
print("sampled action:", action)
print("log probability of the sampled action:", action_log_probability)

env.close()
#%% md
# #### sample_one_episode function
#%% md
# This function plays one episode using the given policy $\pi_\theta$ and return its rollouts. The function adheres to a fixed horizon $T$, which represents the maximum number of steps in the episode.
#%%
def sample_one_episode(
        env: gym.Env, policy_nn: PolicyNetwork, max_episode_duration: int
) -> Tuple[List[np.ndarray], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    policy_nn_adversary : PolicyNetwork
        The adversary policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.

    Returns
    -------
    Tuple[List[np.ndarray], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state_t = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state_t)

    for t in range(max_episode_duration):

        # Sample a discrete action and its log probability from the policy network based on the current state
        # action_t, log_prob_action_t = TODO...

        # Execute the sampled action in the environment, which returns the new state, reward, and whether the episode has terminated or been truncated
        # state_t, reward_t, terminated, truncated = TODO...

        # Check if the episode is done, either due to termination (reaching a terminal state) or truncation (reaching a maximum number of steps)
        done = terminated or truncated

        # Append the new state, action, action log probability and reward to their respective lists

        episode_states.append(state_t)
        episode_actions.append(action_t)
        episode_log_prob_actions.append(log_prob_action_t)
        episode_rewards.append(float(reward_t))

        if done:
            break

    return episode_states, episode_actions, episode_rewards, episode_log_prob_actions
#%% md
# Testing this function on the untrained agent.
#%%
NUM_EPISODES = 3

env =

for episode_index in range(NUM_EPISODES):
# policy_nn = TODO...
# policy_nn_adversary = TODO...
# episode_states, episode_actions, episode_rewards, episode_log_prob_actions = TODO...

env.close()
#%%
episode_states
#%%
episode_actions
#%%
episode_rewards
#%% md
# #### Test Function
#%% md
# `avg_return_on_multiple_episodes` function tests the given policy $\pi_\theta$ on `num_episodes` episodes (for fixed horizon $T$) and returns the average reward on the `num_episodes` episodes.
# 
# The function `avg_return_on_multiple_episodes` is designed to play multiple episodes of a given environment using a specified policy neural network and calculate the average return. It takes as input the environment to play in, the policy neural networks to use, the number of episodes to play and the maximum duration of an episode.
# In each episode, it uses the `sample_one_episode` function to play the episode and collect the rewards. The function then returns the average of these cumulated rewards.
# 
# `avg_return_on_multiple_episodes` will be used for evaluating the performance of a policy over multiple episodes.
#%%
def avg_return_on_multiple_episodes(
        env: gym.Env,
        policy_nn: PolicyNetwork,
        policy_nn_adversary: PolicyNetwork,
        num_test_episode: int,
        max_episode_duration: int,
) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    plocy_nn_adversary : PolicyNetwork
        The adversary policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.

    Returns
    -------
    float
        The average return.
    """

    # TODO...

    return average_return
#%% md
# Testing this function on the untrained agent.
#%%
env =

# TODO...

print(average_return)

env.close()
#%% md
# ### 3.2 Train Function
#%% md
# `train_reinforce_discrete` trains a policy network using the REINFORCE algorithm in the given environment. This function takes as input the environment, the number of training episodes, the number of tests to perform per episode, the maximum duration of an episode, and the learning rate for the optimizer.
# 
# The function first initializes a policy network and an AdamW optimizer. Then, for each training episode, it generates an episode using the current policies (current player and adversary) and calculates the return at each time step. It uses this return and the log probability of the action taken at that time step to compute the loss, which is the negative of the product of the return and the log probability. This loss is used to update the policy network parameters using gradient ascent.
# 
# After each training episode, the function tests the current policy by playing a number of test episodes and calculating the average return. This average return is added to a list for monitoring purposes.
# 
# The function returns the trained policy network and the list of average returns for each episode. This function encapsulates the main loop of the REINFORCE algorithm, including the policy update step.
#%%
def train_reinforce_discrete(
        env: gym.Env,
        num_train_episodes: int,
        num_test_per_episode: int,
        max_episode_duration: int,
        learning_rate: float,
) -> Tuple[PolicyNetwork, List[float]]:
    """
    Train a policy using the REINFORCE algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    learning_rate : float
        The initial step size.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    policy_nn = PolicyNetwork(observation_dim, num_actions).to(device)
    policy_nn_adversary = PolicyNetwork(observation_dim, num_actions).to(device
    optimizer = torch.optim.AdamW(policy_nn.parameters(), lr=learning_rate)

    for episode_index in tqdm(range(num_train_episodes)):
        # TODO...

        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(
            env=env,
            policy_nn=policy_nn,
            policy_nn_adversary=policy_nn_adversary,
            num_test_episode=num_test_per_episode,
            max_episode_duration=max_episode_duration,
        )

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return policy_nn, episode_avg_return_list
#%% md
# #### Training the agent
#%%
env =

NUMBER_OF_TRAININGS = DEFAULT_NUMBER_OF_TRAININGS
reinforce_trains_result_list: List[List[Union[int, float]]] = [[], [], []]

for train_index in range(NUMBER_OF_TRAININGS):
    reinforce_policy_nn, episode_reward_list = train_reinforce_discrete(
        env=env,
        num_train_episodes=150,
        num_test_per_episode=5,
        max_episode_duration=500,
        learning_rate=0.01,
    )

    reinforce_trains_result_list[0].extend(range(len(episode_reward_list)))
    reinforce_trains_result_list[1].extend(episode_reward_list)
    reinforce_trains_result_list[2].extend([train_index for _ in episode_reward_list])

reinforce_trains_result_df = pd.DataFrame(
    np.array(reinforce_trains_result_list).T,
    columns=["num_episodes", "mean_final_episode_reward", "training_index"],
)
reinforce_trains_result_df["agent"] = "REINFORCE"

torch.save(reinforce_policy_nn, MODELS_DIR / "reinforce_policy_network.pth")

env.close()
#%% md
# #### Plots
#%%
g = sns.relplot(
    x="num_episodes",
    y="mean_final_episode_reward",
    kind="line",
    hue="agent",
    estimator=None,
    units="training_index",
    data=reinforce_trains_result_df,
    height=7,
    aspect=2,
    alpha=0.5,
)
plt.savefig(PLOTS_DIR / "reinforce_trains_results.png")
#%%
all_trains_result_df = pd.concat(
    [
        dqn2_trains_result_df,
        reinforce_trains_result_df,
    ]
)
g = sns.relplot(
    x="num_episodes",
    y="mean_final_episode_reward",
    kind="line",
    hue="agent",
    data=all_trains_result_df,
    height=7,
    aspect=2,
)
plt.savefig(PLOTS_DIR / "trains_results_agg.png")
#%% md
# ### Testing the policy against a random policy
#%%
NUM_EPISODES = 3
)

env =
# TODO...

env.close()
#%% md
# ### Testing the policy against our Java ID Alpha Beta agent
#%%
NUM_EPISODES = 3
)

env =
# TODO...

env.close()
#%%
reinforce_trains_result_df
#%% md
# #### Score
#%%
train_score_reinforce = reinforce_trains_result_df[["num_episodes", "mean_final_episode_reward"]].groupby(
    "num_episodes").mean().max()
train_score_reinforce
#%% md
# ## Hyperparameters optimization with Optuna
# 
# Optuna is an open-source hyperparameter optimization framework designed to automate the process of searching for the best hyperparameters in machine learning models. It is highly efficient and flexible, supporting various optimization algorithms. Optuna works with Python-based machine learning libraries like PyTorch, TensorFlow, and Scikit-learn. Optuna’s core feature is its ability to perform dynamic search spaces and pruning, allowing faster convergence by terminating poorly performing trials early.
# Optuna supports distributed optimization for large-scale tuning.
# 
# ### Installation
# 
# ```
# pip install optuna
# ```
# 
# ### Official documentation
# 
# - Optuna GitHub: [https://github.com/optuna/optuna](https://github.com/optuna/optuna)
# - Optuna Documentation: [https://optuna.org](https://optuna.org)
# 
# ### Example of usage with PyTorch
# 
# Here's an example of how to use Optuna to optimize the hyperparameters of a simple neural network with PyTorch.
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset


# Define the PyTorch model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Objective function for Optuna
def objective(trial):
    # Hyperparameters to be tuned
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = Net(input_size=28 * 28, hidden_size=hidden_size, output_size=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy dataset
    X = torch.randn(100, 28 * 28)
    y = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # Training loop
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    return loss.item()


# Optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Show best hyperparameters
print(study.best_trial)
#%% md
# This example creates a basic neural network and tunes the `hidden_size` and learning rate (`lr`) using Optuna.
#%% md
# ## 4. Gradient-Free Optimization - CEM / ES
#%% md
# ### 4.1 Objective Function
#%% md
# In Reinforcement Learning, by convention the score is a reward to maximize whereas in mathematical optimization the score is a cost to minimize; the objective function will therefore return the opposite of the reward as the score of evaluated policies.
#%%
class ObjectiveFunction:
    """
    Objective function for evaluating a policy in a given environment.

    Parameters
    ----------
    env : gym.Env
        The environment in which to evaluate the policy.
    policy : torch.nn.Module
        The policy to evaluate.
    adversary_policy : torch.nn.Module
        The adversary policy.
    num_episodes : int, optional
        The number of episodes to run for each evaluation, by default 1.
    max_time_steps : float, optional
        The maximum number of time steps per episode, by default float("inf").
    minimization_solver : bool, optional
        Whether the solver is a minimization solver, by default True.

    Attributes
    ----------
    env : gym.Env
        The environment in which to evaluate the policy.
    policy : torch.nn.Module
        The policy to evaluate.
    num_episodes : int
        The number of episodes to run for each evaluation.
    max_time_steps : float
        The maximum number of time steps per episode.
    minimization_solver : bool
        Whether the solver is a minimization solver.
    num_evals : int
        The number of evaluations performed.
    """

    def __init__(
            self,
            env: gym.Env,
            policy: torch.nn.Module,
            adversary_policy: torch.nn.Module,
            num_episodes: int = 1,
            max_time_steps: float = float("inf"),
            minimization_solver: bool = True,
    ):
        self.env = env
        self.policy = policy
        self.adversary_policy = adversary_policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver

        self.num_evals = 0

    def eval(self, policy_params: np.ndarray, num_episodes: Optional[int] = None,
             max_time_steps: Optional[float] = None) -> float:
        """
        Evaluate a policy.

        Parameters
        ----------
        policy_params : np.ndarray
            The parameters of the policy to evaluate.
        num_episodes : int, optional
            The number of episodes to run for each evaluation, by default None.
        max_time_steps : float, optional
            The maximum number of time steps per episode, by default None.

        Returns
        -------
        float
            The average total rewards over the evaluation episodes.
        """
        self.policy.set_params(policy_params)

        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes

        if max_time_steps is None:
            max_time_steps = self.max_time_steps

        average_total_rewards = 0

        for i_episode in range(num_episodes):
            total_rewards = 0.0
            observation, info = self.env.reset()

            for t in range(max_time_steps):
                action = self.policy(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_rewards += reward

                done = terminated or truncated

                if done:
                    break

            #print(f"Episode {i_episode} total rewards: {total_rewards}")

            average_total_rewards += float(total_rewards) / num_episodes

        if self.minimization_solver:
            average_total_rewards *= -1.0

        return average_total_rewards  # Optimizers do minimization by default...

    def __call__(self, policy_params: np.ndarray, num_episodes: Optional[int] = None,
                 max_time_steps: Optional[float] = None) -> float:
        """
        Evaluate a policy.

        Parameters
        ----------
        policy_params : np.ndarray
            The parameters of the policy to evaluate.
        num_episodes : int, optional
            The number of episodes to run for each evaluation, by default None.
        max_time_steps : float, optional
            The maximum number of time steps per episode, by default None.

        Returns
        -------
        float
            The average total rewards over the evaluation episodes.
        """
        return self.eval(policy_params, num_episodes, max_time_steps)
#%% md
# ## 4.2 CEM optimization algorithm
#%% md
# `cem_uncorrelated` function searches the best $\theta$ parameters with a Cross Entropy Method, using the objective function defined above.
# $\mathbb{P}$ can be defined as an multivariate normal distribution $\mathcal{N}\left( \boldsymbol{\mu}, \boldsymbol{\sigma^2} \boldsymbol{\Sigma} \right)$ where $\boldsymbol{\mu}$ and $\boldsymbol{\sigma^2} \boldsymbol{\Sigma}$ are vectors i.e. we use one mean and one variance parameters per dimension of $\boldsymbol{\theta}$.
#%% md
# **Cross Entropy**
# 
# **Input**:<br>
# $\quad\quad$ $f$: the objective function<br>
# $\quad\quad$ $\mathbb{P}$: family of distribution<br>
# $\quad\quad$ $\boldsymbol{\theta}$: initial parameters for the proposal distribution $\mathbb{P}$<br>
# 
# **Algorithm parameter**:<br>
# $\quad\quad$ $m$: sample size<br>
# $\quad\quad$ $m_{\text{elite}}$: number of samples to use to fit $\boldsymbol{\theta}$<br>
# 
# **FOR EACH** iteration<br>
# $\quad\quad$ samples $\leftarrow \{ \boldsymbol{x}_1, \dots, \boldsymbol{x}_m \}$ with $\boldsymbol{x}_i \sim \mathbb{P}(\boldsymbol{\theta}) ~~ \forall i \in 1\dots m$<br>
# $\quad\quad$ elite $\leftarrow $ { $m_{\text{elite}}$ best samples } $\quad$ (i.e. select best samples according to $f$)<br>
# $\quad\quad$ $\boldsymbol{\theta} \leftarrow $ fit $\mathbb{P}(\boldsymbol{\theta})$ to the elite samples<br>
# 
# **RETURN** $\boldsymbol{\theta}$
#%%
def cem_uncorrelated(
        objective_function: Callable[[np.ndarray], float],
        mean_array: np.ndarray,
        var_array: np.ndarray,
        max_iterations: int = 500,
        sample_size: int = 50,
        elite_frac: float = 0.2,
        print_every: int = 10,
        success_score: float = float("inf"),
        num_evals_for_stop: Optional[int] = None,
        hist_dict: Optional[dict] = None,
) -> np.ndarray:
    """
    Cross-entropy method.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The function to maximize.
    mean_array : np.ndarray
        The initial proposal distribution (mean vector).
    var_array : np.ndarray
        The initial proposal distribution (variance vector).
    max_iterations : int, optional
        Number of training iterations, by default 500.
    sample_size : int, optional
        Size of population at each iteration, by default 50.
    elite_frac : float, optional
        Rate of top performers to use in update with elite_frac ∈ ]0;1], by default 0.2.
    print_every : int, optional
        How often to print average score, by default 10.
    success_score : float, optional
        The score at which to stop the optimization, by default float("inf").
    num_evals_for_stop : Optional[int], optional
        Number of evaluations for stopping criteria, by default None.
    hist_dict : Optional[dict], optional
        Dictionary to log the history, by default None.

    Returns
    -------
    np.ndarray
        The optimized mean vector.
    """
    assert 0.0 < elite_frac <= 1.0

    n_elite = math.ceil(sample_size * elite_frac)

    for iteration_index in range(0, max_iterations):

        # SAMPLE A NEW POPULATION OF SOLUTIONS (X VECTORS) ####################

        x_array = np.random.normal(mean_array, np.sqrt(var_array), size=(sample_size, len(mean_array)))

        # EVALUATE SAMPLES AND EXTRACT THE BEST ONES ("ELITE") ################

        score_array = np.array([objective_function(x) for x in x_array])

        sorted_indices_array = np.argsort(
            score_array)
        elite_indices_array = sorted_indices_array[
                              :n_elite]

        elite_x_array = x_array[elite_indices_array]

        # FIT THE NORMAL DISTRIBUTION ON THE ELITE POPULATION #################

        mean_array = np.mean(elite_x_array, axis=0)
        var_array = np.var(elite_x_array, axis=0)
        score = np.min(score_array)

        # PRINT STATUS ########################################################

        if iteration_index % print_every == 0:
            print("Iteration {}\tScore {}".format(iteration_index, score))

        if hist_dict is not None:
            hist_dict[iteration_index] = [score] + mean_array.tolist() + var_array.tolist()

        # STOPPING CRITERIA ####################################################

        if num_evals_for_stop is not None:
            score = objective_function(mean_array)

        # `num_evals_for_stop = None` may be used to fasten computations but it introduces bias...
        if score <= success_score:
            break

    return mean_array
#%% md
# ### Training the agents (DQN v2 and REINFORCE) with CEM
#%% md
# ### DQN v2
#%%
env =

dqnv2_cem = QNetwork(env.observation_space.shape[0])
dqnv2_cem_adversary = QNetwork(env.observation_space.shape[0])

objective_function = ObjectiveFunction(
    env=env, policy=dqnv2_cem, adversary_policy=dqnv2_cem_adversary, num_episodes=10, max_time_steps=1000
)
#%%
%%time

hist_dict = {}

num_params = len(dqnv2_cem.get_params())

init_mean_array = np.random.random(num_params)
init_var_array = np.ones(num_params) * 100.0

optimized_policy_params_dqnv2_cem = cem_uncorrelated(
    objective_function=objective_function,
    mean_array=init_mean_array,
    var_array=init_var_array,
    max_iterations=30,
    sample_size=50,
    elite_frac=0.1,
    print_every=1,
    success_score=-500,
    num_evals_for_stop=None,
    hist_dict=hist_dict,
)

env.close()
#%%
df = pd.DataFrame.from_dict(
    hist_dict,
    orient="index",
    columns=["score", "mu1", "mu2", "mu3", "mu4", "var1", "var2", "var3", "var4"],
)
ax = df.score.plot(title="Average reward", figsize=(20, 5))
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.savefig(PLOTS_DIR / "dqnv2_cem_avg_reward_wrt_iterations.png")
plt.show()
#%%
ax = df[["mu1", "mu2", "mu3", "mu4"]].plot(
    title="Theta w.r.t training steps", figsize=(20, 5)
);
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "dqnv2_cem_params_wrt_iterations.png")
plt.show()
#%%
ax = df[["var1", "var2", "var3", "var4"]].plot(
    logy=True, title="Variance w.r.t training steps", figsize=(20, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "dqnv2_cem_var_wrt_iterations.png")
plt.show()
#%%
print("Optimized weights: ", optimized_policy_params_dqnv2_cem)
#%% md
# ### REINFORCE
#%%
reinforce_policy_nn_cem = PolicyNetwork(env.observation_space.shape[0])
reinforce_policy_nn_cem_adversary = PolicyNetwork(env.observation_space.shape[0])
objective_function = ObjectiveFunction(
    env=env, policy=dqnv2_cem, adversary_policy=reinforce_policy_nn_cem_adversary, num_episodes=10, max_time_steps=1000
)
#%%
%%time

hist_dict = {}

num_params = len(reinforce_policy_nn_cem.get_params())

optimized_policy_params_reinforce_policy_nn_cem = cem_uncorrelated(
    objective_function=objective_function,
    mean_array=init_mean_array,
    var_array=init_var_array,
    max_iterations=30,
    sample_size=50,
    elite_frac=0.1,
    print_every=1,
    success_score=-500,
    num_evals_for_stop=None,
    hist_dict=hist_dict,
)

env.close()
#%%
df = pd.DataFrame.from_dict(
    hist_dict,
    orient="index",
    columns=["score", "mu1", "mu2", "mu3", "mu4", "var1", "var2", "var3", "var4"],
)
ax = df.score.plot(title="Average reward", figsize=(20, 5))
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_cem_avg_reward_wrt_iterations.png")
plt.show()
#%%
ax = df[["mu1", "mu2", "mu3", "mu4"]].plot(
    title="Theta w.r.t training steps", figsize=(20, 5)
);
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_cem_params_wrt_iterations.png")
plt.show()
#%%
ax = df[["var1", "var2", "var3", "var4"]].plot(
    logy=True, title="Variance w.r.t training steps", figsize=(20, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_cem_var_wrt_iterations.png")
plt.show()
#%%
print("Optimized weights: ", optimized_policy_params_reinforce_policy_nn_cem)
#%% md
# ### Testing the trained agents againt a random policy
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=dqnv2_cem, adversary_policy=)

print("Average Reward (DQN v2 CEM): ",
      -objective_function.eval(optimized_policy_params_dqnv2_cem, num_episodes=NUM_EPISODES, max_time_steps=200))
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=reinforce_policy_nn_cem, adversary_policy=)

print("Average Reward (REINFORCE CEM): ",
      -objective_function.eval(optimized_policy_params_reinforce_policy_nn_cem, num_episodes=NUM_EPISODES, max_time_steps=200))
#%% md
# ### Testing the trained agents againt our Java ID Alpha Beta agent
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=dqnv2_cem, adversary_policy=)

print("Average Reward (DQN v2 CEM): ",
      -objective_function.eval(optimized_policy_params_dqnv2_cem, num_episodes=NUM_EPISODES, max_time_steps=200))
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=reinforce_policy_nn_cem, adversary_policy=)

print("Average Reward (REINFORCE CEM): ",
      -objective_function.eval(optimized_policy_params_reinforce_policy_nn_cem, num_episodes=NUM_EPISODES, max_time_steps=200))
#%% md
# ## 4.3 (1+1)-SA-ES optimization algorithm
#%% md
# `saes_1_1` function searchs the best $\theta$ parameters with a (1+1)-SA-ES algorithm, using the objective function defined above.
#%% md
# **(1+1)-SA-ES**
# 
# **Input**:<br>
# $\quad\quad$ $f$: the objective function<br>
# $\quad\quad$ $\boldsymbol{x}$: initial solution<br>
# 
# **Algorithm parameter**:<br>
# $\quad\quad$ $\tau$: self-adaptation learning rate<br>
# 
# **FOR EACH** generation<br>
# $\quad\quad$ 1. mutation of $\sigma$ (current individual strategy) : $\sigma' \leftarrow \sigma ~ e^{\tau \mathcal{N}(0,1)}$<br>
# $\quad\quad$ 2. mutation of $\boldsymbol{x}$ (current solution) : $\boldsymbol{x}' \leftarrow \boldsymbol{x} + \sigma' ~ \mathcal{N}(0,1)$<br>
# $\quad\quad$ 3. eval $f(\boldsymbol{x}')$<br>
# $\quad\quad$ 4. survivor selection $\boldsymbol{x} \leftarrow \boldsymbol{x}'$ and $\sigma \leftarrow \sigma'$ if $f(\boldsymbol{x}') \leq f(\boldsymbol{x})$<br>
# 
# **RETURN** $\boldsymbol{x}$
#%%
def saes_1_1(
        objective_function: Callable[[np.ndarray], float],
        x_array: np.ndarray,
        sigma_array: np.ndarray,
        max_iterations: int = 500,
        tau: Optional[float] = None,
        print_every: int = 10,
        success_score: float = float("inf"),
        num_evals_for_stop: Optional[int] = None,
        hist_dict: Optional[dict] = None,
) -> np.ndarray:
    """
    (1+1)-Self-Adaptive Evolution Strategy (SA-ES) optimization algorithm.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The function to minimize.
    x_array : np.ndarray
        The initial solution vector.
    sigma_array : np.ndarray
        The initial strategy parameter vector (step sizes).
    max_iterations : int, optional
        The maximum number of iterations, by default 500.
    tau : Optional[float], optional
        The self-adaptation learning rate, by default None.
    print_every : int, optional
        How often to print the current score, by default 10.
    success_score : float, optional
        The score at which to stop the optimization, by default float("inf").
    num_evals_for_stop : Optional[int], optional
        Number of evaluations for stopping criteria, by default None.
    hist_dict : Optional[dict], optional
        Dictionary to log the history, by default None.

    Returns
    -------
    np.ndarray
        The optimized solution vector.
    """
    # Number of dimension of the solution space
    d = x_array.shape[0]

    if tau is None:
        # Self-adaptation learning rate
        tau = 1.0 / (2.0 * d)

    score = objective_function(x_array)

    for iteration_index in range(0, max_iterations):
        # 1. Mutation of sigma (current "individual strategy")
        new_sigma_array = sigma_array * np.exp(tau * np.random.normal(0, 1, size=d))

        # 2. Mutation of x (current solution)
        new_x_array = x_array + new_sigma_array * np.random.normal(0, 1, size=d)

        # 3. Eval f(x')
        new_score = objective_function(new_x_array)

        # 4. survivor selection (we follow the ES convention and do minimization)
        if new_score <= score:  # You may try `new_score < score` for less exploration
            score = new_score
            x_array = new_x_array.copy()
            sigma_array = new_sigma_array.copy()

        # PRINT STATUS ########################################################

        if iteration_index % print_every == 0:
            print("Iteration {}\tScore {}".format(iteration_index, score))

        if hist_dict is not None:
            hist_dict[iteration_index] = [score] + x_array.tolist() + sigma_array.tolist()

        # STOPPING CRITERIA ####################################################

        if num_evals_for_stop is not None:
            score = objective_function(x_array)

        # `num_evals_for_stop = None` may be used to fasten computations but it introduces bias...
        if score <= success_score:
            break

    return x_array
#%% md
# ### Training the agents (DQN v2 and REINFORCE) with (1+1)-SA-ES
#%% md
# ### DQN v2
#%%
env =

dqnv2_saes_1_1 = QNetwork(env.observation_space.shape[0])
dqnv2_saes_1_1_adversary = QNetwork(env.observation_space.shape[0])

objective_function = ObjectiveFunction(
    env=env, policy=dqnv2_saes_1_1, adversary_policy=dqnv2_saes_1_1_adversary, num_episodes=10, max_time_steps=1000
)
#%%
%%time

hist_dict = {}

num_params = len(dqnv2_saes_1_1.get_params())

initial_solution_array = np.random.random(num_params)
initial_sigma_array = np.ones(num_params) * 1.0

optimized_policy_params_dqnv2_saes_1_1 = saes_1_1(
    objective_function=objective_function,
    x_array=initial_solution_array,
    sigma_array=initial_sigma_array,
    tau=0.001,
    max_iterations=1000,
    print_every=100,
    success_score=-500,
    num_evals_for_stop=None,
    hist_dict=hist_dict,
)

env.close()
#%%
df = pd.DataFrame.from_dict(
    hist_dict,
    orient="index",
    columns=[
        "score",
        "mu1",
        "mu2",
        "mu3",
        "mu4",
        "sigma1",
        "sigma2",
        "sigma3",
        "sigma4",
    ],
)
ax = df.score.plot(title="Average reward", figsize=(30, 5))
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.savefig(PLOTS_DIR / "dqnv2_saes_1_1_avg_reward_wrt_iterations.png")
plt.show()
#%%
ax = df[["mu1", "mu2", "mu3", "mu4"]].plot(
    title="Theta w.r.t training steps", figsize=(30, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "dqnv2_saes_1_1_params_wrt_iterations.png")
plt.show()
#%%
ax = df[["sigma1", "sigma2", "sigma3", "sigma4"]].plot(
    logy=True, title="Sigma w.r.t training steps", figsize=(30, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "dqnv2_saes_1_1_var_wrt_iterations.png")
plt.show()
#%%
print("Optimized weights: ", optimized_policy_params_dqnv2_saes_1_1)
#%% md
# ### REINFORCE
#%%
env =

reinforce_policy_nn_saes_1_1 = PolicyNetwork(env.observation_space.shape[0])
reinforce_policy_nn_saes_1_1_adversary = PolicyNetwork(env.observation_space.shape[0])
objective_function = ObjectiveFunction(
    env=env, policy=reinforce_policy_nn_saes_1_1, adversary_policy=reinforce_policy_nn_saes_1_1_adversary, num_episodes=10, max_time_steps=1000
)
#%%
%%time

hist_dict = {}

num_params = len(reinforce_policy_nn_saes_1_1.get_params())

initial_solution_array = np.random.random(num_params)
initial_sigma_array = np.ones(num_params) * 1.0

optimized_policy_params_reinforce_policy_nn_saes_1_1 = saes_1_1(
    objective_function=objective_function,
    x_array=initial_solution_array,
    sigma_array=initial_sigma_array,
    tau=0.001,
    max_iterations=1000,
    print_every=100,
    success_score=-500,
    num_evals_for_stop=None,
    hist_dict=hist_dict,
)

env.close()
#%%
df = pd.DataFrame.from_dict(
    hist_dict,
    orient="index",
    columns=[
        "score",
        "mu1",
        "mu2",
        "mu3",
        "mu4",
        "sigma1",
        "sigma2",
        "sigma3",
        "sigma4",
    ],
)
ax = df.score.plot(title="Average reward", figsize=(30, 5))
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_saes_1_1_avg_reward_wrt_iterations.png")
plt.show()
#%%
ax = df[["mu1", "mu2", "mu3", "mu4"]].plot(
    title="Theta w.r.t training steps", figsize=(30, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_saes_1_1_params_wrt_iterations.png")
plt.show()
#%%
ax = df[["sigma1", "sigma2", "sigma3", "sigma4"]].plot(
    logy=True, title="Sigma w.r.t training steps", figsize=(30, 5)
)
plt.xlabel("Training Steps")
plt.savefig(PLOTS_DIR / "reinforce_policy_nn_saes_1_1_var_wrt_iterations.png")
plt.show()
#%%
print("Optimized weights: ", optimized_policy_params_reinforce_policy_nn_saes_1_1)
#%% md
# ### Testing the trained agents against a random policy
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=dqnv2_saes_1_1, adversary_policy=)

print("Average Reward (DQN v2 (1+1)-SA-ES): ",
      -objective_function.eval(optimized_policy_params_dqnv2_saes_1_1, num_episodes=NUM_EPISODES, max_time_steps=200))

env.close()
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=reinforce_policy_nn_saes_1_1, adversary_policy=)

print("Average Reward (REINFORCE (1+1)-SA-ES): ",
      -objective_function.eval(optimized_policy_params_reinforce_policy_nn_saes_1_1, num_episodes=NUM_EPISODES, max_time_steps=200))

env.close()
#%% md
# ### Testing the trained agents against our Java ID Alpha Beta agent
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=dqnv2_saes_1_1, adversary_policy=)

print("Average Reward (DQN v2 (1+1)-SA-ES): ",
      -objective_function.eval(optimized_policy_params_dqnv2_saes_1_1, num_episodes=NUM_EPISODES, max_time_steps=200))

env.close()
#%%
NUM_EPISODES = 3

env =

objective_function = ObjectiveFunction(env=env, policy=reinforce_policy_nn_saes_1_1, adversary_policy=)

print("Average Reward (REINFORCE (1+1)-SA-ES): ",
      -objective_function.eval(optimized_policy_params_reinforce_policy_nn_saes_1_1, num_episodes=NUM_EPISODES, max_time_steps=200))

env.close()
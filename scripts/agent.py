import torch 
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer, PrioritizedReplay
import numpy as np
import random
import copy

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size,
                      action_size,
                      n_step,
                      per, 
                      munchausen,
                      D2RL,
                      random_seed,
                      hidden_size,
                      BUFFER_SIZE = int(1e6),  # replay buffer size
                      BATCH_SIZE = 128,        # minibatch size
                      GAMMA = 0.99,            # discount factor
                      TAU = 1e-3,              # for soft update of target parameters
                      LR_ACTOR = 1e-4,         # learning rate of the actor 
                      LR_CRITIC = 1e-4,        # learning rate of the critic
                      WEIGHT_DECAY = 0,#1e-2        # L2 weight decay
                      LEARN_EVERY = 1,
                      LEARN_NUMBER = 1,
                      EPSILON = 1.0,
                      EPSILON_DECAY = 1,
                      device = "cuda"
                      ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step =n_step
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        self.EPSILON_DECAY = EPSILON_DECAY
        self.device = device
        self.seed = random.seed(random_seed)
        
        print("Using: ", device)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        print("Actor: \n", self.actor_local)
        print("\nCritic: \n", self.critic_local)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = EPSILON
        # Replay memory
        self.memory = ReplayBuffer( BUFFER_SIZE, BATCH_SIZE, n_step=n_step, device=device, seed=random_seed, gamma=GAMMA)
        

    def step(self, state, action, reward, next_state, done, timestamp):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE and timestamp % self.LEARN_EVERY == 0:
            for _ in range(self.LEARN_NUMBER):
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        assert state.shape == (1,3), "shape: {}".format(state.shape)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return action #np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states.to(self.device))
        Q_targets_next = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     
        
        # ----------------------- update epsilon and noise ----------------------- #
        self.epsilon *= self.EPSILON_DECAY
        self.noise.reset()
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
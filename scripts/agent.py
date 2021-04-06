import torch 
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .networks import Actor, Critic, DeepActor, DeepCritic, IQN, DeepIQN
from .replay_buffer import ReplayBuffer, PrioritizedReplay
import numpy as np
import random
import copy
from .ICM import ICM, Inverse, Forward

# TODO: Check for batch norm comparison! batch norm seems to have a big impact on final performance
#       Also check if normal gaussian noise is enough. -> D4PG paper says there is no difference maybe chooseable parameter for the implementation

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size,
                      action_size,
                      n_step,
                      per, 
                      munchausen,
                      distributional,
                      D2RL,
                      noise_type,
                      curiosity,
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
                      device = "cuda",
                      frames = 100000,
                      worker=1
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
        self.per = per
        self.munchausen = munchausen
        self.n_step = n_step
        self.distributional = distributional
        self.D2RL = D2RL
        self.curiosity = curiosity[0]
        self.reward_addon = curiosity[1]
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_NUMBER = LEARN_NUMBER
        self.EPSILON_DECAY = EPSILON_DECAY
        self.device = device
        self.seed = random.seed(random_seed)
        # distributional Values
        self.N = 32
        self.entropy_coeff = 0.001
        # munchausen values
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        
        self.eta = torch.FloatTensor([.1]).to(device)
        
        print("Using: ", device)
        
        # Actor Network (w/ Target Network)
        if not self.D2RL:
            self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        else:
            self.actor_local = DeepActor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
            self.actor_target = DeepActor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if self.distributional:
            if not self.D2RL:
                self.critic_local = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
                self.critic_target = IQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
            else:
                self.critic_local = DeepIQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
                self.critic_target = DeepIQN(state_size, action_size, layer_size=hidden_size, device=device, seed=random_seed, dueling=None, N=self.N).to(device)
        else:
            if not self.D2RL:
                self.critic_local = Critic(state_size, action_size, random_seed).to(device)
                self.critic_target = Critic(state_size, action_size, random_seed).to(device)
            else:
                self.critic_local = DeepCritic(state_size, action_size, random_seed).to(device)
                self.critic_target = DeepCritic(state_size, action_size, random_seed).to(device)

        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        print("Actor: \n", self.actor_local)
        print("\nCritic: \n", self.critic_local)

        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.icm = ICM(inverse_m, forward_m, device=device)#.to(device)
            print(inverse_m, forward_m)
            
        # Noise process
        self.noise_type = noise_type
        if noise_type == "ou":
            self.noise = OUNoise(action_size, random_seed)
            self.epsilon = EPSILON
        else:
            self.epsilon = 0.3
        print("Use Noise: ", noise_type)
        # Replay memory
        if per:
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, device=device, seed=random_seed, gamma=GAMMA, n_step=n_step, parallel_env=worker, beta_frames=frames)

        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_step=n_step, parallel_env=worker, device=device, seed=random_seed, gamma=GAMMA)
        
        if distributional:
            self.learn = self.learn_distribution
        else:
            self.learn = self.learn_

        print("Using PER: ", per)    
        print("Using Munchausen RL: ", munchausen)
        
    def step(self, state, action, reward, next_state, done, timestamp, writer):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE and timestamp % self.LEARN_EVERY == 0:
            for _ in range(self.LEARN_NUMBER):
                experiences = self.memory.sample()
                
                losses = self.learn(experiences, self.GAMMA)
            writer.add_scalar("Critic_loss", losses[0], timestamp)
            writer.add_scalar("Actor_loss", losses[1], timestamp)
            if self.curiosity:
                writer.add_scalar("ICM_loss", losses[2], timestamp)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        assert state.shape == (state.shape[0],self.state_size), "shape: {}".format(state.shape)
        self.actor_local.eval()
        with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            if self.noise_type == "ou":
                action += self.noise.sample() * self.epsilon
            else:
                action += self.epsilon * np.random.normal(0, scale=1)
        return action #np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn_(self, experiences, gamma):
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
        states, actions, rewards, next_states, dones, idx, weights = experiences
        icm_loss = 0
        # calculate curiosity
        if self.curiosity:
            icm_loss, forward_pred_err = self.icm.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"
            
            if self.reward_addon == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()

        # ---------------------------- update critic ---------------------------- #
        if not self.munchausen:
            # Get predicted next-state actions and Q values from target models
            with torch.no_grad():
                actions_next = self.actor_target(next_states.to(self.device))
                Q_targets_next = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (gamma**self.n_step * Q_targets_next * (1 - dones))
        else:
            with torch.no_grad():
                actions_next = self.actor_target(next_states.to(self.device))
                q_t_n = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                # this might cause some instability (?) needs to be tested
                logsum = torch.logsumexp(\
                    q_t_n /self.entropy_tau, 1).unsqueeze(-1) #logsum trick
                assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                tau_log_pi_next = (q_t_n  - self.entropy_tau*logsum)
                
                pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1)
                # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                Q_target = (self.GAMMA**self.n_step * (pi_target * (q_t_n-tau_log_pi_next)*(1 - dones)))
                assert Q_target.shape == (self.BATCH_SIZE, 1), "has shape: {}".format(Q_target.shape)

                q_k_target = self.critic_target(states, actions)
                tau_log_pik = q_k_target - self.entropy_tau*torch.logsumexp(\
                                                                        q_k_target/self.entropy_tau, 1).unsqueeze(-1)
                assert tau_log_pik.shape == (self.BATCH_SIZE, 1), "shape instead is {}".format(tau_log_pik.shape)
                # calc munchausen reward:
                munchausen_reward = (rewards + self.alpha*torch.clamp(tau_log_pik, min=self.lo, max=0))
                assert munchausen_reward.shape == (self.BATCH_SIZE, 1)
                # Compute Q targets for current states 
                Q_targets = munchausen_reward + Q_target
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        if self.per:
            td_error =  Q_targets - Q_expected
            critic_loss = (td_error.pow(2)*weights.to(self.device)).mean().to(self.device)
        else:
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
        if self.per:
            self.memory.update_priorities(idx, np.clip(abs(td_error.data.cpu().numpy()),-1,1))
        # ----------------------- update epsilon and noise ----------------------- #
        
        self.epsilon *= self.EPSILON_DECAY
        
        if self.noise_type == "ou": self.noise.reset()
        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy(), icm_loss

    
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


    def learn_distribution(self, experiences, gamma):
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
            states, actions, rewards, next_states, dones, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            # Get max predicted Q values (for next states) from target model
            if not self.munchausen:
                with torch.no_grad():
                    next_actions = self.actor_local(next_states)
                    Q_targets_next, _ = self.critic_target(next_states, next_actions, self.N)
                    Q_targets_next = Q_targets_next.transpose(1,2)
                # Compute Q targets for current states 
                Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
            else:
                with torch.no_grad():
                    #### CHECK FOR THE SHAPES!!
                    actions_next = self.actor_target(next_states.to(self.device))
                    Q_targets_next, _ = self.critic_target(next_states.to(self.device), actions_next.to(self.device), self.N)

                    q_t_n = Q_targets_next.mean(1)
                    # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                    # this might cause some instability (?) needs to be tested
                    logsum = torch.logsumexp(\
                        q_t_n /self.entropy_tau, 1).unsqueeze(-1) #logsum trick
                    assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                    tau_log_pi_next = (q_t_n  - self.entropy_tau*logsum).unsqueeze(1)
                    
                    pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)
                    # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                    Q_target = (self.GAMMA**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1)))).transpose(1,2)
                    assert Q_target.shape == (self.BATCH_SIZE, self.action_size, self.N), "has shape: {}".format(Q_target.shape)

                    q_k_target = self.critic_target.get_qvalues(states, actions)
                    tau_log_pik = q_k_target - self.entropy_tau*torch.logsumexp(\
                                                                            q_k_target/self.entropy_tau, 1).unsqueeze(-1)
                    assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
                    # calc munchausen reward:
                    munchausen_reward = (rewards + self.alpha*torch.clamp(tau_log_pik, min=self.lo, max=0)).unsqueeze(-1)
                    assert munchausen_reward.shape == (self.BATCH_SIZE, self.action_size, 1)
                    # Compute Q targets for current states 
                    Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            Q_expected, taus = self.critic_local(states, actions, self.N)
            assert Q_targets.shape == (self.BATCH_SIZE, 1, self.N)
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
    
            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
            
            if self.per:
                critic_loss = (quantil_l.sum(dim=1).mean(dim=1, keepdim=True)*weights.to(self.device)).mean()
            else:
                critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local.get_qvalues(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)                     
            if self.per:
                self.memory.update_priorities(idx, np.clip(abs(td_error.sum(dim=1).mean(dim=1,keepdim=True).data.cpu().numpy()),-1,1))
            # ----------------------- update epsilon and noise ----------------------- #
            
            self.epsilon *= self.EPSILON_DECAY
            
            if self.noise_type == "ou": self.noise.reset()
            return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

        

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

def calc_fraction_loss(FZ_,FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    assert not gradients.requires_grad
    if weights != None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1)*weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss
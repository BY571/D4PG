import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.relu(self.batch_norm(self.fc1(state)))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size+action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.batch_norm(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QVN(nn.Module):
    """Quantile Value Network"""
    def __init__(self, state_size, action_size, layer_size, n_step, device, seed, dueling=False, N=32):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(self.input_shape)
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.dueling = dueling
        self.device = device

        # Network Architecture

        self.head = nn.Linear(self.input_shape[0], layer_size) 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.cos_layer_out = layer_size
        weight_init([self.head, self.ff_1])
        if dueling:
            self.advantage = nn.Linear(layer_size, action_size)
            self.value = nn.Linear(layer_size, 1)
            weight_init([self.ff_1])
        else:
            self.ff_2 = nn.Linear(layer_size, action_size)    
            weight_init([self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self,taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos
    
    def forward(self, input):
        """Calculate the state embeddings"""
        return torch.relu(self.head(input))
        
    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        
         """
        if embedding==None:
            x = self.forward(input)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)   
        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size)
    
    

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):
        """
        Calculates tau, tau_ and the entropy
        
        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """

        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy



class DeepActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DeepActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.batch_norm = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.relu(self.batch_norm(self.fc1(state)))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class DeepCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DeepCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.batch_norm = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.batch_norm(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
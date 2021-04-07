import torch
import numpy as np 
import torch.nn as nn 
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

class Inverse(nn.Module):
    """
    1. (first submodel) encodes the state and next state into feature space.
    2. (second submodel) the inverse approximates the action taken by the given state and next state in feature size
    
    returns the predicted action and the encoded state for the Forward Model and the encoded next state to train the forward model!
    
    optimizing the Inverse model by the loss between actual action taken by the current policy and the predicted action by the inverse model
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Inverse, self).__init__()

        self.state_size = state_size

        self.encoder = nn.Sequential(nn.Linear(state_size, 128),
                                        nn.ELU())
        self.layer1 = nn.Linear(2*128, hidden_size)                                         
        self.layer2 = nn.Linear(hidden_size, action_size)
    
    def calc_input_layer(self):
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return x.flatten().shape[0]
    
    def forward(self, enc_state, enc_next_state):
        """
        Input: state s and state s' as torch Tensors with shape: (batch_size, state_size)
        Output: action probs with shape (batch_size, action_size)
        """
        x = torch.cat((enc_state, enc_next_state), dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        dist = Normal(loc=x, scale=torch.FloatTensor([0.1]).to(x.device))
        action = dist.sample()
        return action

    
class Forward(nn.Module):
    """
  
    """
    def __init__(self, state_size, action_size, output_size, hidden_size=256, device="cuda:0"):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.device = device
        self.forwardM = nn.Sequential(nn.Linear(output_size+self.action_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size,output_size))
    
    def forward(self, state, action):
        """
        Input: state s embeddings and action a as torch Tensors with shape
        s: (batch_size, embedding_size), 
        a: (batch_size, action_size)
        
        Output:
        encoded state s' prediction by the forward model with shape: (batch_size, embedding_size)
        
        Gets as inputs the aciton taken from the policy and the encoded state by the encoder in the inverse model.
        The froward model trys to predict the encoded next state. 
        Returns the predicted encoded next state.
        Gets optimized by the MSE between the actual encoded next state and the predicted version of the forward model!
         """
        # One-hot-encoding for the discrete actions 
        # ohe_action = torch.zeros(action.shape[0], self.action_size).to(action.device)
        # indices = torch.stack((torch.arange(action.shape[0]).to(action.device), action.squeeze().long()), dim=0)
        # indices = indices.tolist()
        # ohe_action[indices] = 1.
        # x = torch.cat((state, ohe_action) ,dim=1)
        #concat state embedding and encoded action

        x = torch.cat((state, action) ,dim=1)
        #assert x.device.type == "cuda"
        return self.forwardM(x)

    
class ICM(nn.Module):
    def __init__(self, inverse_model, forward_model, learning_rate=1e-3, lambda_=0.1, beta=0.2, device="cuda:0"):
        super(ICM, self).__init__()
        self.inverse_model = inverse_model.to(device)
        self.forward_model = forward_model.to(device)
        self.device = device
        
        self.forward_scale = 1.
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = torch.FloatTensor([beta]).to(device)
        self.lambda_ = lambda_
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.MSELoss(reduction='none') # CrossEntropyLoss for discrete
        self.optimizer = optim.Adam(list(self.forward_model.parameters())+list(self.inverse_model.parameters()), lr=1e-3)

    def calc_errors(self, state1, state2, action):
        """
        Input: Torch Tensors state s, state s', action a with shapes        print(enc_state1.shape)
        s: (batch_size, state_size)
        s': (batch_size, state_size)
        a: (batch_size, 1)
        
        """
        #assert state1.device.type == "cuda" and state2.device.type == "cuda" and action.device.type == "cuda"
        enc_state1 = self.inverse_model.encoder(state1).view(state1.shape[0],-1)
        enc_state2 = self.inverse_model.encoder(state2).view(state1.shape[0],-1)

        #assert enc_state1.shape == (32,1152), "Shape is {}".format(enc_state1.shape)
        # calc forward error 
        forward_pred = self.forward_model(enc_state1.detach(), action)
        assert not action.requires_grad, "action should not require grad!"
        assert forward_pred.shape == enc_state2.shape, "forward_pred and enc_state2 dont have the same shape"
        forward_pred_err = 1/2 * self.forward_loss(forward_pred, enc_state2.detach()).sum(dim=1).unsqueeze(dim=1)
        
        # calc prediction error
        pred_action = self.inverse_model(enc_state1, enc_state2)
        #assert pred_action.shape ==  action.flatten().long().shape, "Pred_action: {} -- Action shape: {}".format(pred_action.shape, action.flatten().long().shape)

        inverse_pred_err = self.inverse_loss(pred_action, action)
        self.optimizer.zero_grad()
        loss = ((1. - self.beta) * inverse_pred_err + self.beta * forward_pred_err).mean()

        loss.backward()
        clip_grad_norm_(self.inverse_model.parameters(), 0.5)
        clip_grad_norm_(self.forward_model.parameters(), 0.5)
        self.optimizer.step()
      
        return loss.detach().cpu().numpy(), forward_pred_err.detach()
    
    def get_intrinsic_reward(self, state, next_state, action):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)

        with torch.no_grad():
            enc_state1 = self.inverse_model.encoder(state)
            enc_state2 = self.inverse_model.encoder(next_state)

            # calc forward error 
            forward_pred = self.forward_model(enc_state1, action)
            assert not action.requires_grad, "action should not require grad!"
            assert forward_pred.shape == enc_state2.shape, "forward_pred and enc_state2 dont have the same shape"
            forward_pred_err = 1/2 * self.forward_loss(forward_pred, enc_state2.detach()).sum(dim=1).unsqueeze(dim=1)
        return forward_pred_err
        
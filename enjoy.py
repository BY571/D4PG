import numpy as np
import random
import gym
import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0
from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
#from  files import MultiPro
from scripts.agent import Agent
from  scripts import MultiPro
import json


def evaluate(eval_runs=5):
    """
    Makes an evaluation run 
    """

    for _ in range(eval_runs):
        state = eval_env.reset()

        rewards = 0
        while True:
            eval_env.render()
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)

            state, reward, done, _ = eval_env.step(action_v[0])
            rewards += reward
            if done:
                print("Episode Rewards: {}".format(rewards))
                break

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run to load the hyperparameter and the model weights, default: test")
    parser.add_argument("--runs", type=int, default=10, help="Number of evaluation runs with the policy, default: 10")
    
    args = parser.parse_args()
    
    with open('runs/'+args.run_name+".json", 'r') as f:
        parameters = json.load(f)
    
    
    print("Parameters: \n", parameters)
    parameters = dotdict(parameters)
    # create eval environement
    eval_env = gym.make(parameters.env)
    eval_env.seed(parameters.seed)
    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]
    
    # create agent
    agent = Agent(state_size=state_size, action_size=action_size, n_step=parameters.nstep, per=parameters.per, munchausen=parameters.munchausen,distributional=parameters.iqn,
                 D2RL=parameters.d2rl, curiosity=(parameters.icm, parameters.add_ir), noise_type=parameters.noise, random_seed=parameters.seed,
                 hidden_size=parameters.layer_size,
                 BATCH_SIZE=parameters.batch_size, BUFFER_SIZE=parameters.replay_memory, GAMMA=parameters.gamma,
                 LR_ACTOR=parameters.lr_a, LR_CRITIC=parameters.lr_c, TAU=parameters.tau, LEARN_EVERY=parameters.learn_every,
                 LEARN_NUMBER=parameters.learn_number, device="cpu", frames=0, worker=parameters.worker) 
    evaluate(args.runs)
    
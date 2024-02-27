# NAO-CONTROL WITH POLICY GRADIENTS

# Authors:
# - Felipe de Jesus Felix Arredondo (A00833150){@tec.mx}[Deep Learning Engineer]
# - Arturo Jose Murra Lopez (A01236090){@tec.mx}[Robotics Engineer]
# - Juan Andrés Sánchez Chaires (A01412353){@tec.mx}[Mechatronics Engineer]
# Description:

# [LIBRARIES]

# Enable warnings
import warnings
warnings.filterwarnings("ignore")

# [Robotics Libraries]
import pybullet as p
import pybullet_data

# [Basic Libraries]
import time
import math
import random

# [Deep Learning Libraries]
import torch
import torch.nn as nn
import torch.nn.functional as F

# [Deep Learning Libraries]{@torch}[Distributions]
from torch.distributions.normal import Normal

# [Gymnasium Libraries]
import gymnasium as gym

# [Stable Baselines3 Libraries]
from stable_baselines3 import PPO


class Buffer(object):

    def __init__(self, gamma=0.99):
        self.buffer = {"states": [],
                       "actions": [],
                       "values": [],
                       "rewards": [],
                       "dones": []}
        
    def store(self, state, action, value, reward, done):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["values"].append(value)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)

    def clear(self):
        self.buffer = {"states": [],
                        "actions": [],
                        "values": [],
                        "rewards": [],
                        "dones": []}
        
    def get(self):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)


class NaoTrainer:

    def __init__(self, env, brain, buffer, optimizer, epochs, batch_size, gamma):
        self.env = env
        self.brain = brain
        self.buffer = buffer
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma

    def collect_experience(self):
        state = self.env.reset()
        done = False
        while not done:
            dist, value = self.brain(torch.tensor(state, dtype=torch.float32))
            action = dist.sample().numpy()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.store(state, action, reward, done)
            state = next_state

    def train(self):
        for epoch in range(self.epochs):
            batch = self.buffer.get()
            for i in range(0, len(batch["states"]), self.batch_size):
                states = batch["states"][i:i+self.batch_size]
                actions = batch["actions"][i:i+self.batch_size]
                values = batch["values"][i:i+self.batch_size]
                rewards = batch["rewards"][i:i+self.batch_size]
                returns = self.calculate_returns(rewards)
                self.optimizer.zero_grad()
                dist = self.brain(states)
                log_probs = dist.log_prob(actions)
                Aloss = -log_probs * returns
                Closs = F.smooth_l1_loss(values, returns)
                loss = Aloss + Closs
                loss.backward()
                self.optimizer.step()
            self.buffer.clear()

    def calculate_returns(self, rewards):
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns


class NaoBrain(nn.Module):
    
        def __init__(self, state_dim, action_dim):
            super(NaoBrain, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.mean = nn.Linear(64, action_dim)
            self.std = nn.Linear(64, action_dim)
            self.value = nn.Linear(64, 1)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            mean = self.mean(x)
            std = self.std(x)
            value = self.value(x)
            return Normal(mean, std), value
        
if __name__ == "__main__":
    pass
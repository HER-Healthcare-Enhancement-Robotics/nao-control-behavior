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
import numpy as np
from collections import deque

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

# [Self-Defined Libraries]
from NaoBot.NAOenv import NaoEnv


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
            #print ("datos jiji",key, type(self.buffer[key]), len(self.buffer[key]))
            if isinstance(self.buffer[key], list) and isinstance(self.buffer[key][0], torch.Tensor):
                batch[key] = torch.stack(self.buffer[key])
            else:
                batch[key] = torch.tensor(np.array(self.buffer[key]), dtype=torch.float32)
        return batch


class NaoTrainer:

    def __init__(self, env, brain, buffer, optimizer, epochs, batch_size, gamma, random_epsilon=1):
        self.env = env
        self.brain = brain
        self.buffer = buffer
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.random_epsilon = random_epsilon
        self.mean_rewards = deque(maxlen=100)

    def collect_experience(self):
        state,_ = self.env.reset()
        done = False
        running_reward = 0
        while not done:
            dist, value = self.brain(torch.tensor(np.array(state), dtype=torch.float32))
            if np.random.rand() < self.random_epsilon:
                action = Normal(torch.zeros_like(dist.mean), .1*torch.ones_like(dist.stddev)).sample()
                self.random_epsilon *= 0.99998
            else:
                action = dist.sample()
            action = torch.clamp(action, -1, 1).detach().numpy()
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.buffer.store(state, action, value, reward, done)
            state = next_state
            running_reward += reward
        self.mean_rewards.append(running_reward)


    def train(self):
        for epoch in range(self.epochs):
            batch = self.buffer.get()
            for i in range(0, len(batch["states"]), self.batch_size):
                states = batch["states"][i:i+self.batch_size]
                actions = batch["actions"][i:i+self.batch_size]
                values = batch["values"][i:i+self.batch_size]
                rewards = batch["rewards"][i:i+self.batch_size]
                returns = self.calculate_returns(rewards)
                # normalize returns
                returns = (returns - returns.mean()) / (returns.std() + 1e-4)
                # Convert the divided by 0 to 0
                returns = torch.where(torch.isnan(returns), torch.mean(returns), returns)
                self.optimizer.zero_grad()
                dist, value = self.brain(states)
                log_probs = dist.log_prob(actions)
                #print ("log_probs",log_probs.shape, "returns",returns.shape, "value",value.shape)
                #print ("Actions:",actions[0])
                Aloss = -log_probs * returns.unsqueeze(1)
                Closs = F.smooth_l1_loss(value, returns.unsqueeze(1))
                loss = (Aloss + .5 * Closs + .001 * dist.entropy()).mean()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.3)
                self.optimizer.step()

        self.buffer.clear()
        print ("Random Epsilon",self.random_epsilon)

    def calculate_returns(self, rewards):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + self.gamma * running_return
            returns[i] = running_return
        return returns


class NaoBrain(nn.Module):
    
        def __init__(self, state_dim, action_dim):
            super(NaoBrain, self).__init__()
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 64)
            self.mean = nn.Linear(64, action_dim)
            self.std = nn.Linear(64, action_dim)
            self.value = nn.Linear(64, 1)
            self.norm_layer = nn.LayerNorm(64)
            self.selu = nn.SELU()
    
        def forward(self, x):
            x = self.norm_layer(F.relu(self.fc1(x)))
            x = self.norm_layer(F.relu(self.fc2(x)))
            x = self.norm_layer(F.relu(self.fc3(x)))
            x = self.norm_layer(F.relu(self.fc4(x)))
            mean = self.mean(x)+1e-4
            std = torch.nn.functional.softplus(self.std(x))+1e-4
            std = torch.ones_like(mean) * std
            value = self.value(x)
            return Normal(mean, std), value
        
class NaoBrainLSTM(nn.Module):
    
        def __init__(self,
                    state_dim,
                    action_dim,
                    hidden_dim,
                    num_layers,
                    dropout):
            super(NaoBrainLSTM, self).__init__()
            self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
if __name__ == "__main__":
    # [Environment Configuration]
    env = NaoEnv(render_mode="human",delta=0.35)
    # [Brain Configuration]
    brain = NaoBrain(env.observation_space.shape[0], env.action_space.shape[0])
    # [Load Model]
    brain.load_state_dict(torch.load("nao.pth"))
    # [Buffer Configuration]
    buffer = Buffer()
    # [Optimizer Configuration]
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.0001)
    # [Trainer Configuration]
    trainer = NaoTrainer(env, brain, buffer, optimizer, 20, 32, 0.99)
    # [Training Loop]
    for _ in range(1000):
        trainer.collect_experience()
        print ("mean reward",np.mean(trainer.mean_rewards))
        trainer.train()
        # [Save Model]
        torch.save(brain.state_dict(), "nao.pth")
    # [Save Model]
    torch.save(brain.state_dict(), "nao.pth")
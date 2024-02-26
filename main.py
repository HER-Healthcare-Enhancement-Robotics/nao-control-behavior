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


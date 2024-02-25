import pybullet as p
import pybullet_data
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class NaoBrain(nn.Module):
    def __init__(self,
                input_size: int = 2,
                output_size: int = 2,
                hidden_size: int = 10,
                 ):
        super(NaoBrain, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_sigma = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        sigma = torch.exp(sigma)
        return Normal(mu, sigma)


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    naoId = p.loadURDF("humanoid/nao.urdf", [0, 0, 1])

    # Create a NaoBrain with the number of joints
    brain = NaoBrain(p.getNumJoints(naoId), p.getNumJoints(naoId))
    jointDelta = 0.1
    while (1):
        dist = brain(torch.ones(1, p.getNumJoints(naoId)))
        print (torch.ones(p.getNumJoints(naoId)).shape)
        action = dist.sample()
        print (action.shape)
        # Add the action to the joints
        for i in range(p.getNumJoints(naoId)):
            p.setJointMotorControl2(naoId, i, p.POSITION_CONTROL, action[0][i].item())

        p.stepSimulation()
        time.sleep(1. / 20.)
    p.disconnect()
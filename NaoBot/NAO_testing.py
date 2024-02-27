import pybullet as p
import pybullet_data
import time
import math

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

def get_neck_position(naoId):
    neck_position, _, _, _, _, _ = p.getLinkState(naoId, 9, computeForwardKinematics=True)
    return neck_position[2]

def calculate_reward(neck_position, threshold=0.45):
    if neck_position > threshold:
        return 1.0
    else:
        return 0.0

def launch_ball(ballId, target_position):
    # Set the orientation to face the target
    target_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    p.resetBasePositionAndOrientation(ballId, target_position, target_orientation)

    ball_velocity = [0, 0, -10]  # Negative velocity to make it fall directly
    p.resetBaseVelocity(ballId, ball_velocity)

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load the plane and Nao
    p.loadURDF("plane.urdf")
    naoId = p.loadURDF("humanoid/nao.urdf", [0, 0, 1])

    # Load the ball URDF
    ballId = p.loadURDF("ball.urdf", [0, 0, 5])

    # Create a NaoBrain with the number of joints
    brain = NaoBrain(p.getNumJoints(naoId), p.getNumJoints(naoId))
    jointDelta = 0.1

    last_launch_time = time.time()

    while True:
        # Get the current neck position
        neck_position = get_neck_position(naoId)

        # Calculate the reward based on the neck position
        reward = calculate_reward(neck_position)

        # Print the current neck position and reward
        print(f"Neck Position: {neck_position}, Reward: {reward}")

        # Perform the policy optimization using the brain
        dist = brain(torch.ones(1, p.getNumJoints(naoId)))
        action = dist.sample()

        # Add the action to the joints
        for i in range(p.getNumJoints(naoId)):
            p.setJointMotorControl2(naoId, i, p.POSITION_CONTROL, action[0][i].item()*jointDelta)

        # Check if 3 seconds have passed since the last ball launch
        if time.time() - last_launch_time >= 3:
            # Launch the ball in front of the Nao
            target_position = p.getBasePositionAndOrientation(naoId)[0]
            launch_ball(ballId, [target_position[0], target_position[1], 5])
            last_launch_time = time.time()

        # Step the simulation
        p.stepSimulation()
        time.sleep(1. / 20.)

    p.disconnect()
# NAO-CONTROL WITH POLICY GRADIENTS

# Authors:
# - Felipe de Jesus Felix Arredondo (A00833150){@tec.mx}[Deep Learning Engineer]
# - Arturo Jose Murra Lopez (A01236090){@tec.mx}[Robotics Engineer]
# - Juan Andrés Sánchez Chaires (A01412353){@tec.mx}[Mechatronics Engineer]
# Description:

# [LIBRARIES]

# [Robotics Libraries]
import pybullet as p
import pybullet_data

# [Basic Libraries]
import time
import math

# [Deep Learning Libraries]
import torch
import torch.nn as nn
import torch.nn.functional as F

# [Deep Learning Libraries]{@torch}[Distributions]
from torch.distributions.normal import Normal

# [Gymnasium Libraries]
import gymnasium as gym


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
    
class NaoEnv(gym.Env):
    def __init__(self, 
                delta=0.1,
                render_mode=False
                ):
        # [Environment Configuration]
        self.metadata = {'render.modes': ['human']}
        self.render_mode = render_mode
        # [PyBullet Configuration]
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # [PyBullet Configuration]{@nao}[URDFs]
        self.planeId = p.loadURDF("plane.urdf")
        self.naoId = p.loadURDF("humanoid/nao.urdf", [0, 0, 1])
        self.ballId = p.loadURDF("ball.urdf", [0, 0, 5])

        # [NaoBrain Configuration]
        self.jointDelta = delta
        self.last_launch_time = time.time()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(p.getNumJoints(self.naoId),))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(p.getNumJoints(self.naoId),))
        self.reward_range = (0, 1)

    def step(self, action):
        # [PyBullet Configuration]{@nao}[Neck Position]
        neck_position = self._get_neck_position(self.naoId)
        # [PyBullet Configuration]{@nao}[Reward]
        reward = self._calculate_reward(neck_position)
        # [PyBullet Configuration]{@nao}[Launch Ball]
        if time.time() - self.last_launch_time >= 3:
            target_position = p.getBasePositionAndOrientation(self.naoId)[0]
            self._launch_ball(self.ballId, [target_position[0], target_position[1], 5])
            self.last_launch_time = time.time()
        # [PyBullet Configuration]{@nao}[Movement]
        self._set_movement(action)
        # [PyBullet Configuration]{@nao}[Step Simulation]
        p.stepSimulation()
        time.sleep(1. / 20.)
        # [PyBullet Configuration]{@nao}[Observation]
        observation = torch.ones(1, p.getNumJoints(self.naoId))
        return observation, reward, False, False, {}
    
    # [Methods]{@nao}[get_neck_position]
    def _get_neck_position(self, naoId):
        neck_position, _, _, _, _, _ = p.getLinkState(naoId, 9, computeForwardKinematics=True)
        return neck_position[2]
    
    # [Methods]{@nao}[calculate_reward]
    def _calculate_reward(self, neck_position, threshold=0.45):
        if neck_position > threshold:
            return 1.0
        else:
            return 0.0
    
    # [Methods]{@nao}[launch_ball]
    def _launch_ball(self, ballId, target_position):
        target_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        p.resetBasePositionAndOrientation(ballId, target_position, target_orientation)
        ball_velocity = [0, 0, -10]
        p.resetBaseVelocity(ballId, ball_velocity)

    # [Methods]{@nao}[movement]
    def _set_movement(self, actions):
        for i in range(p.getNumJoints(self.naoId)):
            p.setJointMotorControl2(self.naoId, i, p.POSITION_CONTROL, actions[0][i].item()*self.jointDelta)

    
    
    def reset(self):
        return torch.ones(1, p.getNumJoints(self.naoId)), {}
    
    def render(self):
        pass

    def close(self):
        p.disconnect()


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
    env = NaoEnv(render_mode=True,delta=.5)
    # Create a NaoBrain with the number of joints
    brain = NaoBrain(env.observation_space.shape[0], env.action_space.shape[0])
    jointDelta = 0.5

    # [Start Simulation]
    state, _ = env.reset()
    # [Start Simulation]{@nao}[Loop]
    while True:
        # [PyBullet Configuration]{@nao}[Neck Position]
        neck_position = get_neck_position(env.naoId)
        # [PyBullet Configuration]{@nao}[Reward]
        reward = calculate_reward(neck_position)
        # [PyBullet Configuration]{@nao}[Print]
        print(f"Neck Position: {neck_position}, Reward: {reward}")
        # [PyBullet Configuration]{@nao}[Policy Optimization]
        dist = brain(state)
        action = dist.sample()
        # [PyBullet Configuration]{@nao}[Add Action]
        state, _, _, _,_ = env.step(action)
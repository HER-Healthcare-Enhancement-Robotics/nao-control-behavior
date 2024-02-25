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

    # [Constructor]{@nao}[Environment Configuration
    def __init__(self, 
                delta=0.1,
                minimum_position=.45,
                render_mode=False
                ):
        # [Environment Configuration]
        self.metadata = {'render.modes': ['human']}
        self.render_mode = render_mode
        self.minimum_position = minimum_position
        # [PyBullet Configuration]
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
            p.setRealTimeSimulation(1)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # [PyBullet Configuration]{@nao}[URDFs]
        self.planeId = p.loadURDF("plane.urdf")
        self.naoId = p.loadURDF("humanoid/nao.urdf", [0, 0, .35])
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
        # [PyBullet Configuration]{@nao}[Done]
        trunc = self._get_done(neck_position, threshold=self.minimum_position * .5)
        done = self._get_done(neck_position, threshold=self.minimum_position)
        # [PyBullet Configuration]{@nao}[Reward]
        reward = self._calculate_reward(neck_position,threshold=self.minimum_position)
        # [PyBullet Configuration]{@nao}[Launch Ball]
        if time.time() - self.last_launch_time >= 3:
            target_position = p.getBasePositionAndOrientation(self.naoId)[0]
            # Generate a random position
            random_position = [random.uniform(-1, 1), random.uniform(-1, 1), 5]
            for i in range(3):
                random_position[i] = random_position[i] + target_position[i]
            self._launch_ball(self.ballId, random_position)
            self.last_launch_time = time.time()
        # [PyBullet Configuration]{@nao}[Movement]
        self._set_movement(action)
        # [PyBullet Configuration]{@nao}[Step Simulation]
        # Make a pause in the simulation
        time.sleep(1./2.)
        p.stepSimulation()
        
        # [PyBullet Configuration]{@nao}[Observation]
        observation = self._get_observation()
        return observation, reward, done, trunc, {}
    
    # [Methods]{@nao}[get_neck_position]
    def _get_neck_position(self, naoId):
        neck_position, _, _, _, _, _ = p.getLinkState(naoId, 9, computeForwardKinematics=True)
        return neck_position[2]
    
    # [Methods]{@nao}[get_observation]
    def _get_observation(self):
        jointStates = []
        for i in range(p.getNumJoints(self.naoId)):
            jointStates.append(p.getJointState(self.naoId, i)[0])
        return torch.tensor(jointStates)
    
    # [Methods]{@nao}[calculate_reward]
    def _calculate_reward(self, neck_position, threshold=0.45):
        if neck_position > threshold:
            return 1.0
        else:
            return -.5
        
    # [Methods]{@nao}[get_done]
    def _get_done(self, neck_position, threshold=0.45):
        if neck_position < threshold:
            return True
        else:
            return False
    
    # [Methods]{@nao}[launch_ball]
    def _launch_ball(self, ballId, target_position):
        target_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        p.resetBasePositionAndOrientation(ballId, target_position, target_orientation)
        ball_velocity = [0, 0, -10]
        p.resetBaseVelocity(ballId, ball_velocity)

    # [Methods]{@nao}[movement]
    def _set_movement(self, actions):
        for i in range(p.getNumJoints(self.naoId)):
            # Consider the jointLimits
            #p.setJointMotorControl2(self.naoId, i, p.POSITION_CONTROL, actions[i].item()*self.jointDelta)
            p.setJointMotorControl2(self.naoId, i, p.POSITION_CONTROL, actions[i].item()*self.jointDelta)
    
    def reset(self):
        return self._get_observation(), {}
    
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

def launch_ball(ballId, start_position,target_position):
    # Set the orientation to face the target
    target_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    p.resetBasePositionAndOrientation(ballId, start_position, target_orientation)
    pos_diff = [target_position[0] - start_position[0], target_position[1] - start_position[1], target_position[2] - start_position[2]]
    norm = math.sqrt(pos_diff[0]**2 + pos_diff[1]**2 + pos_diff[2]**2)
    normalized_diff = [pos_diff[0]/norm, pos_diff[1]/norm, pos_diff[2]/norm]
    vel_mag = 10
    ball_velocity = [normalized_diff[0]*vel_mag, normalized_diff[1]*vel_mag, normalized_diff[2]*vel_mag]
    p.resetBaseVelocity(ballId, ball_velocity)
    # Change position to a random position

if __name__ == "__main__":
    env = NaoEnv(render_mode=False,delta=.5)
    # Create a NaoBrain with the number of joints
    brain = NaoBrain(env.observation_space.shape[0], env.action_space.shape[0])
    
    jointDelta = 0.5

    # [Start Simulation]
    state, _ = env.reset()
    # [Start Simulation]{@nao}[Loop]
    while True:
        # [PyBullet Configuration]{@nao}[Get action]
        dist = brain(state)
        action = dist.sample()
        # [PyBullet Configuration]{@nao}[Add action to the joints]
        state, reward, done, trunc, _ = env.step(action)
        # [PyBullet Configuration]{@nao}[Print]
        print (f"Reward: {reward}, Position: {get_neck_position(env.naoId)}")
        #print(f"Neck Position: {get_neck_position(env.naoId)}, Reward: {reward}")
        if done:
            print ("Done")
            break
    # [End Simulation]
    env.close()
        
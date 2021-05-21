# Standard Library Imports
import os
import gym
import copy
import random
import sys
sys.path.append("..")

# External Dependencies
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

# Deep Learning and Bayesian Deep Learning packages
import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers
from deepbayes_prealpha import PosteriorModel
from deepbayes_prealpha.analyzers import IBP_state
from deepbayes_prealpha.analyzers import IBP_prob

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *




# In[ ]:

# Custom obstacles for HOV
tri1 = np.array([[-1,-1], [1,-1], [0,-0.5]])*0.2
tri2 = np.array([[-1,1], [1,1], [0,-0.0]])*0.1
   
obstacles1 = np.asarray([np.asarray([0.75,0.35])])
obstacles2 = np.asarray([])

OBSTACLE = [0.75,0.35]

def PointInTriangle(pt, tri):
    '''checks if point pt(2) is inside triangle tri(3x2). @Developer'''
    a = 1/(-tri[1,1]*tri[2,0]+tri[0,1]*(-tri[1,0]+tri[2,0])+         tri[0,0]*(tri[1,1]-tri[2,1])+tri[1,0]*tri[2,1])
    s = a*(tri[2,0]*tri[0,1]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*pt[0]+         (tri[0,0]-tri[2,0])*pt[1])
    if s<0: return False
    else: t = a*(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]+(tri[0,1]-tri[1,1])*pt[0]+               (tri[1,0]-tri[0,0])*pt[1])
    return ((t>0) and (1-s-t>0))

"""
The handcrafted Fan et al environment we consider
"""
"""
The handcrafted Fan et al environment we consider
"""
import math
class Zigzag_env:
    def __init__(self):
        self.x = 1.5 + random.uniform(-0.025, 0.025)
        self.y = 0.5 + random.uniform(-0.025, 0.025)
        self.z = 0.25
        self.theta = -2.65
        self.t = 0
        self.T = 30
        self.col = False

    def step(self, act):
        pre = np.linalg.norm([self.x, self.y]) 
        #print("step (t=%s): "%(self.t), self.x, self.y)
        self.x = self.x + (math.cos(self.theta + 3.14 + 0.2)*act[0])
        self.y = self.y + (math.sin(self.theta + 3.14 + 0.2)*act[0])
        self.z = self.z + act[1]
        self.z = self.z - 0.005 # gravity term
        self.theta = self.theta + act[2]
        self.t += 1 
        
        # Check obstacle 
        #if(self.z <= 0.0):
        #    print("Grounded")
        #else:
        #    print("Altitude ", self.z)
        #self.col = True
        if(VERSION != 0):
            collide = self.check_collision([self.x, self.y])
            if(collide == 1):
                print("Collided")
                self.col = True
        return [self.x, self.y, self.z, self.theta], pre - np.linalg.norm([self.x, self.y, self.z]) , self.complete(), 0    

    def complete(self):
        state = np.asarray([self.x, self.y, self.z])
        if(np.linalg.norm(state) < 0.15 or self.x < 0.1):
            return True
        if(self.t==self.T or self.t > self.T):
            print("Timeout")
            return True
        return self.col

    def reset(self):
        self.x = 1.5 + random.uniform(-0.025, 0.025)
        self.y = 0.5 + random.uniform(-0.025, 0.025)
        self.z = 0.25
        self.theta = -2.65
        self.t = 0
        self.col = False
        return np.asarray([self.x, self.y, self.z, self.theta])
    
    def check_collision(self, pt):
            for obs in obstacles1:
                col = PointInTriangle(pt, tri1+obs)
                if(col == True or col == 1):
                    return 1
            for obs in obstacles2:
                col = PointInTriangle(pt, tri2+obs)
                if(col == True or col == 1):
                    return 1
            return 0

    def action_space_sample(self):
        return np.random.rand(3)
    def observation_space_sample(self):
        return np.random.rand(4)

env = Zigzag_env()
pos_ind = 0
momentum = 0.0
VERSION = 1
pos_ind = 0
state_0 = env.reset()

print("Example observation: ", state_0)

initial_state = state_0 # This is now the template for the "STATE" that we will pass throughout

action_dims = len(env.action_space_sample())
observe_dims = len(env.observation_space_sample())

print("Observation example and dimensions: ", env.observation_space_sample(),  len(env.observation_space_sample()))
print("Action example and dimensions: ", env.action_space_sample(),  len(env.action_space_sample()))

model_input_dims = observe_dims + action_dims
model_output_dims = observe_dims
print(model_input_dims, model_output_dims)

control_input_dims = observe_dims
control_output_dims = action_dims
print(control_input_dims, control_output_dims)


# Load BNN
bayes_model = PosteriorModel("Posteriors/BayesDyn_HOV_v0_RUN")

# Load DNN
control_model = PosteriorModel("Posteriors/Control_HOV_v0_RUN", deterministic=True)

# The 25 controls the number of runs we do
trajectories = []
for traj in trange(25, desc="Simulating the Control Loop"):
    done = False
    prev_obs = env.reset()
    time = 0; sim = []
    sim.append([prev_obs[0], prev_obs[1]])
    print(prev_obs)
    while(not done):
        act = control_model.predict(np.asarray([prev_obs]))
        act = np.squeeze(act)
        a = copy.deepcopy(act)
        a[1] = np.clip(act[1], -0.025, 0.025)
        act = a
        #obs = bayes_model.predict(np.asarray([np.concatenate((prev_obs, act))]))
        #prev_obs = np.squeeze(prev_obs) + np.squeeze(obs)
        prev_obs, _, _, _ = env.step(act)
        sim.append([prev_obs[0], prev_obs[1]])
        if(prev_obs[0] < 0.2 or time > 30):
            print(prev_obs, act)
            break
        print(prev_obs, act)
        time += 1
    env.reset()
    time = 0
    trajectories.append(sim)

np.save("simulated_HOV_v2_%s"%(25), np.asarray(trajectories))

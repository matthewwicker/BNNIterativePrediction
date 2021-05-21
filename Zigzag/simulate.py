# Standard Library Imports
import os
import gym
import sys
import copy
import random
sys.path.append("..")

# External Library Imports
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

# Deep Learning and Bayesian Deep Learning Libraries
import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers
from deepbayes_prealpha import PosteriorModel
from deepbayes_prealpha.analyzers import IBP_state
from deepbayes_prealpha.analyzers import IBP_prob

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# Custom obstacles for zigzag
tri1 = np.array([[-1,-1], [1,-1], [0,1.25]])*0.5
tri2 = np.array([[-1,1], [1,1], [0,-1.25]])*0.5

obstacles1 = np.asarray([np.asarray([0.5,-1.25]),
                       np.asarray([1.0,-1.25])])
obstacles2 = np.asarray([np.asarray([0.25,1.0]),
                        np.asarray([0.75,0.5]),
                        np.asarray([1.25,0.75])])



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
import math
class Zigzag_env:
    def __init__(self):
        self.x = 1.75 + random.uniform(-0.1, -0.0)
        self.y = -0.85 +  random.uniform(-0.1, -0.0)
        self.theta = -0.3
        self.t = 0
        self.T = 80
        self.col = False

    def step(self, act):
        pre = np.linalg.norm([self.x, self.y]) 
        #print("step (t=%s): "%(self.t), self.x, self.y)
        self.x = self.x + (math.cos(self.theta)*act[0])#*0.1
        self.y = self.y + (math.sin(self.theta)*act[0])#*0.1
        self.theta = self.theta + act[1]
        self.t += 1 
        # Check obstacle 
        collide = self.check_collision([self.x, self.y])
        if(collide == 1):
            print("Collided")
            self.col = True
        return [self.x, self.y, self.theta], pre - np.linalg.norm([self.x, self.y]) , self.complete(), 0    

    def complete(self):
        state = np.asarray([self.x, self.y])
        if(np.linalg.norm(state) < 0.15 or self.x < -0.1):
            return True
        if(self.t==self.T or self.t > self.T):
            print("Timeout")
            return True
        return self.col

    def reset(self):
        self.x = 1.75 + random.uniform(-0.1, -0.0)
        self.y = -0.85 + random.uniform(-0.1, -0.0)
        self.theta = -0.3
        self.t = 0
        self.col = False
        return np.asarray([self.x, self.y, self.theta])
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
        return np.random.rand(2)
    def observation_space_sample(self):
        return np.random.rand(3)

env = Zigzag_env()
pos_ind = 0
momentum = -0.0
state_0 = env.reset()
initial_state = state_0 # This is now the template for the "STATE" that we will pass throughout

action_dims = len(env.action_space_sample())
observe_dims = len(env.observation_space_sample())

print("DIMENSIONS OBS: ", env.observation_space_sample(),  len(env.observation_space_sample()))
print("DIMENSIONS ACT: ", env.action_space_sample(),  len(env.action_space_sample()))

model_input_dims = observe_dims + action_dims
model_output_dims = observe_dims
print(model_input_dims, model_output_dims)

control_input_dims = observe_dims
control_output_dims = action_dims
print(control_input_dims, control_output_dims)

# Load BNN
bayes_model = PosteriorModel("Posteriors/BayesDyn_ZIG_v1_RUN")
# Load DNN
control_model = PosteriorModel("Posteriors/Control_ZIG_v1_RUN", deterministic=True)

trajectories = []
for traj in trange(25, desc="Simulating the Control Loop"):
    done = False
    prev_obs = env.reset()
    time = 0; sim = []
    while(not done):
        act = control_model.predict(np.asarray([prev_obs]))
        act = np.squeeze(act)
        # Here you can choose to either run with the BNN or the true Env
        #obs = bayes_model.predict(np.asarray([np.concatenate((prev_obs, act))]))
        #prev_obs = np.squeeze(prev_obs) + np.squeeze(obs)
        prev_obs, reward, done, collisions = env.step(act)
        sim.append([prev_obs[0], prev_obs[1]])
        if(prev_obs[0] < -0.1):
            print(prev_obs, act)
            break
        print(prev_obs, act)
        time += 1
    env.reset()
    time = 0
    trajectories.append(sim)

np.save("simulated_ZIG_v0_%s"%(25), np.asarray(trajectories))

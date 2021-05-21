# Author: Matthew Wicker

import os
import gym
import copy
import sys
sys.path.append("..")

import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers
from deepbayes_prealpha import PosteriorModel
from deepbayes_prealpha.analyzers import IBP_state
from deepbayes_prealpha.analyzers import IBP_prob

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

pos_ind = 0
momentum = -0.0
HORIZON = 50
OBSTACLE = [0.4,0.5]

class P2D_env:
    def __init__(self):
        self.x = 1.0 + np.random.normal(0.0, 0.025)
        self.y = 0.9 + np.random.normal(0.0, 0.025)
        self.dx = 0.0
        self.dy = 0.0
        # Point Mass Dynamics Control:
        self.h = 0.35      # time resolution
        self.T = HORIZON       # Time for each trajectory
        self.eta = 0.75    # friction coef
        self.m = 4.0      # mass of the car
        self.t = 0        # current time

    def step(self, act):
        pt = np.asarray([self.x, self.y])
        vt = np.asarray([self.dx, self.dy])
        ut = np.asarray(act)
        vt_1 = ((1-((self.h*self.eta)/self.m))*vt) + ((self.h/self.m)*ut)
        pt_1 = pt + (self.h*vt_1)
        if(np.linalg.norm(pt_1 - OBSTACLE) < 0.05):
            print("COLLIDED")
            self.dx, self.dy = 0, 0
        self.x, self.y = pt_1[0], pt_1[1]
        self.dx, self.dy = vt_1[0], vt_1[1]
        self.t += 1
        state = np.asarray([self.x, self.y, self.dx, self.dy])
        return state, np.linalg.norm(pt) - np.linalg.norm(pt_1), self.complete(), 0

    def complete(self):
        state = np.asarray([self.x, self.y])
        if(np.linalg.norm(state) < 0.05):
            return True
        if(self.t==self.T):
            return True

    def reset(self):
        self.x = 1.0 + np.random.normal(0.0, 0.025)
        self.y = 0.9 + np.random.normal(0.0, 0.025)
        self.dx = 0.0
        self.dy = 0.0
        self.t = 0
        return np.asarray([self.x, self.y, self.dx, self.dy])

    def action_space_sample(self):
        return np.random.rand(2)
    def observation_space_sample(self):
        return np.random.rand(4)

env = P2D_env()


pos_ind = 0
state_0 = env.reset()
#print("HERE IS STATE 0 ", env.robot_pos, env.goal_pos, env.robot_rot)
print("OBS VAL: ", state_0)
print("this VAL: ", state_0[pos_ind:pos_ind-2])

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
bayes_model = PosteriorModel("Posteriors/BayesDyn_P2D_v1_RUN")

# Load DNN
control_model = PosteriorModel("Posteriors/Control_P2D_v1_RUN", deterministic=True)

trajectories = []
for traj in trange(25, desc="Simulating the Control Loop"):
    done = False
    prev_obs = env.reset()
    time = 0; sim = []
    while(not done):
        act = control_model.predict(np.asarray([prev_obs]))
        act = np.squeeze(act)
        #obs = bayes_model.predict(np.asarray([np.concatenate((prev_obs, act))]))
        #prev_obs = np.squeeze(prev_obs) + np.squeeze(obs)
        prev_obs, _, _, _ = env.step(act)
        sim.append([prev_obs[0], prev_obs[1]])
        if(prev_obs[0] < 0.2 and prev_obs[1] < 0.2):
            print(prev_obs, act)
            break
        print(prev_obs, act)
        time += 1
        if(time > 30):
            print("FAILED TO REACH")
            break
    env.reset()
    time = 0
    trajectories.append(sim)
    print("    DONE    ")

np.save("simulated_P2D_v1_%s"%(10), np.asarray(trajectories))

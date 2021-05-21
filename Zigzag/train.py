#!/usr/bin/env python
# coding: utf-8

# Standard Library Imports
import os
import sys
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

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
GOAL_STATE = [0.0,0.0]
OBSTACLE = [10.00, 10.00]
HORIZON = 25
EPOCHS = 15
EPISODES = 8
EXPLORE = 25
pos_ind = 0

# Custom obstacles for zigzag
tri1 = np.array([[-1,-1], [1,-1], [0,1.25]])*0.5
tri2 = np.array([[-1,1], [1,1], [0,-1.25]])*0.5

obstacles1 = np.asarray([np.asarray([0.5,-1]),
                       np.asarray([1.0,-1])])

obstacles2 = np.asarray([np.asarray([0.25,1]),
                        np.asarray([0.75,0.75]),
                        np.asarray([1.25,1])])


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
        self.x = 1.75 + random.uniform(-0.1, 0.1)
        self.y = -0.5 +  random.uniform(-0.1, 0.1)
        self.theta = -0.30
        self.t = 0
        self.T = 80
        self.col = False

    def step(self, act):
        pre = np.linalg.norm([self.x, self.y]) 
        self.x = self.x + (math.cos(self.theta)*act[0])#*0.1
        self.y = self.y + (math.sin(self.theta)*act[0])#*0.1
        self.theta = self.theta + act[1]
        self.t += 1 

        # Check obstacle 
        collide = self.check_collision([self.x, self.y])
        if(collide == 1):
            print("Collided")
            #self.reset()
            self.col = True
        return [self.x, self.y, self.theta], pre - np.linalg.norm([self.x, self.y]) , self.complete(), 0    

    def complete(self):
        state = np.asarray([self.x, self.y])
        if(np.linalg.norm(state) < 0.15 or self.x < 0.1):
            return True
        if(self.t==self.T or self.t > self.T):
            print("Timeout")
            return True
        return self.col

    def reset(self):
        self.x = 1.75 + random.uniform(-0.1, 0.1)
        self.y = -0.5 + random.uniform(-0.1, 0.1)
        self.theta = -0.30 #-0.35
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
state_0 = env.reset()

# Collect some basic stats about the env
# =======================================================================================
# =======================================================================================
# =======================================================================================
#				SETTING UP THE BNN DYNAMICS MODEL
# =======================================================================================
# =======================================================================================
# =======================================================================================
action_dims = len(env.action_space_sample())
observe_dims = len(env.observation_space_sample()) # + len(env.robot_pos)

print("DIMENSIONS OBS: ", env.observation_space_sample(),  len(env.observation_space_sample()))
print("DIMENSIONS ACT: ", env.action_space_sample(),  len(env.action_space_sample()))

model_input_dims = observe_dims + action_dims
model_output_dims = observe_dims
print(model_input_dims, model_output_dims)

control_input_dims = observe_dims
control_output_dims = action_dims
print(control_input_dims, control_output_dims)

# Set up the BNN structure
model = Sequential()
model.add(Dense(64, input_shape=(model_input_dims,), activation="linear", dtype='float32'))
model.add(Dense(model_output_dims, activation="tanh"))
learning_rate = 0.0125; decay=0.0
opt = optimizers.VariationalOnlineGuassNewton()
loss = tf.keras.losses.mean_squared_error
bayes_model = opt.compile(model, loss_fn=loss, epochs=70, learning_rate=learning_rate,
                          decay=decay, inflate_prior=0.001, mode="regression")


def reward_function(state_0, dyn_pred, goal):
    state_0 = np.squeeze(state_0)
    dyn_pred = np.squeeze(dyn_pred)
    pos = (state_0 + dyn_pred)[:,pos_ind:pos_ind+2]
    penalty = 0
    for obs in obstacles1:
        point = 2
        d = 100/ np.linalg.norm( pos - (tri1+obs)[point], axis=1 ) # distance to point of triangle
        d = np.clip(d, 0, 700)
        penalty -= d
    for obs in obstacles2:
        point = 2
        d = 100/ np.linalg.norm( pos - (tri2+obs)[point], axis=1 ) # distance to point of triangle
        d = np.clip(d, 0, 700)
        penalty -= d
    d = 100/ np.linalg.norm( pos - np.asarray([1.25, -0.75]), axis=1) # distance to point of triangle
    d = np.clip(d, 0, 700)
    penalty -= d
    #print(penalty)
    return -(2.0*np.linalg.norm(pos, axis=1)) + (penalty/800)


def sample_action_sequence(num_seqs, horizon):
    actions_1 = np.asarray([-0.05, -0.025])*1.0
    actions_2 = np.asarray([0.03, 0.015, 0.0075, 0.0, -0.0075])*1.0
    seqs = []
    for i in trange(num_seqs):
        a_seq = []
        for t in range(horizon):
            a_seq.append([np.random.choice(actions_1), np.random.choice(actions_2)])
        seqs.append(a_seq)
    return np.asarray(seqs)

global_action_sequences = sample_action_sequence(8000, 3)

def return_MPC_action(state_0, dyn_model, action_sequences, gamma=0.8):
    rewards = -1
    state = np.asarray([state_0 for i in range(len(action_sequences))])
    seq_reward = 0
    disc = 1.0
    for i in range(len(action_sequences[0])):
        act = action_sequences[:,i]
        inp = np.asarray([np.concatenate((state, act), axis=1)])
        state_new = np.squeeze(dyn_model.predict(inp))
        seq_reward = reward_function(state, state_new, GOAL_STATE)
        disc *= gamma
        state = state_new
        if(type(rewards) == int):
            rewards = disc * seq_reward
        else:
            rewards += disc * seq_reward
    best_seq = np.argmax(rewards, axis=0)
    return action_sequences[best_seq][0]

# Define function to sample {(s,a) -> s} dataset
def gen_dynamics_dataset(env, size, dyn_model, X=[], y=[], fully_rand=False):
    X_train = list(X); y_train = list(y)
    appended = 0
    for traj in trange(size, desc="Generating Dynamics Dataset"):
        done = False
        prev_obs = env.reset()
        #prev_obs = np.concatenate((obs,env.robot_pos))
        while(not done):
            if(traj < 5 or fully_rand==True):
                act = env.action_space_sample()
            else:
                act = return_MPC_action(prev_obs, dyn_model, global_action_sequences)
            obs, reward, done, collisions = env.step(act)
            x = np.concatenate((prev_obs,act))
            y = np.asarray(obs) - np.asarray(prev_obs)
            X_train.append(x); y_train.append(y)
            prev_obs = obs
    return np.asarray(X_train), np.asarray(y_train)

# Define a metric for how we are doing on the reward signal
def estimate_reward(env, dyn_model, size=5):
    reward = 0
    final_state = []
    rot_info = []
    for traj in trange(size):
        done = False
        prev_obs = env.reset()
        counter = 0
        while(not done):
            act = return_MPC_action(prev_obs, dyn_model, global_action_sequences) 
            obs, r, done, collisions = env.step(act)
            print("STATE: %s ACT: %s "%(obs, act))
            rot_info.append(np.squeeze(act)[1])
            reward += r; counter += 1
            prev_obs = obs
        final_state.append(obs)
        print("final state ", obs[pos_ind:pos_ind+2])
    np.save("sampled_actions", global_action_sequences)
    reward/=size
    return reward, np.mean(final_state, axis=0), np.mean(rot_info)


# =======================================================================================
# =======================================================================================
# =======================================================================================
#                    TRAINING THE DYNAMICS MODEL WITH RANDOM SHOOTING
# =======================================================================================
# =======================================================================================
# =======================================================================================

#### Train the BNN on the new dataset
X_dyn_train, y_dyn_train = gen_dynamics_dataset(env, 20, bayes_model, fully_rand=True)
bayes_model.train(X_dyn_train, y_dyn_train, X_dyn_train[0:100], y_dyn_train[0:100])
bayes_model.epochs = int(50)

print("After Initialization REWARD: ", estimate_reward(env, bayes_model, 2))

#### Use controller to explore the real model & collect data
for iteration in range(EPISODES):
    X_dyn_train, y_dyn_train = gen_dynamics_dataset(env, EXPLORE, bayes_model, X_dyn_train, y_dyn_train)
    bayes_model.train(X_dyn_train, y_dyn_train, X_dyn_train[0:100], y_dyn_train[0:100])
    print("After %s REWARD: "%(iteration), estimate_reward(env, bayes_model, 2))
    global_action_sequences = sample_action_sequence(8000, 3)

print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
print(" ~~~DYNAMICS LEARNED. LEARNING CONTROLLER~~~ ")
print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")

# Define function to sample {(s,a) -> s} dataset
def gen_control_dataset(env, size, dyn_model, X=[], y=[], fully_rand=False):
    X_train = list(X); y_train = list(y)
    appended = 0
    for traj in trange(size, desc="Generating Control Dataset"):
        done = False
        prev_obs = env.reset()
        while(not done):
            act = return_MPC_action(prev_obs, dyn_model, global_action_sequences)
            obs, reward, done, collisions = env.step(act)
            x = obs #np.concatenate((prev_obs,act))
            prev_obs = np.squeeze(prev_obs); act = np.squeeze(act)
            X_train.append(np.asarray(prev_obs)) 
            y_train.append(np.asarray(act))
            prev_obs = obs
    return np.asarray(X_train), np.asarray(y_train)


controller = Sequential()
controller.add(Dense(32, input_shape=(control_input_dims,), activation="linear", dtype='float32'))
controller.add(Dense(control_output_dims, activation="tanh"))
learning_rate = 0.00075; decay=0.00
opt = optimizers.StochasticGradientDescent()
loss = tf.keras.losses.mean_squared_error
control_model = opt.compile(controller, loss_fn=loss, epochs=EPOCHS*3, learning_rate=learning_rate,
                          decay=decay, robust_train=0, mode="regression")

X_con_train, y_con_train = [], []
for iteration in range(EPISODES):
    X_con_train, y_con_train = gen_control_dataset(env, EXPLORE*2, bayes_model,X_con_train, y_con_train)
    print(X_con_train.shape, y_con_train.shape)
    control_model.train(X_con_train, y_con_train, X_con_train, y_con_train)

# Save BNN
bayes_model.save("Posteriors/BayesDyn_ZIG_v1_RUN")
# Save DNN
control_model.save("Posteriors/Control_ZIG_v1_RUN")


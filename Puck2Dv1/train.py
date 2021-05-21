# Standard Libarary Imports
import os
import gym
import sys
import random
sys.path.append("..")

# External Libaray Imports
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

# Deep Learning and Bayesian Deep Learning Imports
import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
GOAL_STATE = [0.0,0.0]
OBSTACLE = [0.4,0.5]
#HORIZON = 25
HORIZON = 35
EPOCHS = 25
EPISODES = 25
EXPLORE = 15

pos_ind = 0
"""
The handcrafted 2D point environment we consider
"""
class P2D_env:
    def __init__(self):
        self.x = 1.0+ np.random.normal(0.0, 0.05)
        self.y = 1.0+ np.random.normal(0.0, 0.05)
        self.dx = 0.0
        self.dy = 0.0
        # Point Mass Dynamics Control:
        self.h = 0.35      # time resolution
        self.T = HORIZON       # Time for each trajectory
        self.eta = 0.75    # friction coef
        self.m = 4.0      # mass of the car
        self.t = 0        # current time

    def step(self, act):
        if(self.t > self.T):
            sys.exit(-1)
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
        self.x = 1.0+ np.random.normal(0.0, 0.05)
        self.y = 1.0+ np.random.normal(0.0, 0.05)
        self.dx = 0.0
        self.dy = 0.0
        self.t = 0
        return np.asarray([self.x, self.y, self.dx, self.dy])

    def action_space_sample(self):
        return np.random.rand(2)
    def observation_space_sample(self):
        return np.random.rand(4)


env = P2D_env()
state_0 = env.reset()
print("HERE IS STATE 0 ", state_0)
print("this VAL: ", state_0[pos_ind:pos_ind+2])

state_0 = env.reset()
print("HERE IS STATE 0 ", state_0)
print("this VAL: ", state_0[pos_ind:pos_ind+2])


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
model.add(Dense(32, input_shape=(model_input_dims,), activation="linear", dtype='float32'))
#model.add(Dense(128,  activation="linear", dtype='float32'))
model.add(Dense(model_output_dims, activation="tanh"))
learning_rate = 0.1; decay=0.0
opt = optimizers.VariationalOnlineGuassNewton()
loss = tf.keras.losses.mean_squared_error
bayes_model = opt.compile(model, loss_fn=loss, epochs=35, learning_rate=learning_rate,
                          decay=decay, inflate_prior=0.00001, mode='regression')


def reward_function(state_0, dyn_pred, goal):
    state_0 = np.squeeze(state_0)
    dyn_pred = np.squeeze(dyn_pred)
    state_1 = state_0 + dyn_pred
    #print("State shape: ", state_1.shape)
    d1 = np.linalg.norm(state_0[:,pos_ind:pos_ind+2] - goal, axis=1)
    d2 = np.linalg.norm(state_1[:,pos_ind:pos_ind+2] - goal, axis=1)
    o1 = np.linalg.norm(state_0[:,pos_ind:pos_ind+2] - OBSTACLE, axis=1)
    o2 = np.linalg.norm(state_1[:,pos_ind:pos_ind+2] - OBSTACLE, axis=1)
    #print("REWARD ",d1-d2)
    #d2 = np.linalg.norm(state_1[:,0:2], axis=1)
    #print("Rward shape: ", d2.shape)
    return (-1*(d2-d1)) + ((0.1/o2)*(o2-o1))


def sample_action_sequence(num_seqs, horizon):
    actions_1 = [0.5, 0.25, 0.0, -0.25, -0.5]
    actions_2 = [0.5, 0.25, 0.0, -0.25,-0.5]
    seqs = []
    for i in trange(num_seqs):
        a_seq = []
        for t in range(horizon):
            a_seq.append([np.random.choice(actions_1), np.random.choice(actions_2)])
        seqs.append(a_seq)
    seqs.append(np.asarray(a_seq) * 0.0)
    return np.asarray(seqs)

total_seqs = 7500
global_action_sequences = sample_action_sequence(total_seqs, 10)
# This will probably be super fucking slow to start
# but we can easily generate faster code for this

def return_MPC_action(state_0, dyn_model, action_sequences, gamma=0.8):
    rewards = -1
    state = np.asarray([state_0 for i in range(len(action_sequences))])
    seq_reward = 0
    disc = 1.0
    for i in range(len(action_sequences[0])):
        act = action_sequences[:,i]
        #print(state.shape, act.shape)
        inp = np.asarray([np.concatenate((state, act), axis=1)])
        state_new = np.squeeze(dyn_model.predict(inp))
        seq_reward = reward_function(state, state_new, GOAL_STATE)
        disc *= gamma
        state = state_new
        if(type(rewards) == int):
            rewards = disc * seq_reward
        else:
            rewards += disc * seq_reward
    #print("Rewards: ", rewards)
    best_seq = np.argmax(rewards, axis=0)
    #print("Best action: ", action_sequences[best_seq][0])
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
                #print("MAKING PREDICTION: ", fully_rand==False)
                act = return_MPC_action(prev_obs, dyn_model, global_action_sequences)
            obs, reward, done, collisions = env.step(act)
            #obs = np.concatenate((obs, env.robot_pos))
            x = np.concatenate((prev_obs,act))
            y = obs - prev_obs
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
        #prev_obs = env.robot_pos
        counter = 0
        while(not done):
            act = return_MPC_action(prev_obs, dyn_model, global_action_sequences) 
            obs, r, done, collisions = env.step(act)
            #obs = env.robot_pos
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

#save_trajectory("pretraining")
#### Train the BNN on the new dataset
X_dyn_train, y_dyn_train = gen_dynamics_dataset(env, 20, bayes_model, fully_rand=True)
bayes_model.train(X_dyn_train, y_dyn_train, X_dyn_train[0:100], y_dyn_train[0:100])
bayes_model.epochs = int(0.75*EPOCHS)

print("After Initialization REWARD: ", estimate_reward(env, bayes_model, 2))

#### Use controller to explore the real model & collect data
for iteration in range(EPISODES):
    X_dyn_train, y_dyn_train = gen_dynamics_dataset(env, EXPLORE, bayes_model, X_dyn_train, y_dyn_train)
    bayes_model.train(X_dyn_train, y_dyn_train, X_dyn_train[0:100], y_dyn_train[0:100])
    print("After %s REWARD: "%(iteration), estimate_reward(env, bayes_model, 2))
    global_action_sequences = sample_action_sequence(total_seqs, 4)



"""
FRAME_ARR = save_trajectory("posttraining1", control_model)
import imageio
images = []
for filename in FRAME_ARR:
    images.append(imageio.imread(filename))
imageio.mimsave('Car_v0_Solution.gif', images)
env.close()
"""

print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
print(" ~~~DYNAMICS LEARNED. LEARNING CONTROLLER~~~ ")
print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# =======================================================================================
# =======================================================================================
# =======================================================================================
#                         TRAINING THE CONTROL MODEL WITH DAGGER
# =======================================================================================
# =======================================================================================
# =======================================================================================


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
            #y = obs - prev_obs
            prev_obs = np.squeeze(prev_obs); act = np.squeeze(act)
            #print(prev_obs.shape, act.shape)
            #print(prev_obs, act)
            X_train.append(np.asarray(prev_obs)) 
            y_train.append(np.asarray(act))
            prev_obs = obs
    return np.asarray(X_train), np.asarray(y_train)

controller = Sequential()
controller.add(Dense(32, input_shape=(control_input_dims,), activation="linear", dtype='float32'))
#controller.add(Dense(64, activation="linear"))
controller.add(Dense(control_output_dims, activation="tanh"))
learning_rate = 0.000075; decay=0.00
opt = optimizers.StochasticGradientDescent()
loss = tf.keras.losses.mean_squared_error
control_model = opt.compile(controller, loss_fn=loss, epochs=EPOCHS*3, learning_rate=learning_rate,
                          decay=decay, robust_train=0, mode='regression')

#model.compile(loss='mean_absolute_error',
#                optimizer=tf.keras.optimizers.Adam(0.001))

X_con_train, y_con_train = [], []
for iteration in range(int(8)):
    X_con_train, y_con_train = gen_control_dataset(env, EXPLORE*2, bayes_model,X_con_train, y_con_train)
    print(X_con_train.shape, y_con_train.shape)
    control_model.train(X_con_train, y_con_train, X_con_train, y_con_train)
   

# Save BNN
bayes_model.save("Posteriors/BayesDyn_P2D_v1_RUN")
# Save DNN
control_model.save("Posteriors/Control_P2D_v1_RUN")

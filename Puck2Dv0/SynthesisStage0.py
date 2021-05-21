# Author: Matthew Wicker

# Learning from previously synthesized actions!
import os
#import gym
import copy
import sys
sys.path.append('..')
#import safety_gym
import random

import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

import deepbayesHF
import deepbayesHF.optimizers as optimizers
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import IBP_state
from deepbayesHF.analyzers import IBP_prob_dyn_m as IBP_prob
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *




# Load BNN
b_model = PosteriorModel("BayesDyn_P2D_v0_EXT")
# Load DNN
c_model = PosteriorModel("Control_P2D_v0_EXT", deterministic=True)

# =========================================
import copy
# Load in Dynamics Dataset
X_train, y_train = np.load('Stage0Datasets/DynSynth_X.npy'), np.load('Stage0Datasets/DynSynth_Y.npy')
# print Dyanamics Dataset
X, y = [], []
for i in range(len(X_train)):
#    X.append(X_train[i])
#    y.append(y_train[i]-0.2)
    v = copy.deepcopy(X_train[i])
    #v[2] = -0.25; v[3] = -0.25
    #v[4] = -0.65; v[5] = -0.65
    #y_train[i][-1] = -0.5;
    #y_train[i][-2] = -0.5;
    y_train[i][0] = -0.75;
    y_train[i][1] = -0.75;
    print(v, y_train[i])
    X.append(v)
    y.append(y_train[i])
#print("Bye bye now!")
#sys.exit()

X_train = np.asarray(X)
y_train = np.asarray(y)

# Set up inference problem
model = Sequential()
model.add(Dense(64, input_shape=(6,), activation="linear", dtype='float32'))
#model.add(Dense(128,  activation="linear", dtype='float32'))
model.add(Dense(4, activation="tanh"))
learning_rate = 0.015; decay=0.3
opt = optimizers.VariationalOnlineGuassNewton()
loss = tf.keras.losses.mean_squared_error
bayes_model = opt.compile(model, loss_fn=loss, epochs=25, learning_rate=learning_rate,
                          decay=decay, inflate_prior=0.001, mode='regression')


bayes_model.prior_mean = b_model.posterior_mean
bayes_model.prior_var = b_model.posterior_var
bayes_model.posterior_mean = b_model.posterior_mean
bayes_model.posterior_var = b_model.posterior_var

# Do inference over the dynamics dataset
bayes_model.train(X_train, y_train, X_train, y_train)

# Save the model as a new model
bayes_model.save("BNN_Stage0")

# =========================================
"""
# Load in Control Dataset
X_train, y_train = np.load('Stage0Datasets/ConSynth_X.npy'), np.load('Stage0Datasets/ConSynth_Y.npy')
y_train = np.clip(y_train, -0.125, 0.1)
# print Control Dataset
for i in range(len(X_train)):
    print(X_train[i], y_train[i])

# Set up inference problem
controller = Sequential()
controller.add(Dense(32, input_shape=(4,), activation="linear", dtype='float32'))
#controller.add(Dense(64, activation="linear"))
controller.add(Dense(2, activation="tanh"))
learning_rate = 0.0005; decay=0.00
opt = optimizers.StochasticGradientDescent()
loss = tf.keras.losses.mean_squared_error
control_model = opt.compile(controller, loss_fn=loss, epochs=20, learning_rate=learning_rate,
                          decay=decay, robust_train=0, mode='regression')


control_model.model.set_weights(c_model.posterior_mean)

# Do training over the control dataset
control_model.train(X_train, y_train, X_train, y_train)

# Save the controller as a new controller
#control_model.save("DNN_Stage0")
"""

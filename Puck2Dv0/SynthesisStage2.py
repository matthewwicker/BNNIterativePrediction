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
import copy


c_model = PosteriorModel("Control_P2D_v0_EXT", deterministic=True)
# Load in Control Dataset
X_train, y_train = np.load('ConSynth_X_train.npy'), np.load('ConSynth_Y_train.npy')
y_train = np.clip(y_train, -1.0, 0.0)
# print Control Dataset
#for i in range(len(X_train)):
#    y_train[i][0] -= 0.5
#    y_train[i][1] -= 0.5
#    print(X_train[i], y_train[i])

# Set up inference problem
controller = Sequential()
controller.add(Dense(32, input_shape=(4,), activation="linear", dtype='float32'))
#controller.add(Dense(64, activation="linear"))
controller.add(Dense(2, activation="tanh"))
learning_rate = 0.0005; decay=0.00
opt = optimizers.StochasticGradientDescent()
loss = tf.keras.losses.mean_squared_error
control_model = opt.compile(controller, loss_fn=loss, epochs=150, learning_rate=learning_rate,
                          decay=decay, robust_train=0, mode='regression')


#control_model.prior_mean = c_model.prior_mean
#control_model.posterior_mean = tf.convert_to_tensor(c_model.posterior_mean)
control_model.model.set_weights(c_model.posterior_mean)

# Do training over the control dataset
control_model.train(X_train, y_train, X_train, y_train)

# Save the controller as a new controller
control_model.save("DNN_Stage2")

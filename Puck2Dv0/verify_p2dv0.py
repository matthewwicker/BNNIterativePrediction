# Standard Library Imports
import os
import copy
import sys
import random
sys.path.append('..')

# External Libarary Imports
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

# Deep Learning and Bayesian Deep Learning Imports
import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers
from deepbayes_prealpha import PosteriorModel
from deepbayes_prealpha.analyzers import IBP_state
from deepbayes_prealpha.analyzers import IBP_prob_dyn_m as IBP_prob

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

"""
The handcrafted 2D point environment we consider
"""
class P2D_env:
    def __init__(self):
        self.x = 2.0
        self.y = 2.0
        self.dx = momentum
        self.dy = momentum
        # Point Mass Dynamics Control:
        self.h = 0.5      # time resolution
        self.T = 20       # Time for each trajectory
        self.eta = 0.5    # friction coef
        self.m = 2.0      # mass of the car
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
        self.x = 2.0
        self.y = 2.0
        self.dx = momentum
        self.dy = momentum
        self.t = 0
        return np.asarray([self.x, self.y, self.dx, self.dy])

    def action_space_sample(self):
        return np.random.rand(2)
    def observation_space_sample(self):
        return np.random.rand(4)

pos_ind = 0
momentum = -0.25
GOAL_STATE = [-0.2, -0.2]
env = P2D_env()
state_0 = env.reset()

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
bayes_model = PosteriorModel("Posteriors/BayesDyn_P2D_v0_VER")
# Load DNN
control_model = PosteriorModel("Posteriors/Control_P2D_v0_VER", deterministic=True)

initial_state = state_0

def verify_state(state, eps, predicate, refine=1.0, s=False):
    s0 = np.asarray(state) 				# Lower bound of the states
    s1 = s0 + eps					# Add epsilon to the postion to get the upper bound
    input_to_alg = (s0, s1)

    initial_state[pos_ind:pos_ind+2] = s0		# Making copies of the state with velocity to pass to bounds
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    s0_pre[pos_ind+2:pos_ind+4] -= 2*eps  		# Subtracting epsilon from momentum to get interval over velocity
    diff = s0_pre - s1_pre

    # Propagating the intervals through the controller to get action intervals
    low, high = np.asarray([s0_pre])+0.5*diff, np.asarray([s0_pre])-0.5*diff
    act = control_model.predict(np.asarray([s0_pre]))
    act_l, act_u = IBP_state(control_model, low, high, control_model.model.get_weights())
    print("Actions: ", act_l, act_u)
    act_l = np.squeeze(act_l)
    act_u = np.squeeze(act_u)

    # Adding the action intervals to state intervals to get full intervals
    s0 = np.concatenate((s0_pre,act_l))
    s1 = np.concatenate((s1_pre,act_u))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    # Computing the probability by propagating intervals through the BNN
    if(refine < 5.0):
        p, outs = IBP_prob(bayes_model, s0, s1, 1.65*(1/refine), int(95*(refine**2)), predicate, inflate=1.0)
    else:
        refine /= 2.0
        p, outs = IBP_prob(bayes_model, s0, s1, 1.85*(1/refine), int(100*(refine**2)), predicate, inflate=1.0)

    return p, outs


def predicate(source_l, source_u, state_l, state_u):

    source_l = source_l[0:observe_dims]
    source_u = source_u[0:observe_dims]

    state_l = source_l + state_l		# Lower Bound on next state
    state_u = source_u + state_u		# Upper Bound on next state
    collision = True				# No collision in this model, so set to True (= Safe)

    # If the upper bound of the postion is in the safe region then we are safe
    goal = (state_u[pos_ind:pos_ind+2] <= SAFE_REGION).all()
    # If all of the velocities are stillsafe then we are safe
    velo = (state_u[pos_ind+2:pos_ind+4] <= momentum).all()
    # Return the condition that position and velocities are safe
    return collision and goal and velo



# Setting up the state space discretization for the verification


from tqdm import tqdm

SAFE_REGION = 0.2			# Goal region (Initial safe region)
states = 50				# Number of states to verify
end_point = 0.65			# End point of the state space to consider
eps = end_point/(states)		# Epsilon (size of each grid space)
non_zero = []
ind_i, ind_j = 0,0
multiplier = 1.0
pre_global_probas = np.zeros((states,states))
print("k = ", states-int(SAFE_REGION/eps) + 1)
global_probas = np.zeros((states,states))
for k in trange(states-int(SAFE_REGION/eps) + 1):
    probas = []
    p_global_probas = []
#    for i in np.flip(np.linspace(0, end_point, num=states)):
    for i in np.linspace(0, end_point, num=states):
        for j in tqdm(np.linspace(0, end_point, num=states)[::-1]):
            #i = 0.30; j=0.0
            #SAFE_REGION = 0.285
            if(global_probas[ind_i][ind_j] != 0):
                probas.append(global_probas[ind_i][ind_j])
                p_global_probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            elif((i < SAFE_REGION and j < SAFE_REGION)):
                probas.append(1.0)
                p_global_probas.append(1.0)
                ind_j += 1; continue
            if(i > (SAFE_REGION + eps) or j > (SAFE_REGION + eps)):
                probas.append(0.0)
                p_global_probas.append(0.0)
                ind_j += 1; continue
            print("State: (%s, %s) (eps: %s) (wc: %s)"%(i,j,eps, multiplier))
            p, outs = verify_state([i, j], eps, predicate, refine=2.0)
            if(p < 0.95):
                p, outs = verify_state([i, j], eps, predicate, refine=5.1)
                if(p < 0.95):
                    p, outs = verify_state([i, j], eps, predicate, refine=6.0)
            print(" ")
            print(" ")
            print("Probability: %s"%(p))
            print(" ")
            print(" ")

            p_global_probas.append(p)
            p *= multiplier
            probas.append(p)
            non_zero.append(p)
            ind_j += 1
        ind_i += 1
        ind_j = 0
    SAFE_REGION += eps
    ind_i = 0
    probas = np.reshape(probas, (states,states))
    p_global_probas = np.reshape(p_global_probas, (states,states))
    global_probas = np.maximum(global_probas, probas)
    pre_global_probas = np.maximum(p_global_probas, pre_global_probas)
    multiplier = min(non_zero)
    print("======================================")
    print("==========EXPANDING GOAL==============")
    print("======================================")
    probas = np.asarray(probas)
    probas = probas.reshape(states,states)
    np.save("probs_P2D_v0_%s_%s"%(states, eps), probas)

probas = np.asarray(probas)
probas = probas.reshape(states,states)
np.save("probs_P2D_v0_%s_%s"%(states, eps), probas)




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

pos_ind = 0
momentum = -0.25
GOAL_STATE = [-0.2, -0.2]


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
#bayes_model = PosteriorModel("BayesDyn_P2D_v0_EXT")
bayes_model = PosteriorModel("BNN_Stage0")
#bayes_model = PosteriorModel("SynthDynBNN")

# Load DNN
#control_model = PosteriorModel("Control_P2D_v0_EXT", deterministic=True)
control_model = PosteriorModel("DNN_Stage0", deterministic=True)
control_model_synth = PosteriorModel("DNN_Stage2", deterministic=True)
#control_model = PosteriorModel("PoorConDNN", deterministic=True)

initial_state = state_0


"""
Below is all of the code to do optimization in the action space of our
BNN dynamics model wrt the output of the DNN control model. This takes
the probabalistic safety into account during our verification and saves
a dataset of new, synthesized actions on which the controller is trained

This new controller can then be re-verified using this same script but
passing the optimized controller.
"""


from deepbayesHF.analyzers import IBP_fixed_w

def reward_function(state_0, dyn_pred, goal):
    state_0 = tf.cast(tf.squeeze(state_0), dtype=tf.float32)
    dyn_pred = tf.cast(tf.squeeze(dyn_pred), dtype=tf.float32)
    state_1 = state_0 + dyn_pred

    d1 = tf.norm(state_0[pos_ind:pos_ind+2] - goal, axis=0)
    d2 = tf.norm(state_1[pos_ind:pos_ind+2] - goal, axis=0)

    #o1 = tf.norm(state_0[pos_ind:pos_ind+2] - OBSTACLE, axis=0)
    #o2 = tf.norm(state_1[pos_ind:pos_ind+2] - OBSTACLE, axis=0)
    #return (-1*(d2-d1)) + ((0.1/o2)*(o2-o1))

    return (-1*(d2-d1))


def gradient_expectation(model, inp, loss_fn, num_models=10):
    gradient_sum = tf.zeros(inp.shape)
    inp = tf.convert_to_tensor(inp)
    val = num_models
    if(num_models < 1):
        num_models = 1
    for i in range(num_models):
        if(model.det or val == -1):
            no_op = 0
        else:
            model.model.set_weights(model.sample())
        # Establish Gradient Tape Context (for input this time)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inp)
            # Get the probabilities
            predictions = model.predict(inp)
            loss = loss_fn([tf.squeeze(inp)[0:-2]], predictions, GOAL_STATE)
            #predictions = model.predict(inp)
            #predictions = tf.concat(predictions, tf.convert_to_tensor([0.0, 0.0]) ) 
            #loss = loss_fn(inp, predictions, GOAL_STATE)
        # Get the gradients
        inp_gradient = tape.gradient(loss, inp)
        #print("GRAD: ", inp_gradient)
        try:
            gradient_sum += inp_gradient
        except:
            gradient_sum += tf.cast(inp_gradient, 'float32')
        if(model.det or val == -1):
            break
    return gradient_sum

def gradient_w(model, inp, loss_fn, w):
    gradient_sum = tf.zeros(np.asarray(inp).shape)
    inp = tf.convert_to_tensor(inp)
    model.model.set_weights(w)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inp)
        predictions = model.predict(inp)
        loss = loss_fn([tf.squeeze(inp)[0:-2]], predictions, GOAL_STATE)
    inp_gradient = tape.gradient(loss, inp)
    inp_gradient = tf.cast(inp_gradient, 'float32')
    return tf.squeeze(inp_gradient)[-2:]


def synth_state(state, eps, predicate, loss=None, grad_iters=35):
    if(loss is None):
        loss = reward_function # This is gobally defined
    s0 = np.asarray(state) # Lower bound
    s1 = s0 + eps
    input_to_alg = (s0, s1)

    initial_state[pos_ind:pos_ind+2] = s0
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    s0_pre[pos_ind+2:pos_ind+4] -= 2*eps
    diff = s0_pre - s1_pre

    act = control_model.predict(np.asarray([s0_pre]))

    act = np.squeeze(act)
    original_act = copy.deepcopy(act)

    s0 = np.concatenate((s0_pre,act))
    s1 = np.concatenate((s1_pre,act))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    #for i in trange(grad_iters, desc="Synthesizing Action"):
    for i in range(grad_iters):
        act = np.squeeze(act)
        s0 = np.concatenate((s0_pre,act))
        s1 = np.concatenate((s1_pre,act))
        s0 = np.asarray([s0])
        s1 = np.asarray([s1])

        w = bayes_model.sample()
        p, outs = IBP_fixed_w(bayes_model, s0, s1, 2.0, w, predicate)
        grad = gradient_w(bayes_model, (s0+s1)/2, loss, w)
        #print(act, p, grad)
        #a = act + (math.ceil(p)*0.5 * grad)
        #a = act - (1.5 * grad[::-1])
        a = act + (1.0 * grad)
        act = a
    print("RESTUL OF SYNTH: ", act)
    return act, original_act

# Here are the dataset variables that we will track during the initial verification

DYNAMICS_X = []
DYNAMICS_Y = []

CONTROL_X = []
CONTROL_Y = []


"""
Below is a modified version of the verification proceedure with a subroutine
for generating synthesized actions. These synthezied actions are then stored
in the dataset (or can be applied to the verification) in order to develop
a safer controller (wrt probabalistic safety).
"""

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
    if(s):
        act = control_model_synth.predict(np.asarray([s0_pre]))
        act_l, act_u = IBP_state(control_model_synth, low, high, control_model_synth.model.get_weights())
        act_l = np.squeeze(act_l)#-0.175
        act_u = np.squeeze(act_u)#-0.175
        print("Synth Actions: ", act_l, act_u)

    else:
        act = control_model.predict(np.asarray([s0_pre]))
        act_l, act_u = IBP_state(control_model, low, high, control_model.model.get_weights())
        act_l = np.squeeze(act_l)#-0.175
        act_u = np.squeeze(act_u)#-0.175
        print("Actions: ", act_l, act_u)


    # Adding the action intervals to state intervals to get full intervals
    s0 = np.concatenate((s0_pre,act_l))
    s1 = np.concatenate((s1_pre,act_u))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    # Computing the probability by propagating intervals through the BNN
    p, outs = IBP_prob(bayes_model, s0, s1, 3.5*(1/refine), int(150*(refine**2)), predicate, inflate=1.0)

    return p, outs


def predicate(source_l, source_u, state_l, state_u):

    source_l = source_l[0:observe_dims]
    source_u = source_u[0:observe_dims]

    #state_l = source_l + state_l		# Lower Bound on next state
    #state_u = source_u + state_u		# Upper Bound on next state
    collision = True				# No collision in this model, so set to True (= Safe)

    #print(state_u)
    # If the upper bound of the postion is in the safe region then we are safe
    goal = (state_u[pos_ind:pos_ind+2] <= SAFE_REGION).all()
    # If all of the velocities are stillsafe then we are safe
    #velo = (state_u[pos_ind+2:pos_ind+4] <= momentum).all()
    # Return the condition that position and velocities are safe
    return collision and goal #and velo



# Setting up the state space discretization for the verification


from tqdm import tqdm
INIT_SAFE_REGION = 0.2
SAFE_REGION = 0.2			# Goal region (Initial safe region)
states = 15				# Number of states to verify
end_point = 0.30			# End point of the state space to consider
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
    for i in np.linspace(0, end_point, num=states):
        for j in tqdm(np.linspace(0, end_point, num=states)[::-1]):
            if(global_probas[ind_i][ind_j] != 0):
                probas.append(global_probas[ind_i][ind_j])
                p_global_probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            elif((i <INIT_SAFE_REGION and j < INIT_SAFE_REGION)):
                probas.append(1.0)
                p_global_probas.append(1.0)
                ind_j += 1; continue
            if(i > (SAFE_REGION + eps) or j > (SAFE_REGION + eps)):
                probas.append(0.0)
                p_global_probas.append(0.0)
                ind_j += 1; continue
            print("State: (%s, %s) (eps: %s) (wc: %s)"%(i,j,eps, multiplier))

            p, outs = verify_state([i, j], eps, predicate, refine=3.5, s=False)
            p1, outs1 = verify_state([i, j], eps, predicate, refine=3.5, s=True)

            print(" ")
            print(" ")
            print("PROB UNSYNTH: %s PROB SYNTH: %s"%(p, p1))
            print(" ")
            print(" ")

            p_global_probas.append(p1)
            #p *= multiplier # - We dont consider this in the synthesis part just to visualize the per-state improvement easier
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

    print("======================================")
    print("==========EXPANDING GOAL==============")
    print("======================================")

    np.save("presynth_P2D_v0_%s_%s"%(states, eps), global_probas)
    np.save("postsynth_P2D_v0_%s_%s"%(states, eps), pre_global_probas)

probas = np.asarray(probas)
probas = probas.reshape(states,states)
#np.save("presynth_probas_P2D_v0_%s_%s"%(states, eps), probas)

### RETRAINING OF MODEL AND CONTROLLER



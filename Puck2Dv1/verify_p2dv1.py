# Standard Library Imports
import os
import sys
import math
import copy
import random
sys.path.append('..')

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
from deepbayes_prealpha.analyzers import IBP_fixed_w
from deepbayes_prealpha.analyzers import IBP_prob_dyn as IBP_prob
from deepbayes_prealpha.analyzers import IBP_prob_dyn
from deepbayes_prealpha.analyzers import IBP_prob_dyn_m

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

pos_ind = 0
momentum = 0.0
OBSTACLE = [0.4,0.5]
GOAL_STATE = [-0.2,-0.2]
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
bayes_model = PosteriorModel("Posteriors/BayesDyn_P2D_v1_VER")
#bayes_model = PosteriorModel("BNN_Stage0")

# Load DNN
control_model = PosteriorModel("Posteriors/Control_P2D_v1_VER", deterministic=True)
#control_model = PosteriorModel("DNN_Stage0", deterministic=True)


def reward_function(state_0, dyn_pred, goal):
    state_0 = tf.cast(tf.squeeze(state_0), dtype=tf.float32)
    dyn_pred = tf.cast(tf.squeeze(dyn_pred), dtype=tf.float32)
    state_1 = state_0 + dyn_pred
    #print("State shape: ", state_1.shape)
    d1 = tf.norm(state_0[pos_ind:pos_ind+2] - goal, axis=0)
    d2 = tf.norm(state_1[pos_ind:pos_ind+2] - goal, axis=0)

    o1 = tf.norm(state_0[pos_ind:pos_ind+2] - OBSTACLE, axis=0)
    o2 = tf.norm(state_1[pos_ind:pos_ind+2] - OBSTACLE, axis=0)
    return (-1*(d2-d1)) + ((0.1/o2)*(o2-o1))

    #return (-1*(d2-d1))

# ============================================
#       Generate input gradient feild
# ============================================

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

#initial_state = [2.0, 2.0, -0.185, -0.185]

# Lets try to do a full IBP pass through these networks

def synth_state(state, eps, predicate, loss=None, grad_iters=20):
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
        p, outs = IBP_fixed_w(bayes_model, s0, s1, 2.25, w, predicate)
        grad = gradient_w(bayes_model, (s0+s1)/2, loss, w)
        #print(act, p, grad)
        #a = act + (math.ceil(p)*0.5 * grad)
        #a = act - (1.5 * grad[::-1])
        a = act + (1.5 * grad)
        act = a
    print("RESTUL OF SYNTH: ", original_act, act)
    return act, original_act

DYNAMICS_X = []
DYNAMICS_Y = []
CONTROL_X = []
CONTROL_Y = []

initial_state = state_0

# Lets try to do a full IBP pass through these networks
def verify_state(state, eps, predicate, p_arr, refine=2.5, s=False):

    s0 = np.asarray(state)                              # Lower bound of the states
    s1 = s0 + eps                                       # Add epsilon to the postion to get the upper bound
    input_to_alg = (s0, s1)

    initial_state[pos_ind:pos_ind+2] = s0               # Making copies of the state with velocity to pass to bounds
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    s0_pre[pos_ind+2:pos_ind+4] -= 2*eps                # Subtracting epsilon from momentum to get interval over velocity
    diff = s0_pre - s1_pre

    # Propagating the intervals through the controller to get action intervals
    low, high = np.asarray([s0_pre])+0.5*diff, np.asarray([s0_pre])-0.5*diff
    act = control_model.predict(np.asarray([s0_pre]))
    act_l, act_u = IBP_state(control_model, low, high, control_model.model.get_weights())
    print("Actions: ", act_l, act_u)
    act_l = np.squeeze(act_l)
    act_u = np.squeeze(act_u)

    if(s):
        global DYNAMICS_X
        global DYNAMICS_Y
        global CONTROL_X
        global CONTROL_Y
        act, _ = synth_state(state, eps, predicate, grad_iters=int(25*refine))
        act_l = np.squeeze(act)
        act_u = np.squeeze(act)
        print("SYNTHED ACTION: ", act_l, act_u)
        CONTROL_X.append(s0_pre); CONTROL_Y.append(act_u)
        res0 = np.concatenate((np.asarray(state) - SAFE_REGION -0.1, act_l))
        res1 = np.concatenate((np.asarray(state) - SAFE_REGION - 0.1, act_l))
        s0 = np.concatenate((s0_pre,act_l))
        s1 = np.concatenate((s1_pre,act_u))
        CONTROL_X.append(s0_pre)
        CONTROL_X.append(s1_pre)
        CONTROL_Y.append(act_u)
        CONTROL_Y.append(act_l)
        DYNAMICS_X.append(s0); DYNAMICS_Y.append(res0)
        DYNAMICS_X.append(s1); DYNAMICS_Y.append(res1)

    # Adding the action intervals to state intervals to get full intervals
    s0 = np.concatenate((s0_pre,act_l))
    s1 = np.concatenate((s1_pre,act_u))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    # Computing the probability by propagating intervals through the BNN
    #if(refine < 5.0):
    #    p, outs = IBP_prob(bayes_model, s0, s1, 1.95*(1/refine), int(95*(refine**2)), predicate, inflate=1.0)
    #else:
    #    print("!!!!!!!!!!!!!!!", refine)
    #    refine /= 2.0
    #    p, outs = IBP_prob(bayes_model, s0, s1, 1.95*(1/refine), int(100*(refine**2)), predicate, inflate=1.0)


    p, outs = IBP_prob_dyn_m(bayes_model, s0, s1, 1.75*(1/refine), int(50*(refine**2)), predicate)
    if(p < 0.9):
        refine += 1.0
        p, outs = IBP_prob_dyn_m(bayes_model, s0, s1, 2.5*(1/refine), int(125*(refine**2)), predicate)
        if(p < 0.9):
            refine += 1.0
            p, outs = IBP_prob_dyn_m(bayes_model, s0, s1, 2.5*(1/refine), int(125*(refine**2)), predicate)
            if(p < 0.9):
                refine += 1.0
                p, outs = IBP_prob_dyn_m(bayes_model, s0, s1, 2.25*(1/refine), int(250*(refine**2)), predicate)
    if(p == 0):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~NOT REACHABLE~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return 0.00000000000000000001

    #Get max and mins here!
    mins = [i[0] for i in outs]
    maxs = [i[1] for i in outs]
    mins = np.min(mins, axis=0)[pos_ind:pos_ind+2]
    maxs = np.max(maxs, axis=0)[pos_ind:pos_ind+2]
    #maxs[0] = min(maxs[0]-2*eps, np.squeeze(s1)[0]-2*eps)
    #maxs[1] = min(maxs[1]-2*eps, np.squeeze(s1)[1]-2*eps)
    maxs[0] = min(maxs[0], np.squeeze(s1)[0])
    maxs[1] = min(maxs[1], np.squeeze(s1)[1])
    # Account for first iteration here:
    print("MAXES: ", maxs)
    print("MINS: ", mins)
    if((maxs < INIT_SAFE_ZONE).all()):
        print("One step reachable. (p=%s)"%(p))
        return p
    else:
        print("CHECK THIS INTERVAL: ", mins, maxs)
        mi_i, ma_i = math.floor(mins[0]/eps), math.ceil(maxs[0]/eps)
        #mi_j, ma_j = math.floor((mins[1]+0.5)/eps), math.ceil((maxs[1]+0.5)/eps)
        mi_j, ma_j = math.floor((mins[1])/eps), math.ceil((maxs[1])/eps)
        # index clipping
        mi_i, ma_i = max(0, mi_i), max(0, ma_i)
        mi_j, ma_j = max(0, mi_j), max(0, ma_j)
        mi_i, ma_i = min(len(p_arr), mi_i), min(len(p_arr), ma_i)-1
        mi_j, ma_j = min(len(p_arr), mi_j), min(len(p_arr), ma_j)-1
        mi_i, ma_i = max(0, mi_i), max(0, ma_i)
        mi_j, ma_j = max(0, mi_j), max(0, ma_j)
        #mi_j, ma_j = -1*mi_j, -1*ma_j
        print("X ", mi_i, ma_i)
        print("Y ", mi_j, ma_j)
        if(mi_i == 0 and ma_i == 0):
#            return p
            ma_i += 1
        if(mi_j == 0 and ma_j == 0):
#            return p 
            ma_j += 1
        try:
            worst_future = np.min(np.asarray(p_arr[mi_i:ma_i,mi_j:ma_j]).flatten())
        except:
            worst_future = 0.0 #p_arr[mi_i,mi_j]
        print("Multi-step reachable. Worst case: %s (p=%s)"%(worst_future, p))
        print(mins, maxs)
        if(worst_future == 0.0):
            return 0.00000000000000000001
        if(p == 0):
            return 0.00000000000000000001
        return p * worst_future


SAFE_REGION = 0.2
INIT_SAFE_ZONE = 0.2

def label(state_l, state_u):
    collision = True
    # Overlap, side
    if(not (state_u[0] <= (OBSTACLE[0]-0.025) or state_l[0] >= (OBSTACLE[0]+0.025)
            or state_u[1] <= (OBSTACLE[1] - 0.025) or state_l[1] >= (OBSTACLE[1] + 0.025))):
        collision = False
    # Obstacle subsumed
    if(state_u[0] < (OBSTACLE[0]+0.025) and state_l[0] > (OBSTACLE[0]-0.025) and 
          state_u[1] < (OBSTACLE[1]+0.025) and state_l[1] > (OBSTACLE[1]-0.025)):
        collision = False
    # State subsumed
    if(state_u[0] > (OBSTACLE[0]+0.025) and state_l[0] < (OBSTACLE[0]-0.025) and 
          state_u[1] > (OBSTACLE[1]+0.025) and state_l[1] < (OBSTACLE[1]-0.025)):
        collision = False
    # Point intersection
    def point_intersection(low, high, pt) :
        if (pt[0] > low[0] and pt[0] < high[0] and pt[1] > low[1] and pt[1] < high[1]):
            return False
        else:
            return True
    pts = [state_l, state_u, [state_l[0], state_u[1]], [state_u[0], state_l[1]]]
    for pt in pts:
        collision = (collision and point_intersection(OBSTACLE, np.asarray(OBSTACLE)+0.025, pt))
    return collision


def predicate(source_l, source_u, state_l, state_u):
    source_l = source_l[0:observe_dims]
    source_u = source_u[0:observe_dims]

    # Start by assuming we are safe wrt the obstacle collision critera
    collision = True

    # Check if we overlap with the obstacle, if so we are unssafe
    collision = collision and label(source_l, source_u)
    collision = collision and label(state_l, state_u)

    # Check that we make it to the safe region in the worst case
    goal = (state_u[pos_ind:pos_ind+2] <= SAFE_REGION).all()

    # Check that the velocity constraint is satisfied
    velo = (state_u[pos_ind+2:pos_ind+4] <= 0.0).all()

    # Return True if all of the conditions are satisfied
    return collision and goal and velo

from tqdm import tqdm
INIT_SAFE_REGION = 0.2
states = 51
end_point = 1.00
eps = end_point/(states-1)
non_zero = []
ind_i, ind_j = 0,0
multiplier = 1.0
global_probas = np.zeros((states,states))
for k in trange(states-int(SAFE_REGION/eps) + 1):
    probas = []
    for i in np.linspace(0, end_point, num=states):
        for j in tqdm(np.linspace(0, end_point, num=states)):
            if(global_probas[ind_i][ind_j] != 0):
                probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            elif((i < INIT_SAFE_REGION and j < INIT_SAFE_REGION)):
                probas.append(1.0)
                ind_j += 1; continue
            if(i > (SAFE_REGION + eps/1.5) or j > (SAFE_REGION + eps/1.5)):
                probas.append(0.0)
                ind_j += 1; continue
            print("State: (%s, %s) (eps: %s)"%(i,j,eps))
            p = verify_state([i, j], eps, predicate, global_probas, refine=2.0, s=False)

            print(" ")
            print(" ")
            print("Probability: %s"%(p))
            print(" ")
            print(" ")

            probas.append(p)
            non_zero.append(p)
            ind_j += 1
        ind_i += 1
        ind_j = 0
    SAFE_REGION += eps
    ind_i = 0
    probas = np.reshape(probas, (states,states))
    #global_zeros = np.argwhere(global_probas < 0.05)
    probas_vals = np.argwhere(probas > 0.00)
    #global_probas = np.maximum(global_probas, probas)
    #global_probas[global_zeros] = 0.00
    global_probas[probas_vals] = probas[probas_vals]
    print("======================================")
    print("==========EXPANDING GOAL==============")
    print("======================================")
    probas = np.asarray(probas)
    probas = probas.reshape(states,states)
    np.save("probs_P2D_v0_%s_%s"%(states, eps), probas)

probas = np.asarray(probas)
probas = probas.reshape(states,states)
np.save("probs_P2D_v0_%s_%s"%(states, eps), probas)


# Standard Library Imports
import os
import sys
import copy
import math
import random
sys.path.append("..")

# External Library Imports
import numpy as np
from tqdm import trange
import matplotlib
import matplotlib.pyplot as plt

# Deep Learning and Bayesian Deep Learning Imports
import deepbayes_prealpha
import deepbayes_prealpha.optimizers as optimizers
from deepbayes_prealpha import PosteriorModel
from deepbayes_prealpha.analyzers import IBP_state
from deepbayes_prealpha.analyzers import IBP_fixed_w
from deepbayes_prealpha.analyzers import IBP_prob_dyn_m as IBP_prob

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
INIT_SAFE_ZONE = 0.2
OBSTACLE = [0.4,0.5]
GOAL_STATE = [-0.2, 0.0]
pos_ind = 0
momentum = -0.15

# Custom obstacles for zigzag
tri1 = np.array([[-1,-1], [1,-1], [0,1.5]])*0.5
tri2 = np.array([[-1,1], [1,1], [0,-1.5]])*0.5

obstacles1 = np.asarray([np.asarray([0.5,-1]),
                       np.asarray([1.0,-1])])
obstacles2 = np.asarray([np.asarray([0.25,1]),
                        np.asarray([0.75,1]),
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
        self.x = 1.75
        self.y = -0.5
        self.theta = -0.5
        self.t = 0
        self.T = 30
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
        self.x = 1.75
        self.y = -0.5
        self.theta = -0.5
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
state_0 = env.reset()
initial_state = state_0 # This is now the template for the "STATE" that we will pass throughout
action_dims = len(env.action_space_sample())
observe_dims = len(env.observation_space_sample())

print("DIMENSIONS OBS: ", env.observation_space_sample(),  len(env.observation_space_sample()))
print("DIMENSIONS ACT: ", env.action_space_sample(),  len(env.action_space_sample()))

model_input_dims = observe_dims + action_dims
model_output_dims = observe_dims

control_input_dims = observe_dims
control_output_dims = action_dims


# Load BNN
#bayes_model = PosteriorModel("BNN_Stage0")
bayes_model = PosteriorModel("Posteriors/BayesDyn_ZIG_v1_RUN")
# Load DNN
control_model = PosteriorModel("Posteriors/Control_ZIG_v1_RUN", deterministic=True)
initial_state = state_0

"""
Synthesis of States
"""
def reward_function(state_0, dyn_pred, goal):
    state_0 = tf.squeeze(state_0)
    dyn_pred = tf.squeeze(dyn_pred)
    state_0 = tf.dtypes.cast(state_0, tf.float32)
    dyn_pred = tf.dtypes.cast(dyn_pred, tf.float32)
    state_1 = state_0 + dyn_pred
    d1 = tf.norm(state_0[pos_ind:pos_ind+2])
    d2 = tf.norm(state_1[pos_ind:pos_ind+2])
    distance_from_middle = state_1[1]
    distance_x = state_1[0]
    return (-1*(d2-d1)) + (tf.math.abs(distance_from_middle)*1.5) # - (distance_x*0.1) # + ((0.5/o2)*(o2-o1))


# Define a metric for how we are doing on the reward signal
def estimate_reward(env, dyn_model, con_model, size=1):
    reward = 0
    final_state = []
    rot_info = []
    for traj in trange(size):
        done = False
        prev_obs = env.reset()
        counter = 0
        while(not done):
            act = con_model.predict(np.asarray([prev_obs])) #return_MPC_action(prev_obs, dyn_model, global_action_sequences) 
            obs, r, done, collisions = env.step(act)
            print(list(obs), obs[0:2])
            rot_info.append(np.squeeze(act)[1])
            reward += r; counter += 1
            prev_obs = obs
        final_state.append(obs)
        print("final state ", obs[0:2])
    reward/=size
    return reward, np.mean(final_state, axis=0), np.mean(rot_info)


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
        # Get the gradients
        inp_gradient = tape.gradient(loss, inp)
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
        predictions = model.model(inp)
        loss = loss_fn([tf.squeeze(inp)[0:-2]], predictions, GOAL_STATE)
    inp_gradient = tape.gradient(loss, inp)
    inp_gradient = tf.cast(inp_gradient, 'float32')
    return tf.squeeze(inp_gradient)[-2:]


# Lets try to do a full IBP pass through these networks
def synth_state(state, eps, predicate, loss=None, grad_iters=35):
    if(loss is None):
        loss = reward_function
    s0 = np.asarray(state) # Lower bound
    s1 = s0 + eps
    input_to_alg = (s0, s1)
    initial_state[pos_ind:pos_ind+2] = s0
    initial_state[pos_ind+2:pos_ind+3] -= 0.1 #s0[1]
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)

    act = control_model.predict(np.asarray([s0_pre]))

    act = np.squeeze(act)
    original_act = copy.deepcopy(act)

    s0 = np.concatenate((s0_pre,act))
    s1 = np.concatenate((s1_pre,act))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    for i in trange(grad_iters, desc="Synthesizing Action"):
        act = np.squeeze(act)
        s0 = np.concatenate((s0_pre,act))
        s1 = np.concatenate((s1_pre,act))
        s0 = np.asarray([s0])
        s1 = np.asarray([s1])

        w = bayes_model.sample()
        p, outs = IBP_fixed_w(bayes_model, s0, s1, 1.25, w, predicate)
        s0, s1 = tf.convert_to_tensor(s0), tf.convert_to_tensor(s1)
        grad = gradient_w(bayes_model, (s0+s1)/2, loss, w)
        a = act + (0.5 * grad)
        act = a
    print("RESTUL OF SYNTH: ", original_act, act)
    return act, original_act


DYNAMICS_X = []
DYNAMICS_Y = []

# Lets try to do a full IBP pass through these networks
def verify_state(state, eps, predicate, p_arr, refine=2.0):

    initial_state = env.reset()
    s0 = np.asarray(state) # Lower bound
    s1 = s0 + eps
    input_to_alg = (s0, s1)

    i_s = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s0
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    initial_state = i_s

    act = control_model.predict(np.asarray([s0_pre]))
    diff = s0_pre - s1_pre
    low, high = np.asarray([s0_pre])+0.5*diff, np.asarray([s0_pre])-0.5*diff
    act = control_model.predict(np.asarray([s0_pre]))
    act_l, act_u = IBP_state(control_model, low, high, control_model.model.get_weights())

    act = np.squeeze(act)
    act_l = np.squeeze(act_l)
    act_u = np.squeeze(act_u)

    s0 = np.concatenate((s0_pre,act_l))
    s1 = np.concatenate((s1_pre,act_u))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    print("Action: ", act_l, act_u)

    p, outs = IBP_prob(bayes_model, s0, s1, 1.5*(1/refine), int(125*(refine**2)), predicate)
    if(p < 0.9):
        refine += 1.0
        p, outs = IBP_prob(bayes_model, s0, s1, 1.5*(1/refine), int(125*(refine**2)), predicate)
    if(p == 0):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~NOT REACHABLE~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return 0
    #Get max and mins here!
    mins = [i[0] for i in outs]
    maxs = [i[1] for i in outs]
    mins = np.min(mins, axis=0)[pos_ind:pos_ind+2]
    maxs = np.max(maxs, axis=0)[pos_ind:pos_ind+2]
    # Account for potential backwards momentum:
    #print("******")
    #print(maxs[0], np.squeeze(s1)[0])
    #print(maxs[1], np.squeeze(s1)[1])
    maxs[0] = min(maxs[0], np.squeeze(s1)[0])-eps
    maxs[1] = min(maxs[1], np.squeeze(s1)[1])-eps
    print("MAXES: ", maxs)
    # Account for first iteration here:
    if((maxs < INIT_SAFE_ZONE).all()):
        print("One step reachable. (p=%s)"%(p))
        return p
    else:
        mi_i, ma_i = math.floor(mins[0]/eps), math.ceil(maxs[0]/eps)
        mi_j, ma_j = math.floor(mins[1]/eps), math.ceil(maxs[1]/eps)
        # index clipping
        mi_i, ma_i = max(0, mi_i), max(0, ma_i)
        mi_j, ma_j = max(0, mi_j), max(0, ma_j)
        mi_i, ma_i = min(len(p_arr), mi_i), min(len(p_arr), ma_i)
        mi_j, ma_j = min(len(p_arr), mi_j), min(len(p_arr), ma_j)
        print("X ", mi_i, ma_i)
        print("Y ", mi_j, ma_j)
        if(mi_i == 0 and ma_i == 0):
            return 1.0
            ma_i += 1
        if(mi_j == 0 and ma_j == 0):
            return 1.0 
            ma_j += 1
        
        worst_future = np.min(np.asarray(p_arr[mi_i:ma_i,mi_j:ma_j]).flatten())
        print("Multi-step reachable. Worst case: %s (p=%s)"%(worst_future, p))
        print(mins, maxs)
        if(worst_future ==0.0):
            return -1
        return p * worst_future

SAFE_REGION = INIT_SAFE_ZONE
def predicate(source_l, source_u, state_l, state_u):
    collision = True
    points_to_check = [state_u[pos_ind:pos_ind+2], state_l[pos_ind:pos_ind+2], 
                          source_u[pos_ind:pos_ind+2], source_l[pos_ind:pos_ind+2], 
                          [state_u[pos_ind], state_l[pos_ind+1]], 
                          [state_l[pos_ind], state_u[pos_ind+1]], 
                          [source_u[pos_ind], source_l[pos_ind+1]], 
                          [source_l[pos_ind], source_u[pos_ind+1]]]
    for pt in points_to_check:
        for obs in obstacles1:
            col = PointInTriangle(pt, tri1+obs)
            if(col == True or col == 1):
               collision = False
        for obs in obstacles2:
            col = PointInTriangle(pt, tri2+obs)
            if(col == True or col == 1):
                collision = False
    goal = (state_u[pos_ind:pos_ind+2] <= SAFE_REGION).all()
    return collision and goal

states = 30
end_point = 0.60
eps = end_point/(states)
non_zero = []
ind_i, ind_j = 0,0
multiplier = 1.0
global_probas = np.zeros((states,states))
from tqdm import trange
for k in trange(56):
    probas = []
    for i in np.linspace(0, end_point, num=states):
        for j in np.linspace(0.0, end_point, num=states):
            if(global_probas[ind_i][ind_j] != 0):
                probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            elif((i < INIT_SAFE_ZONE and j < INIT_SAFE_ZONE)):
                probas.append(1.0)
                ind_j += 1; continue
            elif((i < SAFE_REGION and j < SAFE_REGION)):
                probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            if(abs(i) > (SAFE_REGION + eps) or abs(j) > (SAFE_REGION + eps)):
                probas.append(0.0)
                ind_j += 1; continue
            print("State: (%s, %s) (eps: %s/mult: %s)"%(i,j,eps,multiplier))
            # (state, eps, predicate, p_arr, refine=1.0)
            p = verify_state([i, j], eps, predicate, global_probas)
            #print("DYNAMICS DATA: ", np.shape(DYNAMICS_X), np.shape(DYNAMICS_Y))
            #np.save("ZigDyn_X", DYNAMICS_X)
            #np.save("ZigDyn_Y", DYNAMICS_Y)
            if(p < 0):
                p = 0.0
            probas.append(p)
            #non_zero.append(p)
            ind_j += 1
        ind_i += 1
        ind_j = 0
    SAFE_REGION += eps
    ind_i = 0
    probas = np.reshape(probas, (states,states))
    global_probas = np.maximum(global_probas, probas)
    #multiplier = min(non_zero)
    np.save("probs_P2D_v1.0_%s_%s"%(states, eps), probas)
    print("======================================")
    print("==========EXPANDING GOAL==============")
    print("======================================")

probas = np.asarray(probas)
probas = probas.reshape(states,states)
print(list(probas))

np.save("probs_P2D_v1.0_%s_%s"%(states, eps), probas)

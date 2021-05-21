# Standard Library Imports
import os
import copy
import math
import random
import sys
sys.path.append("..")

# External Library Dependancies
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
from deepbayes_prealpha.analyzers import IBP_prob_dyn_m as IBP_prob_dyn

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


# Custom obstacles for HOV
tri1 = np.array([[-1,-1], [1,-1], [0,-0.5]])*0.2
tri2 = np.array([[-1,1], [1,1], [0,-0.0]])*0.1
   
obstacles1 = np.asarray([np.asarray([0.75,-0.1])])
obstacles2 = np.asarray([])

OBSTACLE = [0.75,-0.1]
INITIAL_THETA = 1.85
INIT_SAFE_ZONE = 0.2
GOAL_STATE = [-0.2, 0.0]
pos_ind = 0

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
        self.x = 1.5 + random.uniform(-0.025, 0.025)
        self.y = 0.5 + random.uniform(-0.025, 0.025)
        self.z = 0.5
        self.theta = INITIAL_THETA
        self.t = 0
        self.T = 30
        self.col = False

    def step(self, act):
        pre = np.linalg.norm([self.x, self.y]) 
        self.x = self.x + (math.cos(self.theta + 3.14 + 0.2)*act[0])
        self.y = self.y + (math.sin(self.theta + 3.14 + 0.2)*act[0])
        self.z = self.z + act[1]
        self.z = self.z - 0.005 # gravity term
        self.theta = self.theta + act[2]
        self.t += 1 

        # Check obstacle 
        #if(self.z <= 0.0):
        #    print("Grounded")
        #else:
        #    print("Altitude ", self.z)
        #self.col = True

        if(VERSION != 0):
            collide = self.check_collision([self.x, self.y])
            if(collide == 1):
                print("Collided")
                self.col = True
        return [self.x, self.y, self.z, self.theta], pre - np.linalg.norm([self.x, self.y, self.z]) , self.complete(), 0    

    def complete(self):
        state = np.asarray([self.x, self.y, self.z])
        if(np.linalg.norm(state) < 0.15 or self.x < 0.1):
            return True
        if(self.t==self.T or self.t > self.T):
            print("Timeout")
            return True
        return self.col

    def reset(self):
        self.x = 1.5 + random.uniform(-0.025, 0.025)
        self.y = 0.5 + random.uniform(-0.025, 0.025)
        self.z = 0.5
        self.theta = INITIAL_THETA
        self.t = 0
        self.col = False
        return np.asarray([self.x, self.y, self.z, self.theta])

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
        return np.random.rand(3)
    def observation_space_sample(self):
        return np.random.rand(4)


env = Zigzag_env()
state_0 = env.reset()
initial_state = state_0 # This is now the template for the "STATE" that we will pass throughout

action_dims = len(env.action_space_sample())
observe_dims = len(env.observation_space_sample())

print("Example observations and dimensions: ", env.observation_space_sample(),  len(env.observation_space_sample()))
print("Example actions and dimensions: ", env.action_space_sample(),  len(env.action_space_sample()))

model_input_dims = observe_dims + action_dims
model_output_dims = observe_dims
print(model_input_dims, model_output_dims)

control_input_dims = observe_dims
control_output_dims = action_dims
print(control_input_dims, control_output_dims)

# Load BNN Model
bayes_model = PosteriorModel("Posteriors/BayesDyn_HOV_v0_RUN")
control_model = PosteriorModel("Posteriors/Control_HOV_v0_RUN", deterministic=True)

# ============================================
#       Generate input gradient feild
# ============================================

def reward_function(state_0, dyn_pred, goal):
    state_0 = tf.cast(state_0, tf.float32)
    dyn_pred = tf.cast(dyn_pred, tf.float32)
    state_0 = tf.squeeze(state_0)
    dyn_pred = tf.squeeze(dyn_pred)
    state_1 = state_0 + dyn_pred

    l1x = tf.norm(state_1[pos_ind:pos_ind+1])
    l1y = tf.norm(state_1[pos_ind+1:pos_ind+2])
    #d1 = np.linalg.norm(state_0[:,pos_ind:pos_ind+2], axis=1)
    #d2 = np.linalg.norm(state_1[:,pos_ind:pos_ind+2], axis=1)
    #o1 = np.linalg.norm(state_0[:,pos_ind:pos_ind+2] - OBSTACLE, axis=1)
    #o2 = np.linalg.norm(state_1[:,pos_ind:pos_ind+2] - OBSTACLE, axis=1)
    distance_z = tf.norm(state_1[2]-0.25)

#    return (-1*(d2-d1)) - distance_z + ((0.25/o2)*(o2-o1))
    return -l1x  - l1y - distance_z #+ ((0.1/o2)*(o2-o1))

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
            predictions = model.predict(inp)
            loss = loss_fn([tf.squeeze(inp)[0:-3]], predictions, GOAL_STATE)
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
        predictions = model.predict(inp)
        loss = loss_fn([tf.squeeze(inp)[0:-3]], predictions, GOAL_STATE)
    inp_gradient = tape.gradient(loss, inp)
    inp_gradient = tf.cast(inp_gradient, 'float32')
    return tf.squeeze(inp_gradient)[-3:]

def synth_state(state, eps, predicate, loss=None, grad_iters=50):
    if(loss is None):
        loss = reward_function # This is gobally defined

    global initial_state 
    s0 = np.asarray(state) # Lower bound
    s1 = s0 + (eps/2.0)
    input_to_alg = (s0, s1)

    i_s = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s0
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    initial_state = i_s

    act = control_model.predict(np.asarray([s0_pre]))

    act = np.squeeze(act)
    original_act = copy.deepcopy(act)

    s0 = np.concatenate((s0_pre,act))
    s1 = np.concatenate((s1_pre,act))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    for i in range(grad_iters):
        act = np.squeeze(act)
        s0 = np.concatenate((s0_pre,act))
        s1 = np.concatenate((s1_pre,act))
        s0 = np.asarray([s0])
        s1 = np.asarray([s1])

        w = bayes_model.sample()
        p, outs = IBP_fixed_w(bayes_model, s0, s1, 2.25, w, predicate)
        grad = gradient_w(bayes_model, (s0+s1)/2, loss, w)

        a = act + (1.0 * grad)
        act = a
    print("RESTUL OF SYNTH: ", original_act, act)
    return act, original_act

CONTROL_DATA_X = []
CONTROL_DATA_Y = []

def verify_state(state, eps, predicate, p_arr, refine=2.5):
    global initial_state
    s0 = np.asarray(state) # Lower bound
    s1 = s0 + eps
    input_to_alg = (s0, s1)

    i_s = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s0
    s0_pre = copy.deepcopy(initial_state)
    initial_state[pos_ind:pos_ind+2] = s1
    s1_pre = copy.deepcopy(initial_state)
    initial_state = i_s

    diff = s0_pre - s1_pre
    low, high = np.asarray([s0_pre])+0.5*diff, np.asarray([s0_pre])-0.5*diff
    act = control_model.predict(np.asarray([s0_pre]))
    act_l, act_u = IBP_state(control_model, low, high, control_model.model.get_weights())

    act = np.squeeze(act)
    act_l = np.squeeze(act_l)
    act_u = np.squeeze(act_u)
    print("ACTION OUTPUT: ", act_l, act_u)

    s0 = np.concatenate((s0_pre,act_l))
    s1 = np.concatenate((s1_pre,act_u))

    s0 = np.asarray([s0])
    s1 = np.asarray([s1])

    p = 0.0
    if(p < 0.95):
        refine += 2.25
        act, _ = synth_state(state, eps, predicate)
        act = np.squeeze(act)
        act_l = np.squeeze(act_l)
        act_u = np.squeeze(act_u)
        s0 = np.concatenate((s0_pre,act_l))
        s1 = np.concatenate((s1_pre,act_u))
        s0 = np.asarray([s0])
        s1 = np.asarray([s1])
        p, outs = IBP_prob_dyn(bayes_model, s0, s1, 3.0*(1/refine), int(100*(refine**2)), predicate)
        # Here we are adding data to a control dataset for the synthesis stages
        global CONTROL_DATA_X
        global CONTROL_DATA_Y
        CONTROL_DATA_X.append(np.squeeze(s0_pre))
        CONTROL_DATA_X.append(np.squeeze(s1_pre))
        CONTROL_DATA_Y.append(act)
        CONTROL_DATA_Y.append(act)

    if(p == 0):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~NOT REACHABLE~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return 0, 0

    mins = [i[0] for i in outs]
    maxs = [i[1] for i in outs]
    mins = np.min(mins, axis=0)[pos_ind:pos_ind+2]
    maxs = np.max(maxs, axis=0)[pos_ind:pos_ind+2]
    maxs[0] = min(maxs[0], np.squeeze(s1)[0])
    maxs[1] = min(maxs[1], np.squeeze(s1)[1])
    print("MAXES: ", maxs)
    if((maxs < INIT_SAFE_ZONE).all()):
        print("   ")
        print("One step reachable. (p=%s)"%(p))
        print("   ")
        return p, p
    else:
        mi_i, ma_i = math.floor(mins[0]/eps), math.ceil(maxs[0]/eps)
        mi_j, ma_j = math.floor((mins[1])/eps), math.ceil((maxs[1])/eps)
        mi_i, ma_i = max(0, mi_i), max(0, ma_i)
        mi_j, ma_j = max(0, mi_j), max(0, ma_j)
        mi_i, ma_i = min(len(p_arr), mi_i), min(len(p_arr), ma_i)
        mi_j, ma_j = min(len(p_arr), mi_j), min(len(p_arr), ma_j)
        mi_i, ma_i = max(0, mi_i), max(0, ma_i)
        mi_j, ma_j = max(0, mi_j), max(0, ma_j)
        print("X ", mi_i, ma_i)
        print("Y ", mi_j, ma_j)
        if(mi_i == 0 and ma_i == 0):
            return p, p
            ma_i += 1
        if(mi_j == 0 and ma_j == 0):
            return p , p
            ma_j += 1
        try:
            worst_future = np.min(np.asarray(p_arr[mi_i:ma_i,mi_j:ma_j]).flatten())
        except:
            worst_future = p_arr[mi_i,mi_j]
        print("Multi-step reachable. Worst case: %s (p=%s)"%(worst_future, p))
        print(mins, maxs)
        if(worst_future ==0.0):
            return 0.0, 0.0 #-1
        return p * worst_future, p

SAFE_REGION = INIT_SAFE_ZONE

def predicate(source_l, source_u, state_l, state_u):
    collision = True
    goal = (state_u[pos_ind:pos_ind+2] <= SAFE_REGION).all()
    goal = goal and (state_u[pos_ind+2] > 0.0)
    # Return the conjunction of these two properties
    return collision and goal



INIT_SAFE_ZONE = 0.2
SAFE_REGION =  INIT_SAFE_ZONE
states = 25
end_point = 0.5
eps = end_point/(states)
non_zero = []
ind_i, ind_j = 0,0
multiplier = 1.0
global_probas = np.zeros((states,states))
pre_global_probas = np.zeros((states,states))


from tqdm import trange
for k in trange(44):
    probas = []
    pre_probas = []
    for j in np.linspace(0.0, end_point, num=states):
        for i in np.linspace(0.0, end_point, num=states):
            if(global_probas[ind_i][ind_j] != 0):
                probas.append(global_probas[ind_i][ind_j])
                pre_probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            elif((i < INIT_SAFE_ZONE and abs(j) < INIT_SAFE_ZONE)):
                probas.append(1.0)
                pre_probas.append(1.0)
                ind_j += 1; continue
            elif((i < SAFE_REGION and abs(j) < SAFE_REGION)):
                probas.append(global_probas[ind_i][ind_j])
                pre_probas.append(global_probas[ind_i][ind_j])
                ind_j += 1; continue
            if(i > (SAFE_REGION + eps) or abs(j) > (SAFE_REGION + eps)):
                probas.append(0.0)
                pre_probas.append(0.0)
                ind_j += 1; continue
            print(i, j)
            print("State: (%s, %s) (eps: %s/mult: %s)"%(i,j,eps,multiplier))
            p, p2 = verify_state([i, j], eps, predicate, global_probas, refine=2.25)
            print(" ")
            print(" ")
            print(" State verified with probability : ", p)
            print(" ")
            print(" ")
            #print("CONTROLLER DATA: ", np.shape(CONTROL_DATA_X), np.shape(CONTROL_DATA_Y))
            #np.save("HoverCon_X", CONTROL_DATA_X)
            #np.save("HoverCon_Y", CONTROL_DATA_Y)
            probas.append(p)
            pre_probas.append(p2)
            ind_j += 1
        ind_i += 1
        ind_j = 0
    SAFE_REGION += eps
    ind_i = 0
    probas = np.reshape(probas, (states,states))
    pre_probas = np.reshape(pre_probas, (states,states))
    global_probas = np.maximum(global_probas, probas)
    pre_global_probas = np.maximum(pre_global_probas, pre_probas)
    np.save("probs_HOV_v0_%s_%s"%(states, eps), probas)
    np.save("preprobs_HOV_v0_%s_%s"%(states, eps), pre_probas)

probas = np.asarray(probas)
probas = probas.reshape(states,states)
np.save("probs_HOV_v0_%s_%s"%(states, eps), probas)



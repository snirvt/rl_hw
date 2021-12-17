

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from Q_learner import get_Q, init_W, get_max_Q
env_name  = 'MountainCar-v0'


c_p = [-1,-0.75, -0.5, -0.25, 0, 0.25, 0.5]
c_v = [-0.04, -0.02, 0, 0.02, 0.04]

centers = [[p, v] for p in c_p for v in c_v]

def epsilon_greedy(W, f_s, eps=0.1):
    if random.random() < eps:
        rand_a = random.sample(range(3),1)[0]
        return rand_a, get_Q(f_s, rand_a, W)
    return get_max_Q(f_s, W)


def get_eligibility_trace(n):
    E = np.zeros(n)
    return E

def get_features(s):
    x = s - centers
    diag_inv = np.diag([10, 1500])
    return np.exp(-0.5 * x @ diag_inv @ x.T).diagonal()


def evaluate_policy(W, env, GAMMA, sample_size = 1000):
    reward_sum = 0
    for _ in range(sample_size):
        s = env.reset()
        step_count = 0
        while True:
            f_s = get_features(s)
            a, _ = get_max_Q(f_s, W)
            s, reward, done, info = env.step(a)
            reward_sum += (GAMMA**step_count) * reward
            step_count += 1
            if done or step_count == 200:
                s = env.reset()
                break
    return reward_sum/sample_size


def SARSA(param_dict):
    GAMMA = param_dict['GAMMA']
    ALPHA = param_dict['ALPHA']
    n_steps = param_dict['n_steps']
    step_size = param_dict['step_size']
    sample_size = param_dict['sample_size']
    env = gym.make(env_name)
    s = env.reset()
    cnt = 0
    eps = 1
    max_pos = float('-inf')
    n_actions = 3
    n_features = len(c_p) * len(c_v)
    W = init_W(n_actions = n_actions, n_features = n_features)
    best_W = deepcopy(W)
    best_value = float('-inf')
    value = []

    for i in range(n_steps):
        if i % step_size == 0:
            value.append(evaluate_policy(W, env, GAMMA, sample_size))
            print('val: {}, eps {}'.format(value[-1], eps))
            if value[-1] > best_value:
                best_value = value[-1]
                best_W = deepcopy(W)
        cnt+=1 
        f_s = get_features(s)
        f_s_action, f_s_value = epsilon_greedy(W, f_s, eps=eps)       
        s_next, reward, done, _ = env.step(f_s_action)
             
        position, velocity = s_next
        if position > max_pos:
            print(f'{i}: {position}')
            max_pos = position
        
        if done and cnt < 199:
            print(f'{i} succ, eps: {eps} lr: {ALPHA}')
            ALPHA *= 0.995

        f_s_next = get_features(s_next)
        best_action, max_next_action_val = get_max_Q(f_s_next, W)
        
        ''' W update '''
        TD_error = reward + GAMMA * max_next_action_val - f_s_value
        grad_w = f_s
        w_delta = ALPHA * TD_error * grad_w
        W[f_s_action] += w_delta
        
        s = s_next
        
        if done or cnt>199:
            eps *= 0.995
            s = env.reset()
            cnt=0

    env.close()
    return W, value, best_W


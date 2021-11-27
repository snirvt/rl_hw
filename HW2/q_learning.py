

import gym
import numpy as np
import gym
import random
from copy import deepcopy

from P_learner import get_Q

env_name  = 'FrozenLake8x8-v1'


def epsilon_greedy(q_s, eps=0.1):
    if random.random() < eps:
        return random.sample(range(len(q_s)),1)[0]
    return np.argmax(q_s)

def get_eligibility_trace(Q):
    E = {}
    for s in Q.keys():
        E[s] = [0]*len(Q[s])
    return E


def evaluate_policy(Q, env, GAMMA, sample_size = 1000):
    reward_sum = 0
    for _ in range(sample_size):
        s = env.reset()
        step_count = 0
        while True:
            a = np.argmax(Q[s])
            s, reward, done, info = env.step(a)
            reward_sum += (GAMMA**step_count) * reward
            step_count += 1
            if done:
                s = env.reset()
                break
    return reward_sum/sample_size


def QLearning(param_dict):
    GAMMA = param_dict['GAMMA']
    ALPHA = param_dict['ALPHA']
    LAMBDA = param_dict['LAMBDA']
    sample_size = param_dict['sample_size']
    # eps = param_dict['eps']
    eps_updater = param_dict['eps_updater']
    decay_rate = param_dict['decay_rate']
    n_steps = param_dict['n_steps']
    step_size = param_dict['step_size']
    elegability_trace = param_dict['elegability_trace']
    
    env = gym.make(env_name, map_name="8x8",is_slippery=True)
    env_t = deepcopy(env)
    Q = get_Q(n_state = 8*8, n_actions=4)
    s = env.reset()
    cnt = 0
    E = get_eligibility_trace(Q)
    value = []
    eps = eps_updater()
    trace = set()
    best_Q = deepcopy(Q)
    best_value = 0
    
    for i in range(n_steps):
        if i % step_size == 0:
            value.append(evaluate_policy(Q, env_t, GAMMA, sample_size))
            print('val: {}, eps {}'.format(value[-1], eps))
            if value[-1] > best_value:
                best_value = value[-1]
                best_Q = deepcopy(Q)
        cnt+=1
        a = epsilon_greedy(Q[s], eps=eps)
        
        trace.add((s,a))
        
        s_next, reward, done, info = env.step(a)
              
        if reward > 0:
            eps = eps_updater()

        delta = reward + GAMMA*Q[s_next][np.argmax(Q[s_next])] - Q[s][a]
        if elegability_trace and LAMBDA > 0:
            for s_,a_ in trace:
                E[s_][a_] = GAMMA * LAMBDA * E[s_][a_]
                if s_ == s and a_ == a:
                    E[s_][a_] += 1
                Q[s_][a_] += ALPHA * (delta) * E[s_][a_]
        else:
            Q[s][a] += ALPHA * (delta)
        
        s = s_next
        if done or cnt==250:
            s = env.reset()
            E = get_eligibility_trace(Q)
            cnt=0
            trace = set()
    env.close()
    return Q, value, best_Q




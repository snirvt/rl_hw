

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from Q_learner import get_Q, init_W, get_max_Q, sample_action, get_mean_w, sample_gaussian_action
env_name  = 'Pendulum-v0'
# env_name  = 'MountainCar-v0'

from sklearn.preprocessing import PolynomialFeatures

# c_1 = [-6, -4, -2,0, 2, 4, 6]
# c_2 = [-6, -4, -2,0, 2, 4, 6]
# c_3 = [0]
# centers = [[c1, c2, c3] for c1 in c_1 for c2 in c_2 for c3 in c_3]

# c_p = [-1,-0.75, -0.5, -0.25, 0, 0.25, 0.5]
# c_v = [-0.04, -0.02, 0, 0.02, 0.04]

# centers = [[p, v] for p in c_p for v in c_v]



def check_valid_matrix(m):   
    return np.all(~np.isfinite(m)) # True wherever pos-inf or neg-inf or nan

def get_features(s):
    # x = s - centers
    # diag_inv = np.diag([10, 1500])
    # return np.concatenate([np.array([1]),np.exp(-0.5 * x @ diag_inv @ x.T).diagonal()])
    poly = PolynomialFeatures(10)
    res = poly.fit_transform(s.reshape(1,-1))
    res[:,1:] = (res[:,1:] - np.mean(res[:,1:]))/np.std(res[:,1:])
    return res.reshape(1,-1)


def evaluate_policy(W, Theta, env, GAMMA, var, sample_size = 1000):
    reward_sum = 0
    for _ in range(sample_size):
        s = env.reset()
        step_count = 0
        while True:
            f_s = get_features(s)
            a, _, _ = sample_gaussian_action(f_s.squeeze(), W.squeeze(), Theta.squeeze(), var)
            s, reward, done, info = env.step([a])
            # reward += 20
            reward_sum += (GAMMA**step_count) * reward
            step_count += 1
            if done or step_count == 200:
                s = env.reset()
                break
    return reward_sum/sample_size


def QAC(param_dict):
    GAMMA = param_dict['GAMMA']
    ALPHA = param_dict['ALPHA']
    BETA = param_dict['BETA']
    n_steps = param_dict['n_steps']
    step_size = param_dict['step_size']
    sample_size = param_dict['sample_size']
    env = gym.make(env_name)
    s = env.reset()
    cnt = 0
    eps = 1
    # n_features = len(centers)
    n_features = 286#35
    n_actions = 1
    W = init_W(n_actions = n_actions, n_features = n_features)
    Theta = init_W(n_actions = n_actions, n_features = n_features)
    
    best_W = deepcopy(W)
    best_Theta = deepcopy(Theta)
    best_value = float('-inf')
    value = []
    var = 2

    f_s = get_features(s)
    f_s_action, f_s_value, f_s_mean = sample_gaussian_action(f_s.squeeze(), W.squeeze(), Theta.squeeze(), var)

    f_s_list = []
    f_s_action_list = []
    reward_list = []
    f_s_next_value_list = []
    f_s_value_list = []
    f_s_mean_list = []
    cnt2 = 1
    replay_buffer = []
    for i in range(n_steps):
        if i % step_size == 0:
            value.append(evaluate_policy(W, Theta, env, GAMMA, var, sample_size))
            print('val: {}, var {}'.format(value[-1], var))
            if value[-1] > best_value:
                best_value = value[-1]
                best_W = deepcopy(W)
        cnt+=1 
        f_s = get_features(s)
        if cnt2 % 100 == 0:
            pass
            # env.render()
            # print(f_s_action)
               
        s_next, reward, done, _ = env.step([f_s_action])
        reward += 20

        if done and cnt < 199:
            print(f'{i} succ, eps: {eps} lr: {ALPHA}')
            ALPHA *= 0.995

        f_s_next = get_features(s_next)
        f_s_next_action, f_s_next_value, f_s_next_mean = sample_gaussian_action(f_s_next.squeeze(), W.squeeze(), Theta.squeeze(), var)
        
        ''' if fails update all actions '''
             
        f_s_list.append(f_s)
        f_s_action_list.append(f_s_action)
        reward_list.append(reward)
        f_s_value_list.append(f_s_value)
        f_s_next_value_list.append(f_s_next_value)
        f_s_mean_list.append(f_s_mean)
        
        batch_size = 1
        if cnt2 % batch_size == 0 and cnt == 1:
            replay_buffer.append([f_s_list, f_s_action_list, reward_list, f_s_next_value_list, f_s_value_list, f_s_mean_list]) 
            if len(replay_buffer)>1:
                    replay_buffer = replay_buffer[-1:]
            
            for buffer in replay_buffer:
                f_s_list = buffer[0]
                f_s_action_list = buffer[1]
                reward_list = buffer[2]
                f_s_next_value_list = buffer[3]
                f_s_value_list = buffer[4]
                f_s_mean_list = buffer[5]
                
                for j in range(len(f_s_list)):
                    f_s_ = f_s_list[j]
                    f_s_action_ = f_s_action_list[j]
                    reward_ = reward_list[j]
                    f_s_value_ = f_s_value_list[j]
                    f_s_next_value_ = f_s_next_value_list[j]
                    f_s_mean_ = f_s_mean_list[j]
                    TD_error = reward_ + GAMMA * f_s_next_value_ - f_s_value_
                    grad_log_theta = (1/f_s_action_) * (1 - f_s_action_) * ((f_s_action_ - f_s_mean_) * f_s)/var
                    Theta += 1/batch_size * ALPHA * grad_log_theta.reshape(Theta.shape) * f_s_value_
                    W += 1/batch_size * BETA * TD_error * f_s_.reshape(W.shape)
            

            f_s_list = []
            f_s_action_list = []
            reward_list = []
            f_s_next_value_list = []
            f_s_value_list = []
            f_s_mean_list = []
            Theta *= (1-ALPHA)
            W *= (1-BETA)
            var *= 0.999 
            if var < 0.1:
                var = 0.1      
        s = s_next
        f_s_action = f_s_next_action
        
        if done or cnt>199 or any(np.isnan(s)):
            s = env.reset()
            cnt=0
            cnt2+=1
    env.close()
    return W, value, best_W


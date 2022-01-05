

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from Q_learner import get_Q, init_W, get_max_Q, sample_action, get_mean_w, sample_gaussian_action
env_name  = 'Pendulum-v0'

from sklearn.preprocessing import PolynomialFeatures

c_1 = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8 ,1]
c_2 = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8 ,1]
c_3 = [-1, 0, 1]
centers = [[c1, c2, c3] for c1 in c_1 for c2 in c_2 for c3 in c_3]


def check_valid_matrix(m):   
    return np.all(~np.isfinite(m)) # True wherever pos-inf or neg-inf or nan

def get_features(s):
    # x = s - centers
    # diag_inv = np.diag([10, 10, 10])
    # return np.concatenate([np.array([1]),s,np.exp(-0.5 * x @ diag_inv @ x.T).diagonal()]).reshape(1,-1)
    poly = PolynomialFeatures(7)
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
            a, _, _ = sample_gaussian_action(f_s.squeeze(), W.squeeze(), Theta.squeeze(), 0)
            s, reward, done, info = env.step([a])
            reward += 20
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
    # n_features = len(centers) + 1 + 3
    n_features = 120
    n_actions = 1
    W = init_W(n_actions = n_actions, n_features = n_features + 1)
    Theta = init_W(n_actions = n_actions, n_features = n_features)
    
    best_W = deepcopy(W)
    best_Theta = deepcopy(Theta)
    best_value = float('-inf')
    value = []
    var = 0.5
    f_s = get_features(s)
    f_s_action, f_s_value, f_s_mean = sample_gaussian_action(f_s.squeeze(), W.squeeze(), Theta.squeeze(), var)

    f_s_list = []
    f_s_next_list = []
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
            print('val: {}, var {}, ALPHA {}, BETA {}'.format(value[-1], var, ALPHA, BETA))
            if value[-1] > best_value:
                best_value = value[-1]
                best_Theta = deepcopy(Theta)
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
 
        batch_size = 1
        if cnt2 % batch_size == 0 and cnt == 1:
            replay_buffer.append([f_s_list, f_s_action_list, reward_list, f_s_next_value_list, f_s_value_list, f_s_mean_list]) 
            if len(replay_buffer)>10:
                    replay_buffer = replay_buffer[-10:]
            # for buffer in replay_buffer:
            for buffer_idx in np.random.choice(range(len(replay_buffer)), len(replay_buffer), False) :
                f_s_list = replay_buffer[buffer_idx][0]
                f_s_action_list = replay_buffer[buffer_idx][1]
                reward_list = replay_buffer[buffer_idx][2]
                f_s_next_value_list = replay_buffer[buffer_idx][3]
                f_s_value_list = replay_buffer[buffer_idx][4]
                f_s_mean_list = replay_buffer[buffer_idx][5]
                
                for j in np.random.choice(range(len(f_s_list)), len(f_s_list), False) :
                    f_s_ = f_s_list[j]
                    f_s_next_ = f_s_next_list[j]
                    f_s_action_ = f_s_action_list[j]
                    reward_ = reward_list[j]
                    f_s_value_ = f_s_value_list[j]
                    f_s_next_value_ = f_s_next_value_list[j]
                    f_s_mean_ = f_s_mean_list[j]
                    
                    f_s_action_, f_s_value_, f_s_mean_ = sample_gaussian_action(f_s_.squeeze(), W.squeeze(), Theta.squeeze(), var)

                    TD_error = reward_ + GAMMA * f_s_next_value_ - f_s_value_
                    grad_log_theta = (1/f_s_action_) * (1 - f_s_action_) * ((f_s_action_ - f_s_mean_) * f_s)/var
                    
                    grad_log_theta = np.nan_to_num(grad_log_theta, copy=False, nan=0.0, posinf=0, neginf=0)
                    TD_error = np.nan_to_num(TD_error, copy=False, nan=0.0, posinf=0, neginf=0)

                    Theta += (1/len(f_s_list)) * ALPHA * grad_log_theta.reshape(Theta.shape) * f_s_value_
                    W += (1/len(f_s_list)) * BETA * TD_error * (np.concatenate([np.array(f_s_mean_.reshape(1,-1)),f_s_],axis=1)).reshape(W.shape)
                    Theta *= (1-ALPHA*3)
                    W *= (1-BETA*3)
                    Theta = np.nan_to_num(Theta, copy=False, nan=0.0, posinf=0, neginf=0)
                    W = np.nan_to_num(W, copy=False, nan=0.0, posinf=0, neginf=0)            

            f_s_list = []
            f_s_next_list = []
            f_s_action_list = []
            reward_list = []
            f_s_next_value_list = []
            f_s_value_list = []
            f_s_mean_list = []
            ALPHA *= 0.999995
            BETA *= 0.999995
            var *= 0.995
            if var < 0.2:
                var = 0.2
                
        f_s_next = get_features(s_next)
        f_s_next_action, f_s_next_value, f_s_next_mean = sample_gaussian_action(f_s_next.squeeze(), W.squeeze(), Theta.squeeze(), var)
        
        ''' if fails update all actions '''
             
        f_s_list.append(f_s)
        f_s_next_list.append(f_s_next)
        f_s_action_list.append(f_s_action)
        reward_list.append(reward)
        f_s_value_list.append(f_s_value)
        f_s_next_value_list.append(f_s_next_value)
        f_s_mean_list.append(f_s_mean)                    
        s = s_next
        f_s_action = f_s_next_action
        
        if done or cnt>199 or any(np.isnan(s)):
            s = env.reset()
            cnt=0
            cnt2+=1
    env.close()
    return Theta, value, best_Theta


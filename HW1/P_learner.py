
import numpy as np
import gym
from cheats import new_reset, modify_env

def get_P():
    env = gym.make('Taxi-v3')
    P = {}
    for state in range(500):
        for action in range(6):
            env.env.s = state
            env.reset
            observation, reward, done, info = env.step(action)
            if state not in P:
                P[state] = {}
            P[state][action] = [(1.0, observation, reward, done)]
    return P





# P = {}
# P_true = env.P
# for state in range(500):
#     for action in range(6):
#         env.env.s = state
#         observation, reward, done, info = env.step(action)
#         if state not in P:
#             P[state] = {}
#         P[state][action] = [(1.0, observation, reward, done)]
#         if P_true[state][action][0][1] != observation and P_true[state][action][0][2] != reward:
#             print('FUCKKKKKKKKKKKKKKKKK')













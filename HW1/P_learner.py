
import numpy as np
import gym
from cheats import new_reset, modify_env

def get_P():
    env = gym.make('Taxi-v3')
    P = {}
    T = {} # terminal states
    for state in range(500):
        for action in range(6):
            env.reset()
            env.env.s = state
            observation, reward, done, info = env.step(action)
            if state not in P:
                P[state] = {}
            if done: # if terminal state
                T[state] = []
            P[state][action] = [(1.0, observation, reward, done)]
    return P, T
# P, T = get_P()
    




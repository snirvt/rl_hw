
import numpy as np
import gym
from cheats import new_reset, modify_env

def get_P():
    env = gym.make('Taxi-v3')
    P = {}
    Terminals = {} # terminal states
    Posibble_Transitions = {}
    for state in range(500):
        for action in range(6):
            env.reset()
            env.env.s = state
            observation, reward, done, info = env.step(action)
            if state not in P:
                P[state] = {}
                Posibble_Transitions[state] = set()
            if done: # if terminal state
                Terminals[state] = []
            P[state][action] = [(1.0, observation, reward, done)]
            Posibble_Transitions[state].add(observation)
    return P, Terminals, Posibble_Transitions
# P, T, PT = get_P()
    


def can_s1_go_to_s2(P, s1, s2):
    for a in range(6):
        if s2 == P[s1][a][0][1]:
            return True
    return False






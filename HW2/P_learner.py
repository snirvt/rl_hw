
import numpy as np
import gym

def get_Q(n_state = 4*4, n_actions=4):
    q_s_a = {}
    for s in range(n_state):
        q_s_a[s] = [0]*n_actions
        # q_s_a[s] = np.random.uniform(low=0, high=0.01, size=(n_actions)).tolist()
        # q_s_a[s] = np.zeros(n_actions)
    return q_s_a
    



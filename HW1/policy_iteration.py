import numpy as np
import gym
from P_learner import get_P


def init_policy(P): ## init policy to uniform action for all states
    policy = {}
    for s in P.keys():            
        policy[s] = np.ones(6)/6
    return policy

def init_value(P): ## init value function to 0 value for all
    value = {}
    for s in P.keys():
        value[s] = 0
    return value



def policy_evaluation(policy, value, P, T, PT, GAMMA=0.95): # evaluating a given policy
    new_value = value.copy()
    for s in P.keys():
        v_temp = 0
        if s not in T: # not in terminal state
            for a in range(6):
                PI_a_s = policy[s][a] ## chance of doing a when in s
                R_a_s = P[s][a][0][2] ## reward of being in s and doing a
                s_next = P[s][a][0][1]
                v_temp += PI_a_s*(R_a_s + GAMMA * value[s_next]) # since P(s,a) is deterministic no need to loop over S'.
        new_value[s] = v_temp
    return new_value


def policy_improvment(value, P, GAMMA=0.95):
    new_policy = {}
    for s in value.keys():
        s_a_rewards = []
        for a in range(6):
            R_a_s = P[s][a][0][2]
            s_next = P[s][a][0][1]
            s_a_rewards.append(R_a_s + GAMMA * value[s_next]) # q(s,a)
        best_a = np.argmax(s_a_rewards) # max_a q(s,a)
        new_policy[s] = np.zeros(6)
        new_policy[s][best_a] = 1
    return new_policy


def policy_iteration(P, T, PT, max_steps = 100, GAMMA=0.95, quit_when_optimal = True, sample_size = 500):
    env_stub = gym.make('Taxi-v3')
    policy = init_policy(P)
    value = init_value(P)
    value_mean = []
    for cnt in range(max_steps):
        old_policy = policy.copy()
        value = policy_evaluation(policy, value, P, T, PT, GAMMA)
        policy = policy_improvment(value, P, GAMMA)
        internal_value_sum = 0
        for _ in range(sample_size):
            env_stub.reset()
            internal_value_sum += value[env_stub.env.s]
        value_mean.append(internal_value_sum/sample_size)

        if quit_when_optimal and all(all(old_policy[s] == policy[s]) for s in P.keys()):
            print('Optimal Policy at: {} steps'.format(cnt))
            break
    return policy, value, value_mean

# P, T, PT = get_P()
# policy, value = policy_iteration(P, T, PT)

import numpy as np
from policy_iteration import policy_iteration
from P_learner import get_P

GAMMA = 0.95
P, T, PT = get_P()
policy, value, value_sum = policy_iteration(P, T, PT, GAMMA)


import gym
# from taxienv import TaxiEnv
# env = TaxiEnv()

env = gym.make('Taxi-v3')
observation = env.reset()
env.render(mode="human")
observation = env.env.s
for _ in range(50):
    action = np.argmax(policy[observation])
    observation, reward, done, info = env.step(action)
    env.render(mode="human")
    print('reward: {}'.format(reward))
    print(done)
    if done:
        observation = env.reset()
        break
env.close()







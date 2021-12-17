

# https://github.com/adibyte95/Mountain_car-OpenAI-GYM/blob/master/prog.py

import numpy as np
import gym
from sklearn.gaussian_process.kernels import RBF

env_name  = 'MountainCar-v0'
env = gym.make(env_name)

action_dict = {'a': 0, 's':1, 'd': 2}

observation = env.reset()
for _ in range(200):
    env.render(mode="human")
    msg = '\n- Accelerate_left :a\n- idle: s\n- Accelerate_right: d\n'
    action = input(msg)
    observation, reward, done, _ = env.step(action_dict[action])
    
    print(observation, reward, done)
    print('reward: {}'.format(reward))
    if done:
        observation = env.reset()
        print(reward)
        break
env.close()


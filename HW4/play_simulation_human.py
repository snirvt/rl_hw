

# https://github.com/adibyte95/Mountain_car-OpenAI-GYM/blob/master/prog.py

import numpy as np
import gym
from sklearn.gaussian_process.kernels import RBF
# from pendulum import PendulumEnv
# env = PendulumEnv()

env_name  = 'Pendulum-v0'
# env_name  = 'MountainCar-v0'
env = gym.make(env_name)

action_dict = {'a': 0, 's':1, 'd': 2}


observation = env.reset()
for _ in range(200):
    env.render(mode="human")
    msg = '\n- Accelerate_left :a\n- idle: s\n- Accelerate_right: d\n'
    action = input(msg)
    # action = float(action)
    observation, reward, done, info = env.step(action_dict[action])
    # observation, reward, done, info = env.step([action])

    print('summary: ',observation, reward, done)
    print('reward: {}'.format(reward))
    if done:
        observation = env.reset()
        print(reward)
        break
env.close()


import numpy as np
import gym
env_name  = 'FrozenLake8x8-v1'

action_dict = {'a': 0, 's':1, 'd': 2, 'w': 3}
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3




env = gym.make(env_name,map_name="4x4",is_slippery=False)
observation = env.reset()
for _ in range(200):
    env.render(mode="human")
    msg = '\n- s: move south\n- w: move north\n- d: move east\n- a: move west\n'
    action = input(msg)
    observation, reward, done, info = env.step(action_dict[action])
    print(observation, reward, done, info)
    print('reward: {}'.format(reward))
    if done:
        observation = env.reset()
env.close()

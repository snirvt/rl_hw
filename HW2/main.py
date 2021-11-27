
import gym

env_name  = 'FrozenLake8x8-v1'
import numpy as np
import gym


from simulation_src_code import FrozenLakeEnv




env = gym.make(env_name)
# env = FrozenLakeEnv()
observation = env.reset()



for _ in range(200):
	env.render(mode="human")
	# msg = '\n- 0: move south\n- 1: move north\n- 2: move east\n- 3: move west\n- 4: pickup passenger\n- 5: drop off passenger\n'
	# action = int(input(msg))
	action = env.action_space.sample() # take a random action

	observation, reward, done, info = env.step(action)

	print('reward: {}'.format(reward))
	if done:
		observation = env.reset()
env.close()

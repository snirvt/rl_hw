import numpy as np
import gym
env = gym.make('Taxi-v3')
observation = env.reset()
for _ in range(200):
	env.render(mode="human")
	msg = '\n- 0: move south\n- 1: move north\n- 2: move east\n- 3: move west\n- 4: pickup passenger\n- 5: drop off passenger\n'
	action = int(input(msg))
	observation, reward, done, info = env.step(action)

	print('reward: {}'.format(reward))
	if done:
		observation = env.reset()
env.close()

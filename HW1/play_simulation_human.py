# pip install gym
# pip install pyglet

import numpy as np

import gym
# from taxienv import TaxiEnv
# env = gym.make("CartPole-v1")
env = gym.make('Taxi-v3')
# env = TaxiEnv()
observation = env.reset()
for _ in range(200):
	env.render(mode="human")
	# action = env.action_space.sample() # your agent here (this takes random actions)
	# print(action)
	msg = '\n- 0: move south\n- 1: move north\n- 2: move east\n- 3: move west\n- 4: pickup passenger\n- 5: drop off passenger\n'
	action = int(input(msg))
	observation, reward, done, info = env.step(action)
	print('reward: {}'.format(reward))
	print(done)
	# obs = np.zeros(4)
	# for iter in env.decode(observation):
	# 	pass
	# env.decode(observation)
	# print(info)

	if done:
		observation = env.reset()
env.close()



# for i in reversed(out):
# 	print(i)



# i=100
# out = []
# out.append(i % 4)
# i = i // 4
# out.append(i % 5)
# i = i // 5
# out.append(i % 5)
# i = i // 5
# out.append(i)
# assert 0 <= i < 5
# return reversed(out)



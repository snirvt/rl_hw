import numpy as np
import matplotlib.pyplot as plt

from q_learning import QLearning
from epsilon import ExponentialDecay, SimpleEpsilon, LinearDecay
env_name  = 'FrozenLake8x8-v1'


render_img = False
print_vals = True
# sample_size = 1
# n_steps = 1000000
# step_size = n_steps+1
# eps = 1
# decay_rate = 0.95
# eps_updater = LinearDecay(eps = 1 , p = 0.99)


# GAMMA = 0.95
# LAMBDA = 0.1
# ALPHA = 0.1
# elegability_trace = True

# param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA, 'LAMBDA': LAMBDA, 'sample_size': sample_size,
#                 'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
#                 'elegability_trace': elegability_trace, 'eps_updater': LinearDecay(eps = 1 , p = 0.95)}
# Q, value_mean  = QLearning(param_dict)



import numpy as np
import matplotlib.pyplot as plt

from q_learning import QLearning
from epsilon import ExponentialDecay, SimpleEpsilon, LinearDecay


fig = plt.figure()

sample_size = 250
n_steps = 1000000
step_size = 10000
eps = 1
decay_rate = 0.95
elegability_trace = True
GAMMA = 0.95
# eps_updater = LinearDecay(eps = 1 , p = 0.99)
# eps_updater = ExponentialDecay(gamma = 0.001 , N = 100)
# eps_updater = SimpleEpsilon(k = 100)

dict_plot = {}

# for LAMBDA in [0.1]:#[0, 0.1, 0.2, 0.3]:
#     for ALPHA in [0.1]:#[0.01, 0.1, 0.3, 0.5]:
#         for p in [0.9, 0.95, 0.99]:
#             param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA, 'LAMBDA': LAMBDA, 'sample_size': sample_size,
#                         'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
#                         'elegability_trace': elegability_trace, 'eps_updater': LinearDecay(eps = 1 , p = p, min_eps=0)}
#             Q, value_mean, bestQ = QLearning(param_dict)
#             plt.plot(np.arange(len(value_mean))*step_size,value_mean, label=(GAMMA, ALPHA, LAMBDA, p))
# plt.legend(title="GAMMA, ALPHA, LAMBDA, p")
# fig.suptitle('Value Function Mean Over Iterations', fontsize=20)
# plt.xlabel('Iteration Number', fontsize=15) 
# plt.ylabel('Value Function', fontsize=15)
# plt.show()
# np.save("bestQ.npy", bestQ)

bestQ = np.load("bestQ.npy", allow_pickle=True).item()

import gym
action_dict = {'a': 0, 's':1, 'd': 2, 'w': 3}
# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

# pass_loc_dict = {0: (0,0), 1:(0,4), 2:(4,0), 3:(4,3), 4:('taxi')}
action_dict = {0: 'LEFT',1: 'DOWN', 2:'RIGHT', 3: 'UP'}
env = gym.make(env_name)
cnt = 0
observation = env.reset()
if render_img:
    env.render(mode="human")
if print_vals:
    row, col = env.s // env.ncol, env.s % env.ncol
    loc = (row, col)
    print(cnt, loc, env.desc[row,col], (env.desc.shape[0]-1,env.desc.shape[1]-1), '_', '_')

taxi_row, taxi_col, pass_loc, dest_idx = env.decode(observation)
observation = env.env.s
total_reward = 0
for num_step in range(100):
    cnt+=1
    action = np.argmax(bestQ[observation])
    observation, reward, done, info = env.step(action)
    if render_img:
        env.render(mode="human")
    if print_vals:
        row, col = env.s // env.ncol, env.s % env.ncol
        loc = (row, col)
        print(cnt, loc, env.desc[row,col], (env.desc.shape[0]-1,env.desc.shape[1]-1) , action_dict[action], reward)
    # print('reward: {}'.format(reward))
    total_reward += reward
    if done:
        observation = env.reset()
        break
env.close()
print('Games over\nNumber of steps: {}\nTotal reward: {}\n'.format(num_step+1, total_reward))

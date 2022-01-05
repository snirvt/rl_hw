import numpy as np
import matplotlib.pyplot as plt

from qac import QAC, get_features
from Q_learner import get_mean_action
env_name  = 'Pendulum-v0'

render_img = False
print_vals = True

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

sample_size = 100
n_steps = 15000
step_size = 1000

eps = 1
decay_rate = 0.95
elegability_trace = True
GAMMA = 1
dict_plot = {}
for ALPHA in [0.00001]:
    for BETA in [0.00001]:
        param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA,'BETA': BETA, 'sample_size': sample_size,
                    'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
                    'elegability_trace': elegability_trace}
        Q, value_mean, bestQ = QAC(param_dict)
        plt.plot(np.arange(len(value_mean))*step_size,value_mean, label=(GAMMA, ALPHA, BETA))
plt.legend(title="GAMMA, ALPHA, BETA")
fig.suptitle('Q Function Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Q Function', fontsize=15)
plt.show()



# np.save("bestQ.npy", bestQ)



# bestQ = np.load("bestQ.npy", allow_pickle=True)


import gym

env = gym.make(env_name)
cnt = 0
observation = env.reset()
if render_img:
    _ = env.render(mode="human")

total_reward = 0
print_list = []
for num_step in range(200):
    cnt+=1
    f_s = get_features(observation)
    action = get_mean_action(f_s, bestQ)
    observation, reward, done, info = env.step(action)
    reward += 20
    if render_img:
        _ = env.render(mode="human")
    if print_vals:
        print_list.append([cnt, *action[0], observation, reward[0]])
    total_reward += reward
    if done:
        observation = env.reset()
        break
env.close()

print('Games over\nNumber of steps: {}\nTotal reward: {}\n'.format(num_step+1, total_reward))
for i in range(len(print_list)-1):
    cnt, action, observation, reward = print_list[i]
    observation_next = print_list[i+1][2]
    print(cnt, format(action,".2f"), tuple(np.round(observation,3).ravel()), tuple(np.round(observation_next,3).ravel()), format(reward,".2f"))




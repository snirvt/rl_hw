import numpy as np
import matplotlib.pyplot as plt

from sarsa import SARSA,get_features,get_max_Q
from epsilon import ExponentialDecay, SimpleEpsilon, LinearDecay
env_name  = 'MountainCar-v0'

render_img = True
print_vals = True

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

sample_size = 250
n_steps = 200000
step_size = 10000
eps = 1
decay_rate = 0.95
elegability_trace = True
GAMMA = 1

dict_plot = {}
for LAMBDA in [0]:
    for ALPHA in [0.02]:
        param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA, 'LAMBDA': LAMBDA, 'sample_size': sample_size,
                    'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
                    'elegability_trace': elegability_trace}
        Q, value_mean, bestQ = SARSA(param_dict)
        plt.plot(np.arange(len(value_mean))*step_size,value_mean, label=(GAMMA, ALPHA, LAMBDA, p))
plt.legend(title="GAMMA, ALPHA, LAMBDA")
fig.suptitle('Value Function Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Value Function', fontsize=15)
plt.show()



# np.save("bestQ.npy", bestQ)






# bestQ = np.load("bestQ.npy", allow_pickle=True)



import gym
target_loc_vel = (0.5,0)
action_dict = {0:'la', 1:'na', 2:'ra'}

env = gym.make(env_name)
cnt = 0
observation = env.reset()
if render_img:
    _ = env.render(mode="human")
if print_vals:
    position, velocity = observation    
    print(cnt, *observation, target_loc_vel, '_', '_')

# observation = env.env.s
total_reward = 0
for num_step in range(200):
    cnt+=1
    f_s = get_features(observation)
    action, _ = get_max_Q(f_s, bestQ)
    observation, reward, done, info = env.step(action)
    if render_img:
        _ = env.render(mode="human")
    if print_vals:
        print(cnt, *observation, target_loc_vel, action_dict[action], reward)
    # print('reward: {}'.format(reward))
    total_reward += reward
    if done:
        observation = env.reset()
        break
env.close()
print('Games over\nNumber of steps: {}\nTotal reward: {}\n'.format(num_step+1, total_reward))



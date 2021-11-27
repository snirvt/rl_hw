

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
eps_updater = LinearDecay(eps = 1 , p = 0.99)
# eps_updater = ExponentialDecay(gamma = 0.001 , N = 100)
# eps_updater = SimpleEpsilon(k = 100)

dict_plot = {}

for LAMBDA in [0, 0.1, 0.2, 0.3, 0.4]:
    for ALPHA in [0.1, 0.2, 0.3, 0.4]:
        param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA, 'LAMBDA': LAMBDA, 'sample_size': sample_size,
                    'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
                    'elegability_trace': elegability_trace, 'eps_updater': LinearDecay(eps = 0.1, p = 0.95, min_eps=0)}
        Q, value_mean, bestQ  = QLearning(param_dict)
        dict_plot[(GAMMA, ALPHA, LAMBDA)] = (np.arange(len(value_mean))*step_size,value_mean, value_mean, (GAMMA, ALPHA, LAMBDA))
        # np.save("dict_plot_discounted_eps_0.1_0.95.npy", dict_plot)
        plt.plot(np.arange(len(value_mean))*step_size,value_mean, label=(GAMMA, ALPHA, LAMBDA))
plt.legend(title="GAMMA, ALPHA, LAMBDA")
fig.suptitle('Value Function Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Value Function', fontsize=15)
plt.show()






dict_plot=np.load("dict_plot_discounted_eps_0.1_0.95.npy", allow_pickle=True).item()
GAMMA = 0.95
fig = plt.figure()
for LAMBDA in [0, 0.2]:
    for ALPHA in [0.1, 0.2, 0.3, 0.4]:
        x = dict_plot[(GAMMA, ALPHA, LAMBDA)][0]
        value = dict_plot[(GAMMA, ALPHA, LAMBDA)][1]
        par = dict_plot[(GAMMA, ALPHA, LAMBDA)][-1]
        plt.plot(x,value, label=par)
plt.legend(title="GAMMA, ALPHA, LAMBDA")
fig.suptitle('Policy Evaluation Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Value Function', fontsize=15)
plt.show()




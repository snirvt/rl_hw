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
n_steps = 100000
step_size = 5000
eps = 1
decay_rate = 0.95
elegability_trace = True
GAMMA = 1

dict_plot = {}
for ALPHA in [0.01]:
    param_dict = {'GAMMA': GAMMA,'ALPHA': ALPHA, 'sample_size': sample_size,
                'eps': eps, 'decay_rate': decay_rate, 'n_steps': n_steps, 'step_size': step_size,
                'elegability_trace': elegability_trace}
    Q, value_mean, bestQ = SARSA(param_dict)
    plt.plot(np.arange(len(value_mean))*step_size,value_mean, label=(GAMMA, ALPHA))
plt.legend(title="GAMMA, ALPHA, LAMBDA")
fig.suptitle('Q Function Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Q Function', fontsize=15)
plt.show()

# np.save('bestQ.npy', bestQ)
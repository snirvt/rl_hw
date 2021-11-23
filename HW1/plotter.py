

import numpy as np
import matplotlib.pyplot as plt
from policy_iteration import policy_iteration
from P_learner import get_P

P, T, PT = get_P()

fig = plt.figure()

sample_size = 1000
for GAMMA in [0.9, 0.95, 0.99]:
    policy, value, value_sum = policy_iteration(P, T, PT,max_steps = 30, GAMMA=GAMMA,
                                                quit_when_optimal = False, sample_size = sample_size)
    plt.plot(value_sum, label=GAMMA)
plt.legend(title="GAMMA")
fig.suptitle('Value Function Mean Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Value Function', fontsize=15)
plt.show()








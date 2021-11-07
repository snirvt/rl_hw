

import numpy as np
import matplotlib.pyplot as plt
from policy_iteration import policy_iteration
from P_learner import get_P

P, T, PT = get_P()

fig = plt.figure()

for GAMMA in [0.95]:
    policy, value, value_sum = policy_iteration(P, T, PT, GAMMA=GAMMA, quit_when_optimal = True)
    plt.plot(value_sum, label=GAMMA)
plt.legend(title="GAMMA")
fig.suptitle('Value Function Sum Over Iterations', fontsize=20)
plt.xlabel('Iteration Number', fontsize=15) 
plt.ylabel('Value Function', fontsize=15)
plt.show()








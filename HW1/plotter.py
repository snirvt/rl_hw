

import numpy as np
import matplotlib.pyplot as plt
from policy_iteration import policy_iteration
from P_learner import get_P

P, T, PT = get_P()
policy, value, value_sum = policy_iteration(P, T, PT)

plt.plot(value_sum)
plt.show()








    
import numpy as np   
from scipy.special import softmax
    
def init_W(n_actions = 3, n_features = 5):
    return np.random.randn(n_actions, n_features)/np.sqrt(n_actions * n_features)
    
def get_Q(f_s, A, W):
    return f_s @ W[A] 

def get_max_Q(f_s, W, Theta, actions, heat):
    max_action_val = float('-inf')
    best_action = -1
    actions_list = []
    values = []
    for i in range(len(actions)):
        actions_list.append(get_Q(f_s, i, Theta))
        values.append(get_Q(f_s, i, W))
    sm = softmax(np.array(actions_list)*heat)
    return sample_action(sm, values, actions)


def sample_action(x, y, actions):
    # actions = [-2, -1,-0.5, 0.5, 1, 2]
    action_idx = np.random.choice(range(len(actions)),p=x)
    return action_idx, y[action_idx]
    
def get_mean_w(f_s, W, actions):
    values = []
    for i in range(len(actions)):
        values.append(get_Q(f_s, i, W))
    return np.mean(values)

    
    
def sample_gaussian_action(f_s, W, Theta, var):
    mean = (f_s @ Theta.T)
    value = (f_s @ W.T)
    action = np.random.normal(mean, var)
    clipped_action = np.tanh(action) * 2
    return clipped_action, value, mean
    
        
    
  


    
    
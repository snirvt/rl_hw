    
import numpy as np   
   
    
def init_W(n_actions = 3, n_features = 5):
    return np.random.rand(n_actions, n_features)
    
def get_Q(f_s, A, W):
    return f_s @ W[A] 

def get_max_Q(f_s, W):
    max_action_val = float('-inf')
    best_action = -1
    for i in range(3):
        if get_Q(f_s, i, W) > max_action_val:  
            max_action_val = f_s @ W[i] 
            best_action = i
    return best_action, max_action_val
  


    
    
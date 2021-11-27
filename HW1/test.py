import numpy as np

def modify_env(env):
    
    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)
    
    env.orig_reset =  env.reset
    env.reset = new_reset
    return env

# def new_reset(env, state=None):
#     env.orig_reset()  
#     if state is not None:
#         env.env.s = state

#     # return np.array(env.env.s)
    
#     env.orig_reset =  env.reset
#     env.reset = new_reset
#     return env





# a = [1,2,3,4,5]#,6,7,8,9,10]
a = [5,4,3,2,1]#,6,7,8,9,10]

b = [1,2,3,4,5]#,6,7,8,9,10]

events = np.array([1, 1.2, 9, 10])
preds = np.array([1.2, 1, 10, 9])
event_obs = np.array([True, True, True, True, True])


sort_idx = np.argsort(events)
events_sort = events[sort_idx]
preds_sort = preds[sort_idx]
event_obs_sort = event_obs[sort_idx]

const = 0.25
p = 0.11


cnt = 0
correct_cnt = 0
for i in range(len(preds_sort)-1):
    if ~event_obs_sort[i]:
        continue
    for j in range(i+1,len(preds_sort)):
        pred_adj = min(preds_sort[i]*(1-p), preds_sort[i] - const) # percentage flexibility by prediction
        # pred_adj = min(preds_sort[i]-events_sort*(1-p), preds_sort[i] - const) # percentage flexibility by real
        if (pred_adj < preds_sort[j]):
            correct_cnt+=1
        if (pred_adj == preds_sort[j]):
            correct_cnt+=0.5
        cnt+=1
        print(preds_sort[i], preds_sort[j])

correct_cnt/cnt


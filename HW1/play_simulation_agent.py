
import numpy as np
from policy_iteration import policy_iteration
from P_learner import get_P

GAMMA = 0.95
render_img = False
print_vals = True
P, T, PT = get_P() ## get the transition matrix
for max_steps in [20]:
    policy, value, value_sum = policy_iteration(P, T, PT, max_steps = max_steps, GAMMA=GAMMA, quit_when_optimal = True)
    inp = input('Done training for {} steps.\npress enter for simulation...'.format(max_steps))


    import gym
    # from taxienv import TaxiEnv
    # env = TaxiEnv()
    pass_loc_dict = {0: (0,0), 1:(0,4), 2:(4,0), 3:(4,3), 4:('taxi')}
    action_dict = {0: 'South',1: 'North', 2:'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}
    env = gym.make('Taxi-v3')
    observation = env.reset()
    if render_img:
        env.render(mode="human")
    if print_vals:
        taxi_row, taxi_col, pass_loc, dest_idx = env.decode(observation)
        taxi_loc = (taxi_row, taxi_col)
        print(taxi_loc, pass_loc_dict[pass_loc], pass_loc_dict[dest_idx], '_', '_')

    taxi_row, taxi_col, pass_loc, dest_idx = env.decode(observation)
    observation = env.env.s
    total_reward = 0
    for num_step in range(50):
        action = np.argmax(policy[observation])
        observation, reward, done, info = env.step(action)
        if render_img:
            env.render(mode="human")
        if print_vals:
            taxi_row, taxi_col, pass_loc, dest_idx = env.decode(observation)
            taxi_loc = (taxi_row, taxi_col)
            if pass_loc != 4:
                print(taxi_loc, pass_loc_dict[pass_loc], pass_loc_dict[dest_idx], action_dict[action], reward)
            else:
                print(taxi_loc, taxi_loc, pass_loc_dict[dest_idx], action_dict[action], reward)
        # print('reward: {}'.format(reward))
        total_reward += reward
        if done:
            observation = env.reset()
            break
    env.close()
    print('Games over\nNumber of steps: {}\nTotal reward: {}\n'.format(num_step+1, total_reward))





# for state in range(500):
#     taxi_row, taxi_col, pass_loc, dest_idx = env.decode(state)
#     print(pass_loc, dest_idx)
    




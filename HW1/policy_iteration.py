import numpy as np

from P_learner import get_P
# def get_policy(P):

def init_policy(P): ## init to action 0 for all
    policy = {}
    for s in P.keys():
        # policy[s] = 0
        policy[s] = np.random.choice(6)
    return policy

def init_value(P): ## init to value 0 for all
    value = {}
    for s in P.keys():
        value[s] = 0
    return value


def policy_evaluation(policy, value, P, T, PT):
    new_value = value.copy()
    for s in P.keys():
        # if s not in T:
        v_temp = 0
        for a in range(6):
            PI_a_s = 1 if policy.get(s) == a else 0 # deterministic policy
            if PI_a_s: # only calculate the relevant action 
                R_a_s = P[s][a][0][2]
                v_temp += R_a_s
                for s_next in P.keys():
                    T_s_a_snext = 1 if s_next in PT[s] else 0 # is snext reachable from s
                    if T_s_a_snext: # only calculate the relevant next states 
                        v_temp += 0.9 * value[s_next]
        new_value[s] = v_temp
    return new_value


def policy_improvment(value, P):
    new_policy = {}
    for s in value.keys():
        s_next_list = [P[s][a][0][1] for a in range(6)]  # get all possible next states
        best_a = np.argmax([value[s_next] for s_next in s_next_list]) # take action (index) with the higest value
        new_policy[s] = best_a
    return new_policy



def policy_iteration(P, T, PT):
    policy = init_policy(P)
    value = init_value(P)

    for _ in range(1000):
        old_policy = policy.copy()
        value = policy_evaluation(policy, value, P, T, PT)
        policy = policy_improvment(value, P)

        if all(old_policy[s] == policy[s] for s in P.keys()):
            break
    return policy, value

P, T, PT = get_P()
policy, value = policy_iteration(P, T, PT)


for s in P.keys():
    if value[s] > 0:
        print(s)

for s in P.keys():
    for a in range(6):
        if P[s][a][0][2] > 0:
            print(s)


# def policy_improvment(V, S, A):
#     policy = {s: A[0] for s in S}

#     for s in S:
#         Q = {}
#         for a in A:
#             Q[a] =  R(s,a) + sum(P(s_next, s , a) * oldv[s_next] for s_next in S)
        
#         policy[s] = max(Q, key=Q.get)
#     return policy




# def policy_evaluation(policy, S, V):
#     # V = {s: A[0] for s in S}

#     while True:
#         oldv = V.copy()

#         for s in S:
#             a = policy[s]
#             V[s] = R(s,a) + sum(P(s_next, s , a) * oldv[s_next] for s_next in S)

#         if all(oldv[s] == V[s] for s in S):
#             break
#     return V


# def policy_iteration(S, A, P, R):
#     policy = {s: A[0] for s in S}

#     V = {}
#     while True:
#         old_policy = policy.copy()
#         V = policy_evaluation(policy, S, V)
#         policy = policy_improvment(V, S, A)

#         if all(old_policy[s] == policy[s] for s in S):
#             break
#     return policy















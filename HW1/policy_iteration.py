

# def get_policy(P):

def init_policy(P): ## init to action 0 for all
    policy = {}
    for s in P.keys():
        policy[s] = 0
    return policy

def init_value(P): ## init to value 0 for all
    value = {}
    for s in P.keys():
        value[s] = 0
    return value


def policy_evaluation(policy, value, P, T):
    new_value = value.copy()
    for s in P.keys():
        if s not in T:
            for a in range(6):
                for s_next in P.keys():
                    PI_a_s = 1 if policy.get(s) == a else 0 # deterministic policy
                    R_a_s = P[s][a][0][2]
                    T_s_a_snext = 
                    new_value = PI_a_s * policy[s]





def policy_iteration(S, A, P, R):
    policy = init_policy(P)
    value = init_value(P)

    for _ in range(100):
        old_policy = policy.copy()
        V = policy_evaluation(policy, S, V)
        policy = policy_improvment(V, S, A)

        if all(old_policy[s] == policy[s] for s in S):
            break
    return policy








def policy_improvment(V, S, A):
    policy = {s: A[0] for s in S}

    for s in S:
        Q = {}
        for a in A:
            Q[a] =  R(s,a) + sum(P(s_next, s , a) * oldv[s_next] for s_next in S)
        
        policy[s] = max(Q, key=Q.get)
    return policy




def policy_evaluation(policy, S, V):
    # V = {s: A[0] for s in S}

    while True:
        oldv = V.copy()

        for s in S:
            a = policy[s]
            V[s] = R(s,a) + sum(P(s_next, s , a) * oldv[s_next] for s_next in S)

        if all(oldv[s] == V[s] for s in S):
            break
    return V


def policy_iteration(S, A, P, R):
    policy = {s: A[0] for s in S}

    V = {}
    while True:
        old_policy = policy.copy()
        V = policy_evaluation(policy, S, V)
        policy = policy_improvment(V, S, A)

        if all(old_policy[s] == policy[s] for s in S):
            break
    return policy















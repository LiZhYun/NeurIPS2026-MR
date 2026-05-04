import numpy as np
from numba import njit

@njit(cache=True)
def prepare_group_cost(group_cost, cost):
    L = cost.shape[0]
    L2 = cost.shape[1]
    for i in range(L):
        for j in range(i + 1, L + 1):
            for k in range(L2 - (j - i - 1)):
                group_cost[i, j, k] = group_cost[i, j - 1, k] + cost[j - 1, k + j - i - 1]

@njit(cache=True)
def nn_dp(G, E, F, Cost, tmin, L, Nt):
    G[0] = 0
    for i in range(tmin, L + 1):
        for k in range(Nt):
            for l in range(i - tmin + 1):
                new_val = G[l] + Cost[l, i, k]
                if new_val < G[i]:
                    G[i] = new_val
                    E[i] = k
                    F[i] = l

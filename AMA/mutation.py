import numpy as np
from numba import njit
from fitness import fitness

@njit(cache=True)
def two_opt_mutation(tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time):
    N = tour.shape[0]
    if N <= 1:
        return tour.copy(), fitness(tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time), 0, 0

    new_tour = tour.copy()

    # Pilih dua titik potong secara acak — benar-benar uniform
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    while j == i:
        j = np.random.randint(0, N)
    if i > j:
        i, j = j, i

    # BALIKKAN SUB-ARRAY SECARA AMAN (Two-Pointer Swap)
    left = i
    right = j
    while left < right:
        tmp = new_tour[left]
        new_tour[left] = new_tour[right]
        new_tour[right] = tmp
        left += 1
        right -= 1

    new_cost = fitness(new_tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time)

    return new_tour, new_cost, i, j
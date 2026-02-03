import numpy as np
from numba import njit
from numba.typed import List
from tsp.representation import is_valid_tour, tour_cost, invert_permutation
from tsp.subtours import make_subtour, initialize_subtours, repair_tour

"""
CROSSOVER OPERATORS:

This module implements OX (Order Crossover) operators for ATSP tours
"""

@njit(cache=True)
def OX(parent1, parent2, distance_matrix):
    n = parent1.shape[0]
    child = np.full(n, -1)

    # 1. pilih cut points
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)

    # 2. copy slice dari parent1
    for k in range(i, j):
        child[k] = parent1[k]

    # 3. isi sisanya dari parent2 (preserve order)
    pos = j % n
    for k in range(n):
        city = parent2[(j + k) % n]
        if not contains(child, city):
            child[pos] = city
            pos = (pos + 1) % n

    cost = tour_cost(child, distance_matrix)
    return child, cost

@njit
def contains(arr, val):
    for x in arr:
        if x == val:
            return True
    return False


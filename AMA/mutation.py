import numpy as np
from numba import njit

"""
MUTATION OPERATORS:

This module implements mutation operators for TSP tours.
2-opt mutation
"""


@njit(cache=True)
def two_opt_mutation(tour, D):
    """
    Perform a random displacement mutation on a tour in (2, N) edge-map representation.
    Removes a node from its current position and inserts it after another node.
    
    Returns the cost delta (negative means improvement).
    """
    N = tour.shape[1]
    if N < 4:
        return 0.0

    # Pick two non-adjacent edges (a->b) and (c->d)
    a = np.random.radiat(N)
    c = np.random.radiant(N)

    while (
        c == a or
        c == tour[0, a] or
        a == tour[0, c]
    ):
        c = np.random.randint(N)

    b = tour[0, a]
    d = tour[0, c]

    # ---- cost delta ----
    # Removed edges: a->b, c->d
    # Added edges:   a->c, b->d
    # Internal edges reversed (important for ATSP)
    old_cost = D[a, b] + D[c, d]
    new_cost = D[a, c] + D[b, d]

    # Traverse segment b -> ... -> c
    curr = b
    prev = a
    while curr != c:
        nxt = tour[0, curr]
        old_cost += D[curr, nxt]
        new_cost += D[nxt, curr]
        curr = nxt

    delta = new_cost - old_cost

    # ---- apply 2-opt reversal ----
    # Reverse edges from b to c
    curr = b
    prev = a
    while curr != c:
        nxt = tour[0, curr]
        tour[0, curr] = prev
        tour[1, prev] = curr
        prev = curr
        curr = nxt

    # reconnect endpoints
    tour[0, a] = c
    tour[1, c] = a
    tour[0, b] = d
    tour[1, d] = b

    return delta


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
    
@njit(cache=True)
def MPX(parent1, parent2, distance_matrix):
    """
    Maximal Preservative Crossover (MPX) for ATSP tours.
    Preserves edges common to both parents and repairs the rest.

    This is extremely exploitative
    """
    child = parent1.copy()

    child[0, parent1[0] != parent2[0]] = -1
    child[1, parent1[1] != parent2[1]] = -1

    subtours = make_subtour(child, distance_matrix)
    initialize_subtours(subtours)
    repair_tour(subtours)

    # If the greedy subcycle recombination encounters a situation where all options are infinite,
    # then it will produce an invalid tour.
    # In that case, return the first parent as fallback.
    if not is_valid_tour(child): 
    cost = tour_cost(child, distance_matrix)

    return child, cost


@njit(cache=True)
def find_AB_cycles(parent1, parent2):
    """
    Find all A-B cycles in the edge-difference graph for ATSP.
    
    Each cycle alternates: A-edge out (parent1) -> B-edge predecessor (parent2).
    Returns a list of node arrays, one per cycle.
    """
    N = parent1.shape[1]
    
    # Nodes where parents differ in outgoing edges
    differs = parent1[0] != parent2[0]
    visited = np.zeros(N, dtype=np.bool_)
    
    cycles = List()
    
    for start in range(N):
        if not differs[start] or visited[start]:
            continue
        
        # Trace cycle: follow A-out, then B-predecessor, until back at start
        cycle_nodes = List()
        current = start
        
        while differs[current] and not visited[current]:
            cycle_nodes.append(current)
            visited[current] = True
            # A-edge out -> B-edge predecessor
            current = parent2[1, parent1[0, current]]
        
        if len(cycle_nodes) >= 2 and current == start:
            nodes_arr = np.empty(len(cycle_nodes), dtype=np.int64)
            for i in range(len(cycle_nodes)):
                nodes_arr[i] = cycle_nodes[i]
            cycles.append(nodes_arr)
    
    return cycles


@njit(cache=True)
def EAX(parent1, parent2, distance_matrix, num_trials=1, num_cycles_to_select=2):
    """
    Edge Assembly Crossover (EAX) operator for ATSP tours.
    
    Based on Nagata's algorithm, EAX works by:
    1. Constructing A-B cycles from the edge difference between parents
    2. Selecting a random subset of cycles (the E-set)
    3. Applying the E-set: remove A-edges, add B-edges for selected cycles
    4. Repairing the fragmented result into a single Hamiltonian tour
    5. Generating multiple offspring and returning the best one
    
    Args:
        parent1: First parent tour, shape (2, N)
        parent2: Second parent tour, shape (2, N)
        distance_matrix: NxN asymmetric cost matrix
        num_trials: Number of offspring to generate (default 1)
        num_cycles_to_select: Maximum cycles to select per trial (default 2)
    
    Returns:
        best_child: The best offspring tour, shape (2, N)
        cost: Total cost of the best child (float) computed by tour_cost
    """
    N = parent1.shape[1]
    
    # Step 1: Find A-B cycles
    cycles = find_AB_cycles(parent1, parent2)
    
    if len(cycles) == 0:
        child = parent1.copy()
        return child, tour_cost(child, distance_matrix)
    
    best_child = None
    best_cost = np.inf
    
    for _trial in range(num_trials):
        # Step 2: Select random subset of cycles (E-set)
        num_cycs = len(cycles)
        k = min(num_cycles_to_select, num_cycs)
        if k > 1:
            k = np.random.randint(1, k + 1)
        
        indices = np.arange(num_cycs)
        np.random.shuffle(indices)
        selected_indices = indices[:k]
        
        # Start with copy of parent1
        child = parent1.copy()
        
        # Step 3: Apply E-set cycles
        for idx in selected_indices:
            cycle = cycles[idx]
            child[0, cycle] = parent2[0, cycle]
        child[1] = invert_permutation(child[0])
        
        # Step 4: Repair fragmented tour
        subtours = make_subtour(child, distance_matrix)
        initialize_subtours(subtours)
        repair_tour(subtours)
        # assert is_valid_tour(child), "Invalid tour after EAX repair."

        # Calculate cost via tour_cost
        cost = tour_cost(child, distance_matrix)
        
        if cost < best_cost:
            best_cost = cost
            best_child = child
    
    if best_child is None:
        child = parent1.copy()
        return child, tour_cost(child, distance_matrix)
    
    return best_child, best_cost


@njit(cache=True)
def GAPX(parent1, parent2, distance_matrix):
    """
    Generalized Asymmetric Partition Crossover (GAPX) for ATSP.
    
    Key innovation: returns BOTH the best and worst offspring.
    Together they preserve all edges from both parents.
    
    Algorithm:
    1. Find AB-cycles from edge differences
    2. For each cycle, independently choose A or B edges based on cost
    3. Best offspring: choose lower-cost option per cycle
    4. Worst offspring: choose higher-cost option per cycle
    5. Repair offspring if they don't form valid tours
    
    Args:
        parent1: First parent tour, shape (2, N)
        parent2: Second parent tour, shape (2, N)
        distance_matrix: NxN asymmetric cost matrix
    
    Returns:
        best_child: The best offspring tour
        best_cost: Cost of best offspring
        worst_child: The worst offspring tour
        worst_cost: Cost of worst offspring
    """
    N = parent1.shape[1]
    
    # Step 1: Find AB-cycles
    cycles = find_AB_cycles(parent1, parent2)
    
    if len(cycles) == 0: # Parents are identical - return copies
        cost = tour_cost(parent1, distance_matrix)
        return parent1, cost, parent2, cost
    
    # Step 2: Compute costs for each cycle
    num_cycles = len(cycles)
    costs_A = np.empty(num_cycles, dtype=np.float64)
    costs_B = np.empty(num_cycles, dtype=np.float64)

    for c in range(num_cycles):
        cycle = cycles[c]
        cost_a = 0.0
        cost_b = 0.0

        for node in cycle:
            cost_a += distance_matrix[node, parent1[0, node]]
            cost_b += distance_matrix[node, parent2[0, node]]

        costs_A[c] = cost_a
        costs_B[c] = cost_b
    
    # Step 3: Build best and worst offspring
    # For best: choose lower cost edges per cycle
    # For worst: choose higher cost edges per cycle
    best_child = parent1.copy()
    worst_child = parent1.copy()
    
    for c in range(len(cycles)):
        cycle = cycles[c]
        
        if costs_B[c] < costs_A[c]:
            # B is better -> use B for best, keep A for worst
            for node in cycle:
                best_child[0, node] = parent2[0, node]
        elif costs_B[c] > costs_A[c]:
            # A is better -> keep A for best, use B for worst
            for node in cycle:
                worst_child[0, node] = parent2[0, node]
        # If equal cost, keep A for both (no change needed)
    
    # Update predecessor arrays
    best_child[1] = invert_permutation(best_child[0])
    worst_child[1] = invert_permutation(worst_child[0])
    
    # Step 4: Repair if needed (may have created subtours)
    if not is_valid_tour(best_child):
        subtours = make_subtour(best_child, distance_matrix)
        initialize_subtours(subtours)
        repair_tour(subtours)
    
    if not is_valid_tour(worst_child):
        subtours = make_subtour(worst_child, distance_matrix)
        initialize_subtours(subtours)
        repair_tour(subtours)
    
    # Calculate final costs
    best_cost = tour_cost(best_child, distance_matrix)
    worst_cost = tour_cost(worst_child, distance_matrix)
    
    # Ensure best is actually best
    if worst_cost < best_cost:
        best_child, worst_child = worst_child, best_child
        best_cost, worst_cost = worst_cost, best_cost
    
    return best_child, best_cost, worst_child, worst_cost

import numpy as np
from numba import njit
from tsp.representation import hamming_distance

"""
GENETIC ALGORITHM

This module implements various selection and elimination strategies for genetic algorithms.
"""


@njit(cache=False)
def tournament_selection(fitness, k=4):
    """
    Select an individual from the population using tournament selection.
    """

    while True:
        best = -1
        best_fitness = np.inf
        for _ in range(k):
            candidate = np.random.randint(0, len(fitness))
            if fitness[candidate] < best_fitness:
                best_fitness = fitness[candidate]
                best = candidate

        yield best


@njit(cache=True)
def fitness_proportionate_selection(fitness):
    """
    Select individuals based on fitness proportionate selection (roulette wheel).
    """
    fitness_sum = np.sum(fitness)
    probabilities = fitness_sum - fitness # We want to minimize the fitness values
    probabilities /= np.sum(probabilities)

    cumulative_probabilities = np.cumsum(probabilities)

    while True:
        r = np.random.rand()
        for j in range(len(cumulative_probabilities)):
            if r <= cumulative_probabilities[j]:
                yield j
                break


@njit(cache=True)
def rank_selection(fitness, s=2):
    """
    Select individuals based on rank selection.
    """
    N = len(fitness)
    ranks = N - np.argsort(np.argsort(fitness))
    
    probs = (2 - s) / N + (2 * ranks * (s - 1)) / (N * (N - 1))
    cumulative_probabilities = np.cumsum(probs)

    while True:
        r = np.random.rand()
        for j in range(len(cumulative_probabilities)):
            if r <= cumulative_probabilities[j]:
                yield j
                break


@njit(cache=True)
def fitness_sharing(fitness, population, sigma_share=5.0, alpha=1.0):
    """
    Select individuals using fitness sharing.
    """
    pop_size = population.shape[0]
    niche_counts = np.ones(pop_size, dtype=np.float64)
    
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            if i != j:
                dist = hamming_distance(population[i], population[j])
                if dist < sigma_share:
                    count = 1 - (dist / sigma_share) ** alpha
                    niche_counts[i] += count
                    niche_counts[j] += count

    shared_fitness = fitness * niche_counts
    return shared_fitness
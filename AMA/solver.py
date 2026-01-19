import numpy as np
import time
from numba import njit, prange
from typing import NamedTuple
from tsp.representation import tour_cost, is_valid_tour, to_city_order, hamming_distance
from tsp.greedy import greedy_cycle
from tsp.selection import tournament_selection, rank_selection, fitness_sharing
from tsp.crossover import MPX, EAX, GAPX
from tsp.mutation import double_bridge, reverse, k_segment_perturbation
from tsp.search import precompute_candidates, lso, LSO_2OPT, LSO_3OPT, LSO_OROPT
from tsp.reporter import Reporter


class Config(NamedTuple):
    distance_matrix: np.ndarray
    candidates: np.ndarray
    population_size: int
    offspring_size: int
    init_temp: float
    tournament_size: int
    window_size: int
    mutation_rates: np.ndarray
    search_iters: np.ndarray


class MemeticATSP:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        population_size: int = 128,
        offspring_size: int = 128,
        init_temp: float = 0.05,
        tournament_size: int = 4,
        window_size: int = 100,
        mutation_rates: tuple = (0.25, 0.25), # (double_bridge, reverse)
        search_iterations: tuple = (1, 1, 1), # (3opt, oropt, 2opt)
    ):
        self.population = None
        self.fitness = None
        self.age = None
        self.global_best_tour = None
        self.global_best_fitness = np.inf
        self.latest_offspring = None
        self.config = None
        self.generation = 0
        self.timings = {
            'offspring': 0.0,
            'mutation': 0.0,
            'search': 0.0,
            'elimination': 0.0
        }
        self.config = Config(
            distance_matrix=distance_matrix,
            candidates=precompute_candidates(distance_matrix, num_candidates=20),
            population_size=population_size,
            offspring_size=offspring_size,
            init_temp=init_temp,
            tournament_size=tournament_size,
            window_size=window_size,
            mutation_rates=np.array(mutation_rates, dtype=np.float64),
            search_iters=np.array(search_iterations, dtype=np.int64),
        )

    def initialize(self):
        """Initialize the population."""
        self.population = initialization(self.config)
        self._update_fitness()
        self.age = np.zeros(self.config.population_size, dtype=np.int64)
        self.generation = 0
        self._update_global_best()

    def _update_fitness(self):
        """Update fitness values for the population."""
        self.fitness = np.array(
            [tour_cost(tour, self.config.distance_matrix) for tour in self.population]
        )

    def _update_global_best(self):
        """Update global best if a better solution is found."""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        if current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_tour = self.population[current_best_idx].copy()

    def step(self):
        """Run one generation of the EA."""
        # Offspring generation
        t_start = time.perf_counter()
        offspring, offspring_fitness = generate_offspring(
            self.population, self.fitness, self.config
        )
        self.timings['offspring'] = time.perf_counter() - t_start
        
        # Mutation
        t_start = time.perf_counter()
        mutation(offspring, offspring_fitness, self.config)
        self.timings['mutation'] = time.perf_counter() - t_start
        
        # Local search (offspring + population)
        t_start = time.perf_counter()
        search(offspring, offspring_fitness, self.config)
        search(self.population, self.fitness, self.config)
        self.timings['search'] = time.perf_counter() - t_start
        
        # Store offspring for diversity tracking
        self.latest_offspring = offspring
        
        # Elimination
        t_start = time.perf_counter()
        self.population, self.fitness, self.age = elimination(
            self.population, self.fitness, self.age, offspring, offspring_fitness, self.config.window_size
        )
        self.timings['elimination'] = time.perf_counter() - t_start
        
        # Update global best
        self._update_global_best()
        
        self.generation += 1
    
    def run(self, num_generations: int, reporter: Reporter = None):
        """Run the EA for a given number of generations.
        
        Args:
            num_generations: Number of generations to run.
            reporter: Optional Reporter instance for logging. If None, a default
                      reporter will be created.
        """
        if reporter is None:
            reporter = Reporter()
        
        if self.population is None:
            self.initialize()
        
        reporter.start(self)
        try:
            reporter.log(self)
            
            for _ in range(num_generations):
                self.step()
                reporter.log(self)
        finally:
            reporter.stop()
            return self.global_best_tour, self.global_best_fitness
                
    @property
    def best_fitness(self) -> float:
        """Return the best fitness in the population."""
        return float(np.min(self.fitness))

    @property
    def mean_fitness(self) -> float:
        """Return the mean fitness of the population."""
        return float(np.mean(self.fitness))

    @property
    def best_tour(self) -> np.ndarray:
        """Return the best tour in the population."""
        return self.population[np.argmin(self.fitness)]

    @property
    def mean_age(self) -> float:
        """Return the mean age of the population."""
        return float(np.mean(self.age))

    @property
    def max_age(self) -> int:
        """Return the maximum age in the population."""
        return int(np.max(self.age))

@njit(cache=True, parallel=True)
def initialization(config):
    """
    Initialize a population of tours using the greedy heuristic.
    """
    N = config.distance_matrix.shape[0]
    population = np.empty((config.population_size, 2, N), dtype=np.int_)

    for i in prange(config.population_size):
        tour = None
        while tour is None:
            tour = greedy_cycle(config.distance_matrix, config.init_temp)
        population[i] = tour

    return population

@njit(cache=True, parallel=True)
def mutation(population, fitness, config):
    """
    Apply double bridge mutation to each tour in the population with given mutation rate.
    Updates fitness values in-place after mutation.
    """
    for i in prange(population.shape[0]):
        if np.random.rand() < config.mutation_rates[0]:
            fitness[i] += double_bridge(population[i, ...], config.distance_matrix)
            # fitness[i] += k_segment_perturbation(population[i, ...], config.distance_matrix, k=5)
        if np.random.rand() < config.mutation_rates[1]:
            fitness[i] += reverse(population[i, ...], config.distance_matrix)
        # fitness[i] = tour_cost(population[i], config.distance_matrix)
        # assert is_valid_tour(population[i]), "Invalid tour after reverse mutation."

@njit(cache=True, parallel=True)
def search(population, fitness, config):
    """
    Apply local search to each tour in the population.
    Updates fitness values in-place after local search.
    """
    for i in prange(population.shape[0]):
        change = 0.0
        for search_type, iters in zip([0, 1, 2], config.search_iters):
            change += lso(population[i, ...], config.distance_matrix, config.candidates, search_type, iters)

        fitness[i] += change
        # fitness[i] = tour_cost(population[i], config.distance_matrix)
        # assert is_valid_tour(population[i]), "Invalid tour after local search."

@njit(cache=True)
def generate_offspring(population, pop_fitness, config):
    """
    Generate offspring using crossover from the population.
    Returns both offspring and their fitness values.
    """
    N = population.shape[2]
    offspring = np.empty((config.offspring_size, 2, N), dtype=np.int_)
    offspring_fitness = np.empty(config.offspring_size, dtype=np.float64)

    # shared_fitness = fitness_sharing(pop_fitness, population, sigma_share=50, alpha=1.0)
    offspring_generator = tournament_selection(pop_fitness, config.tournament_size)
    # offspring_generator = rank_selection(pop_fitness, 2)

    for i in range(config.offspring_size):
        p1 = population[next(offspring_generator)]
        p2 = population[next(offspring_generator)]
        if np.random.rand() < 0.8:
            child, cost = EAX(p1, p2, config.distance_matrix)
        else:
            child, cost, _other, _cost = GAPX(p1, p2, config.distance_matrix)
        # assert is_valid_tour(child), "Generated invalid tour in crossover."
        offspring[i] = child
        offspring_fitness[i] = cost

    return offspring, offspring_fitness


@njit(cache=True)
def elimination(population, population_fitness, population_age, offspring, offspring_fitness, window_size):
    """
    Eliminate individuals to maintain population size, using Restricted Tournament Replacement (RTR).
    Returns the new population, updated fitness values, and updated ages.
    """
    pop_size = population.shape[0]

    new_population = population.copy()
    new_fitness = population_fitness.copy()
    new_age = population_age + 1  # Increment age for all individuals

    for i in range(offspring.shape[0]):
        offspring_tour = offspring[i]
        offspring_fit = offspring_fitness[i]

        # Select random window
        start_idx = np.random.randint(0, pop_size)
        indices = [(start_idx + j) % pop_size for j in range(window_size)]

        # Find the most similar individual in the window
        best_idx = -1
        best_distance = np.inf

        for idx in indices:
            dist = hamming_distance(offspring_tour, new_population[idx])
            if dist < best_distance:
                best_distance = dist
                best_idx = idx

        # Replace if offspring is better
        if offspring_fit < new_fitness[best_idx]:
            new_population[best_idx] = offspring_tour
            new_fitness[best_idx] = offspring_fit
            new_age[best_idx] = 0  # Reset age for new offspring

    return new_population, new_fitness, new_age
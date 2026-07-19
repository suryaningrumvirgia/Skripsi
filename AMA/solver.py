import time
import numpy as np
from numba import njit, prange
from typing import NamedTuple
import pandas as pd

from simulator import hitung_detail_rute
from search import AdaptiveLocalSearch
from fitness import fitness
from selection import select_parents
from crossover import OX
from random_immigrant import apply_random_immigrant_scheme
from reporter import Reporter
from mutation import two_opt_mutation


class Config(NamedTuple):
    time_matrix: np.ndarray
    demands: np.ndarray
    sla_limits: np.ndarray
    max_capacity: float
    penalty_rate: float
    service_time: float
    start_time: float
    population_size: int
    pc: float
    pm: float
    ns_max: int
    ls_size: int
    delta: float
    time_matrix_global: np.ndarray = None
    customer_map: list = None
    pelanggan_baru: list = None
    time_matrix_global: np.ndarray = None
    customer_map: list = None
    pelanggan_baru: list = None
    dist_to_depot: np.ndarray = None
    dist_from_depot: np.ndarray = None
    is_new_customer: np.ndarray = None
    initial_load: float = 0.0


class MemeticSolver:
    def __init__(self, config: Config):
        self.config = config
        self.num_customers = config.time_matrix.shape[0] - 1

        self.population = None
        self.fitness = None
        self.global_best_tour = None
        self.global_best_fitness = np.inf

        self.generation = 0
        self.als = AdaptiveLocalSearch(time_matrix=config.time_matrix, delta=config.delta)

        self.als.history = []
        self.neighbor_history = []
        self.als.generation_history = []
        self.evolution_log = []

        self.reporter = Reporter("results/log.json")

    def initialize(self, customer_map=None, init_phase="static", seed=None):
        static_idx = np.arange(1, self.config.time_matrix.shape[0], dtype=np.int32)

        self.population = np.empty((self.config.population_size, self.num_customers), dtype=np.int32)

        for i in range(self.config.population_size):
            tour = np.copy(static_idx)
            np.random.shuffle(tour)
            self.population[i] = tour

        self.reporter.save_initial_population(
            self.population,
            self.config.start_time,
            customer_map=customer_map,
            init_phase=init_phase,
            seed=seed
        )

        self._update_fitness()
        self._update_global_best()

    def _update_fitness(self):
        self.fitness = np.array([
            fitness(
                giant_tour=tour,
                time_matrix=self.config.time_matrix,
                q_array=self.config.demands,
                sla_limit_array=self.config.sla_limits,
                start_time=self.config.start_time,
                max_capacity=self.config.max_capacity,
                penalty_rate=self.config.penalty_rate,
                service_time=self.config.service_time
            )
            for tour in self.population
        ])

    def _update_global_best(self):
        idx = np.argmin(self.fitness)

        if self.fitness[idx] < self.global_best_fitness:
            self.global_best_fitness = self.fitness[idx]
            self.global_best_tour = self.population[idx].copy()

    def step(self):
        offspring, offspring_fit, all_parents, all_children, all_mutations = generate_offspring(
            self.population,
            self.fitness,
            self.config.population_size,
            self.config.pc,
            self.config.pm,
            self.config.time_matrix,
            self.config.demands,
            self.config.sla_limits,
            self.config.start_time,
            self.config.max_capacity,
            self.config.penalty_rate,
            self.config.service_time,
        )

        self.generation_parents = all_parents
        self.generation_children = all_children
        self.generation_mutations = all_mutations

        combined_pop = np.vstack((self.population, offspring))

        combined_fit = np.concatenate([
            self.fitness,
            offspring_fit
        ])

        best_indices = np.argsort(combined_fit)[:self.config.population_size]

        self.population = combined_pop[best_indices]
        self.fitness = combined_fit[best_indices]

        elite_idx = np.argmin(self.fitness)

        elite_pop = self.population[elite_idx:elite_idx+1].copy()
        elite_fit = self.fitness[elite_idx:elite_idx+1].copy()

        elite_pop, elite_fit = adaptive_local_search_phase(
            elite_pop,
            elite_fit,
            self.als,
            self.config,
            self.generation,
            elite_indices=[elite_idx]
        )

        self.population[elite_idx] = elite_pop[0]
        self.fitness[elite_idx] = elite_fit[0]

        self.population, self.fitness, _ = apply_random_immigrant_scheme(
            self.population,
            self.fitness,
            self.num_customers,
            self.config.time_matrix,
            self.config.demands,
            self.config.sla_limits,
            self.config.start_time,
            self.config.max_capacity,
            self.config.penalty_rate,
            self.config.service_time
        )

        self._update_global_best()
        self.generation += 1

        return all_parents, all_children, all_mutations

    def run(self, num_generations, max_stagnant_gen=None, max_time_seconds=None, log_filename=None, customer_map=None, init_phase=None, seed=None):
        if self.population is None:
            # Inisialisasi populasi awal
            self.initialize(customer_map=customer_map, init_phase=init_phase, seed=seed)

        reporter = None
        if log_filename:
            reporter = Reporter(filename=log_filename)
            reporter.start()

        start_clock = time.time()
        stagnant_counter = 0
        previous_best = float("inf")

        # ALS pada populasi awal
        elite_idx = np.argmin(self.fitness)

        elite_pop = self.population[elite_idx:elite_idx+1].copy()
        elite_fit = self.fitness[elite_idx:elite_idx+1].copy()

        elite_pop, elite_fit = adaptive_local_search_phase(
            elite_pop,
            elite_fit,
            self.als,
            self.config,
            generation=0,
            elite_indices=[elite_idx]
        )

        self.population[elite_idx] = elite_pop[0]
        self.fitness[elite_idx] = elite_fit[0]

        self._update_global_best()
        previous_best = self.global_best_fitness

        self.generation = 1

        # Evolusi
        for gen in range(num_generations):
            all_parents, all_children, all_mutations = self.step()
            
            for pair_idx in range(len(all_parents)):
                p1, p2 = all_parents[pair_idx]
                c = all_children[pair_idx]
                m = all_mutations[pair_idx]

                self.evolution_log.append({
                    "generation": gen + 1,
                    "pair_index": pair_idx,
                    "parent_1": p1.tolist(),
                    "parent_2": p2.tolist(),
                    "cut_point": c["cut_point"],
                    "child_1": c["child1"].tolist(),
                    "child_2": c["child2"].tolist(),
                    "mutation_child_1": m["mutation_child1"].tolist(),
                    "mutation_child_2": m["mutation_child2"].tolist(),
                    "mutation_cut_1": m["mutation_cut_1"],
                    "mutation_cut_2": m["mutation_cut_2"],
                })

            current_best = self.global_best_fitness

            if current_best == previous_best:
                stagnant_counter += 1
            else:
                stagnant_counter = 0
                previous_best = current_best

            best_route_lokal = self.global_best_tour.tolist()

            waktu_tempuh, pelanggaran_sla, total_penalty, _, arrival_log = hitung_detail_rute(
                best_route_lokal,
                self.config.time_matrix,
                self.config.demands,
                self.config.sla_limits,
                self.config.start_time,
                self.config.max_capacity,
                self.config.penalty_rate,
                service_time=self.config.service_time
            )

            if reporter:
                reporter.log(
                    generation=gen + 1,
                    fitness_array=self.fitness,
                    rho_SI=self.als.rho_SI, 
                    rho_MI=self.als.rho_MI,
                    best_route=best_route_lokal,
                    waktu_tempuh=waktu_tempuh,
                    pelanggaran_sla=pelanggaran_sla,
                    penalty_cost=total_penalty
                )

            if max_stagnant_gen and stagnant_counter >= max_stagnant_gen:
                # print(f"[ STOP ] Konvergensi tercapai di Gen-{gen+1}")
                break

            # if max_time_seconds and (time.time() - start_clock) >= max_time_seconds:
                # print(f"[ STOP ] Batas waktu habis di Gen-{gen+1}")
                # break

        if reporter:
            reporter.stop()

        return self.global_best_tour, self.global_best_fitness


def generate_offspring(population, fitness_array, population_size, pc, pm, 
                       time_matrix, demands, sla_limits, start_time,
                       max_capacity, penalty_rate, service_time):
    
    N_customers = population.shape[1]
    
    offspring = np.empty((population_size, N_customers), dtype=np.int32)
    offspring_fitness = np.empty(population_size, dtype=np.float64)

    all_parents = []      
    all_children = []     
    all_mutations = [] 

    for i in range(0, population_size, 2):
        parent1_idx, parent2_idx = select_parents(fitness_array)

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        n = parent1.shape[0]

        # Pilih cut points
        cut_i = np.random.randint(0, n - 1)
        cut_j = np.random.randint(cut_i + 1, n)

        # Crossover
        if np.random.rand() < pc:
            child1 = OX(parent1, parent2, cut_i, cut_j) 
            child2 = OX(parent2, parent1, cut_i, cut_j)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()   

        if np.any(child1 == -1):
            print("Child1 belum penuh!", child1)

        if np.any(child2 == -1):
            print("Child2 belum penuh!", child2)         
        
        # Mutasi
        mutation_result_1 = two_opt_mutation(
            child1, time_matrix, demands, sla_limits, 
            start_time, max_capacity, penalty_rate, service_time
        ) if np.random.rand() < pm else (child1.copy(), None, -1, -1)

        mutation_child1 = mutation_result_1[0]
        mutation_cut_1 = (mutation_result_1[2], mutation_result_1[3]) # (i, j)

        mutation_result_2 = two_opt_mutation(
            child2, time_matrix, demands, sla_limits, 
            start_time, max_capacity, penalty_rate, service_time
        ) if np.random.rand() < pm else (child2.copy(), None,  -1, -1)

        mutation_child2 = mutation_result_2[0]
        mutation_cut_2 = (mutation_result_2[2], mutation_result_2[3])
    
        # Hitung fitness dari anak yang sudah melewati tahap mutasi
        mutation_child1_fit = fitness(mutation_child1, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time)
        offspring[i] = mutation_child1
        offspring_fitness[i] = mutation_child1_fit

        # Lakukan hal yang sama untuk anak kedua (jika belum melebihi batas populasi)
        if i + 1 < population_size:
            mutation_child2_fit = fitness(mutation_child2, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time)
            offspring[i+1] = mutation_child2
            offspring_fitness[i+1] = mutation_child2_fit

        all_parents.append((parent1.copy(), parent2.copy()))
        all_children.append({
            "child1": child1.copy(),
            "child2": child2.copy(),
            "cut_point": (cut_i, cut_j)
        })
        all_mutations.append({
            "mutation_child1": mutation_child1.copy(),
            "mutation_child2": mutation_child2.copy(),
            "mutation_cut_1": mutation_cut_1,
            "mutation_cut_2": mutation_cut_2,
        })
            
    return offspring, offspring_fitness, all_parents, all_children, all_mutations

def adaptive_local_search_phase(population, fitness_array, als, config, generation, elite_indices=None):
    pop_size = population.shape[0]

    for i in prange(pop_size):
        original_idx = elite_indices[i]
        s = population[i].copy()
        f_s = fitness_array[i]

        for siklus in range(config.ls_size):

            # ===============================
            # simpan kondisi awal cycle
            # ===============================
            s_before = s.copy()

            f_before = f_s

            rho_si_old = als.rho_SI
            rho_mi_old = als.rho_MI

            S_tour = []
            S_delta = []
            S_operator = []
            S_rand = []
            S_edge = []

            # ===============================
            # generate neighborhood
            # ===============================
            ns = 0
            while ns < config.ns_max:
                new_tour = s.copy()

                # pilih operator berdasarkan probabilitas adaptif
                operator_type, rand = als.select_operator()

                if operator_type == "SI":
                    new_tour, delta, edge_info = als.single_inversion(new_tour)
                else:
                    new_tour, delta, edge_info = als.multiple_inversion(new_tour)

                # simpan candidate
                S_tour.append(new_tour)
                S_delta.append(delta)
                S_operator.append(operator_type)
                S_rand.append(rand)
                S_edge.append(edge_info)

                # log neighbor
                als.neighbor_history.append({
                    "generation": generation,
                    "individual": original_idx,
                    "cycle": siklus,
                    "neighbor": ns,

                    "operator": operator_type,
                    "random": rand,

                    "tour_before": edge_info["tour_before"],
                    "tour_after": edge_info["tour_after"],

                    "i_move": edge_info["move_i"],
                    "j_move": edge_info["move_j"],

                    "delta": delta,

                    "a": edge_info["a"],
                    "b": edge_info["b"],
                    "c": edge_info["c"],
                    "d": edge_info["d"],

                    "ab": edge_info["ab"],
                    "ac": edge_info["ac"],
                    "bd": edge_info["bd"],
                    "cd": edge_info["cd"],
                    
                    "selected": False      # sementara
                })

                ns += 1

            # ===============================
            # pilih best neighbor (min delta)
            # ===============================
            best_idx = np.argmin(S_delta)
            s_star = S_tour[best_idx]
            best_delta = S_delta[best_idx]
            best_operator = S_operator[best_idx] 
            best_rand = S_rand[best_idx]
            als.neighbor_history[-config.ns_max + best_idx]["selected"] = True

            # ===============================
            # full fitness evaluation
            # ===============================
            f_star = fitness(
                s_star,
                config.time_matrix,
                config.demands,
                config.sla_limits,
                config.start_time,
                config.max_capacity,
                config.penalty_rate,
                config.service_time
            )

            eta = 0.0
            improved = False

            # ===============================
            # Terima kalau lebih baik
            # ===============================
            if f_star < f_s:
                eta = als.compute_eta(f_s, f_star)

                # Reward operator
                als.record_eta(best_operator, eta)

                # Update solusi
                s = s_star.copy()
                f_s = f_star
                improved = True

            # ===============================
            # Log cycle history
            # ===============================
            als.history.append({
                "generation": generation,
                "individual": original_idx,
                "cycle": siklus,

                "s_before": s_before,
                "f_ini": f_before,

                "rho_SI_before": rho_si_old,
                "rho_MI_before": rho_mi_old,

                "random": best_rand,
                "operator": best_operator,

                "best_neighbor": s_star.copy(),
                "best_delta": best_delta,
                "num_neighbors": len(S_tour),

                "f_imp": f_star,
                "eta": eta,
                "improved": improved,
            })

        # Simpan hasil individu setelah semua local search cycle selesai
        population[i] = s
        fitness_array[i] = f_s

    # ==========================================
    # Update operator probability per generation
    # ==========================================
    als.update_probabilities_per_generation(generation)
    
    return population, fitness_array
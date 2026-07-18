import numpy as np
from numba import njit
from fitness import fitness

@njit(cache=True)
def calculate_diversity_index(fitness_array):
    # Buat mask untuk menyaring nilai yang inf
    valid_fitness = fitness_array[~np.isinf(fitness_array)]
    
    # Deteksi jika populasi rusak total (semua inf)
    if len(valid_fitness) == 0:
        return 0.0

    f_best = np.min(valid_fitness)
    f_avg = np.mean(valid_fitness)

    if f_avg == 0:
        return 0.0

    xi = abs((f_avg - f_best)) / f_avg

    return np.minimum(xi, 1.0)

@njit(cache=True)
def calculate_immigrant_ratio(xi, ri_min=0.01, ri_max=0.1, alpha=1.0):
    ri = alpha * (1 - xi) * (ri_max - ri_min) + ri_min
    
    return np.minimum(ri, ri_max)

@njit(cache=True)
def generate_random_immigrant(num_customers):
    """
    Menghasilkan satu kromosom acak berformat 1D Giant Tour.
    """
    # Menghasilkan array [1, 2, ..., num_customers]
    tour = np.arange(1, num_customers + 1, dtype=np.int32)
    # Acak urutannya
    np.random.shuffle(tour)
    return tour

@njit(cache=True)
def apply_random_immigrant_scheme(population, fitness_array, num_customers, 
                                  time_matrix, demands, sla_limits, start_time,
                                  max_capacity, penalty_rate, service_time, 
                                  ri_min=0.01, ri_max=0.1, alpha=1.0):
    """
    Fungsi Utama: Mengganti individu terburuk dengan kromosom acak (Immigrant).
    """
    pop_size = population.shape[0]
    
    # 1. Hitung keragaman dan rasio immigrant
    xi = calculate_diversity_index(fitness_array)
    ri = calculate_immigrant_ratio(xi, ri_min, ri_max, alpha)
    
    # 2. Tentukan berapa banyak individu yang akan diganti
    num_immigrants = int(np.round(ri * pop_size))
    
    # Jika tidak ada yang perlu diganti, kembalikan populasi asli (ditambah ri untuk Reporter)
    if num_immigrants == 0:
        return population, fitness_array, ri
        
    # 3. Cari individu terburuk (fitness terbesar)
    sorted_indices = np.argsort(fitness_array)
    worst_indices = sorted_indices[-num_immigrants:]
    
    # 4. Lakukan penggantian (Replacement)
    for idx in worst_indices:
        # Bangkitkan imigran baru
        new_immigrant = generate_random_immigrant(num_customers)
        
        # Hitung fitness-nya (SEKARANG DENGAN PARAMETER LENGKAP)
        new_fitness = fitness(new_immigrant, time_matrix, demands, sla_limits, 
                             start_time, max_capacity, penalty_rate, service_time)
        
        # Ganti di populasi utama
        population[idx] = new_immigrant
        fitness_array[idx] = new_fitness
        
    return population, fitness_array, ri
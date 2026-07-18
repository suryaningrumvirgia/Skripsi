import numpy as np
from numba import njit
from fitness import fitness

@njit(cache=True)
def two_opt_mutation(tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time):
    N = tour.shape[0]
    if N <= 1:
        return tour.copy(), fitness(tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time), 0, 0  # Jika hanya ada satu node, kembalikan salinan tour dan biayanya
    
    # Duplikasi rute asli
    new_tour = tour.copy()

    # Pilih dua titik potong secara acak
    i = np.random.randint(0, N - 1)
    j = np.random.randint(i + 1, N)

    # BALIKKAN SUB-ARRAY SECARA AMAN (Two-Pointer Swap)
    # Menghilangkan bug angka kembar akibat overlap memori slicing Numba
    left = i
    right = j
    while left < right:
        # Tukar posisi elemen
        tmp = new_tour[left]
        new_tour[left] = new_tour[right]
        new_tour[right] = tmp
        
        # Gerakkan penunjuk ke dalam
        left += 1
        right -= 1

    # HITUNG BIAYA DENGAN PARAMETER LENGKAP
    old_cost = fitness(tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time)
    new_cost = fitness(new_tour, time_matrix, demands, sla_limits, start_time, max_capacity, penalty_rate, service_time)

    return new_tour, new_cost, i, j
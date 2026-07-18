import numpy as np
from numba import njit

@njit(cache=True)
def binary_tournament_selection(fitness):
    N = len(fitness)
    
    # Pilih 2 kandidat secara acak
    candidate1 = np.random.randint(0, N)
    candidate2 = np.random.randint(0, N)
    
    # Karena ini masalah MINIMISASI (waktu + SLA penalti), 
    # fitness TERKECIL adalah pemenangnya.
    if fitness[candidate1] < fitness[candidate2]:
        return candidate1
    else:
        return candidate2

@njit(cache=True)
def select_parents(fitness):
    """
    Fungsi pembantu untuk memilih DUA induk (parents) 
    sekaligus untuk proses Crossover.
    """
    parent1_idx = binary_tournament_selection(fitness)
    
    # Pastikan parent2 berbeda dari parent1 (mencegah inbreeding/crossover dengan diri sendiri)
    # Gunakan loop dengan batas maksimal untuk mencegah infinite loop jika populasi seragam
    parent2_idx = binary_tournament_selection(fitness)
    attempts = 0
    while parent2_idx == parent1_idx and attempts < 10:
        parent2_idx = binary_tournament_selection(fitness)
        attempts += 1
        
    return parent1_idx, parent2_idx
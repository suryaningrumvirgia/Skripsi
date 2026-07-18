import numpy as np
from numba import njit

# OX (Order Crossover)

@njit(cache=True)
def OX(parent1, parent2, cut_i, cut_j):
    n = parent1.shape[0]
    if n <= 1:
        return parent1.copy()  # Jika hanya ada satu node, kembalikan salinan parent1
    else:
        child = np.full(n, -1, dtype=np.int32)
        
        # Tambahkan + 1 agar indeks ID pelanggan (1 sampai n) tidak Out of Bounds
        in_child = np.zeros(n + 1, dtype=np.bool_)

        # Copy slice dari parent1
        for k in range(cut_i, cut_j + 1):
            child[k] = parent1[k]
            in_child[parent1[k]] = True

        # Isi sisanya dari parent2 dari kiri ke kanan
        parent2_idx = 0

        for pos in range(n):

            # kalau posisi sudah terisi hasil copy, lewati
            if child[pos] != -1:
                continue

            # cari gen parent2 berikutnya yang belum ada di child
            while parent2_idx < n and in_child[parent2[parent2_idx]]:
                parent2_idx += 1

            if parent2_idx < n:
                child[pos] = parent2[parent2_idx]
                in_child[parent2[parent2_idx]] = True
                parent2_idx += 1

        return child


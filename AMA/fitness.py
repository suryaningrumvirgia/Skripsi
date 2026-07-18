import numpy as np
from numba import njit
import pandas as pd

@njit(cache=True)
def fitness(
    giant_tour,
    time_matrix,
    q_array,
    sla_limit_array,
    start_time,
    max_capacity,
    penalty_rate,
    service_time,
):
    current_time = start_time
    current_load = 0.0
    total_penalty = 0.0
    prev_node = 0

    for node in giant_tour:

        # Jika kapasitas tidak cukup,
        # kembali ke depot untuk reload
        if current_load + q_array[node] > max_capacity:

            if prev_node != 0:
                current_time += time_matrix[prev_node, 0]

            prev_node = 0
            current_load = 0.0

        # Berangkat ke pelanggan
        current_time += time_matrix[prev_node, node]

        arrival_time = current_time

        # Penalti SLA
        late = max(0.0, arrival_time - sla_limit_array[node])
        total_penalty += penalty_rate * late

        # Service
        current_time += service_time

        current_load += q_array[node]
        prev_node = node

    # Kembali ke depot setelah pelanggan terakhir
    if prev_node != 0:
        current_time += time_matrix[prev_node, 0]

    total_time = current_time - start_time

    return total_time + total_penalty
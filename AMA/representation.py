import pandas as pd
import numpy as np
from numba import njit

def siapkan_data_vrp(file_pelanggan, waktu_batas_statis=0):
    # Membaca dan memisahkan pelanggan statis & dinamis
    df = pd.read_excel(file_pelanggan, sheet_name='Sheet2', index_col=0)
    df = df.sort_values(by='Arrival Time').reset_index(drop=True)
    
    statis_mask = df['Arrival Time'] <= waktu_batas_statis
    num_static_customers = int(statis_mask.sum())
    
    df['numba_id'] = np.arange(1, len(df) + 1)
    
    df_statis = df[statis_mask]
    df_dinamis = df[~statis_mask]
    
    return df, df_statis, df_dinamis, num_static_customers

@njit(cache=True)
def is_valid_static_tour(tour, num_static_customers):
    # Mengecek validitas rute statis awal
    if tour.shape[0] != num_static_customers:
        return False
        
    visited = np.zeros(num_static_customers + 1, dtype=np.int_)
    
    for i in range(tour.shape[0]):
        node = tour[i]
        if node <= 0 or node > num_static_customers:
            return False
        visited[node] += 1
        if visited[node] > 1:
            return False
            
    return True
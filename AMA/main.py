import numpy as np
import time
import random
import pandas as pd
import os
from numba import njit
from pathlib import Path
from solver import Config, MemeticSolver
from reporter import Reporter, generate_sla_report, export_fitness_trace
from visualize import gambar_rute_osrm
from initial_route import data_titik
from simulator import hitung_detail_rute, hitung_eta_rute, cari_titik_terkunci
from representation import siapkan_data_vrp, is_valid_static_tour 

@njit
def _seed_numba(seed):
    np.random.seed(seed)

def muat_data_excel(file_matriks, file_pelanggan):
    print(f"[ INFO ] Membaca matriks dari: {file_matriks}")
    print(f"[ INFO ] Membaca data pelanggan dari: {file_pelanggan}")

    try:
        df_customers = pd.read_excel(
            file_pelanggan,
            sheet_name='Sheet2',
            index_col=0,
            engine='openpyxl'
        )

        cutoff_statis = 7.0 * 3600  # 07:00 dalam detik

        # Hitung waktu siap pesanan
        df_customers['ready_time'] = (
            df_customers['Order Time'] +
            df_customers['Prep Time']
        )

        # Pisahkan statis dan dinamis
        df_statis = df_customers[df_customers['ready_time'] <= cutoff_statis]
        df_dinamis = df_customers[df_customers['ready_time'] > cutoff_statis]

        num_static = len(df_statis)
        num_customers = len(df_customers)

        last_order_statis = (
            df_statis['ready_time'].max()
            if not df_statis.empty else 0
        )

        # Matriks waktu
        df_time = pd.read_excel(
            file_matriks,
            sheet_name='Sheet1',
            index_col=0,
            engine='openpyxl'
        )
        time_matrix = df_time.to_numpy(dtype=np.float64)

        # Demand & SLA
        demands = df_customers['Jumlah Pesanan'].to_numpy(dtype=np.float64)
        sla_limits = df_customers['SLA Limit'].to_numpy(dtype=np.float64)

        return (
            time_matrix, demands, sla_limits,
            num_customers, num_static,
            df_customers, df_statis,
            df_dinamis, last_order_statis
        )

    except Exception as e:
        print(f"[ ERROR ] Terjadi kesalahan saat membaca Excel: {e}")
        exit(1)

def hitung_beban_tersisa(rute_saat_ini, demands_global):
    """
    Menghitung total muatan yang masih dibawa kurir 
    berdasarkan sisa pelanggan dalam antrean rute_saat_ini.
    """
    beban = 0.0
    for node_id in rute_saat_ini:
        # Asumsikan demands_global adalah list/array yang bisa diakses dengan node_id
        beban += demands_global[node_id]
    return beban

def main():
    # ==========================================
    # 1. PERSIAPAN DATA UTAMA (GLOBAL)
    # ==========================================
    NAMA_FILE_MATRIKS = "Matriks_Waktu_Detik.xlsx"
    NAMA_FILE_PELANGGAN = "DATA.xlsx"
    time_matrix_global, demands_global, sla_limits_global, num_customers, num_static, df_customers, df_statis, df_dinamis, last_order_statis = muat_data_excel(NAMA_FILE_MATRIKS, NAMA_FILE_PELANGGAN)

    param_dasar = {
        "max_capacity": 10.0,
        "penalty_rate": 1,
        "service_time": 120,
        "population_size": 20,
        "pc": 0.8,
        "pm": 0.2,
        "ns_max": 10,
        "ls_size": 5,
        "delta": 0.1,
    }

    seeds = list(range(1, 11))

    all_results = []
    all_routes = []
    all_seed_details = []
    all_seed_logs = []
    all_violations = []
    all_seed_trip = []

    for seed in seeds:
        all_als_history = []
        all_als_generation_history = []
        all_als_neighbor_history = []
        seed_logs = []
        seed_trip = []
        arrival_log_seed = []
        total_statis = 0.0
        total_dinamis = 0.0
        total_waktu_tempuh_keseluruhan = 0.0

        np.random.seed(seed)
        _seed_numba(seed)
        start_seed_time = time.time()

        # ==========================================
        # 2. FASE 1: RUTE AWAL STATIS
        # ==========================================        
        list_id_statis = df_statis.index.tolist()
        if 0 in list_id_statis:
            list_id_statis.remove(0)

        num_static = len(list_id_statis)

        idx_statis = [0] + list_id_statis
        
        time_matrix_statis = np.array(time_matrix_global[np.ix_(idx_statis, idx_statis)], dtype=np.float64)
        
        demands_statis = np.array(demands_global[idx_statis], dtype=np.float64).flatten()
        sla_limits_statis = np.array(sla_limits_global[idx_statis], dtype=np.float64).flatten()

        config_statis = Config(time_matrix=time_matrix_statis, demands=demands_statis, sla_limits=sla_limits_statis, start_time=7*3600, **param_dasar)
        
        solver_statis = MemeticSolver(config_statis)
        best_route_statis_lokal, _ = solver_statis.run(
            num_generations=100, 
            max_stagnant_gen=50,     
            # max_time_seconds=120,      
            log_filename=f"output/konvergensi_epoch_0_seed_{seed}.csv",
            customer_map=idx_statis,
            init_phase="static",
            seed=seed
        )

        for h in solver_statis.als.history:
            h["epoch"] = 0
            all_als_history.append(h)

        for g in solver_statis.als.generation_history:
            row = g.copy()
            row["seed"] = seed
            row["epoch"] = 0
            all_als_generation_history.append(row)

        for n in solver_statis.als.neighbor_history:
            n["epoch"] = 0
            all_als_neighbor_history.append(n)

        df_evolution_statis = pd.DataFrame(solver_statis.evolution_log)
        df_evolution_statis["epoch"] = 0
        all_evolution_log = [df_evolution_statis]
            
        best_route_statis = [idx_statis[i] for i in best_route_statis_lokal]

        waktu_tempuh, pelanggaran_sla, total_penalty, rute_visual, arrival_log = hitung_detail_rute(
            tour=best_route_statis,
            time_matrix=time_matrix_global,
            demands=demands_global,
            sla_limits=sla_limits_global,
            start_time=7*3600,
            max_capacity=param_dasar["max_capacity"],
            penalty_rate=param_dasar["penalty_rate"],
            service_time=param_dasar["service_time"],
            start_node=0
        )

        pending_arrival_log = arrival_log
        seed_trip.append(rute_visual.copy())
        
        is_valid = is_valid_static_tour(best_route_statis_lokal, num_static)

        seed_logs.append({
            "epoch": 0,
            "jam": "07.00",
            "pelanggan_baru": list_id_statis,
            "titik_kunci": [],
            "sisa_antrean": [],
            "titik_awal": 0,
            "rute_baru": rute_visual,
            "waktu_tempuh": waktu_tempuh / 60
        })

        # ==========================================
        # 3. FASE 2: LOOPING MESIN WAKTU (DINAMIS)
        # ==========================================
        jadwal_dinamis = []
        if not df_dinamis.empty:
            interval_detik = 1200
            df_dinamis['Epoch Time'] = np.ceil(df_dinamis['ready_time'] / interval_detik) * interval_detik
            
            for waktu, group in df_dinamis.groupby('Epoch Time'):
                jam_str = f"{int(waktu)//3600:02d}:{(int(waktu)%3600)//60:02d}"
                jadwal_dinamis.append({
                    "jam": jam_str, 
                    "waktu_detik": int(waktu), 
                    "pelanggan_baru": group.index.tolist() 
                })

        # --- PERSIAPAN REPORTER DINAMIS ---
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        
        epoch_counter = 1 # Penghitung epoch

        # Variabel Pelacak untuk menyambung rute
        rute_historis_final = [0]
        rute_saat_ini = best_route_statis
        waktu_berangkat_saat_ini = last_order_statis if not df_statis.empty else 0
        titik_awal_saat_ini = 0  
        total_run_cost = 0

        # --- LOOPING EPOCH UTAMA ---
        for event in jadwal_dinamis:
            waktu_event = event["waktu_detik"]
            pelanggan_baru = event["pelanggan_baru"]

            rute_dengan_awal = [titik_awal_saat_ini] + rute_saat_ini
            jadwal_saat_ini = hitung_eta_rute(rute_dengan_awal, waktu_berangkat_saat_ini, time_matrix_global, param_dasar["service_time"])

            titik_kunci, sisa_antrean = cari_titik_terkunci(jadwal_saat_ini, waktu_event)

            if titik_kunci is not None:
                realized_travel = (jadwal_saat_ini[titik_kunci] + param_dasar["service_time"]) - waktu_berangkat_saat_ini
            elif rute_saat_ini:
                non_depot_nodes = [n for n in rute_saat_ini if n != 0]
                if non_depot_nodes:
                    last_node = non_depot_nodes[-1]
                    waktu_tiba_depot = jadwal_saat_ini[last_node] + param_dasar["service_time"] + time_matrix_global[last_node, 0]
                    realized_travel = waktu_tiba_depot - waktu_berangkat_saat_ini
                else:
                    realized_travel = 0.0
            else:
                realized_travel = 0.0

            total_waktu_tempuh_keseluruhan += realized_travel

            if titik_kunci is None:
                confirmed_customers = set(rute_saat_ini)
            else:
                confirmed_customers = set()
                for node in rute_saat_ini:
                    confirmed_customers.add(node)
                    if node == titik_kunci:
                        break

            realized_penalty = 0.0

            for entry in pending_arrival_log:
                if entry["customer"] in confirmed_customers:
                    arrival_log_seed.append(entry)
                    realized_penalty += entry["late_seconds"] * param_dasar["penalty_rate"]
            pending_arrival_log = []  # reset; diisi ulang setelah solve epoch ini

            total_run_cost += realized_travel + realized_penalty

            if titik_kunci is None:
                rute_historis_final.extend(rute_saat_ini)
                if len(rute_historis_final) == 0 or rute_historis_final[-1] != 0:
                    rute_historis_final.append(0)
                rute_saat_ini = [] 
                titik_awal_saat_ini = 0 
                sisa_antrean = []
            else:
                for node in rute_saat_ini:
                    rute_historis_final.append(node)
                    if node == titik_kunci:
                        break
                titik_awal_saat_ini = titik_kunci 

            idx_dinamis = [titik_awal_saat_ini] + sisa_antrean + pelanggan_baru

            # Siapkan Array Jarak Depot
            idx_array = np.array(idx_dinamis, dtype=np.int32)
            dist_to_depot = time_matrix_global[idx_array, 0].copy()
            dist_from_depot = time_matrix_global[0, idx_array].copy()

            # Siapkan Flag Pelanggan Baru
            is_new_customer = np.array([node in pelanggan_baru for node in idx_dinamis], dtype=np.bool_)

            # Hitung beban saat ini (Sisa muatan dari rute sebelumnya)
            beban_saat_ini = hitung_beban_tersisa(rute_saat_ini, demands_global)

            time_matrix_dinamis = time_matrix_global[np.ix_(idx_dinamis, idx_dinamis)].copy() 
            demands_dinamis = np.array(demands_global[idx_dinamis], dtype=np.float64).flatten()
            sla_limits_dinamis = np.array(sla_limits_global[idx_dinamis], dtype=np.float64).flatten()
            demands_dinamis[0] = 0.0 

            # Jalankan optimasi
            config_dinamis = Config(
                time_matrix=time_matrix_dinamis, demands=demands_dinamis, 
                sla_limits=sla_limits_dinamis, start_time=waktu_event, 
                dist_to_depot=dist_to_depot, dist_from_depot=dist_from_depot,
                is_new_customer=is_new_customer, initial_load=beban_saat_ini, **param_dasar)
            solver_dinamis = MemeticSolver(config_dinamis)
            
            jumlah_tujuan_lokal = len(idx_dinamis) - 1

            best_fit = 0.0 # Default fitness
            if jumlah_tujuan_lokal == 0:
                best_route_dinamis_lokal = np.array([], dtype=np.int32) 
            elif jumlah_tujuan_lokal == 1:
                titik_tujuan = idx_dinamis[1]
                best_route_dinamis_lokal = np.array([1], dtype=np.int32)
            else:
                best_route_dinamis_lokal, best_fit = solver_dinamis.run(
                    num_generations=100, 
                    max_stagnant_gen=50,      
                    # max_time_seconds=120,      
                    log_filename=f"output/konvergensi_epoch_{epoch_counter}_seed_{seed}.csv",
                    customer_map=idx_dinamis,
                    init_phase=f"dynamic_{epoch_counter}",
                    seed=seed
                )

                for h in solver_dinamis.als.history:
                    h["epoch"] = epoch_counter
                    all_als_history.append(h)

                for g in solver_dinamis.als.generation_history:
                    row = g.copy()
                    row["seed"] = seed
                    row["epoch"] = epoch_counter
                    all_als_generation_history.append(row)

                for n in solver_dinamis.als.neighbor_history:
                    n["epoch"] = epoch_counter
                    all_als_neighbor_history.append(n)

                df_evolution_epoch = pd.DataFrame(solver_dinamis.evolution_log)
                df_evolution_epoch["epoch"] = epoch_counter
                all_evolution_log.append(df_evolution_epoch)


            # Terjemahkan dan update rute untuk iterasi selanjutnya
            rute_saat_ini = [idx_dinamis[i] for i in best_route_dinamis_lokal]
            waktu_berangkat_saat_ini = waktu_event

            # Terjemahkan ID (Ini hanya untuk rute logis, belum ada Depot-nya)
            rute_mentah = [idx_dinamis[i] for i in best_route_dinamis_lokal]

            # Hitung detail rute (Di sini angka 0 disisipkan secara otomatis!)
            waktu_tempuh_epoch, pelanggaran_sla_epoch, total_penalty_epoch, rute_logis, arrival_log = hitung_detail_rute(
                rute_mentah, 
                time_matrix_global, 
                demands_global,  
                sla_limits_global, 
                start_time=waktu_event,
                max_capacity=param_dasar["max_capacity"],
                penalty_rate =param_dasar["penalty_rate"],
                service_time=param_dasar["service_time"],
                pelanggan_baru=pelanggan_baru,
                start_node=titik_awal_saat_ini
            )

            pending_arrival_log = arrival_log

            seed_logs.append({
                "epoch": epoch_counter,
                "jam": event["jam"],
                "pelanggan_baru": pelanggan_baru,
                "titik_kunci": titik_kunci,
                "sisa_antrean": sisa_antrean,
                "titik_awal": titik_awal_saat_ini,
                "rute_baru": rute_logis,
                "waktu_tempuh": waktu_tempuh_epoch / 60
            })

            seed_trip.append(rute_logis.copy())

            rute_saat_ini = rute_logis[1:]
        
            epoch_counter += 1 # Lanjut ke epoch berikutnya
            

        total_waktu_tempuh_keseluruhan += waktu_tempuh_epoch
        total_run_cost += waktu_tempuh_epoch + total_penalty_epoch

        rute_historis_final.extend(rute_saat_ini)
        if rute_historis_final[-1] != 0:
            rute_historis_final.append(0)

        arrival_log_seed.extend(pending_arrival_log)
        pending_arrival_log = []

        all_seed_details.append({
            "seed": seed,
            "fitness": total_run_cost,
            "route": rute_historis_final.copy(),
            "num_static": num_static,
            "num_dynamic": len(df_dinamis),
            "runtime": time.time() - start_seed_time,
            "violations": all_violations,
            "arrival_log": arrival_log_seed
        })

        os.makedirs("logs", exist_ok=True)

        df = pd.DataFrame(all_als_history)
        df.to_csv(f"logs/als_history_seed_{seed}.csv", index=False)

        df_gen = pd.DataFrame(all_als_generation_history)
        df_gen.to_csv(f"logs/als_generation_seed_{seed}.csv", index=False)

        df_ns = pd.DataFrame(all_als_neighbor_history)
        df_ns.to_csv(f"logs/als_neighbor_history_{seed}.csv", index=False)

        df_evolution_full = pd.concat(all_evolution_log, ignore_index=True)
        df_evolution_full.to_csv(f"results/evolution_log_seed_{seed}.csv", index=False)

        time.sleep(1)

        all_seed_logs.append(seed_logs)
        all_results.append(total_run_cost)
        all_routes.append(rute_historis_final.copy())
        all_seed_trip.append(seed_trip)

    # ==========================================
    # 4. HASIL TERBAIK & VISUALISASI PETA
    # ==========================================    
    best_idx = np.argmin(all_results)
    # idx = 9
    best_detail = all_seed_details[best_idx]

    arrival_log = best_detail["arrival_log"]

    violations = generate_sla_report(arrival_log, verbose=False)

    export_fitness_trace(
        arrival_log,
        "logs/fitness_example.csv"
    )

    print("\n" + "="*50)
    print("BEST SEED DETAIL REPORT")
    print("="*50)

    print(f"Seed              : {best_detail['seed']}")
    print(f"Fitness           : {best_detail['fitness']}")
    print(f"Best Route        : {best_detail['route']}")
    print(f"Jumlah Statis     : {best_detail['num_static']}")
    print(f"Jumlah Dinamis    : {best_detail['num_dynamic']}")
    print(f"Runtime           : {best_detail['runtime']:.2f} detik")
    print(f"Jumlah Violation  : {len(violations)}")
    
    best_logs = all_seed_logs[best_idx]

    print("\n" + "="*50)
    print("DETAIL PERGERAKAN BEST SEED")
    print("="*50)

    for log in best_logs:
        print(f"\nEpoch {log['epoch']} ({log['jam']})")
        print(f"Pesanan Baru     : {log['pelanggan_baru']}")
        print(f"Titik Terkunci   : {log['titik_kunci']}")
        print(f"Sisa Antrean     : {log['sisa_antrean']}")
        print(f"Posisi Kurir     : {log['titik_awal']}")
        print(f"Rute Baru        : {log['rute_baru']}")
        print(f"Waktu Tempuh     : {log['waktu_tempuh']}")

    rute_per_kendaraan = all_seed_trip[best_idx]
    print(f"\nTotal Waktu Tempuh Keseluruhan: {(total_waktu_tempuh_keseluruhan) /  60:.2f} menit")

    if violations:
        print(f"\nTotal pelanggaran SLA: {len(violations)}\n")

        for v in violations:
            sla_h = int(v["sla_limit"] // 3600)
            sla_m = int((v["sla_limit"] % 3600) // 60)

            arr_h = int(v["arrival_time"] // 3600)
            arr_m = int((v["arrival_time"] % 3600) // 60)

            print(
                f"Pelanggan {v['customer']} | "
                f"SLA {sla_h:02d}:{sla_m:02d} | "
                f"Tiba {arr_h:02d}:{arr_m:02d} | "
                f"Terlambat {v['late_minutes']:.2f} menit"
            )
        total_late = np.sum([v['late_minutes'] for v in violations])
        print(f"Total keterlambatan: {total_late:.2f} menit")
    else:
        print("\nTidak ada pelanggaran SLA.")

    gambar_rute_osrm(
        rute_per_kendaraan,
        data_titik,
        output_filename="best_seed_route.html"
    )

if __name__ == "__main__":
    main()
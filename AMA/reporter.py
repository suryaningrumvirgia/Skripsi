import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from random_immigrant import calculate_diversity_index, calculate_immigrant_ratio

def export_fitness_trace(arrival_log, filename):
    df = pd.DataFrame(arrival_log)
    df.to_csv(filename, index=False)

def generate_sla_report(arrival_log, verbose=False):
    violations = [
        row
        for row in arrival_log
        if row["late_seconds"] > 0
    ]

    if verbose:
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
            total_late = np.sum(v['late_minutes'])
            print(f"Total keterlambatan: {total_late:.2f} menit")
        else:
            print("\nTidak ada pelanggaran SLA.")

    return violations

class Reporter:
    def __init__(self, filename: str = "default_log.csv"):
        self.log_path = Path(filename)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.start_time = None
        self.file = None
        self.writer = None  # csv.DictWriter, dibuat saat baris pertama ditulis
        self.config = None  # disimpan terpisah, tidak dicampur ke baris CSV

    def save_initial_population(self, population, start_time, customer_map=None, init_phase="static", seed=None):
        seed = f"_seed_{seed}" if seed is not None else ""
        init_path = self.log_path.parent / f"{init_phase}_{seed}_initial_population.txt"

        with open(init_path, "w", encoding="utf-8") as f:
            f.write(f"Start time: {start_time}\n\n")

            for i, tour in enumerate(population):
                if customer_map is not None:
                    translated_tour = [customer_map[x] for x in tour]
                else:
                    translated_tour = tour.tolist()

                f.write(f"Population {i+1}: {translated_tour}\n")

    def start(self, config=None):
        """Membuka file CSV untuk logging per-generasi."""
        self.start_time = time.time()
        self.file = open(self.log_path, "w", encoding="utf-8", newline="")
        self.writer = None  # header ditulis otomatis saat log() pertama dipanggil

        if config is not None:
            # Metadata config disimpan sebagai file JSON terpisah,
            # supaya tidak merusak struktur kolom CSV.
            self.config = config
            config_path = self.log_path.with_name(self.log_path.stem + "_config.json")
            with open(config_path, "w", encoding="utf-8") as cf:
                json.dump(config, cf, indent=2)

    def log(self, generation, fitness_array, rho_SI, rho_MI, best_route, waktu_tempuh, pelanggaran_sla, penalty_cost):
        """Mencatat status algoritma per generasi sebagai satu baris CSV."""
        elapsed = time.time() - self.start_time

        xi = calculate_diversity_index(fitness_array)
        ri = calculate_immigrant_ratio(xi)

        # best_route diserialisasi jadi string tunggal (mis. "[1, 2, 3]")
        # supaya komanya tidak dianggap pemisah kolom oleh CSV/Excel.
        if isinstance(best_route, np.ndarray):
            route_str = str(best_route.tolist())
        else:
            route_str = str(best_route)

        data = {
            "generation": generation,
            "time_elapsed_s": elapsed,
            "best_fitness": float(np.min(fitness_array)),
            "mean_fitness": float(np.mean(fitness_array[np.isfinite(fitness_array)])),
            "max_fitness": float(np.max(fitness_array)),
            "diversity_index": float(xi),
            "immigrant_ratio": float(ri),
            "rho_SI": rho_SI,
            "rho_MI": rho_MI,
            "best_route": route_str,
            "waktu_tempuh": waktu_tempuh,
            "pelanggaran_sla": pelanggaran_sla,
            "penalty_cost": penalty_cost,
        }

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(data.keys()))
            self.writer.writeheader()

        self.writer.writerow(data)

    def stop(self):
        """Menutup file."""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
# AMA — Adaptive Memetic Algorithm untuk Dynamic Vehicle Routing Problem

Program ini menyelesaikan Vehicle Routing Problem (VRP) dinamis menggunakan Adaptive Memetic Algorithm (kombinasi Genetic Algorithm dan Local Search). Studi kasus utama adalah optimasi rute pengiriman kurir dengan mempertimbangkan kapasitas kendaraan, batas waktu layanan (SLA), serta pelanggan yang muncul secara dinamis (order baru) di tengah proses pengiriman.

## Latar Belakang

Dalam pengiriman barang/makanan berbasis kurir, pesanan tidak selalu diketahui sejak awal — sebagian pelanggan bersifat statis (sudah diketahui sebelum kurir berangkat) dan sebagian lagi dinamis (muncul setelah kurir mulai berjalan). Program ini mensimulasikan skenario tersebut dan mencari rute optimal yang meminimalkan waktu tempuh sekaligus menjaga keterlambatan (pelanggaran SLA) seminimal mungkin.

## Fitur

- Optimasi rute menggunakan Adaptive Memetic Algorithm (GA + Local Search adaptif)
- Mendukung pelanggan statis dan dinamis (re-optimasi saat ada order baru)
- Constraint kapasitas kendaraan dan SLA (batas waktu layanan)
- Skema random immigrant untuk menjaga keragaman populasi GA
- Simulasi pergerakan kurir per-epoch (posisi, antrean, waktu tempuh)
- Pelaporan hasil (fitness trace, ringkasan SLA, log pergerakan)
- Visualisasi rute di atas peta menggunakan OSRM

## Struktur Folder

```
AMA/
├── main.py              # Entry point program
├── solver.py             # Implementasi Memetic Algorithm (Config & MemeticSolver)
├── fitness.py            # Fungsi evaluasi fitness rute
├── selection.py           # Seleksi parent (GA)
├── crossover.py           # Operator crossover (Order Crossover / OX)
├── mutation.py            # Operator mutasi (2-opt)
├── random_immigrant.py    # Skema random immigrant
├── search.py              # Adaptive Local Search
├── simulator.py           # Simulasi pergerakan kurir & perhitungan rute
├── representation.py      # Representasi data VRP
├── initial_route.py       # Pembentukan rute awal & titik pelanggan
├── matrix.py               # Pengolahan matriks jarak/waktu
├── graph.py                # Struktur graf pendukung
├── reporter.py             # Pembuatan laporan (SLA, fitness trace)
├── visualize.py             # Visualisasi rute (OSRM)
├── data/                    # Data input (Excel: DATA.xlsx, Matriks_Waktu_Detik.xlsx)
├── output/                  # Hasil keluaran program
├── results/                 # Hasil ringkasan/eksperimen
├── logs/                    # Log eksekusi
└── Peta/                    # Output visualisasi peta rute
```

## Teknologi yang Digunakan

- **Python 3**
- `numpy` — komputasi numerik
- `pandas` & `openpyxl` — membaca data dari Excel
- `numba` — percepatan komputasi (JIT compilation)
- `requests` — komunikasi dengan OSRM API
- `folium` — visualisasi peta rute
- OSRM (Open Source Routing Machine) — dijalankan lokal lewat Docker, dipakai untuk menghitung geometri rute jalan yang ditampilkan di peta

## Instalasi

1. Clone repository ini:
   ```bash
   git clone https://github.com/suryaningrumvirgia/Skripsi.git
   cd Skripsi/AMA
   ```

2. Install dependency Python yang dibutuhkan:
   ```bash
   pip install numpy pandas openpyxl numba requests folium
   ```

## Setup OSRM (Wajib untuk Visualisasi Peta)

Program ini butuh server OSRM yang jalan secara lokal untuk menggambar rute di atas peta (`visualize.py` memanggil `http://127.0.0.1:5000`). Tanpa ini, program tetap bisa menghitung rute optimal, tapi bagian visualisasi peta akan gagal dengan pesan error `Tidak bisa terhubung ke OSRM Lokal!`.

1. Pastikan Docker sudah terinstall.
2. Download data peta (file `.osm.pbf`) sesuai wilayah yang dibutuhkan, contoh untuk wilayah Indonesia bisa didapat dari [Geofabrik](https://download.geofabrik.de/asia/indonesia.html).
3. Proyek ini menggunakan profile `motorcycle.lua`, yaitu profile `bicycle.lua` yang sudah dimodifikasi pada kecepatan (speed) kendaraannya. 

4. Proses data peta dengan Docker menggunakan profile `motorcycle.lua` (ganti `nama-file` dan path profile sesuai lokasi file kamu):
   ```bash
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /data/profiles/motorcycle.lua /data/nama-file.osm.pbf
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/nama-file.osrm
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/nama-file.osrm
   ```
5. Jalankan server OSRM di port 5000:
   ```bash
   docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/nama-file.osrm
   ```
6. Pastikan server berjalan (bisa dicek lewat `http://127.0.0.1:5000` di browser atau lewat `curl`) sebelum menjalankan `main.py`.

> Server OSRM ini harus tetap aktif selama program `main.py` berjalan, karena pemanggilan peta terjadi di akhir proses (setelah rute terbaik ditemukan).

## Cara Menggunakan

1. Siapkan data input di folder `data/`:
   - `DATA.xlsx` — data pelanggan (order time, prep time, jumlah pesanan, SLA limit, dsb.)
   - `Matriks_Waktu_Detik.xlsx` — matriks waktu tempuh antar titik (dalam detik)

2. Jalankan program:
   ```bash
   python main.py
   ```

3. Program akan menjalankan simulasi untuk beberapa seed, menampilkan detail rute terbaik, log pergerakan kurir per-epoch, ringkasan pelanggaran SLA (jika ada), serta menyimpan visualisasi rute ke `Peta/best_seed_route.html`.

## Parameter Utama

Parameter algoritma dapat disesuaikan di `main.py` melalui dictionary `param_dasar`:

| Parameter | Keterangan |
|---|---|
| `max_capacity` | Kapasitas maksimum kendaraan |
| `penalty_rate` | Bobot penalti untuk pelanggaran SLA |
| `service_time` | Waktu layanan per pelanggan (detik) |
| `population_size` | Ukuran populasi GA |
| `pc` | Probabilitas crossover |
| `pm` | Probabilitas mutasi |
| `ns_max` | Batas neighborhood search |
| `ls_size` | Ukuran local search |
| `delta` | Parameter adaptif algoritma |

## Output

- Ringkasan fitness dan rute terbaik per seed
- Detail pergerakan kurir tiap epoch (pesanan baru, titik terkunci, sisa antrean)
- Ringkasan pelanggaran SLA (jika ada) beserta total keterlambatan
- Peta rute interaktif (`Peta/best_seed_route.html`)

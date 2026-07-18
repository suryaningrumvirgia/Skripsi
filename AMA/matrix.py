import pandas as pd
import requests
import sys

# 1. Membaca file excel (memulai dari baris kedua sesuai header=1)
df = pd.read_excel('DATA.xlsx', sheet_name='Sheet2', index_col=0, engine='openpyxl')

# Periksa apakah ada nilai NaN pada kolom koordinat
if df[['Longitude', 'Latitude']].isnull().any(axis=1).any():
    missing = df[df[['Longitude', 'Latitude']].isnull().any(axis=1)][['Longitude', 'Latitude']]
    print("Peringatan: ditemukan nilai koordinat kosong (NaN) pada baris berikut:")
    print(missing)
    # Buang baris yang mengandung NaN untuk menghindari pembuatan URL dengan 'nan,nan'
    df = df[~df[['Longitude', 'Latitude']].isnull().any(axis=1)].reset_index(drop=True)
    print(f"Baris dengan NaN dihapus. Titik tersisa: {len(df)}")

# Buat list koordinat [(lon1, lat1), (lon2, lat2), ...]
koordinat = df[['Longitude', 'Latitude']].to_numpy().tolist()
jumlah_titik = len(koordinat)

# 2. Rakit koordinat menjadi format URL (lon,lat;lon,lat;...)
koordinat_string = ";".join([f"{lon},{lat}" for lon, lat in koordinat])

# 3. Tentukan URL Server
url = f"http://localhost:5000/table/v1/bicycle/{koordinat_string}"
params = {"annotations": "duration"}

# 4. Kirim permintaan ke OSRM
try:
    print(f"Menghubungi server OSRM untuk {jumlah_titik} titik...")
    response = requests.get(url, params=params)
    # Jika server mengembalikan error, tampilkan isi respon untuk debugging
    if response.status_code != 200:
        print("Response status:", response.status_code)
        print("Response body:", response.text)
    response.raise_for_status() # Cek jika ada error HTTP

    data = response.json()
    
    # OSRM mengembalikan waktu dalam satuan DETIK (List of Lists)
    matriks_waktu_detik = data['durations']

    # 5. Mengubah ke DataFrame Pandas (Matriks n x n)
    # Bagi 60 agar satuannya berubah menjadi menit
    df_waktu = pd.DataFrame(matriks_waktu_detik) / 60

    # Memberi nama indeks baris dan kolom agar lebih informatif
    nama_titik = [f"{i}" for i in range(jumlah_titik)]
    df_waktu.index = nama_titik
    df_waktu.columns = nama_titik

    print(f"\n--- MATRIKS WAKTU TEMPUH n x n ({jumlah_titik}x{jumlah_titik}) DALAM MENIT ---")
    print(df_waktu) # Menampilkan seluruh matriks n x n

    # Menyimpan output matriks ke file Excel
    df_waktu.to_excel('Output_Matriks_Waktu.xlsx')

    # Menyimpan output matriks dalam detik ke file Excel
    df_waktu_detik = pd.DataFrame(matriks_waktu_detik)
    df_waktu_detik.to_excel('Matriks_Waktu_Detik.xlsx')

except requests.exceptions.ConnectionError:
    print("Gagal terhubung! Pastikan Docker OSRM sudah 'Up' dan port 5000 terbuka.")
except requests.exceptions.HTTPError as e:
    print("HTTP error terjadi:", e)
    sys.exit(1)
except KeyError:
    print("Error: Respon dari OSRM tidak mengandung data 'durations'. Periksa kembali parameter Anda.")

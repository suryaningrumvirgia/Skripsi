import folium
import requests
import pandas as pd
import os

# ==========================================
# MEMBACA DATA EXCEL
# ==========================================
#df = pd.read_excel("DATA.xlsx", index_col=0, sheet_name="Sheet3", engine="openpyxl")
# df = df.dropna(subset=["Longitude", "Latitude"])
# df = df.reset_index(drop=True)

def load_data_titik(file_path):
    df = pd.read_excel(file_path, index_col=0, sheet_name="Sheet2",  engine="openpyxl")

    data_titik = {
        idx: (row["Longitude"], row["Latitude"])
        for idx, row in df.iterrows()
    }

    return data_titik

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_titik = load_data_titik(os.path.join(BASE_DIR, "data", "DATA.xlsx"))

print(f"[INFO] Total titik valid: {len(data_titik)}")

def ambil_jalur_rute(partisi_titik, data_koordinat):
    string_koordinat = ";".join(
        [f"{data_koordinat[idx][0]},{data_koordinat[idx][1]}" for idx in partisi_titik]
    )
    url = f"http://localhost:5000/route/v1/bicycle/{string_koordinat}"
    params = {"geometries": "geojson", "overview": "full"}

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            res_data = response.json()
            titik_jalan_osrm = res_data["routes"][0]["geometry"]["coordinates"]
            jalur_folium = [[lat, lon] for lon, lat in titik_jalan_osrm]
            return jalur_folium
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Tidak dapat terhubung ke OSRM")
        return None

# ==========================================
# INISIALISASI PETA
# ==========================================
m = folium.Map(location=[-6.9284, 107.7749], zoom_start=13, tiles="OpenStreetMap")

# Marker Depo
folium.Marker(
    location=[data_titik[0][1], data_titik[0][0]],
    tooltip="DEPO UTAMA",
    icon=folium.DivIcon(
        html=f'''
            <div style="
                background: red;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                border: 2px solid white;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 1px 2px rgba(0,0,0,0.4);
            ">
                <i class="fa fa-home" style="font-size: 12px;"></i> 
            </div>
        '''
    )
).add_to(m)

# Definisi Partisi
rute_kurir = { 
    "Partisi 1": {"warna": "green", "titik": [0, 3, 1, 4, 2, 0]},
    "Partisi 2": {"warna": "purple", "titik": [0, 5, 6, 7, 0]},
    "Partisi 3": {"warna": "black", "titik": [0, 8, 10, 9, 11, 12, 0]},
    "Partisi 4": {"warna": "red", "titik": [0, 13, 15, 18, 14, 0]},
    "Partisi 5": {"warna": "gray", "titik": [0, 16, 19, 21, 0]},
    "Partisi 6": {"warna": "orange", "titik":[0, 17, 25, 24, 20, 0]},
    "Partisi 7": {"warna": "brown", "titik": [0, 23, 26, 22, 0]},
    "Partisi 8": {"warna": "pink", "titik": [0, 28, 29, 27, 0]},
    "Partisi 9": {"warna": "yellow", "titik": [0, 30, 31, 32, 0]},
    "Partisi 10": {"warna": "blue", "titik": [0, 33, 0]},
}

# ==========================================
# PROSES & GAMBAR DI PETA
# ==========================================
for nama_partisi, info in rute_kurir.items():
    # Membuat grup per rute agar bisa di-toggle (LayerControl)
    fg = folium.FeatureGroup(name=nama_partisi)
    
    warna = info["warna"]
    titik_rute = info["titik"]
    jalur = ambil_jalur_rute(info["titik"], data_titik)

    if jalur:
        # Tambahkan jalur ke grup
        folium.PolyLine(jalur, color=warna, weight=4, opacity=0.7, tooltip=nama_partisi).add_to(fg)
        
        # Tambahkan marker urutan ke grup
    for i, titik_idx in enumerate(titik_rute[:-1]):
        if titik_idx == 0: continue
        
        lat, lon = data_titik[titik_idx][1], data_titik[titik_idx][0]
        
        # Menyiapkan teks informasi untuk tooltip (mendukung format HTML dasar)
        info_tooltip = f"""
        <div style="font-family: Arial; font-size: 14px;">
            <b>{nama_partisi}</b><br>
            Pelanggan Node: <b>{titik_idx}</b><br>
            Urutan Kunjungan: <b>Ke-{i}</b>
        </div>
        """
        
        # Menambahkan marker dengan parameter tooltip
        folium.Marker(
            location=[lat, lon],
            tooltip=folium.Tooltip(info_tooltip, sticky=True),
            icon=folium.DivIcon(
                html=f'<div style="background:{warna}; color:white; border-radius:50%; width:20px; height:20px; text-align:center; font-size:12px; font-weight:bold; border:2px solid white;">{i}</div>'
            )
        ).add_to(fg)
            
    fg.add_to(m)

# ==========================================
# FINALISASI
# ==========================================
os.makedirs("peta", exist_ok=True)
folium.LayerControl(collapsed=False).add_to(m)
m.save("peta/Peta_Rute_Awal.html")

print("[SUKSES] Peta berhasil dibuat: Peta_Rute_Awal.html")
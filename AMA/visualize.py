import requests
import folium
import numpy as np
from initial_route import data_titik

def gambar_rute_osrm(routes, coords, output_filename="peta_rute_dvrp.html"):
    print("\n[ INFO ] Menghubungi server OSRM Lokal (Port 5000) untuk menggambar peta...")

    depot_lon, depot_lat = coords[0]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=13)

    # Marker depot
    folium.Marker(
        location=[depot_lat, depot_lon],
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

    warna_rute = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'black', 'grey', 'pink', 'yellow']

    for idx, rute in enumerate(routes):
        fg = folium.FeatureGroup(name=f"Partisi {idx+1}")
        warna = warna_rute[idx % len(warna_rute)]
        titik_koordinat = []

        for seq, node in enumerate(rute):
            lon, lat = coords[node]
            titik_koordinat.append(f"{lon},{lat}")

            if node != 0:
                folium.Marker(
                    location=[lat, lon],
                    tooltip=(
                        f"<b>Rute Kendaraan {idx+1}</b>"
                        f"<br>Stop ke-{seq}"
                        f"<br>Pelanggan {node}"
                    ),
                    icon=folium.DivIcon(
                        html=f"""
                        <div style="
                            background:{warna};
                            color:white;
                            border-radius:50%;
                            width:26px;
                            height:26px;
                            text-align:center;
                            line-height:26px;
                            font-size:12px;
                            font-weight:bold;
                            border:2px solid white;
                        ">
                        {seq}
                        </div>
                        """
                    )
                ).add_to(fg)

        koordinat_string = ";".join(titik_koordinat)

        url = f"http://127.0.0.1:5000/route/v1/driving/{koordinat_string}?overview=full&geometries=geojson"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data["code"] == "Ok":
                    jalur_geojson = data["routes"][0]["geometry"]

                    folium.GeoJson(
                        jalur_geojson,
                        style_function=lambda x, warna=warna: {
                            'color': warna,
                            'weight': 5,
                            'opacity': 0.8
                        }
                    ).add_to(fg)

        except requests.exceptions.ConnectionError:
            print("[ FATAL ERROR ] Tidak bisa terhubung ke OSRM Lokal!")
            return

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_filename)
    print(f"[ INFO ] Peta berhasil disimpan sebagai: {output_filename}")
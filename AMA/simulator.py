def hitung_detail_rute(
    tour, time_matrix, demands, sla_limits,
    start_time, max_capacity, penalty_rate,
    pelanggan_baru=None, service_time=120.0, start_node=0
):
    if pelanggan_baru is None:
        pelanggan_baru = []

    current_time = start_time
    pelanggaran_sla = 0.0
    total_penalty = 0.0
    beban_kendaraan = 0.0
    node_sekarang = start_node
    sudah_ambil_stok_baru = False

    rute_visual = [start_node]
    arrival_log = []

    for node in tour:
        butuh_depot_inventori = (
            node in pelanggan_baru and not sudah_ambil_stok_baru
        )

        butuh_depot_kapasitas = (
            beban_kendaraan + demands[node] > max_capacity
        )

        if butuh_depot_inventori or butuh_depot_kapasitas:
            if node_sekarang != 0:
                current_time += time_matrix[node_sekarang][0]
                rute_visual.append(0)

            node_sekarang = 0
            beban_kendaraan = 0.0
            sudah_ambil_stok_baru = True

        current_time += time_matrix[node_sekarang][node]
        arrival_time = current_time

        lateness = max(0.0, arrival_time - sla_limits[node])

        arrival_log.append({
            "customer": node,
            "from": node_sekarang,
            "arrival_time": arrival_time,
            "departure_time": arrival_time + service_time,
            "travel_time": time_matrix[node_sekarang][node],
            "sla_limit": sla_limits[node],
            "late_seconds": lateness,
            "late_minutes": lateness / 60,
            "demand": demands[node],
            "vehicle_load": beban_kendaraan + demands[node]
        })

        rute_visual.append(node)
        node_sekarang = node
        beban_kendaraan += demands[node]

        lateness = max(0.0, arrival_time - sla_limits[node])

        pelanggaran_sla += lateness
        total_penalty += penalty_rate * lateness

        current_time += service_time

    current_time += time_matrix[node_sekarang][0]
    rute_visual.append(0)

    waktu_tempuh = current_time - start_time

    return waktu_tempuh, pelanggaran_sla, total_penalty, rute_visual, arrival_log
    

def hitung_eta_rute(rute, waktu_berangkat, time_matrix, service_time):
    """Menghitung jadwal tiba dan selesai di setiap titik dalam rute."""
    jadwal = {}
    waktu_sekarang = waktu_berangkat
    
    for i in range(len(rute) - 1):
        node_asal = rute[i]
        node_tujuan = rute[i+1]
        
        waktu_tempuh = time_matrix[node_asal][node_tujuan]
        waktu_tiba = waktu_sekarang + waktu_tempuh
        
        if node_tujuan != 0:
            jadwal[node_tujuan] = waktu_tiba
            waktu_sekarang = waktu_tiba + service_time   # <-- service_time cuma untuk pelanggan
        else:
            waktu_sekarang = waktu_tiba                  # <-- depot: tanpa service_time
        
    return jadwal


def cari_titik_terkunci(jadwal_rute, waktu_pesanan_masuk):
    """Mencari pelanggan mana yang sedang dituju kurir (Committed Node)."""
    titik_terkunci = None
    rute_sisa = []
    
    for node, waktu_tiba in jadwal_rute.items():
        if waktu_tiba > waktu_pesanan_masuk:
            if titik_terkunci is None:
                titik_terkunci = node
            else:
                rute_sisa.append(node)
                
    return titik_terkunci, rute_sisa
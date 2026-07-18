jadwal_saat_ini = hitung_eta_rute(rute_dengan_awal, waktu_berangkat_saat_ini, time_matrix_global, param_dasar["service_time"])

            titik_kunci, sisa_antrean = cari_titik_terkunci(jadwal_saat_ini, waktu_event)
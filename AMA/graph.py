import networkx as nx
import matplotlib.pyplot as plt
from matrix import df_waktu  

# 1. Membuat graf lengkap (Complete Graph) dengan 25 titik (0 sampai 24)
jumlah_titik = 24
G = nx.complete_graph(jumlah_titik)

# 2. Mengatur ukuran kanvas gambar agar cukup besar
plt.figure(figsize=(50, 50), dpi=150)  

# 3. Mengatur posisi titik menjadi melingkar (Circular Layout)
posisi = nx.circular_layout(G)

# 4. Menggambar Node (Titik)
nx.draw_networkx_nodes(G, posisi, 
                       node_color='lightskyblue', 
                       node_size=80, 
                       edgecolors='white')

# 5. Menggambar Label (Nomor 0 - 21)
nx.draw_networkx_labels(G, posisi, 
                        font_size=7, 
                        font_family="times new roman",)

# 6. Menggambar Edges (Garis antar titik)
nx.draw_networkx_edges(G, posisi, 
                       alpha=0.3, 
                       width=0.9, 
                       edge_color='black')

# 7. Menambahkan label pada setiap edge dengan nilai waktu tempuh dari matriks_waktu
edge_labels = {}
for u, v in G.edges():
    # Mengambil nilai dari matriks_waktu (baris u, kolom v)
    edge_labels[(u, v)] = round(df_waktu.iloc[u, v], 1) 

nx.draw_networkx_edge_labels(G, posisi, edge_labels=edge_labels, font_size=3.5)

# Mengatur aspek rasio agar graf terlihat proporsional
plt.axis('equal')

# Menghilangkan bingkai sumbu X dan Y agar lebih bersih
plt.axis('off')

# Menampilkan graf
plt.tight_layout()
plt.show()
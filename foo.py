import networkx as nx
import matplotlib.pyplot as plt

# 1. Crear el grafo dirigido múltiple con 5 nodos
G = nx.MultiDiGraph()

# 2. Definir conexiones de ida y vuelta con pesos diferentes
conexiones = [
    ("A", "B", 5),
    ("B", "A", 3),
    ("B", "C", 4),
    ("C", "B", 8),
    ("C", "D", 2),
    ("D", "C", 1),
    ("D", "E", 6),
    ("E", "D", 9),
    ("E", "A", 7),
    ("A", "E", 4),
]

for origen, destino, peso in conexiones:
    G.add_edge(origen, destino, weight=peso)

# 3. Posicionamiento circular para que las curvas se distribuyan de forma simétrica
pos = nx.circular_layout(G)

# 4. Dibujar los nodos
nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=900)
nx.draw_networkx_labels(G, pos, font_weight="bold", font_size=12)

# 5. Dibujar las aristas curvas
# Usamos una curvatura notable para separar bien los caminos
nx.draw_networkx_edges(
    G,
    pos,
    connectionstyle="arc3, rad=0.25",
    arrowsize=20,
    edge_color="dimgray",
    width=2,
)

# 6. Dibujar las etiquetas encima de las aristas curvas
# Para MultiDiGraph, calculamos posiciones desplazadas (label_pos) para que acompañen a la curva
etiquetas_ida = {
    (u, v): d["weight"] for u, v, k, d in G.edges(keys=True, data=True) if k == 0
}
etiquetas_vuelta = {
    (u, v): d["weight"] for u, v, k, d in G.edges(keys=True, data=True) if k == 1
}

# Dibujar etiquetas desplazadas hacia el inicio de la curva (0.3)
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=etiquetas_ida,
    label_pos=0.3,
    font_color="darkred",
    font_weight="bold",
    font_size=11,
    rotate=True,  # Alinea el texto con la dirección de la arista
)

# Dibujar etiquetas desplazadas hacia el final de la curva (0.7) para que no se empalmen
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=etiquetas_vuelta,
    label_pos=0.7,
    font_color="darkblue",
    font_weight="bold",
    font_size=11,
    rotate=True,
)

plt.axis("off")
plt.tight_layout()

plt.show()

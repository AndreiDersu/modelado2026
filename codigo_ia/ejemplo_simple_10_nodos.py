"""Ejemplo simplificado de 10 nodos para el modelo de propagación "Lo que arde".

Este script implementa un grafo dirigido G = (N, A) con n = 10 zonas/nodos forestales,
cumpliendo la condición estructural del concurso: (i, j) in A <=> (j, i) in A,
con probabilidades de propagación asimétricas p_ij in [0, 1].

Visualización:
- Muestra el valor numérico exacto de p_ij sobre cada arista dirigida.
- Evita el solapamiento de textos calculando el punto medio exacto de la curva Bézier
  para cada arco dirigido (curvatura opuesta entre ida y vuelta).
- Sin mapas de calor, enfocado en la claridad visual de la red.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Configuración del archivo de salida
BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_OUTPUT: Path = BASE_DIR / "ejemplo_10_nodos.png"

# Conexiones forestales: (zona_u, zona_v, prob_u_hacia_v, prob_v_hacia_u)
# Cada arista física posee transmisión bidireccional asimétrica p_ij != p_ji
CONEXIONES_FORESTALES: Final[list[tuple[int, int, float, float]]] = [
    # Anillo perimetral de parcelas adyacentes
    (0, 1, 0.75, 0.40),
    (1, 2, 0.65, 0.85),
    (2, 3, 0.50, 0.70),
    (3, 4, 0.80, 0.35),
    (4, 5, 0.60, 0.55),
    (5, 6, 0.70, 0.90),
    (6, 7, 0.45, 0.65),
    (7, 8, 0.85, 0.30),
    (8, 9, 0.90, 0.50),
    (9, 0, 0.55, 0.75),
    # Conexión transversal (sendero / cortafuego interno entre parcelas 0 y 5)
    (0, 5, 0.30, 0.40),
]


def construir_grafo_forestal() -> nx.DiGraph:
    """Construye el grafo dirigido G = (N, A) con 10 nodos y probabilidades p_ij.

    Returns:
        nx.DiGraph con 10 nodos y arcos con atributo 'prob'.
    """
    grafo = nx.DiGraph()
    grafo.add_nodes_from(range(10))

    for u, v, p_uv, p_vu in CONEXIONES_FORESTALES:
        grafo.add_edge(u, v, prob=p_uv)
        grafo.add_edge(v, u, prob=p_vu)

    return grafo


def calcular_posicion_etiqueta_bezier(
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    rad: float = 0.22,
) -> tuple[float, float, float]:
    """Calcula la coordenada (x, y) y ángulo de rotación para la etiqueta sobre la curva Bézier.

    En Matplotlib 'arc3, rad=rad', el punto de control central C se desplaza perpendicularmente
    a la cuerda u -> v. El punto medio sobre el arco a t = 0.5 se calcula exactamente como:
        x_label = (x_u + x_v)/2 + 0.5 * rad * dy
        y_label = (y_u + y_v)/2 - 0.5 * rad * dx

    Args:
        pos_u: Coordenada [x, y] del nodo origen.
        pos_v: Coordenada [x, y] del nodo destino.
        rad: Parámetro de curvatura del arco.

    Returns:
        Tupla con (x_label, y_label, angulo_rotacion_grados).
    """
    dx = pos_v[0] - pos_u[0]
    dy = pos_v[1] - pos_u[1]

    mid_x = (pos_u[0] + pos_v[0]) / 2.0
    mid_y = (pos_u[1] + pos_v[1]) / 2.0

    label_x = mid_x + 0.5 * rad * dy
    label_y = mid_y - 0.5 * rad * dx

    # Ángulo tangente a la cuerda para alinear la lectura del texto
    angulo = float(np.degrees(np.arctan2(dy, dx)))
    if angulo > 90.0:
        angulo -= 180.0
    elif angulo < -90.0:
        angulo += 180.0

    return label_x, label_y, angulo


def dibujar_grafo_simple(
    grafo: nx.DiGraph,
    guardar_en: Path | None = None,
    mostrar: bool = True,
) -> None:
    """Genera y visualiza el grafo de 10 nodos con números sobre cada arco sin encimarse.

    Args:
        grafo: Grafo dirigido con atributos 'prob' en cada arista.
        guardar_en: Ruta donde guardar la figura en PNG.
        mostrar: Si es True, abre la ventana interactiva con plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Disposición circular simétrica: garantiza máxima separación entre los 10 nodos
    pos = nx.circular_layout(grafo)

    # 1. Dibujar los 10 nodos (zonas del bosque)
    nx.draw_networkx_nodes(
        grafo,
        pos,
        ax=ax,
        node_size=1250,
        node_color="#2b6cb0",
        edgecolors="#1a365d",
        linewidths=2.5,
    )

    # Etiquetas de los nodos (identificadores 0 a 9)
    nx.draw_networkx_labels(
        grafo,
        pos,
        ax=ax,
        font_size=13,
        font_color="white",
        font_weight="bold",
        font_family="sans-serif",
    )

    # 2. Dibujar aristas dirigidas curvas y etiquetas numéricas
    rad_curvatura = 0.22

    for u, v, data in grafo.edges(data=True):
        probabilidad = data["prob"]

        # Arco dirigido con flecha estilizada
        nx.draw_networkx_edges(
            grafo,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad_curvatura}",
            edge_color="#4a5568",
            width=1.8,
            arrowsize=18,
            arrowstyle="-|>",
            min_source_margin=25,
            min_target_margin=25,
        )

        # Ubicación analítica exacta de la etiqueta sobre el arco
        pos_u = np.array(pos[u])
        pos_v = np.array(pos[v])
        lx, ly, angulo = calcular_posicion_etiqueta_bezier(
            pos_u, pos_v, rad=rad_curvatura
        )

        # Texto con valor numérico de probabilidad sobre la arista
        ax.text(
            lx,
            ly,
            f"{probabilidad:.2f}",
            fontsize=9.5,
            fontweight="bold",
            color="#1a202c",
            ha="center",
            va="center",
            rotation=angulo,
            rotation_mode="anchor",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="#cbd5e0",
                lw=0.9,
                alpha=0.95,
            ),
        )

    ax.set_title(
        'Modelo Forestal Simplificado (10 Zonas - "Lo que arde")\n'
        r"Probabilidades de Propagación $p_{ij}$ sobre cada arista dirigida",
        fontsize=14,
        fontweight="bold",
        pad=22,
    )
    ax.axis("off")
    plt.tight_layout()

    if guardar_en:
        guardar_en.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(guardar_en, dpi=200, bbox_inches="tight")
        print(f"[✓] Gráfico guardado exitosamente en: {guardar_en.resolve()}")

    if mostrar:
        print("[i] Mostrando ventana interactiva de visualización...")
        plt.show()


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Grafo simplificado de 10 nodos con probabilidades sobre aristas ('Lo que arde')."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta donde guardar la imagen (por defecto: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana interactiva (útil para guardar directamente la imagen)",
    )

    args = parser.parse_args()
    grafo = construir_grafo_forestal()
    dibujar_grafo_simple(grafo, guardar_en=args.output, mostrar=not args.no_show)


if __name__ == "__main__":
    main()

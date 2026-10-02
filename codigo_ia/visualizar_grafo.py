"""Visualización de grafos dirigidos con probabilidades de transición.

Este módulo carga datos de interacciones dirigidas con probabilidades (u v prob)
y genera representaciones visuales de alta calidad:
1. Topología de red con disposición espacial optimizada y flechas curvas bidireccionales.
2. Matriz térmica (heatmap) de probabilidades de transición P(u -> v).
3. Vista de panel combinado (dashboard).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Rutas predeterminadas
BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DATA_PATH: Path = BASE_DIR / "Datos" / "Datos" / "graph_003_probs.txt"
DEFAULT_INICIO_PATH: Path = BASE_DIR / "Datos" / "Datos" / "inicio.txt"


def cargar_grafo(ruta_archivo: Path) -> nx.DiGraph:
    """Carga un grafo dirigido con pesos de probabilidad desde un archivo de texto.

    Args:
        ruta_archivo: Ruta al archivo con formato `origen destino probabilidad`.

    Returns:
        nx.DiGraph con los nodos y aristas cargados con atributo 'weight'.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si los datos del archivo tienen un formato incorrecto.
    """
    if not ruta_archivo.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_archivo}")

    grafo = nx.DiGraph()
    with ruta_archivo.open("r", encoding="utf-8") as f:
        for num_linea, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) < 3:
                raise ValueError(
                    f"Línea {num_linea} inválida en {ruta_archivo}: '{linea}'. "
                    "Se esperan 3 columnas: origen destino probabilidad."
                )
            u, v = int(partes[0]), int(partes[1])
            prob = float(partes[2])
            grafo.add_edge(u, v, weight=prob)

    return grafo


def cargar_probabilidades_iniciales(ruta_archivo: Path) -> dict[int, float]:
    """Carga las probabilidades iniciales por nodo si existe el archivo inicio.txt.

    Args:
        ruta_archivo: Ruta al archivo inicio.txt con formato `nodo probabilidad`.

    Returns:
        Diccionario con las probabilidades iniciales mapeadas por nodo.
    """
    if not ruta_archivo.is_file():
        return {}

    prob_iniciales: dict[int, float] = {}
    with ruta_archivo.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) >= 2:
                prob_iniciales[int(partes[0])] = float(partes[1])
    return prob_iniciales


def crear_matriz_adyacencia(grafo: nx.DiGraph) -> tuple[np.ndarray, list[int]]:
    """Genera una matriz NumPy de probabilidades de transición P(u -> v).

    Args:
        grafo: Grafo dirigido ponderado.

    Returns:
        Tupla con (matriz 2D de NxN con probabilidades, lista ordenada de nodos).
    """
    nodos = sorted(grafo.nodes())
    n = len(nodos)
    nodo_a_idx = {nodo: i for i, nodo in enumerate(nodos)}
    matriz = np.zeros((n, n), dtype=float)

    for u, v, data in grafo.edges(data=True):
        i, j = nodo_a_idx[u], nodo_a_idx[v]
        matriz[i, j] = data.get("weight", 1.0)

    return matriz, nodos


def dibujar_topologia(
    grafo: nx.DiGraph,
    ax: plt.Axes,
    pos: dict[int, np.ndarray] | None = None,
    semilla: int = 42,
    titulo: str | None = None,
) -> dict[str, Any]:
    """Dibuja la red en el eje ax con flechas curvas y colores según probabilidad.

    Args:
        grafo: Grafo dirigido a dibujar.
        ax: Eje de Matplotlib donde se renderizará el grafo.
        pos: Diccionario opcional de posiciones de nodos. Si es None, se calcula spring_layout.
        semilla: Semilla aleatoria para la reproducibilidad de la disposición visual.
        titulo: Título del subgráfico (opcional).

    Returns:
        Diccionario con elementos visuales clave (mappable para colorbar, posiciones).
    """
    if pos is None:
        pos = nx.spring_layout(grafo, k=1.3, iterations=200, seed=semilla)

    aristas = list(grafo.edges(data=True))
    pesos = [data["weight"] for _, _, data in aristas]

    # Nodos
    nx.draw_networkx_nodes(
        grafo,
        pos,
        ax=ax,
        node_size=850,
        node_color="#2980b9",
        edgecolors="#1b4f72",
        linewidths=2.5,
    )

    # Etiquetas de los nodos
    nx.draw_networkx_labels(
        grafo,
        pos,
        ax=ax,
        font_size=11,
        font_color="white",
        font_weight="bold",
        font_family="sans-serif",
    )

    # Aristas curvas dirigidas (rad=0.15 permite distinguir ambos sentidos u->v y v->u)
    nx.draw_networkx_edges(
        grafo,
        pos,
        ax=ax,
        connectionstyle="arc3,rad=0.15",
        edge_color=pesos,
        edge_cmap=plt.cm.viridis,
        edge_vmin=0.0,
        edge_vmax=1.0,
        width=[1.2 + 2.8 * w for w in pesos],
        arrowsize=16,
        arrowstyle="-|>",
        min_source_margin=15,
        min_target_margin=15,
    )

    # Barra de color para las probabilidades de arista
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Probabilidad de Interacción $P(u \\to v)$", fontsize=11)

    if titulo is None:
        num_nodos = grafo.number_of_nodes()
        num_aristas = grafo.number_of_edges()
        titulo = f"Topología del Grafo Dirigido ({num_nodos} Nodos, {num_aristas} Aristas)"

    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    return {"pos": pos, "mappable": sm}


def dibujar_matriz_calor(
    matriz: np.ndarray,
    nodos: list[int],
    ax: plt.Axes,
    titulo: str | None = None,
) -> None:
    """Dibuja la matriz de probabilidades de transición como un mapa de calor.

    Args:
        matriz: Matriz bidimensional de probabilidades.
        nodos: Lista ordenada de identificadores de nodo.
        ax: Eje de Matplotlib para dibujar el mapa de calor.
        titulo: Título del subgráfico (opcional).
    """
    n = len(nodos)
    im = ax.imshow(matriz, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nodos, fontsize=9)
    ax.set_yticklabels(nodos, fontsize=9)

    ax.set_xlabel("Nodo Destino ($v$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nodo Origen ($u$)", fontsize=11, fontweight="bold")

    if titulo is None:
        titulo = "Matriz de Probabilidades de Transición ($P_{uv}$)"
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)

    # Anotar valores no nulos en las celdas
    for i in range(n):
        for j in range(n):
            val = matriz[i, j]
            if val > 0:
                color_texto = "white" if val > 0.65 else "#1c2833"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color_texto,
                    fontsize=7.0,
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probabilidad $P(u \\to v)$", fontsize=11)


def visualizar(
    ruta_datos: Path,
    modo: str = "dashboard",
    guardar_como: Path | None = None,
    mostrar: bool = True,
) -> None:
    """Función principal para generar y mostrar la visualización del grafo.

    Args:
        ruta_datos: Archivo con los datos del grafo.
        modo: 'dashboard' (grafo + matriz), 'grafo' (solo red) o 'matriz' (solo matriz).
        guardar_como: Ruta para guardar la imagen generada.
        mostrar: Si es True, ejecuta plt.show() para ventana interactiva.
    """
    grafo = cargar_grafo(ruta_datos)
    matriz, nodos = crear_matriz_adyacencia(grafo)

    if modo == "dashboard":
        fig, axes = plt.subplots(
            1, 2, figsize=(21, 9.5), gridspec_kw={"width_ratios": [1.15, 1]}
        )
        dibujar_topologia(
            grafo,
            ax=axes[0],
            titulo="Topología de Interacción (Flechas Curvas $u \\to v$)",
        )
        dibujar_matriz_calor(
            matriz,
            nodos,
            ax=axes[1],
            titulo="Matriz de Probabilidades ($P_{uv}$)",
        )
        fig.suptitle(
            f"Visualización de Red y Probabilidades: {ruta_datos.name}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
    elif modo == "grafo":
        fig, ax = plt.subplots(figsize=(12, 10))
        dibujar_topologia(
            grafo,
            ax=ax,
            titulo=f"{grafo.number_of_nodes()} Nodos, {grafo.number_of_edges()} Aristas Dirigidas Ponderadas",
        )
        fig.suptitle(
            f"Topología de Red Dirigida: {ruta_datos.name}",
            fontsize=15,
            fontweight="bold",
            y=0.96,
        )
    elif modo == "matriz":
        fig, ax = plt.subplots(figsize=(10, 9))
        dibujar_matriz_calor(
            matriz,
            nodos,
            ax=ax,
            titulo=f"Probabilidades $P(u \\to v)$ entre {len(nodos)} Nodos",
        )
        fig.suptitle(
            f"Matriz de Probabilidades: {ruta_datos.name}",
            fontsize=15,
            fontweight="bold",
            y=0.96,
        )
    else:
        raise ValueError(
            f"Modo no reconocido: {modo}. Usa 'dashboard', 'grafo' o 'matriz'."
        )

    plt.tight_layout()

    if guardar_como:
        guardar_como.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(guardar_como, dpi=300, bbox_inches="tight")
        print(f"[✓] Gráfico guardado exitosamente en: {guardar_como.resolve()}")

    if mostrar:
        print("[i] Mostrando ventana interactiva de visualización...")
        plt.show()


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Visualizador de grafos dirigidos con probabilidades de transición."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al archivo de datos (por defecto: Datos/Datos/graph_003_probs.txt)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["dashboard", "grafo", "matriz"],
        default="dashboard",
        help="Modo de visualización: 'dashboard' (ambos), 'grafo' o 'matriz'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE_DIR / "grafo_003_visualizacion.png",
        help="Ruta donde guardar la imagen generada",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir ventana interactiva (útil para entornos headless o scripts)",
    )

    args = parser.parse_args()

    visualizar(
        ruta_datos=args.file,
        modo=args.mode,
        guardar_como=args.output,
        mostrar=not args.no_show,
    )


if __name__ == "__main__":
    main()

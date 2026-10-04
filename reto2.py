from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from loaddata import quickstart
from wildfire_simulator import firecut

"""
Reto 2: Contención Dinámica

Modela la propagación del fuego como un frente de onda determinista sobre un
grafo, en este caso equivalente a uno no dirigido. Se colocan los cortafuegos en tiempo real para mitigar el avance del incendio.
"""

GRAPH_SELECTION: int | str = 7  # Grafo a cargar (3, 4, 5, 7, 8, 10)
INITIAL_NODE: int = 1  # Nodo inicial de ignición en t = 0
K_FIREWALLS: int = 2  # Número de cortafuegos disponibles por paso temporal


def load_graph(
    path_or_id: Path | int | str = GRAPH_SELECTION,
) -> tuple[int, NDArray[np.int64]]:
    """
    Carga la matriz de adyacencia de G.
    """
    N, G, _, _, _, _ = quickstart(graph=path_or_id)
    adj_matrix: NDArray[np.int64] = (G > 0).astype(np.int64)
    return N, adj_matrix


def select_firewalls_by_neighbor_degrees(
    A: NDArray[np.int64],
    Q: NDArray[np.int64],
    I_t: NDArray[np.int64],
    k: int = 2,
) -> list[tuple[int, int]]:
    """Selecciona hasta k aristas (u, v) con u en I_t y v susceptible.

    Criterio heurístico:
        Maximiza el grado libre de v: d_sano(v) = sum_w A[v, w] * (1 - Q[w]).
        Aísla prioritariamente aquellos nodos que poseen mayor cantidad de
        conexiones sanas hacia el resto de la red.
    """
    active_nodes = np.flatnonzero(I_t)
    healthy_nodes = 1 - Q

    # Grado no quemado de cada nodo en el grafo: vector d = A @ (1 - Q)
    healthy_degrees = A @ healthy_nodes

    candidate_edges: list[tuple[int, int, int]] = []

    for u in active_nodes:
        # Vecinos sanos de u: conectados en A y aún no quemados
        healthy_neighbors = np.flatnonzero((A[u, :] > 0) & (healthy_nodes > 0))
        for v in healthy_neighbors:
            candidate_edges.append((int(u), int(v), int(healthy_degrees[v])))

    # Ordenamiento descendente según salidas sanas de v
    candidate_edges.sort(key=lambda item: item[2], reverse=True)

    return [(u, v) for u, v, _ in candidate_edges[:k]]


def propagation_step(
    A: NDArray[np.int64],
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Evolución temporal del frente de onda: I_{t+1} = [I_t * A > 0] * (1 - Q_t)."""
    contact = (active_boundary @ A) > 0
    new_ignitions = (contact & (burned == 0)).astype(np.int64)

    next_burned = burned + new_ignitions
    next_active_boundary = new_ignitions

    return next_burned, next_active_boundary


def simulate_challenge2(
    A: NDArray[np.int64],
    initial_node: int = INITIAL_NODE,
    k: int = K_FIREWALLS,
    max_steps: int = 50,
    verbose: bool = True,
) -> tuple[int, int, NDArray[np.int64]]:
    """Ejecuta la propagación del fuego con colocación dinámica de cortafuegos.

    Args:
        A: Matriz de adyacencia binaria simétrica.
        initial_node: Vértice inicial del incendio.
        k: Cortafuegos a colocar por paso.
        max_steps: Límite superior de pasos temporales.
        verbose: Si es True, imprime la traza de la simulación paso a paso.

    Returns:
        (total_quemados, tiempo_total, vector_quemados_final)
    """
    A_work = A.copy()
    N = A_work.shape[0]

    # Condiciones iniciales en t = 0
    Q: NDArray[np.int64] = np.zeros(N, dtype=np.int64)
    I_t: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

    Q[initial_node] = 1
    I_t[initial_node] = 1

    if verbose:
        print("=" * 65)
        print("MATRIZ DE ADYACENCIA INICIAL A (Grafo no dirigido):")
        print(A_work)
        print("=" * 65)
        print(f"ESTADO INICIAL (t=0) - Fuego en el vértice {initial_node}:")
        print(f"Vector Q_0: {Q}")
        print(f"Frontera I_0: {I_t}")
        print("=" * 65)

    t: int = 1
    while np.any(I_t) and t <= max_steps and not np.all(Q == 1):
        if verbose:
            print(f"\n==================== PASO DE TIEMPO t = {t} ====================")

        firewalls = select_firewalls_by_neighbor_degrees(A_work, Q, I_t, k=k)

        if firewalls:
            if verbose:
                print(f"Cortafuegos colocados en t={t}: {firewalls}")
            A_work = firecut(A_work, firewalls)
        else:
            if verbose:
                print("No hay aristas expuestas disponibles para cortar.")

        Q, I_t = propagation_step(A_work, Q, I_t)

        if verbose:
            print("-" * 65)
            print(f"RESULTADOS AL FINAL DEL TIEMPO t = {t}:")
            print(f"Vector Q_{t}: {Q}")
            print(f"Frontera I_{t}: {I_t}")
            print("-" * 65)

        t += 1

    total_burned = int(np.sum(Q))
    total_time = t - 1

    if verbose:
        print(f"\nTotal de nodos quemados: {total_burned} de {N}")
        print(f"Tiempo total transcurrido: t = {total_time}")

    return total_burned, total_time, Q


def main() -> None:
    N, A = load_graph(GRAPH_SELECTION)
    simulate_challenge2(A, initial_node=INITIAL_NODE, k=K_FIREWALLS, verbose=True)


if __name__ == "__main__":
    main()

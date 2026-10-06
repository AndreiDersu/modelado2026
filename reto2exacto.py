from itertools import combinations
import numpy as np
from numpy.typing import NDArray

from loaddata import quickstart
from wildfire_core import firecut, get_frontier_edges

"""
Reto 2: Contención Dinámica (Búsqueda Exacta con Poda Branch & Bound)

Determina la secuencia óptima global de cortes k=2 por etapa temporal sobre
el modelo de propagación determinista en grafo no dirigido.
"""

# Selección de grafo (por defecto Grafo 4: graph_004_probs.txt)
GRAPH_SELECTION: int | str = 4


def load_graph(
    graph_id: int | str = GRAPH_SELECTION,
) -> tuple[int, NDArray[np.int64]]:
    """Carga la matriz de adyacencia A del grafo (simétrica para no dirigido)"""
    N, G, _, _, _, _ = quickstart(graph=graph_id)
    adj_matrix: NDArray[np.int64] = (G > 0).astype(np.int64)
    return N, adj_matrix


def step(
    A: NDArray[np.int64],
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Avanza un paso de tiempo en la propagación determinista.

    Al ser completamente determinista, no es necesario usar el "wilfire()", como
    en el caso de 1, 3 y 4
    """
    neighbors = (active_boundary @ A) > 0
    I_t = (neighbors & (burned == 0)).astype(np.int64)

    next_burned = burned + I_t
    next_active_boundary = I_t

    return next_burned, next_active_boundary


best_total_burned = float("inf")
best_cut_sequence: list[tuple[int, tuple[int, int], tuple[int, int]]] = []


def find_optimal_rec(
    A: NDArray[np.int64],
    Q: NDArray[np.int64],
    I_t: NDArray[np.int64],
    t: int,
    current_sequence: list[tuple[int, tuple[int, int], tuple[int, int]]],
    N: int,
) -> None:
    """Usa el algortimo de Branch and Bound"""

    global best_total_burned, best_cut_sequence

    current_total_burned = int(np.sum(Q))

    # Algoritmo de Branch and Bound:
    if current_total_burned >= best_total_burned:
        return

    # Condición de parada: Se extinguio el fuego o se quemo todo el grafo
    if not np.any(I_t) or np.all(Q == 1):
        if current_total_burned < best_total_burned:
            best_total_burned = current_total_burned
            best_cut_sequence = list(current_sequence)
        return

    # Reutiliza wildfire_core para identificar las aristas de frontera (u in I_t, v susceptible)
    candidate_edges = get_frontier_edges(I_t, Q, A.astype(np.float64))

    # Si hay al menos 2 aristas candidatas, exploramos las combinaciones
    if len(candidate_edges) >= 2:
        for comb in combinations(candidate_edges, 2):
            A_next = firecut(A, comb)
            Q_next, I_next = step(A_next, Q, I_t)

            current_sequence.append((t, comb[0], comb[1]))
            find_optimal_rec(A_next, Q_next, I_next, t + 1, current_sequence, N)
            current_sequence.pop()

    elif len(candidate_edges) == 1:
        single_cut = candidate_edges[0]
        A_next = firecut(A, (single_cut,))
        Q_next, I_next = step(A_next, Q, I_t)

        current_sequence.append((t, single_cut, single_cut))
        find_optimal_rec(A_next, Q_next, I_next, t + 1, current_sequence, N)
        current_sequence.pop()
    else:
        # No hay aristas expuestas; propagar normalmente
        Q_next, I_next = step(A, Q, I_t)
        find_optimal_rec(A, Q_next, I_next, t + 1, current_sequence, N)


if __name__ == "__main__":
    N, A = load_graph(GRAPH_SELECTION)

    initial_node: int = 1

    Q_0: NDArray[np.int64] = np.zeros(N, dtype=np.int64)
    I_0: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

    Q_0[initial_node] = 1
    I_0[initial_node] = 1

    find_optimal_rec(A, Q_0, I_0, t=1, current_sequence=[], N=N)

    print(f"Mínimo total de nodos en Q: {best_total_burned}")
    for paso, c1, c2 in best_cut_sequence:
        print(f"t={paso}: cortafuegos: ({c1}, {c2})")


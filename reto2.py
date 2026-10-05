from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from loaddata import quickstart
from wildfire_core import firecut, get_frontier_edges
from heuristics import dynamic_risk_bound

"""
Reto 2: Contencion Dinamica

Modela la propagación del fuego como un frente de onda determinista sobre un
grafo, en este caso equivalente a uno no dirigido. Se colocan los cortafuegos en tiempo real para mitigar el avance del incendio.
"""

GRAPH_SELECTION: int | str = 7  # Grafo a cargar (3, 4, 5, 7, 8, 10)
INITIAL_NODE: int = 1  # Nodo inicial de ignición en t = 0
K_FIREWALLS: int = 2  # Número de cortafuegos disponibles por paso temporal


def load_graph(
    path_or_id: Path | int | str = GRAPH_SELECTION,
) -> tuple[int, NDArray[np.int64]]:
    """Carga la matriz de adyacencia de G binarizada desde loaddata."""
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

    Reutiliza get_frontier_edges de wildfire_core para aislar la frontera.
    Maximiza el grado libre de v: d_sano(v) = sum_w A[v, w] * (1 - Q[w]).
    """
    candidates = get_frontier_edges(I_t, Q, A.astype(np.float64))
    if not candidates:
        return []

    # Vector d_sano = A @ (1 - Q)
    healthy_degrees = A @ (1 - Q)

    scored_edges: list[tuple[int, int, int]] = []
    for u, v in candidates:
        # El nodo susceptible v es el que no pertenece a Q
        target = v if Q[v] == 0 else u
        origin = u if target == v else v
        scored_edges.append((origin, target, int(healthy_degrees[target])))

    # Orden descendente según conexiones sanas del nodo objetivo
    scored_edges.sort(key=lambda item: item[2], reverse=True)
    return [(u, v) for u, v, _ in scored_edges[:k]]


def select_firewalls_by_risk_bound(
    A: NDArray[np.int64],
    Q: NDArray[np.int64],
    I_t: NDArray[np.int64],
    k: int = 2,
) -> list[tuple[int, int]]:
    """Alternativa: selecciona k cortafuegos de forma secuencial voraz (Greedy k-step)

    reutilizando dynamic_risk_bound de heuristics.py sobre la matriz de adyacencia.
    """
    chosen_cuts: list[tuple[int, int]] = []
    A_eval = A.astype(np.float64)

    for _ in range(k):
        candidates = get_frontier_edges(I_t, Q, A_eval)
        if not candidates:
            break

        best_edge = candidates[0]
        min_risk = float("inf")

        for edge in candidates:
            A_cand = firecut(A_eval, (edge,))
            risk = dynamic_risk_bound(I_t, Q, A_cand)
            if risk < min_risk:
                min_risk = risk
                best_edge = edge

        chosen_cuts.append(best_edge)
        A_eval = firecut(A_eval, (best_edge,))

    return chosen_cuts


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
    use_risk_heuristic: bool = False,
) -> tuple[int, int, NDArray[np.int64]]:
    """Ejecuta la propagación del fuego con colocación dinámica de cortafuegos.

    Args:
        A: Matriz de adyacencia binaria simétrica.
        initial_node: Vértice inicial del incendio.
        k: Cortafuegos a colocar por paso.
        max_steps: Límite superior de pasos temporales.
        verbose: Si es True, imprime la traza de la simulación paso a paso.
        use_risk_heuristic: Alterna entre heurística de grados y dynamic_risk_bound.

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

        if use_risk_heuristic:
            firewalls = select_firewalls_by_risk_bound(A_work, Q, I_t, k=k)
        else:
            firewalls = select_firewalls_by_neighbor_degrees(A_work, Q, I_t, k=k)

        if firewalls:
            if verbose:
                print(f"Cortafuegos colocados en t={t}: {firewalls}")
            A_work = firecut(A_work.astype(np.float64), firewalls).astype(np.int64)
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
    from loaddata import EJEMPLOS_GRAFOS

    print("Reto 2: conexion dinamica aproximada")

    for gid in EJEMPLOS_GRAFOS:
        N, A = load_graph(gid)
        quemados, tiempo, _ = simulate_challenge2(
            A,
            initial_node=INITIAL_NODE,
            k=K_FIREWALLS,
            verbose=False,
            use_risk_heuristic=True,
        )
        salvados = N - quemados
        print(
            f"Grafo {gid:03d} | Quemados: {quemados:2d} | Salvados: {salvados:2d} | Pasos: {tiempo}"
        )


if __name__ == "__main__":
    main()

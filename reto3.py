import numpy as np
from numpy.typing import NDArray
from loaddata import EJEMPLOS_GRAFOS, quickstart
from wildfire_simulator import firecut, step


def get_frontier_edges(
    I_t: NDArray[np.int64],
    Q_t: NDArray[np.int64],
    G_mat: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Identifica las aristas no dirigidas entre el frente activo y nodos sanos."""
    active_nodes = np.flatnonzero(I_t)
    healthy_nodes = np.flatnonzero(Q_t == 0)
    candidates: set[tuple[int, int]] = set()

    for i in active_nodes:
        for j in healthy_nodes:
            if G_mat[i, j] > 0.0 or G_mat[j, i] > 0.0:
                candidates.add((int(min(i, j)), int(max(i, j))))
    return list(candidates)


def select_best_edge_dynamic(
    I_t: NDArray[np.int64],
    Q_t: NDArray[np.int64],
    G_mat: NDArray[np.float64],
) -> tuple[int, int] | None:
    """Selecciona la mejor arista (k=1) minimizando el riesgo cuadrático

    local emitido por el frente activo.
    """
    candidates = get_frontier_edges(I_t, Q_t, G_mat)
    if not candidates:
        return None

    best_edge = candidates[0]
    min_risk = float("inf")

    # Vector fila del frente activo I_t
    i_vec = I_t.astype(np.float64)

    for edge in candidates:
        g_cand = firecut(G_mat, (edge,))
        g2 = g_cand @ g_cand
        np.fill_diagonal(g2, 0.0)

        # Flujo de riesgo emitido a 1 y 2 pasos hacia nodos susceptibles
        projected_risk = i_vec @ (g_cand + g2)
        # Ponderar solo hacia nodos no quemados
        total_risk = float(np.sum(projected_risk * (1 - Q_t)))

        if total_risk < min_risk:
            min_risk = total_risk
            best_edge = edge

    return best_edge


def dinamic_wildfire(
    G_init: NDArray[np.float64],
    initial_node: int,
    rng: np.random.Generator,
    with_firefighters: bool = True,
    endtime: int = 100,
) -> int:
    """Funciona de manera similar a la funcion wildfire_simulator.wildfire(), pero cambian las condiciones iniciales e incorpora la colocacion dinamica de cortafuegos."""
    n = G_init.shape[0]
    G_work = G_init.copy()

    # Establece las condiciones iniciales de Q_0 e I_0
    Q: NDArray[np.int64] = np.zeros(n, dtype=np.int64)
    I_t: NDArray[np.int64] = np.zeros(n, dtype=np.int64)

    Q[initial_node] = 1
    I_t[initial_node] = 1

    t: int = 0

    # Hasta que el tiempo alcance el limite o I_0 = 0, calcula la siguiente franja de nodos quemados. Si hay bomberos, tambien coloca los cortafuegos
    while np.any(I_t) and t < endtime:
        if with_firefighters:
            cut = select_best_edge_dynamic(I_t, Q, G_work)
            if cut is not None:
                G_work = firecut(G_work, (cut,))

        Q, I_t, _ = step(Q, I_t, G_work, rng)
        t += 1

    return int(np.sum(Q))


def experiment3(tries: int = 10_000, initial_node: int = 1) -> None:
    """
    Corre una simulación para los 6 grafos, usa el metodo de montacarlo para calcular las zonas salvas medias.
    """

    print(f"Reto 3 q={initial_node}, k=1, {tries} simulaciones de monte carlo")
    print(f"{'Grafo'}, {'Sin bomberos'}, {'Con Bomberos'}, {'Salvadas (Media)'}")

    for gid in EJEMPLOS_GRAFOS:
        N, G, _, _, seed, graph_name = quickstart(graph=gid)

        # 1. Simulación sin bomberos
        rng_base = np.random.default_rng(seed)
        q_base = np.array(
            [
                dinamic_wildfire(G, initial_node, rng_base, with_firefighters=False)
                for _ in range(tries)
            ]
        )
        mean_base = float(np.mean(q_base))

        # 2. Simulación con política de bomberos
        rng_strat = np.random.default_rng(seed)
        q_strat = np.array(
            [
                dinamic_wildfire(G, initial_node, rng_strat, with_firefighters=True)
                for _ in range(tries)
            ]
        )
        mean_strat = float(np.mean(q_strat))

        zonas_salvadas = mean_base - mean_strat
        print(f"{graph_name}, {mean_base}, {mean_strat}, {zonas_salvadas}")


if __name__ == "__main__":
    experiment3(tries=2_000, initial_node=1)

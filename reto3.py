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


def dynamic_risk_bound(
    I_t: NDArray[np.int64],
    Q_t: NDArray[np.int64],
    g_cand: NDArray[np.float64],
) -> float:
    """
    Establece la cota de riesgo superior, de manera similar que risk_bound() original en prevention.py

    Ahora, el vector de riesgo es: I_t(G'+G'^2), osea que se cambia p0 por I_t y se elimina el caso de t=0.

    Sin embargo, para impedir el daño por retropropagacion, se mutiplica por un factor (1-Q_t) antes de hacer la suma
    de las componentes.
    """

    g2 = g_cand @ g_cand
    np.fill_diagonal(g2, 0.0)

    risk: NDArray[np.float64] = I_t.astype(np.float64) @ (g_cand + g2)

    return float(np.sum(risk * (1 - Q_t)))


def select_best_edge_dynamic(
    I_t: NDArray[np.int64],
    Q_t: NDArray[np.int64],
    G_mat: NDArray[np.float64],
) -> tuple[int, int] | None:
    """Selecciona la mejor arista (k=1) minimizando la cota de riesgo superior.

    Es similar al firewall_algorithm(), tambien del reto 1, solo que calcula las fronteras de manera dinamica,
    y en el primer filtro usa la cota de riesgo superior dinamica en vez de la estatica y no usa la combinatoria.

    Se decido cambiar de algoritmo (inicialmente primero se usaba combinatoria y luego monte carlo). Las razones son similares a las dificultades presentadas
    para el reto 1 con s mayor a 2.

    El metodo de monte carlo original fue cambiado por Greedy Argmin, el cual es mucho menos costo computacionalmente, por lo que es mas apto si
    consideramos que en una estrategia dinamica la velocidad de reaccion debe de ser prioritaria.

    """

    # Selecciona las posibles aristas donde poner cortafuegos
    candidates = get_frontier_edges(I_t, Q_t, G_mat)

    # Checa que poner cortafuegos sea opcion
    if not candidates:
        return None

    # Se selecciona el primero de la lista como prueba
    best_edge = candidates[0]

    # Para el Greedy Argmin. Solo es para el caso inicial
    min_risk = float("inf")

    for edge in candidates:
        g = firecut(G_mat, (edge,))
        total_risk = dynamic_risk_bound(I_t, Q_t, g)

        # Greedy Argmin

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
    """Funciona de manera similar a la funcion wildfire() de wildfire_simulator.py, pero cambian las condiciones iniciales e incorpora la colocacion dinamica de cortafuegos."""
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


def monte_carlo(): ...


def experiment3(tries: int = 10_000, initial_node: int = 1) -> None:
    """
    Corre una simulación para los 6 grafos, usa el metodo de montacarlo para calcular las zonas salvas medias.
    """

    print(f"Reto 3 q={initial_node}, k=1, {tries} simulaciones de monte carlo")

    for gid in EJEMPLOS_GRAFOS:
        N, G, _, _, seed, graph_name = quickstart(graph=gid)

        # Se calcula las zonas quemadas promedio sin cortafuegos dinamicos G
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
        mean_start = float(np.mean(q_strat))

        zonas_salvadas = mean_base - mean_start
        print(
            f"Grafo: {graph_name}, Sin bomberos: {mean_base}, Con bomberos: {mean_strat}, Salvadas (Media): {zonas_salvadas}"
        )


if __name__ == "__main__":
    experiment3(tries=2_000, initial_node=1)

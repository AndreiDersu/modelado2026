import numpy as np
from itertools import combinations
from numpy.typing import NDArray

from wildfire_core import firecut, undirected_edges, get_frontier_edges


def risk_bound(n: int, p0, g) -> float:
    """
    Establece la cota de riesgo superior, definido como:

    R_2 = ||p0(I+G'+G'^2)||_1,

    la cual es una aproximacion lineal de segundo orden del riesgo real de incendio para una matriz de probabilidad G' con cortafuegos establecidos. Esta formula no involucra calculos complejos y es vectorizable, lo que la combierte en una eleccion ventajosa como cota superior.

    El objetivo es minimizar R_2 cambiando la matriz la distribucion de cortafuegos de G'.
    """

    I: NDArray[np.int64] = np.identity(n, dtype=int)
    g2: NDArray[np.float64] = g @ g
    np.fill_diagonal(g2, 0.0)

    risk: NDArray[np.float64] = p0 @ (I + g + g2)

    return float(risk.sum())


def first_filter(
    n: int, p0: NDArray[np.float64], g: NDArray[np.float64], top: int = 50, k: int = 4
) -> list:
    """
    Este metodo corresponde al primer filtrado de combinaciones de cortafuegos a partir de la cota de riesgo superior de la matriz G'.

    Haciendo uso de la lista de aristas no dirigidas (del metodo anterior), se computa la cota de riesgo superior de cada combinación k de cortafuefuegos con el correspondiente numero de aristas.

    De esta manera, tras ordenar las elecciones de cortafuegos de menor a mayor cota de riesgo superior, se devuelve un top (50 por ejemplo) de candidatos con menor cota de riesgo.

    """

    edges = undirected_edges(n, g)

    results: list = []

    for cuts in combinations(edges, k):
        g_mod = firecut(g, cuts)
        val = risk_bound(n, p0, g_mod)
        results.append((val, cuts))

    results.sort(key=lambda x: x[0])
    return results[:top]


def set_firewall(m: NDArray[np.float64], candidate: list, verbose=False):
    """
    Este metodo selecciona un candidato de la lista de candidatos potenciales y realiza los cortes en la matriz G para obtener la respectiva matriz G'
    """

    score, cuts = candidate

    if verbose:
        print(f"risk: {score:.4f}")
        for i, j in cuts:
            print(f"({i} <-> {j})")

    return firecut(m, cuts)


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

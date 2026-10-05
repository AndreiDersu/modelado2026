import numpy as np
from numpy.typing import NDArray

"""
Funciones base reutilizadas en el reto de codigo, como el metodo para colocar cortafuegos, encontrar las aristas no dirigidas, 
encontrar las aristas de la frontera en llamas, y el avance de cada paso de la simulacion del incendio forestal.

"""


def firecut(m: NDArray[np.float64], cuts: tuple) -> NDArray[np.float64]:
    """
    Metodo para establecer los cortafuegos / cortar las aristas a la matriz "m" desde la lista "cuts". Estableciendo un corte en (i,j), entonces: m_ij, m_ji son igualados a cero.
    """

    m_cut = m.copy()
    for cut in cuts:
        i = cut[0]
        j = cut[1]

        m_cut[i, j] = 0
        m_cut[j, i] = 0

    return m_cut


def undirected_edges(n: int, g: NDArray[np.float64]) -> list[tuple[int, int]]:
    """
    Devuelve la lista de aristas no dirigidas. Por las condiciones del problema, dada nuestra matriz de probabilidades G, si G_ij != 0, entonces G_ji != 0.

    Este metodo regresa entonces aristas (i,j) a partir de los valores G_ij != 0.
    """

    edges: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if g[i, j] > 0.0 or g[j, i] > 0.0:
                edges.append((i, j))
    return edges


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


def step(
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
    G_mat: NDArray[np.float64],
    generator: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Define el avance para cada paso de la simulación. Calcula la probabilidad de ignición a partir de I_t.
    Por eficiencia, se usa las formula vectorizada:

    P(i) = 1-prod_{j=1}^n(1-G_ij) ^ (I_tj)

    Para calcular las probabilidades de que el fuego se pase. Al final se multiplica por (1-Q) para evitar
    aquellos nodos ya quemados.
    """

    # Probabilidad de ignición por los vecinos activos en este paso
    gamma: NDArray[np.float64] = 1.0 - np.prod(
        (1.0 - G_mat) ** active_boundary[:, np.newaxis], axis=0
    )
    delta: NDArray[np.float64] = gamma * (1 - burned)

    sample: NDArray[np.float64] = generator.random(size=burned.shape[0])
    new_ignitions: NDArray[np.int64] = (sample < delta).astype(np.int64)

    next_burned: NDArray[np.int64] = burned + new_ignitions
    next_active_boundary: NDArray[np.int64] = new_ignitions

    return next_burned, next_active_boundary, delta

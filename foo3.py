import numpy as np
from pathlib import Path
from numpy.typing import NDArray

from loaddata import quickstart

"""
Simulacion de la propogacion del fuego
"""

N, G, prob_0, rng, seed, graph_name = quickstart()
T: int = 2


def firecut(m: NDArray[np.float64], cuts: tuple) -> NDArray[np.float64]:
    """
    Metodo para establecer los cortafuegos / cortar las aristas a la matriz "m" desde la lista "cuts". Estableciendo un corte en (i,j), entonces tanto m_ij, m_ji son igualados a cero.
    """

    m_cut = m.copy()
    for cut in cuts:
        i = cut[0]
        j = cut[1]

        m_cut[i, j] = 0
        m_cut[j, i] = 0

    return m_cut


def step(
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
    G_mat: NDArray[np.float64],
    generator: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:

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


def wildfire(
    verbose: bool = False,
    endtime: int = T,
    g: NDArray[np.float64] = G,
) -> tuple[int, int]:

    Q: NDArray[np.int64] = np.zeros(N, dtype=np.int64)
    I_t: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

    l_0: int = int(rng.choice(N, p=prob_0))
    Q[l_0] = 1
    I_t[l_0] = 1

    if verbose:
        print("t=0")
        print(f"Nodo Inicial: {l_0}")
        print(f"Q_0 = {Q}")
        print(f"I_0 = {I_t}")

    t: int = 1
    while np.any(I_t) and t <= endtime:
        Q, I_t, delta = step(Q, I_t, g, rng)
        if verbose:
            print("-" * 50)
            print(f"t={t}")
            print(f"Q_{t}={Q}")
            print(f"I_{t}={I_t}")
            print(f"Vector delta: {delta}")
        t += 1

    return int(np.sum(Q)), l_0


def monte_carlo(
    tries: int = 10000, endtime: int = T, g: NDArray[np.float64] = G
) -> tuple[float, float]:

    total_q = np.empty(tries, dtype=np.int64)
    for i in range(tries):
        total_q[i] = wildfire(verbose=False, endtime=endtime, g=g)[0]
    return float(np.mean(total_q)), float(np.median(total_q))


def main(tries: int, endtime: int, firewall: tuple, g=G):

    g_cut = firecut(m=g, cuts=candidate)

    # Simulacion unica
    Q, l_0 = wildfire(verbose=True, endtime=endtime, g=g_cut)
    print(f"\nNodo inicial: {l_0}")
    print(f"Zonas quemadas: {Q}")

    # Monte Carlo
    media, mediana = monte_carlo(tries=tries, endtime=endtime, g=g_cut)
    print("-" * 50)
    print(f"Monte Carlo ({tries} corridas) | Media: {media:.4f} | Mediana: {mediana}")


if __name__ == "__main__":
    candidate = ((3, 15), (6, 16), (6, 17), (16, 17))
    main(tries=10_000, endtime=2, firewall=candidate)

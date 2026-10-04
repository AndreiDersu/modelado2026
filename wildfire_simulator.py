import numpy as np
from pathlib import Path
from numpy.typing import NDArray

from loaddata import quickstart

"""
Preparativos: Simulacion de la propogacion del fuego

En este modulo se encuentra el simulador de propagacion de fuego que se va a usar para el resto de problemas. 

A partir del simulador  de propagacion de fuego, esta el metodo de monte carlo para calcular el promedio de zonas incendiadas tras correr la simulacion un numero alto de veces.

"""

N, G, prob_0, rng, seed, graph_name = quickstart()
T: int = 2


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


def step(
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
    G_mat: NDArray[np.float64],
    generator: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Define el avance para cada paso de la simulación. Calcula la probabilidad de ignición a partir de I_t

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


def wildfire(
    verbose: bool = False,
    endtime: int = T,
    g: NDArray[np.float64] = G,
) -> tuple[int, int]:
    """
    Simula el incendio forestal usando las condiciones planteadas en el planteamiento general. Por cada intesación del tiempo t, calcula el vector de zonas quemadas Q_t, el vector de nodos con la capacidad de pasar el fuego I_t y el vector de probabilidad que se usó para I_t.

    Al finalizar devuelve en numero total de zonas quemadas y el nodo en el que se inició el fuego.

    Si verbose esta activado, entonces muestra en la terminal los parametros de la simulacion para cada tiempo t, util para hacer pruebas individuales.

    """

    # Establece las condiciones iniciales de Q_0 e I_0
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

    # Hasta que el tiempo alcance el limite o I_0 = 0, calcula la siguiente franja de nodos quemados:
    t: int = 1
    while np.any(I_t) and t <= endtime:
        Q, I_t, prob = step(Q, I_t, g, rng)
        if verbose:
            print("-" * 50)
            print(f"t={t}")
            print(f"Q_{t}={Q}")
            print(f"I_{t}={I_t}")
            print(f"Bajo el vector de probabilidad: {prob}")
        t += 1

    # Regresa el total de nodos quemados y el nodo del inicio del incendio
    return int(np.sum(Q)), l_0


def monte_carlo(
    tries: int = 10_000, endtime: int = T, g: NDArray[np.float64] = G
) -> tuple[float, float]:
    """
    Corre la simuilacion de incendio forestal un gran numero de veces para determinar el promedio de zonas incendiadas.
    """

    total_q = np.empty(tries, dtype=np.int64)
    for i in range(tries):
        total_q[i] = wildfire(verbose=False, endtime=endtime, g=g)[0]
    return float(np.mean(total_q)), float(np.median(total_q))


def experiment(tries: int, endtime: int, firewall: "tuple|None", g=G):
    """
    Corre una simulación individual y el método de monte carlo para sacar un promedio. Esta función se usa para simulaciones individuales.
    """

    if firewall is not None:
        g_cut = firecut(m=g, cuts=candidate)
    else:
        g_cut = g

    # Simulacion unica
    Q, l_0 = wildfire(verbose=True, endtime=endtime, g=g_cut)
    print(f"\nNodo inicial: {l_0}")
    print(f"Zonas quemadas: {Q}")

    # Monte Carlo
    media, mediana = monte_carlo(tries=tries, endtime=endtime, g=g_cut)
    print("-" * 50)
    print(f"Monte Carlo ({tries} corridas) | Media: {media:.4f} | Mediana: {mediana}")


if __name__ == "__main__":
    # Ejemplo, grafo 3 usando el cortafuegos a continuacion:
    candidate = ((3, 15), (6, 16), (6, 17), (16, 17))
    experiment(tries=10_000, endtime=20, firewall=None)

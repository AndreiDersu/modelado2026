import numpy as np
from itertools import combinations
from numpy.typing import NDArray

from foo3 import monte_carlo, firecut
from loaddata import quickstart

N, G, prob_0, rng, seed, graph_name = quickstart()


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


def prevention_algorithm(
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


def firewall_algorithm(tries: int = 10_000, endtime: int = 2, ncandidates: int = 20):
    """
    Primero, evalua las mejores matrices de probabilidad G' con k cortafuegos establecidos a partir de la cota superior de riesgo y la primera fase de filtrado de candidatos

    Segundo, se usa el metodo de monte carlo en la muestra reducida de mejores candidatos para determinar tras una cantidad de corridas, cual es la de menor promedio de zonas quemadas. Aquel candidato de cortafuegos es seleccionado como la opcion.

    """

    candidates = prevention_algorithm(n=N, p0=prob_0, g=G, top=ncandidates)

    result: NDArray = np.empty(ncandidates)
    for i in range(0, len(candidates)):
        g = set_firewall(G, candidates[i])
        mean, median = monte_carlo(tries=tries, endtime=endtime, g=g)

        result[i] = mean

    bestidx: int = int(result.argmin())
    # print(f"{best}\n{candidates[bestidx]}")

    return bestidx, float(result.min()), candidates[bestidx][1]


if __name__ == "__main__":
    ncandidates = 100
    tries = 1_000

    print("Calculando... Esto puede llevar algunos segundos")

    index, median, candidates = firewall_algorithm(tries=tries, ncandidates=ncandidates)

    print("\033[H\033[2J", end="")

    print(
        f"Semilla:{seed}\nGrafica: {graph_name}\nTiradas Monta carlo{tries}\nNumero de candidatos{ncandidates}\n\nIndice del candidato: {index}\nDaño medio: {median}\nCortafuegos elegido: {candidates}"
    )

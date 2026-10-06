from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray

from loaddata import quickstart
from wildfire_core import firecut, step
from heuristics import select_best_edge_dynamic

"""
Simulacion de la propogacion del fuego

En este modulo se encuentra el simulador de propagacion de fuego que se va a usar para el resto de problemas. 

A partir del simulador  de propagacion de fuego, esta el metodo de monte carlo para calcular el promedio de zonas incendiadas tras correr la simulacion un numero alto de veces.

El algoritmo es el mismo para el reto 1, 3, 4 y 5

Algunos elementos de la logica de la simulacion estan el el modulo de "wildfire_core.py".

"""


def wildfire(
    g: NDArray[np.float64],
    endtime: "int | None" = 2,
    generator: "np.random.Generator | None" = None,
    verbose: bool = False,
    initial_node: "int | Sequence[int] | None" = None,
    with_firefighters: bool = False,
    max_steps: int = 100,
    p0: "NDArray[np.float64] | None" = None,
) -> tuple[int, list[int]]:
    """
    Simula el incendio forestal usando las condiciones planteadas en el planteamiento general. Por cada intesación del tiempo t, calcula el vector de zonas quemadas Q_t, el vector de nodos con la capacidad de pasar el fuego I_t y el vector de probabilidad que se usó para I_t.

    Al finalizar devuelve en numero total de zonas quemadas y los nodos en los que se inició el fuego.

    Si verbose esta activado, entonces muestra en la terminal los parametros de la simulacion para cada tiempo t, util para hacer pruebas individuales.

    Posteriormente, en el desarrollo del reto 3 y reto 5, se generalizo la funcion para poder ser utilizada tambien ahi. Se le añadieron las vairables de max_steps, with_firefighters y la posiblilidad de escoger multiples focos iniciales.
    """

    if generator is None:
        generator = np.random.default_rng()

    # Establece las condiciones iniciales de Q_0 e I_0

    n = g.shape[0]
    g_work = g.copy()

    Q: NDArray[np.int64] = np.zeros(n, dtype=np.int64)
    I_t: NDArray[np.int64] = np.zeros(n, dtype=np.int64)

    # Establece la simulacion para el nodo o conjunto de nodos iniciales
    if initial_node is None:
        if p0 is not None:
            prob_0 = p0
        else:
            _, _, prob_0, _, _, _ = quickstart()
        l_init = [int(generator.choice(n, p=prob_0))]
    elif isinstance(initial_node, (int, np.integer)):
        l_init = [int(initial_node)]
    else:
        l_init = [int(u) for u in initial_node]

    for u in l_init:
        Q[u] = 1
        I_t[u] = 1

    if verbose:
        print("t=0")
        print(f"Nodo Inicial: {l_init}")
        print(f"Q_0 = {Q}")
        print(f"I_0 = {I_t}")

    # Hasta que el tiempo alcance el limite o I_0 = 0, calcula la siguiente franja de nodos quemados:

    t: int = 1
    while np.any(I_t):
        if endtime is not None and t > endtime:
            break

        if t > max_steps:
            break

        # Intervención activa de los bomberos (Reto 3 y 4)
        if with_firefighters:
            edge_to_cut = select_best_edge_dynamic(I_t, Q, g_work)
            if edge_to_cut is not None:
                g_work = firecut(g_work, (edge_to_cut,))

        # Se propaga sobre la matriz modificada con los cortes
        Q, I_t, prob = step(Q, I_t, g_work, generator)

        if verbose:
            print("-" * 50)
            print(f"t={t}")
            print(f"Q_{t}={Q}")
            print(f"I_{t}={I_t}")
            print(f"Bajo el vector de probabilidad: {prob}")
        t += 1

    # Regresa el total de nodos quemados y los nodos del inicio del incendio
    return int(np.sum(Q)), l_init


def monte_carlo(
    g: NDArray[np.float64],
    tries: int = 10_000,
    endtime: "int | None" = 2,
    initial_node: "int | Sequence[int] | None" = None,
    with_firefighters: bool = False,
    generator: "np.random.Generator | None" = None,
    p0: "NDArray[np.float64] | None" = None,
) -> float:
    """
    Corre la simuilacion de incendio forestal un gran numero de veces para determinar el promedio de zonas incendiadas.
    """
    if generator is None:
        generator = np.random.default_rng()

    if initial_node is None and p0 is None:
        _, _, p0, _, _, _ = quickstart()

    total_q = np.empty(tries, dtype=np.int64)

    for i in range(tries):
        q_val, _ = wildfire(
            g=g,
            initial_node=initial_node,
            endtime=endtime,
            with_firefighters=with_firefighters,
            generator=generator,
            p0=p0,
        )
        total_q[i] = q_val

    return float(np.mean(total_q))


def experiment(g, tries: int, endtime: int, firewall: "tuple|None", generator):
    """
    Corre una simulación individual y el método de monte carlo para sacar un promedio. Esta funcion se usa para simulaciones individuales realizadas mientras se planteaba la resolucion del reto 1.
    """

    if firewall is not None:
        g_cut = firecut(m=g, cuts=firewall)
    else:
        g_cut = g

    # Simulacion unica
    Q, l_0 = wildfire(g=g_cut, endtime=endtime, generator=generator, verbose=True)
    print(f"\nNodo inicial: {l_0}")
    print(f"Zonas quemadas: {Q}")

    # Monte Carlo
    media = monte_carlo(tries=tries, endtime=endtime, g=g_cut, generator=generator)
    print("-" * 50)
    print(f"Monte Carlo ({tries} corridas) | Media: {media:.4f}")


if __name__ == "__main__":
    # Ejemplo, grafo 3 usando el cortafuegos a continuacion:

    _, G, _, rng, _, _ = quickstart(graph=3)

    candidate = ((6, 13), (10, 16), (11, 19), (13, 16))
    experiment(tries=10_000, endtime=2, firewall=candidate, g=G, generator=rng)

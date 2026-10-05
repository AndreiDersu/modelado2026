import time
import numpy as np
from numpy.typing import NDArray

from loaddata import EJEMPLOS_GRAFOS, quickstart
from heuristics import first_filter, set_firewall
from wildfire_simulator import monte_carlo

"""
Reto 1: Políticas de prevención

Calcula para cada grafo el daño medio sin intervención (línea base con G intacto en s=2 etapas), las aristas optimas cortadas (k=4) y daño medio resultante con cortafuegos.

Tambien calcula el numero esperado de zonas salvadas.

El codigo de la simulacion del incendio forestal se encuentra en wildfire_sumulator.py
El codigo del algoritmo de prevencion se encuentra en prevention.py
El codigo para cargar las graficas y datos iniciales se encuentra en loaddata.py

"""


def firewall_algorithm(
    n: int,
    p0,
    g,
    generator,
    tries: int = 10_000,
    endtime: int = 2,
    ncandidates: int = 20,
):
    """
    Primero, evalua las mejores matrices de probabilidad G' con k cortafuegos establecidos a partir de la cota superior de riesgo y la primera fase de filtrado de candidatos

    Segundo, se usa el metodo de monte carlo en la muestra reducida de mejores candidatos para determinar tras una cantidad de corridas, cual es la de menor promedio de zonas quemadas. Aquel candidato de cortafuegos es seleccionado como la opcion.

    """

    candidates = first_filter(n=n, p0=p0, g=g, top=ncandidates)

    result: NDArray = np.empty(len(candidates))
    for i in range(0, len(candidates)):
        g_cut = set_firewall(g, candidates[i])
        rng = np.random.default_rng(67)
        mean = monte_carlo(
            tries=tries,
            endtime=endtime,
            g=g_cut,
            generator=rng,
            p0=p0,
        )

        result[i] = mean

    bestidx: int = int(result.argmin())

    return bestidx, float(result.min()), candidates[bestidx][1]


if __name__ == "__main__":
    ncandidates: int = 100
    tries: int = 2_000
    s_stages: int = 2

    print("Cargando reto 1... Esto puede tomar algunos segundos")
    print(
        f"Configuracion: k=4 cortafuegos, s={s_stages} etapas, {tries} simulaciones de monte carlo"
    )

    resumen_resultados: list[dict] = []

    for gid in EJEMPLOS_GRAFOS:
        t0 = time.time()

        # Cargar graficas y cosas
        N, G, prob_0, rng, seed, graph_name = quickstart(graph=gid)

        # Se calcula las zonas quemadas promedio sin cortafuegos G
        basedamage = monte_carlo(
            tries=tries, endtime=s_stages, g=G, generator=rng, p0=prob_0
        )

        # Se selecciona la mejor configuracion para la prevencion de cortafuegos G'
        rng = np.random.default_rng(seed)
        best_idx, firewalldamage, firewalls = firewall_algorithm(
            tries=tries,
            endtime=s_stages,
            ncandidates=ncandidates,
            n=N,
            p0=prob_0,
            g=G,
            generator=rng,
        )

        # Se calcula el numero esperado de zonas salvadas
        efficiency = basedamage - firewalldamage
        ctime = time.time() - t0

        print("-" * 70)
        print(f"Grafo: {graph_name} (N = {N}) | Semilla: {seed}")
        print(f"promedio de zonas quemadas sin bomberos: {basedamage:.4f} zonas")
        print(f"Promedio de zonas quemadas con cortafuegos: {firewalldamage:.4f} zonas")
        print(f"Num. esperado de zonas salvadas: {efficiency:.4f} zonas")
        print(f"Aristas cortadas elegidas: {firewalls}")
        print(f"Tiempo de cómputo: {ctime:.2f} s")

import time
import numpy as np
from loaddata import EJEMPLOS_GRAFOS, quickstart
from prevention import firewall_algorithm
import wildfire_simulator as ws
from wildfire_simulator import monte_carlo

"""
Reto 1: Políticas de prevención

Calcula para cada grafo el daño medio sin intervención (línea base con G intacto en s=2 etapas), las aristas optimas cortadas (k=4) y daño medio resultante con cortafuegos.

Tambien calcula el numero esperado de zonas salvadas.

El codigo de la simulacion del incendio forestal se encuentra en wildfire_sumulator.py
El codigo del algoritmo de prevencion se encuentra en prevention.py
El codigo para cargar las graficas y datos iniciales se encuentra en loaddata.py

"""

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
        ws.N = N
        ws.prob_0 = prob_0
        ws.rng = rng

        # Se calcula las zonas quemadas promedio sin cortafuegos G
        ws.rng = np.random.default_rng(seed)
        basedamage, _ = monte_carlo(tries=tries, endtime=s_stages, g=G)

        # Se selecciona la mejor configuracion para la prevencion de cortafuegos G'
        ws.rng = np.random.default_rng(seed)
        best_idx, firewalldamage, firewalls = firewall_algorithm(
            tries=tries,
            endtime=s_stages,
            ncandidates=ncandidates,
            n=N,
            p0=prob_0,
            g=G,
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

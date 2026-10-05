import numpy as np
from loaddata import EJEMPLOS_GRAFOS, quickstart
from heuristics import select_best_edge_dynamic
from wildfire_simulator import monte_carlo

"""
Reto 4: Intelegencia para el mal

La logica es practicamente la misma que la usada para el reto 3, lo unico que cambia es el objetivo del programa. 
Ahora se corre el monte carlo para escoger la zona de peor 

"""


def experiment4(tries: int = 2_000) -> None:
    """
    Corre una simulación para los 6 grafos, usa el metodo de monte carlo para calcular la mejor zona para iniciar el fuego.

    El codigo es practicamente el mismo que para el reto 3, lo unico que cambia es el objetivo al fin y al cabo, pero no hay
    funciones o logica nueva
    """

    print(f"Reto 4, {tries} simulaciones de monte carlo")

    for gid in EJEMPLOS_GRAFOS:
        N, G, _, _, seed, graph_name = quickstart(graph=gid)

        best_l = None
        max_damage = -1.0
        damages_per_node = np.zeros(N)

        # El incendiario evalua laz zonas quemadas esperadas para cada posible foco l
        for l in range(N):
            rng = np.random.default_rng(seed + l)
            damage = monte_carlo(
                g=G,
                initial_node=l,
                endtime=None,
                tries=tries,
                generator=rng,
                with_firefighters=True,
            )

            damages_per_node[l] = damage

            # Aqui se repite un poco la idea usada para el codigo dinamico del reto 3
            if damage > max_damage:
                max_damage = damage
                best_l = l

        Q_0 = np.zeros(N, dtype=np.int64)
        I_0 = np.zeros(N, dtype=np.int64)
        Q_0[best_l] = 1
        I_0[best_l] = 1
        first_firewall = select_best_edge_dynamic(I_0, Q_0, G)

        print(f"\nGrafo: {graph_name}")
        print(f"Zona elegida: N= {best_l}, Q esperado: {max_damage}")
        print(f"Primer cortafuegos (t=0): {first_firewall}")


if __name__ == "__main__":
    experiment4(tries=2_000)

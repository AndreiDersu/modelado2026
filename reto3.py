import numpy as np
from numpy.typing import NDArray
from loaddata import EJEMPLOS_GRAFOS, quickstart
from wildfire_simulator import monte_carlo


def experiment3(tries: int = 10_000, initial_node: int = 1) -> None:
    """
    Corre una simulación para los 6 grafos, usa el metodo de montacarlo para calcular las zonas salvas medias.
    """

    print(f"Reto 3 q={initial_node}, k=1, {tries} simulaciones de monte carlo")

    for gid in EJEMPLOS_GRAFOS:
        N, G, _, _, seed, graph_name = quickstart(graph=gid)

        # Primero se calcula las zonas quemadas promedio sin cortafuegos dinamicos, ie matriz G (hasta extinción)
        rng_base = np.random.default_rng(seed)

        mean_base = monte_carlo(
            g=G,
            initial_node=initial_node,
            endtime=None,
            tries=tries,
            generator=rng_base,
            with_firefighters=False,
        )

        # Simulación con cortafuegos dinamicos, ie matriz G'_t (hasta extinción)
        rng_strat = np.random.default_rng(seed)

        mean = monte_carlo(
            g=G,
            initial_node=initial_node,
            endtime=None,
            tries=tries,
            generator=rng_strat,
            with_firefighters=True,
        )

        zonas_salvadas = mean_base - mean
        print(
            f"Grafo: {graph_name}, Sin bomberos: {mean_base}, Con bomberos: {mean}, Salvadas (Media): {zonas_salvadas}"
        )


if __name__ == "__main__":
    experiment3(tries=2_000, initial_node=1)

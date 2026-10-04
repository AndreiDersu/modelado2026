import time
import wildfire_simulator as ws
from loaddata import EJEMPLOS_GRAFOS, quickstart
from prevention import firewall_algorithm

if __name__ == "__main__":
    ncandidates = 100
    tries = 1_000

    print("Calculando... Esto puede llevar algunos segundos\n")

    for gid in EJEMPLOS_GRAFOS:
        t0 = time.time()

        N, G, prob_0, rng, seed, graph_name = quickstart(graph=gid)
        ws.N = N
        ws.prob_0 = prob_0
        ws.rng = rng

        index, median, candidates = firewall_algorithm(
            tries=tries, ncandidates=ncandidates, n=N, p0=prob_0, g=G
        )

        t_total = time.time() - t0

        print("=" * 50)
        print(
            f"Semilla:{seed}\nGrafica: {graph_name}\nTiradas Monta carlo: {tries}\nNumero de candidatos: {ncandidates}\n\nIndice del candidato: {index}\nDaño medio: {median}\nCortafuegos elegido: {candidates}\nTiempo de ejecucion: {t_total:.2f}s"
        )

    print("=" * 50)

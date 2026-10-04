import numpy as np
from itertools import combinations
from pathlib import Path
from loaddata import cargar_grafo, cargar_probabilidades_iniciales
from numpy.typing import NDArray


RNG_SEED = 67
if RNG_SEED is not None:
    rng = np.random.default_rng(RNG_SEED)
else:
    rng = np.random.default_rng()

BASE_DIR: Path = Path(__file__).resolve().parent
GRAPH_FILE: Path = (
    BASE_DIR / "Datos" / "Datos" / "graph_003_probs.txt"
    if (BASE_DIR / "Datos" / "Datos" / "graph_003_probs.txt").is_file()
    else BASE_DIR / "Datos" / "Datos" / "graph_010_probs.txt"
)
INICIO_FILE: Path = BASE_DIR / "Datos" / "Datos" / "inicio.txt"


N, G = cargar_grafo(GRAPH_FILE)
prob_0: NDArray[np.float64] = cargar_probabilidades_iniciales(INICIO_FILE, N)
T: int = 2


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


def ady_transform(m: NDArray[np.float64]) -> NDArray[np.int64]:

    return (m != 0).astype(int)


def foofoo(node: int):
    H = G[node, :]

    return H, H.sum()


def foofoo2(node: int):

    H = G @ G[node, :]

    return H, H.sum()


def firecut(m: NDArray[np.float64], cuts: tuple) -> NDArray[np.float64]:

    m_cut = m.copy()
    for cut in cuts:
        i = cut[0]
        j = cut[1]

        m_cut[i, j] = 0
        m_cut[j, i] = 0

    return m_cut


def risk_bound(p0=prob_0, g=G) -> float:

    I: NDArray[np.int64] = np.identity(N, dtype=int)
    g2: NDArray[np.float64] = g @ g
    np.fill_diagonal(g2, 0.0)

    risk: NDArray[np.float64] = p0 @ (I + g + g2)

    return float(risk.sum())


def undirected_edges(g: NDArray[np.float64]) -> list[tuple[int, int]]:

    edges: list[tuple[int, int]] = []

    for u in range(N):
        for v in range(u + 1, N):
            if g[u, v] > 0.0 or g[v, u] > 0.0:
                edges.append((u, v))
    return edges


def prevention_algorithm(p0=prob_0, g=G, top: int = 50, k: int = 4):

    edges = undirected_edges(g)

    results: list = []

    for cuts in combinations(edges, k):
        g_mod = firecut(g, cuts)
        val = risk_bound(p0, g_mod)
        results.append((val, cuts))

    results.sort(key=lambda x: x[0])
    return results[:top]


def firewall(m: NDArray[np.float64], candidate: list, verbose=False):

    score, cuts = candidate

    if verbose:
        print(f"risk: {score:.4f}")
        for i, j in cuts:
            print(f"({i} <-> {j})")

    return firecut(m, cuts)


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


def experiment(tries: int, endtime: int, candidate: list):
    g = firewall(G, candidate)

    Q, l_0 = wildfire(verbose=True, endtime=endtime, g=g)
    print(f"\nNodo inicial: {l_0}")
    print(f"Zonas quemadas: {Q}")

    # Evaluación de Monte Carlo
    media, mediana = monte_carlo(tries=tries, endtime=endtime, g=g)
    print("-" * 50)
    print(f"Monte Carlo ({tries} corridas) | Media: {media:.4f} | Mediana: {mediana}")


def main(tries: int = 1000, endtime: int = T, ncandidates: int = 20):

    candidates = prevention_algorithm(top=ncandidates)

    result: NDArray = np.empty(ncandidates)
    for i in range(0, len(candidates)):
        g = firewall(G, candidates[i])
        mean, median = monte_carlo(tries=tries, endtime=endtime, g=g)

        result[i] = mean

    bestidx: int = int(result.argmin())
    best: list = [bestidx, result.min()]
    print(f"{best}\n{candidates[bestidx]}")


if __name__ == "__main__":
    # main(tries=10_000, endtime=2, ncandidates=100)

    candidates = prevention_algorithm(top=100)
    experiment(tries=10_000, endtime=2, candidate=candidates[84])
    # print(prevention_algorithm())
    # print(prevention_algorithm(top=1))

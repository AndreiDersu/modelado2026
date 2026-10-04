import numpy as np
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
    BASE_DIR / "Datos" / "Datos" / "graph_006_probs.txt"
    if (BASE_DIR / "Datos" / "Datos" / "graph_006_probs.txt").is_file()
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


def wildfire(verbose: bool = False, endtime: int = T) -> tuple[int, int]:
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
        Q, I_t, delta = step(Q, I_t, G, rng)
        if verbose:
            print("-" * 50)
            print(f"t={t}")
            print(f"Q_{t}={Q}")
            print(f"I_{t}={I_t}")
            print(f"Vector delta: {delta}")
        t += 1

    return int(np.sum(Q)), l_0


def monte_carlo(tries: int = 10000, endtime: int = T) -> tuple[float, float]:
    total_q = np.empty(tries, dtype=np.int64)
    for i in range(tries):
        total_q[i] = wildfire(verbose=False, endtime=endtime)[0]
    return float(np.mean(total_q)), float(np.median(total_q))


def main(tries: int = 1000, endtime: int = T):
    # Corrida única demostrativa
    Q, l_0 = wildfire(verbose=True, endtime=endtime)
    print(f"\nNodo inicial: {l_0}")
    print(f"Zonas quemadas: {Q}")

    # Evaluación de Monte Carlo
    media, mediana = monte_carlo(tries=tries, endtime=endtime)
    print("-" * 50)
    print(f"Monte Carlo ({tries} corridas) | Media: {media:.4f} | Mediana: {mediana}")


if __name__ == "__main__":
    main(tries=10_000, endtime=2)

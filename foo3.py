from pathlib import Path
from time import sleep
import numpy as np
from numpy.typing import NDArray

"""
Lo mismo que el archivo foo2.py, pero ahora con los datos oficiales. EN este ejemplo se usa el grafico 6.
"""

rng = np.random.default_rng()

# Código IA
# --------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
GRAPH_FILE: Path = (
    BASE_DIR / "Datos" / "Datos" / "graph_006_probs.txt"
    if (BASE_DIR / "Datos" / "Datos" / "graph_006_probs.txt").is_file()
    else BASE_DIR / "Datos" / "Datos" / "graph_010_probs.txt"
)
INICIO_FILE: Path = BASE_DIR / "Datos" / "Datos" / "inicio.txt"


def cargar_grafo(ruta_archivo: Path) -> tuple[int, NDArray[np.float64]]:
    """Carga la matriz de probabilidades G a partir de los datos del grafo dirigido.

    G[j, i] almacena la probabilidad p_{ji} de propagación desde el nodo j hacia el nodo i.
    """
    if not ruta_archivo.is_file():
        raise FileNotFoundError(f"Archivo de grafo no encontrado: {ruta_archivo}")

    aristas: list[tuple[int, int, float]] = []
    nodos: set[int] = set()

    with ruta_archivo.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) >= 3:
                u, v = int(partes[0]), int(partes[1])
                prob = float(partes[2])
                aristas.append((u, v, prob))
                nodos.add(u)
                nodos.add(v)

    n: int = max(nodos) + 1 if nodos else 0
    matriz_g: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
    for u, v, prob in aristas:
        matriz_g[u, v] = prob

    return n, matriz_g


def cargar_probabilidades_iniciales(ruta_archivo: Path, n: int) -> NDArray[np.float64]:
    """Carga el vector de distribución de probabilidad de ignición inicial sum_l p_l = 1."""
    if not ruta_archivo.is_file():
        return np.full(n, 1.0 / n, dtype=np.float64)

    probs: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    with ruta_archivo.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) >= 2:
                nodo = int(partes[0])
                prob = float(partes[1])
                if 0 <= nodo < n:
                    probs[nodo] = prob

    total: float = float(probs.sum())
    if total > 0.0:
        probs /= total
    else:
        probs = np.full(n, 1.0 / n, dtype=np.float64)
    return probs


# Fin del código IA ---------------------------------------------------------------------------------

# Cargar número de nodos N y matriz G desde los datos del grafo
N, G = cargar_grafo(GRAPH_FILE)
T: int = 5

# Initial node
prob_0: NDArray[np.float64] = cargar_probabilidades_iniciales(INICIO_FILE, N)
l_0: int = int(rng.choice(N, p=prob_0))

# Q: Total burned nodes vector
Q: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

# I_t: Burned boundary
I_t: NDArray[np.int64] = np.zeros(N, dtype=np.int64)


def step(
    burned: NDArray[np.int64] = Q,
    active_boundary: NDArray[np.int64] = I_t,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:

    # En este modelo, delta es el vector de probabilidad de que cada nodo x_i se queme

    # Compute the total probability that node x_i burns out in the next step using eq1
    gamma: NDArray[np.float64] = 1.0 - np.prod(
        (1.0 - G) ** active_boundary[:, np.newaxis], axis=0
    )

    # The computed probabilities without burned nodes
    delta: NDArray[np.float64] = gamma * (1 - burned)

    # Choose the following nodes to be burned
    sample: NDArray[np.float64] = rng.random(size=N)
    new_ignitions: NDArray[np.int64] = (sample < delta).astype(np.int64)

    # Q += I_t
    next_burned: NDArray[np.int64] = burned + new_ignitions

    # I_t
    next_active_boundary: NDArray[np.int64] = new_ignitions

    return next_burned, next_active_boundary, delta


def main() -> None:
    global Q, I_t

    Q[l_0] = 1
    I_t[l_0] = 1

    print(G)
    print("-" * 50)

    print("t=0")
    print(f"{Q=}")
    print(f"{I_t=}")

    t: int = 1
    while np.any(I_t) or t != T:
        Q, I_t, delta = step(Q, I_t)
        print("-" * 50)
        print(f"t={t}")
        print(f"{Q=}")
        print(f"{I_t=}")
        print(f"Probabilities vector: {delta}")
        t += 1

    print("-" * 50)
    print(f"End at t={t - 1}.")


if __name__ == "__main__":
    main()

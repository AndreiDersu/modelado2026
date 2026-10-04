from pathlib import Path
import numpy as np
from numpy.typing import NDArray


RNG_SEED = 67

GRAPH_NAME: str = "graph_005_probs"

BASE_DIR: Path = Path(__file__).resolve().parent
GRAPH_FILE: Path = (
    BASE_DIR / "Datos" / GRAPH_NAME
    if (BASE_DIR / "Datos" / GRAPH_NAME).is_file()
    else BASE_DIR / "Datos" / "graph_010_probs.txt"
)

INICIO_FILE: Path = BASE_DIR / "Datos" / "Datos" / "inicio.txt"


def cargar_grafo(ruta_archivo: Path) -> tuple[int, NDArray[np.float64]]:
    if not ruta_archivo.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")

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


def quickstart(rng_seed: "int|None" = RNG_SEED):

    if RNG_SEED is not None:
        rng = np.random.default_rng(RNG_SEED)
    else:
        rng = np.random.default_rng()

    N, G = cargar_grafo(GRAPH_FILE)
    prob_0: NDArray[np.float64] = cargar_probabilidades_iniciales(INICIO_FILE, N)

    return (N, G, prob_0, rng, rng_seed, GRAPH_NAME)

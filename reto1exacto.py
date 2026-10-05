from itertools import combinations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

BASE_DIR: Path = Path(_file_).resolve().parent
GRAPH_FILE: Path = BASE_DIR / "Datos" / "Datos" / "graph_004_probs.txt"


def cargar_grafo(ruta_archivo: Path) -> tuple[int, NDArray[np.int64]]:
    """Carga la matriz de adyacencia A del grafo (simétrica para no dirigido)."""
    if not ruta_archivo.is_file():
        raise FileNotFoundError(f"Archivo de grafo no encontrado: {ruta_archivo}")

    aristas: list[tuple[int, int]] = []
    nodos: set[int] = set()

    with ruta_archivo.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split()
            if len(partes) >= 2:
                u, v = int(partes[0]), int(partes[1])
                aristas.append((u, v))
                nodos.add(u)
                nodos.add(v)

    n: int = max(nodos) + 1 if nodos else 0
    matriz_adj: NDArray[np.int64] = np.zeros((n, n), dtype=np.int64)
    for u, v in aristas:
        matriz_adj[u, v] = 1
        matriz_adj[v, u] = 1

    return n, matriz_adj


def step_propagacion(
    A: NDArray[np.int64],
    burned: NDArray[np.int64],
    active_boundary: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Avanza un paso de tiempo en la propagación determinista."""
    vecinos_alcanzables = (active_boundary @ A) > 0
    new_ignitions = (vecinos_alcanzables & (burned == 0)).astype(np.int64)

    next_burned = burned + new_ignitions
    next_active_boundary = new_ignitions

    return next_burned, next_active_boundary


# Variables globales para almacenar la mejor solución global encontrada
mejor_total_quemados = float("inf")
mejor_secuencia_cortes = []


def buscar_optimo_rec(
    A: NDArray[np.int64],
    Q: NDArray[np.int64],
    I_t: NDArray[np.int64],
    t: int,
    secuencia_actual: list[tuple[int, tuple[int, int], tuple[int, int]]],
    N: int,
) -> None:
    global mejor_total_quemados, mejor_secuencia_cortes

    total_quemados_actual = int(np.sum(Q))

    # PODA (Branch and Bound): Si ya quemamos igual o más nodos que el mejor resultado global, descartamos la rama
    if total_quemados_actual >= mejor_total_quemados:
        return

    # Condición de parada: Se extinguió el fuego o se quemó todo el grafo
    if not np.any(I_t) or np.all(Q == 1):
        if total_quemados_actual < mejor_total_quemados:
            mejor_total_quemados = total_quemados_actual
            mejor_secuencia_cortes = list(secuencia_actual)
        return

    # Identificar aristas expuestas desde la frontera activa hacia nodos no quemados
    nodos_activos = np.where(I_t == 1)[0]
    aristas_candidatas = []
    for u in nodos_activos:
        for v in range(N):
            if A[u, v] == 1 and Q[v] == 0:
                aristas_candidatas.append((u, v))

    # Si hay al menos 2 aristas candidatas, exploramos las combinaciones
    if len(aristas_candidatas) >= 2:
        for comb in combinations(aristas_candidatas, 2):
            A_next = A.copy()
            for u, v in comb:
                A_next[u, v] = 0
                A_next[v, u] = 0

            Q_next, I_next = step_propagacion(A_next, Q, I_t)

            secuencia_actual.append((t, comb[0], comb[1]))
            buscar_optimo_rec(A_next, Q_next, I_next, t + 1, secuencia_actual, N)
            secuencia_actual.pop()

    elif len(aristas_candidatas) == 1:
        u, v = aristas_candidatas[0]
        A_next = A.copy()
        A_next[u, v] = 0
        A_next[v, u] = 0

        Q_next, I_next = step_propagacion(A_next, Q, I_t)

        secuencia_actual.append((t, (u, v), (u, v)))
        buscar_optimo_rec(A_next, Q_next, I_next, t + 1, secuencia_actual, N)
        secuencia_actual.pop()
    else:
        # No hay aristas expuestas; propagar normalmente
        Q_next, I_next = step_propagacion(A, Q, I_t)
        buscar_optimo_rec(A, Q_next, I_next, t + 1, secuencia_actual, N)


def main() -> None:
    global mejor_total_quemados, mejor_secuencia_cortes

    N, A = cargar_grafo(GRAPH_FILE)

    nodo_inicio: int = 1

    Q_0: NDArray[np.int64] = np.zeros(N, dtype=np.int64)
    I_0: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

    Q_0[nodo_inicio] = 1
    I_0[nodo_inicio] = 1

    print("Iniciando búsqueda exhaustiva optimizada con poda (Branch & Bound)...")
    buscar_optimo_rec(A, Q_0, I_0, t=1, secuencia_actual=[], N=N)

    print("\n" + "=" * 65)
    print("RESULTADO DE LA CONFIGURACIÓN GLOBAL ÓPTIMA")
    print("=" * 65)
    print(f"Mínimo total de nodos quemados posible: {mejor_total_quemados} de {N}")
    print("\nSecuencia óptima de cortafuegos por paso:")
    for paso, c1, c2 in mejor_secuencia_cortes:
        print(f"  Tiempo t={paso}: Cortar aristas {c1} y {c2}")
    print("=" * 65)


if _name_ == "_main_":
    main()

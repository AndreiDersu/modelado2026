from pathlib import Path
import numpy as np
from numpy.typing import NDArray

"""
Carga las graficas, la semilla (para los sucesos aleatorios) y las condiciones iniciales.
No hay nada relevante para las solucion de los problemas aqui, los metodos son mas o menos estandar.
"""

RNG_SEED: int = 67

# Catálogo de los 6 ejemplos de grafos disponibles en Datos/
EJEMPLOS_GRAFOS: dict[int, str] = {
    3: "graph_003_probs.txt",
    4: "graph_004_probs.txt",
    5: "graph_005_probs.txt",
    7: "graph_007_probs.txt",
    8: "graph_008_probs.txt",
    10: "graph_010_probs.txt",
}

# Lista ordenada de los grafos
LISTA_GRAFOS: list[str] = [
    "graph_003_probs.txt",  # [0] Grafo 3
    "graph_004_probs.txt",  # [1] Grafo 4
    "graph_005_probs.txt",  # [2] Grafo 5
    "graph_007_probs.txt",  # [3] Grafo 7
    "graph_008_probs.txt",  # [4] Grafo 8
    "graph_010_probs.txt",  # [5] Grafo 10
]

GRAFICA_SELECCIONADA: int | str = 5

BASE_DIR: Path = Path(__file__).resolve().parent

# Archivo de probabilidades iniciales
INICIO_FILE: Path = (
    BASE_DIR / "Datos" / "inicio.txt"
    if (BASE_DIR / "Datos" / "inicio.txt").is_file()
    else BASE_DIR / "Datos" / "Datos" / "inicio.txt"
)


def resolver_archivo_grafo(seleccion: int | str | Path) -> tuple[Path, str]:
    """Resuelve la ruta completa y el nombre base de la gráfica a partir de un ID,
    índice, nombre o Path.

    Opciones admitidas:
      - Entero ID: 3, 4, 5, 7, 8, 10
      - Entero índice: 0 a 5 (mapeados a LISTA_GRAFOS)
      - Cadena con número: "3", "003", "10", etc.
      - Cadena con nombre: "graph_005_probs", "graph_005_probs.txt", etc.
      - Objeto Path directo

    Returns:
        (ruta_archivo, nombre_stem)
    """
    datos_dir = BASE_DIR / "Datos"

    if isinstance(seleccion, Path):
        if seleccion.is_file():
            return seleccion, seleccion.stem
        candidato = datos_dir / seleccion.name
        if candidato.is_file():
            return candidato, candidato.stem

    if isinstance(seleccion, int):
        if seleccion in EJEMPLOS_GRAFOS:
            archivo = EJEMPLOS_GRAFOS[seleccion]
            return datos_dir / archivo, Path(archivo).stem
        if 0 <= seleccion < len(LISTA_GRAFOS):
            archivo = LISTA_GRAFOS[seleccion]
            return datos_dir / archivo, Path(archivo).stem
        raise ValueError(
            f"Grafo no válido: {seleccion}. "
            f"Opciones válidas por ID: {list(EJEMPLOS_GRAFOS.keys())} "
            f"o por índice: 0 a {len(LISTA_GRAFOS) - 1}."
        )

    sel_str = str(seleccion).strip()

    if sel_str.isdigit():
        val = int(sel_str)
        if val in EJEMPLOS_GRAFOS:
            archivo = EJEMPLOS_GRAFOS[val]
            return datos_dir / archivo, Path(archivo).stem
        if 0 <= val < len(LISTA_GRAFOS):
            archivo = LISTA_GRAFOS[val]
            return datos_dir / archivo, Path(archivo).stem

    # Archivo exacto en Datos/
    if (datos_dir / sel_str).is_file():
        ruta = datos_dir / sel_str
        return ruta, ruta.stem

    # Archivo con .txt agregado
    if (datos_dir / f"{sel_str}.txt").is_file():
        ruta = datos_dir / f"{sel_str}.txt"
        return ruta, Path(sel_str).stem

    # Búsqueda difusa en la lista conocida
    sel_clean = sel_str.lower().removesuffix(".txt")
    for archivo in LISTA_GRAFOS:
        stem = Path(archivo).stem
        if sel_clean == stem or sel_clean in stem:
            return datos_dir / archivo, stem

    raise FileNotFoundError(
        f"No se encontró el archivo de grafo para '{seleccion}'. "
        f"Opciones válidas: {list(EJEMPLOS_GRAFOS.keys())} o nombres: {LISTA_GRAFOS}"
    )


# Resolución inicial para variables a nivel de módulo
GRAPH_FILE, GRAPH_NAME = resolver_archivo_grafo(GRAFICA_SELECCIONADA)


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


pass


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


def quickstart(
    rng_seed: int | None = RNG_SEED,
    graph: int | str | Path | None = None,
    **kwargs: object,
) -> tuple[
    int, NDArray[np.float64], NDArray[np.float64], np.random.Generator, int | None, str
]:
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    # Si se pasa un grafo específico se usa ese; si no, se usa GRAPH_NAME / GRAFICA_SELECCIONADA
    target = graph
    if target is None and "grafo" in kwargs:
        target = kwargs["grafo"]  # type: ignore[assignment]
    if target is None and "graph_name" in kwargs:
        target = kwargs["graph_name"]  # type: ignore[assignment]
    if target is None:
        target = GRAPH_NAME

    ruta_archivo, nombre_str = resolver_archivo_grafo(target)
    N, G = cargar_grafo(ruta_archivo)
    prob_0: NDArray[np.float64] = cargar_probabilidades_iniciales(INICIO_FILE, N)

    return (N, G, prob_0, rng, rng_seed, nombre_str)

"""Simulación de propagación estocástica de incendios ("Lo que arde").

Implementación de la dinámica discreta de ignición estocástica sobre redes dirigidas
conforme a la formulación probabilística exacta de eventos independientes:
    P(i arde en t+1) = 1 - prod_{j activo} (1 - p_{ji})
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calcular_propagacion(
    matriz_prob: NDArray[np.float64],
    estado_activo: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calcula las probabilidades de propagación e ignición para la siguiente etapa.

    Args:
        matriz_prob: Matriz (N, N) donde G[j, i] = P(fuego propaga de j hacia i).
        estado_activo: Vector binario (N,) donde 1 indica nodo ardiendo en la etapa actual.

    Returns:
        Tupla (gamma, delta):
            - gamma: Probabilidad de que el nodo i sea alcanzado por al menos un frente activo.
            - delta: Probabilidad de que el nodo i sea un nuevo frente de fuego (I_{t+1}).
    """
    nodos_activos = estado_activo == 1

    if not np.any(nodos_activos):
        n = matriz_prob.shape[0]
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    # Probabilidad de que todos los intentos de propagación hacia i fallen: prod (1 - p_{ji})
    prob_no_arde: NDArray[np.float64] = np.prod(
        1.0 - matriz_prob[nodos_activos, :], axis=0
    )

    # Probabilidad complementaria exacta de ignición
    gamma: NDArray[np.float64] = 1.0 - prob_no_arde

    # Probabilidad de encendido como nuevo frente (solo zonas susceptibles que no estaban ardiendo)
    delta: NDArray[np.float64] = gamma * (1 - estado_activo)

    return gamma, delta


def main() -> None:
    """Ejecución de demostración con matriz aleatoria de 5 zonas."""
    rng = np.random.default_rng(seed=43)
    n = 5

    # G[j, i] = probabilidad de que el fuego pase de j hacia i
    g: NDArray[np.float64] = rng.random(size=(n, n))
    np.fill_diagonal(g, 0.0)

    # Estado activo beta en la etapa t (1 = ardiendo, 0 = susceptible)
    beta: NDArray[np.int64] = rng.integers(low=0, high=2, size=n)

    gamma, delta = calcular_propagacion(g, beta)

    print("Matriz de probabilidades G (origen fila j -> destino columna i):")
    print(np.round(g, 4))
    print("-" * 55)
    print(f"Estado activo en t  (beta):  {beta}")
    print(f"Prob. impacto fuego (gamma): {np.round(gamma, 4)}")
    print(f"Prob. nuevo frente  (delta): {np.round(delta, 4)}")


if __name__ == "__main__":
    main()

import numpy as np
from numpy.typing import NDArray
from time import sleep

"""
Ejemplo simple de la propagación del fuego a través del tiempo en caso de no intervención de los bomberos. En este ejemplo el grafo es de 4 nodos
"""

rng = np.random.default_rng(seed=43)

dice = np.random.default_rng()

N: int = 4

# Initial node
l_0 = rng.integers(low=0, high=N)

# graph G(N,(j,i)) as a matrix
G: NDArray[np.float64] = rng.random(size=(N, N))
np.fill_diagonal(G, 0.0)

kappa = np.zeros((N, N))
kappa[2][3] = G[2][3]
kappa[3][2] = G[3][2]

G[2][3] = 0
G[3][2] = 0

G[1][0] = 0
G[0][1] = 0

A = G - kappa


# Q: Total burned nodes vector
Q: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

# I_t: Burned boundary
I_t: NDArray[np.int64] = np.zeros(N, dtype=np.int64)

B = (G != 0).astype(int)

# G[2][0] = 0
# G[0][2] = 0


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
    sample: NDArray[np.float64] = dice.random(size=N)
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
    print("-" * 501)

    print("t=0")
    print(f"{Q=}")
    print(f"{I_t=}")

    t: int = 1
    while np.any(I_t):
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
    # print(G)
    # print(B @ B)

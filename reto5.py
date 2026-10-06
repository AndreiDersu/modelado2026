from itertools import combinations
import numpy as np
from numpy.typing import NDArray

from loaddata import EJEMPLOS_GRAFOS, quickstart
from heuristics import dynamic_risk_bound, select_best_edge_dynamic
from wildfire_simulator import monte_carlo

"""
Reto 5: Ataque Coordinado con Focos Múltiples (m focos simultáneos)

Determina el conjunto óptimo de vértices que maximiza el
daño esperado del fuego frente a la intervención reactiva de los bomberos (k=1).
"""

M_FIRES: int = 2  # Numero de focos simultaneos del piromano
TRIES_MC: int = 1_000  # Simulaciones Monte Carlo por candidato evaluado


def multi_fire_risk_bound(
    S: tuple[int, ...],
    G: NDArray[np.float64],
    N: int,
) -> float:
    """Calcula la cota analitica de destruccion proyectada a 2 pasos

    reutilizando dynamic_risk_bound(I_0, Q_0, G) con I_0 = Q_0.
    """
    I_0 = np.zeros(N, dtype=np.int64)
    I_0[list(S)] = 1
    return dynamic_risk_bound(I_0, I_0, G)


def solve_challenge5(
    gid: int,
    m: int = M_FIRES,
    tries: int = TRIES_MC,
    top_candidates: int = 25,
) -> tuple[tuple[int, ...], float, tuple[int, int] | None]:
    """Resuelve el Reto 5 para un grafo mediante filtrado analitico y Monte Carlo."""
    N, G, _, _, seed, graph_name = quickstart(graph=gid)

    all_combinations = list(combinations(range(N), m))

    # Primero hace un filtro de cual zona provoca mayor destruccion
    scored_candidates: list[tuple[float, tuple[int, ...]]] = []
    for comb in all_combinations:
        score = multi_fire_risk_bound(comb, G, N)
        scored_candidates.append((score, comb))

    # Ordenar de mayor a menor daño proyectado
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates_to_eval = [comb for _, comb in scored_candidates[:top_candidates]]

    # Fase 2: Evaluación precisa Monte Carlo con intervención defensiva
    best_comb = candidates_to_eval[0]
    max_damage = -1.0

    for idx, comb in enumerate(candidates_to_eval):
        # Números aleatorios comunes para consistencia estadística
        rng = np.random.default_rng(seed + idx)
        expected_damage = monte_carlo(
            g=G,
            initial_node=comb,
            endtime=None,
            tries=tries,
            generator=rng,
            with_firefighters=True,
        )

        if expected_damage > max_damage:
            max_damage = expected_damage
            best_comb = comb

    # Primer cortafuegos que colocarán los bomberos en t=0 en respuesta a best_comb
    I_0 = np.zeros(N, dtype=np.int64)
    Q_0 = np.zeros(N, dtype=np.int64)
    I_0[list(best_comb)] = 1
    Q_0[list(best_comb)] = 1
    first_defense = select_best_edge_dynamic(I_0, Q_0, G)

    return best_comb, max_damage, first_defense


def solve_challenge5_greedy(
    gid: int,
    m: int = M_FIRES,
    tries: int = TRIES_MC,
) -> tuple[tuple[int, ...], float, tuple[int, int] | None]:
    r"""Resuelve el Reto 5 mediante selección voraz secuencial."""
    N, G, _, _, seed, _ = quickstart(graph=gid)
    selected_nodes: list[int] = []

    for step_idx in range(m):
        best_v = None
        max_v_damage = -1.0

        candidates = [v for v in range(N) if v not in selected_nodes]
        for v in candidates:
            cand = tuple(selected_nodes + [v])
            rng = np.random.default_rng(seed + step_idx * N + v)
            damage = monte_carlo(
                g=G,
                initial_node=cand,
                endtime=None,
                tries=tries,
                generator=rng,
                with_firefighters=True,
            )
            if damage > max_v_damage:
                max_v_damage = damage
                best_v = v

        if best_v is not None:
            selected_nodes.append(best_v)

    best_comb = tuple(selected_nodes)

    rng = np.random.default_rng(seed)
    final_damage = monte_carlo(
        g=G,
        initial_node=best_comb,
        endtime=None,
        tries=tries,
        generator=rng,
        with_firefighters=True,
    )

    I_0 = np.zeros(N, dtype=np.int64)
    Q_0 = np.zeros(N, dtype=np.int64)
    I_0[list(best_comb)] = 1
    Q_0[list(best_comb)] = 1
    first_defense = select_best_edge_dynamic(I_0, Q_0, G)

    return best_comb, final_damage, first_defense


if __name__ == "__main__":
    for gid in EJEMPLOS_GRAFOS:
        N, _, _, _, _, name = quickstart(graph=gid)
        best_nodes, expected_q, first_cut = solve_challenge5(gid, m=M_FIRES)

        print(f"Grafo: {name} (N={N})")
        print(f"  Focos óptimos del pirómano: {best_nodes}")
        print(f"  Daño medio final (E[Q]): {expected_q:.4f} zonas quemadas")
        print(f"  Primer cortafuegos defensivo: {first_cut}")
        print("-" * 70)

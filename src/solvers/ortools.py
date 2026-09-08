import numpy as np
from ortools.sat.python import cp_model


def solve_wflo_ortools(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    time_limit_s=300,
    num_workers=8,
    coefficient_scale=1000,
    min_wake_loss=0.0,
):
    """
    Risolve il WFLO con OR-Tools usando:

        min sum_{i<j} L_ij z_i z_j

    subject to:

        sum_i z_i = n_turbines

        z_i + z_j <= 1
        per ogni coppia che viola la distanza minima.
    """

    L = np.asarray(
        wake_loss_matrix,
        dtype=float,
    )

    n = L.shape[0]

    model = cp_model.CpModel()

    z = [
        model.NewBoolVar(f"z_{i}")
        for i in range(n)
    ]

    # ==========================================
    # Vincolo: esattamente 81 turbine
    # ==========================================

    model.Add(
        sum(z) == n_turbines
    )

    # ==========================================
    # Vincolo: distanza minima
    # ==========================================

    for i, j in invalid_pairs:

        model.Add(
            z[int(i)] + z[int(j)] <= 1
        )

    # ==========================================
    # Wake objective
    # ==========================================

    objective_terms = []

    for i in range(n):

        for j in range(i + 1, n):

            loss = L[i, j]

            # Eventuale pruning delle interazioni
            # trascurabili.
            if loss <= min_wake_loss:
                continue

            coeff = int(
                round(
                    loss * coefficient_scale
                )
            )

            if coeff == 0:
                continue

            # y_ij = z_i AND z_j
            y_ij = model.NewBoolVar(
                f"y_{i}_{j}"
            )

            model.AddMultiplicationEquality(
                y_ij,
                [z[i], z[j]],
            )

            objective_terms.append(
                coeff * y_ij
            )

    model.Minimize(
        sum(objective_terms)
    )

    # ==========================================
    # Solve
    # ==========================================

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        time_limit_s
    )

    solver.parameters.num_search_workers = (
        num_workers
    )

    status = solver.Solve(model)

    print(
        "Status:",
        solver.StatusName(status),
    )

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        raise RuntimeError(
            "Nessuna soluzione trovata."
        )

    solution = np.array(
        [
            solver.Value(var)
            for var in z
        ],
        dtype=int,
    )

    wake_objective = (
        solver.ObjectiveValue()
        / coefficient_scale
    )

    print(
        f"Turbine selezionate: "
        f"{solution.sum()}"
    )

    print(
        f"Pairwise wake loss: "
        f"{wake_objective}"
    )

    return solution, solver
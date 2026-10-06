import datetime
import numpy as np
from ortools.sat.python import cp_model
from ortools.math_opt.python import mathopt

def solve_wflo_cpsat(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    time_limit_s=300,
    num_workers=8,
    coefficient_scale=1000,
    min_wake_loss=0.0,
    warmstart=None,
):
    """
    Risolve il WFLO con OR-Tools usando:

        min sum_{i<j} L_ij z_i z_j

    subject to:

        sum_i z_i = n_turbines

        z_i + z_j <= 1
        per ogni coppia che viola la distanza minima.

    Parameters
    ----------
    wake_loss_matrix : array (n, n)
        Matrice simmetrica dei wake loss L_ij tra le coppie di
        candidate locations.

    invalid_pairs : iterable di (i, j)
        Coppie che violano la distanza minima; generano il vincolo
        z_i + z_j <= 1.

    n_turbines : int
        Numero esatto di turbine da posizionare.

    time_limit_s : int
        Tempo massimo di ricerca in secondi.

    num_workers : int, default 8
        Numero di worker CP-SAT.

    coefficient_scale : int, default 1000
        Fattore float -> intero per i coefficienti dell'obiettivo.

    min_wake_loss : float, default 0.0
        Interazioni con loss <= min_wake_loss vengono ignorate.

    warmstart : array-like (n,), optional
        Vettore binario iniziale da fornire a CP-SAT come solution
        hint. Deve avere la stessa lunghezza del numero di candidate
        locations.

        Esempio:
            [0, 1, 0, ..., 1]

        Il warm start NON viene imposto come vincolo: CP-SAT puo'
        allontanarsi liberamente dalla soluzione iniziale.

    Returns
    -------
    solution : np.ndarray (n,)
        Vettore binario delle turbine selezionate.

    solver : cp_model.CpSolver
        Istanza del solver dopo la risoluzione.
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
    # Warm start / solution hint
    # ==========================================

    if warmstart is not None:

        warmstart = np.asarray(
            warmstart,
            dtype=int,
        )

        if warmstart.shape != (n,):
            raise ValueError(
                f"warmstart deve avere shape ({n},), "
                f"ricevuto {warmstart.shape}."
            )

        if not np.all(
            np.isin(
                warmstart,
                [0, 1],
            )
        ):
            raise ValueError(
                "warmstart deve essere un vettore binario."
            )

        for i in range(n):
            model.AddHint(
                z[i],
                int(warmstart[i]),
            )

    # ==========================================
    # Vincolo: esattamente n_turbines
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


def solve_wflo_gscip(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    time_limit_s=300,
    min_wake_loss=0.0,
):
    """
    Solve WFLO using GSCIP through OR-Tools MathOpt.

    Objective:
        min sum_{i<j} L_ij * z_i * z_j

    Constraints:
        sum_i z_i = n_turbines
        z_i + z_j <= 1   for every invalid spacing pair

    Unlike the pywraplp implementation, quadratic terms are passed
    directly to GSCIP: no auxiliary y_ij variables and no McCormick
    linearisation are introduced.
    """

    L = np.asarray(
        wake_loss_matrix,
        dtype=float,
    )

    n = L.shape[0]

    if L.shape != (n, n):
        raise ValueError(
            "wake_loss_matrix must be square."
        )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = mathopt.Model(
        name="wflo_gscip"
    )

    # Binary turbine-selection variables
    z = [
        model.add_binary_variable(
            name=f"z_{i}"
        )
        for i in range(n)
    ]

    # --------------------------------------------------
    # Cardinality constraint
    #
    # sum_i z_i = K
    # --------------------------------------------------

    model.add_linear_constraint(
        sum(z) == n_turbines,
        name="cardinality",
    )

    # --------------------------------------------------
    # Minimum-spacing constraints
    #
    # z_i + z_j <= 1
    # --------------------------------------------------

    for i, j in invalid_pairs:
        i = int(i)
        j = int(j)

        model.add_linear_constraint(
            z[i] + z[j] <= 1,
            name=f"spacing_{i}_{j}",
        )

    # --------------------------------------------------
    # Quadratic wake-loss objective
    #
    # min sum_{i<j} L_ij z_i z_j
    #
    # IMPORTANT:
    # We only use the upper triangle because L is symmetric.
    # --------------------------------------------------

    objective = 0.0
    n_quadratic_terms = 0

    for i in range(n):
        for j in range(i + 1, n):

            loss = float(L[i, j])

            if loss <= min_wake_loss:
                continue

            objective += (
                loss
                * z[i]
                * z[j]
            )

            n_quadratic_terms += 1

    model.minimize_quadratic_objective(
        objective
    )

    # --------------------------------------------------
    # Solver parameters
    # --------------------------------------------------

    params = mathopt.SolveParameters(
        time_limit=datetime.timedelta(
            seconds=time_limit_s
        ),
        enable_output=True,
    )

    # --------------------------------------------------
    # Solve with GSCIP
    # --------------------------------------------------

    result = mathopt.solve(
        model,
        mathopt.SolverType.GSCIP,
        params=params,
    )

    # --------------------------------------------------
    # Check termination
    # --------------------------------------------------

    if result.termination.reason not in (
        mathopt.TerminationReason.OPTIMAL,
        mathopt.TerminationReason.FEASIBLE,
    ):
        raise RuntimeError(
            "GSCIP: no feasible solution. "
            f"Termination: {result.termination}"
        )

    # --------------------------------------------------
    # Extract solution
    # --------------------------------------------------

    values = result.variable_values()

    solution = np.array(
        [
            round(values[z[i]])
            for i in range(n)
        ],
        dtype=int,
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print(
        "Termination:",
        result.termination.reason,
    )

    print(
        f"Turbine selezionate: "
        f"{solution.sum()}"
    )

    print(
        f"Quadratic wake terms: "
        f"{n_quadratic_terms:,}"
    )

    print(
        f"Pairwise wake loss: "
        f"{result.objective_value():,.6f}"
    )

    return solution, model
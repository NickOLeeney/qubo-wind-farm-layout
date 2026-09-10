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

    Parameters
    ----------
    wake_loss_matrix : array (n, n)
        Matrice simmetrica dei wake loss L_ij tra le coppie di
        candidate locations.
    invalid_pairs : iterable di (i, j)
        Coppie che violano la distanza minima; generano il vincolo
        z_i + z_j <= 1.
    n_turbines : int
        Numero esatto di turbine da posizionare (vincolo di
        cardinalita' sum_i z_i == n_turbines).
    time_limit_s : int
        Tempo massimo di ricerca in secondi (max_time_in_seconds).

    num_workers : int, default 8
        Numero di thread di ricerca paralleli
        (num_search_workers). CP-SAT e' un solver "portfolio":
        lancia piu' strategie diverse in parallelo e condivide le
        soluzioni migliori tra i thread. Piu' workers = spesso
        soluzioni buone piu' in fretta, ma NON e' uno speedup
        lineare. Attenzione alla RAM: ogni worker tiene una copia
        delle strutture di ricerca, quindi su modelli molto grandi
        ridurre num_workers abbassa il consumo di memoria.

    coefficient_scale : int, default 1000
        Fattore di conversione float -> intero. CP-SAT lavora solo
        con interi, mentre i wake loss L_ij sono float: ogni loss
        viene moltiplicato per coefficient_scale e arrotondato
        (fixed-point). Con 1000 si mantengono ~3 cifre decimali.
        Trade-off: scale troppo basso -> perdita di precisione e
        coppie arrotondate a coeff == 0 (scartate); scale troppo
        alto -> coefficienti interi enormi che possono rallentare
        presolve e propagazione. L'obiettivo finale viene diviso
        di nuovo per questo fattore per tornare in scala fisica.

    min_wake_loss : float, default 0.0
        Soglia di pruning: le coppie con loss <= min_wake_loss
        vengono saltate (niente variabile y_ij, niente vincolo,
        niente termine nell'obiettivo). Con 0.0 non c'e' quasi
        pruning e tutte le ~n(n-1)/2 coppie generano una variabile
        ausiliaria: su griglie fitte (es. 100 m, n ~ 3600) questo
        crea milioni di variabili e puo' saturare la RAM. Alzare
        la soglia elimina le interazioni fisicamente trascurabili
        (turbine molto lontane) riducendo drasticamente la
        dimensione del modello, al prezzo di un'approssimazione.

    Notes
    -----
    Il termine quadratico z_i * z_j non e' nativo in CP-SAT: ogni
    prodotto viene linearizzato con una variabile ausiliaria
    y_ij = z_i AND z_j (AddMultiplicationEquality) piu' i relativi
    vincoli. Il numero di y_ij cresce col quadrato di n, quindi e'
    la voce dominante nel consumo di memoria del modello.

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
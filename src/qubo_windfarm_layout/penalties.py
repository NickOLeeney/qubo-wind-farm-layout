import sys
import scs
import numpy as np
import cvxpy as cp
from scipy.spatial import cKDTree

sys.path.append("..")
from src.qubo_windfarm_layout.model import compute_aep_from_coords, get_farm_area, get_mask
from src.qubo_windfarm_layout.evaluation import load_layout_coordinates

 
def _compute_reference_wake_matrix(
    layout_path,
):
    """
    Costruisce la pairwise wake-loss matrix L_ref
    per un qualsiasi layout YAML compatibile con
    il formato benchmark.

    Può essere:
      - base.yaml
      - ortools_200m.yaml
      - qualsiasi altro layout salvato nello stesso formato
    """

    locations = load_layout_coordinates(
        layout_path
    )

    locations = np.asarray(
        locations,
        dtype=float,
    )

    n = len(locations)

    # AEP singola turbina
    A0 = compute_aep_from_coords(
        [locations[0]]
    )

    L_ref = np.zeros(
        (n, n),
        dtype=float,
    )

    for i in range(n):

        for j in range(i + 1, n):

            aep_pair = compute_aep_from_coords([
                locations[i],
                locations[j],
            ])

            loss = max(
                0.0,
                2 * A0 - aep_pair,
            )

            L_ref[i, j] = loss
            L_ref[j, i] = loss

    return locations, L_ref, A0


def suggest_cardinality_penalty_from_layout(
    layout_path,
    safety_factor=1.5,
):
    """
    Stima lambda_cardinality da un qualsiasi layout
    YAML di riferimento.

    lambda_N =
        safety_factor *
        max_i sum_{j != i} L_ij
    """

    locations, L_ref, A0 = (
        _compute_reference_wake_matrix(
            layout_path
        )
    )

    marginal_losses = L_ref.sum(
        axis=1
    )

    max_marginal_loss = float(
        marginal_losses.max()
    )

    lambda_cardinality = (
        safety_factor
        * max_marginal_loss
    )

    print(
        f"Reference turbines: {len(locations)}"
    )

    print(
        f"Max marginal wake loss: "
        f"{max_marginal_loss:,.3f}"
    )

    print(
        f"Safety factor: "
        f"{safety_factor}"
    )

    print(
        f"Suggested lambda cardinality: "
        f"{lambda_cardinality:,.3f}"
    )

    return lambda_cardinality


def suggest_spacing_penalty_from_layout(
    layout_path,
    safety_factor=1.2,
):
    """
    Stima lambda_spacing da un qualsiasi layout
    YAML di riferimento.

    lambda_S =
        safety_factor *
        sum_{i<j} L_ij
    """

    locations, L_ref, A0 = (
        _compute_reference_wake_matrix(
            layout_path
        )
    )

    # L_ref è simmetrica:
    # prendiamo solo la parte triangolare superiore.
    reference_wake_objective = float(
        np.triu(
            L_ref,
            k=1,
        ).sum()
    )

    lambda_spacing = (
        safety_factor
        * reference_wake_objective
    )

    print(
        f"Reference turbines: {len(locations)}"
    )

    print(
        f"Reference pairwise wake objective: "
        f"{reference_wake_objective:,.3f}"
    )

    print(
        f"Safety factor: "
        f"{safety_factor}"
    )

    print(
        f"Suggested lambda spacing: "
        f"{lambda_spacing:,.3f}"
    )

    return lambda_spacing


def layout_yaml_to_z(yaml_path, grid_resolution, tol=1.0):
    """
    Ricostruisce il vettore binario z dalla griglia candidati e un layout YAML.

    Parameters
    ----------
    yaml_path : str | Path
        Path al file YAML del layout salvato.
    grid_resolution : int
        Risoluzione della griglia in metri (deve coincidere con quella usata
        quando il layout è stato generato).
    tol : float
        Tolleranza in metri per il matching posizione → candidato.

    Returns
    -------
    z : np.ndarray, shape (n_candidates,), dtype int
        Vettore binario: z[i] = 1 se il candidato i è selezionato.
    candidate_locations : np.ndarray, shape (n_candidates, 2)
        Griglia completa dei candidati nello stesso ordine di z.
    """
    _, farm_area = get_farm_area()
    X, Y, mask = get_mask(farm_area=farm_area, grid_resolution=grid_resolution)
    candidate_locations = np.column_stack([X[mask], Y[mask]])

    selected = load_layout_coordinates(yaml_path)

    tree = cKDTree(candidate_locations)
    dists, indices = tree.query(selected, k=1)

    if np.any(dists > tol):
        bad = np.where(dists > tol)[0]
        raise ValueError(
            f"{len(bad)} posizioni del YAML non trovate nella griglia "
            f"(distanza max = {dists.max():.2f} m > tol={tol} m). "
            "Verifica che grid_resolution coincida con quella usata dal solver."
        )

    z = np.zeros(len(candidate_locations), dtype=int)
    z[indices] = 1
    return z, candidate_locations


def compute_lambda_from_lb(
    wake_loss_matrix,
    z_reference,
    lower_bound,
    delta=1e-6,
):
    L = np.asarray(wake_loss_matrix, dtype=float)
    x = np.asarray(z_reference, dtype=float)

    # H_wake = sum_{i<j} L_ij x_i x_j
    # L è simmetrica, quindi dividiamo per 2
    Q = L / 2

    upper_bound = float(x @ Q @ x)
    print(f"Upper Bound: {upper_bound}")
    lambda_ = upper_bound - lower_bound + delta

    return lambda_


def sdp_fixed_cardinality(
    wake_loss_matrix,
    n_turbines=80,
    max_iters=20_000,
    strong=True,
):
    L = np.asarray(wake_loss_matrix, dtype=float)
    n = L.shape[0]

    # Unica variabile lifted:
    #
    # Y = [ 1   x.T ]
    #     [ x    X  ]
    #
    Y = cp.Variable(
        (n + 1, n + 1),
        PSD=True,
    )

    x = Y[0, 1:]
    X = Y[1:, 1:]

    constraints = [
        Y[0, 0] == 1,

        # x_i^2 = x_i nella formulazione lifted
        cp.diag(X) == x,

        # cardinalità
        cp.sum(x) == n_turbines,

        # x >= 0 è già implicato dalla PSD
        x <= 1,
    ]

    # Tightening opzionale.
    # Costa N^2 disuguaglianze.
    if strong:
        constraints.append(X >= 0)

    # L è simmetrica:
    #
    # 0.5 * sum_ij L_ij X_ij
    # = sum_{i<j} L_ij X_ij
    objective = cp.Minimize(
        0.5 * cp.sum(cp.multiply(L, X))
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    problem.solve(
        solver=cp.SCS,
        linear_solver=scs.LinearSolver.CPU_INDIRECT,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iters=max_iters,
        verbose=True,
    )

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"SDP failed: {problem.status}"
        )

    return float(problem.value)


def sdp_spacing_violation(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    max_iters=20_000,
):
    L = np.asarray(wake_loss_matrix, dtype=float)
    n = L.shape[0]

    scale = np.max(np.abs(L))
    L_scaled = L / scale

    Y = cp.Variable(
        (n + 1, n + 1),
        PSD=True,
    )

    x = Y[0, 1:]
    X = Y[1:, 1:]

    invalid_pairs = np.asarray(
        invalid_pairs,
        dtype=int,
    )

    ii = invalid_pairs[:, 0]
    jj = invalid_pairs[:, 1]

    constraints = [
        Y[0, 0] == 1,
        cp.diag(X) == x,
        cp.sum(x) == n_turbines,
        x <= 1,
        X >= 0,

        cp.sum(X[ii, jj]) >= 1,
    ]

    problem = cp.Problem(
        cp.Minimize(
            0.5 * cp.sum(
                cp.multiply(L_scaled, X)
            )
        ),
        constraints,
    )

    problem.solve(
        solver=cp.SCS,
        linear_solver=scs.LinearSolver.CPU_INDIRECT,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iters=max_iters,
        verbose=True,
    )

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"SDP failed: {problem.status}"
        )

    return float(problem.value * scale)


def get_penalties(grid_resolution, max_iters, penalty_type, save_memory):

    # Setup
    REFERENCE_LAYOUT = f"../results/layouts/cpsat_200_3600s.yaml"
    REFERENCE_WAKE_MATRIX = f"../results/precomputed/wake_loss_200m.npz"
    MIN_DISTANCE = 396  # 2 * 198 m

    _, farm_area = get_farm_area()
    X_grid, Y_grid, mask = get_mask(farm_area=farm_area, grid_resolution=grid_resolution)
    candidate_locations = np.column_stack([X_grid[mask], Y_grid[mask]])

    invalid_pairs = get_invalid_pairs(
    candidate_locations=candidate_locations,
    min_distance=MIN_DISTANCE,
    )
    print(f"Candidati totali : {len(candidate_locations)}")
    print(f"Coppie non valide: {len(invalid_pairs)}")

    # Step 1: retrieve upper bound feasible solution
    z_reference_up, candidate_locations = layout_yaml_to_z(
    yaml_path=REFERENCE_LAYOUT, grid_resolution=grid_resolution
    )

    print(f"Candidati totali   : {len(candidate_locations)}")
    print(f"Turbine selezionate: {z_reference_up.sum()}")

    # Step 2: compute lower bound using a relaxed solution
    wake_loss = load_wake_loss_data(f"../results/precomputed/wake_loss_{grid_resolution}m.npz")
    wake_loss_matrix = wake_loss["wake_loss_matrix"]

    if penalty_type == "cardinality":
        LB = sdp_fixed_cardinality(
        wake_loss_matrix,
        n_turbines=80,
        max_iters=max_iters
        )
    elif penalty_type == "spacing":
        LB = sdp_spacing_violation(
        wake_loss_matrix,
        invalid_pairs,
        n_turbines=81,
        max_iters=max_iters
        )

    print(f"LB {penalty_type} = {LB}")

    # Step 3: compute lambda
    wake_loss_up = load_wake_loss_data(REFERENCE_WAKE_MATRIX)
    wake_loss_matrix_up = wake_loss_up["wake_loss_matrix"]

    _lambda = compute_lambda_from_lb(wake_loss_matrix=wake_loss_matrix_up, z_reference=z_reference_up, lower_bound=LB)

    print(f"Lambda {penalty_type} = {_lambda}")

    return _lambda

if __name__ == "__main__":
    import argparse
    from src.solvers.utils import get_invalid_pairs
    from src.qubo_windfarm_layout.evaluation import load_layout_coordinates
    from src.qubo_windfarm_layout.model import get_farm_area, get_mask, load_wake_loss_data

    parser = argparse.ArgumentParser(description="Compute QUBO penalty coefficients.")
    parser.add_argument(
        "--max-iters",
        type=int,
        required=True,
        help="Maximum SDP solver iterations (default: 1000).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        required=True,
        help="Grid resolution in metres (default: 300).",
    )
    parser.add_argument(
            "--penalty-type",
            type=str,
            choices=["cardinality", "spacing"],
            required=True,
            help="Penalty type to compute",
        )

    parser.add_argument(
                "--save-memory",
                type=bool,
                choices=[True, False],
                required=True,
                default=True,
                help="Decide matrix quantization",
            )
    
    args = parser.parse_args()

    get_penalties(grid_resolution=args.grid_resolution, max_iters=args.max_iters, penalty_type=args.penalty_type, save_memory=args.save_memory)
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


def layout_yaml_to_z(yaml_path, grid_resolution=200, tol=1.0):
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

    lambda_ = upper_bound - lower_bound + delta

    return lambda_


def sdp_fixed_cardinality(
    wake_loss_matrix,
    n_turbines=80,
    max_iters= 20_000
):
    L = np.asarray(wake_loss_matrix, dtype=float)
    Q = L / 2

    n = Q.shape[0]

    x = cp.Variable(n)
    X = cp.Variable((n, n), symmetric=True)

    Y = cp.bmat([
        [np.ones((1, 1)), cp.reshape(x, (1, n), order="C")],
        [cp.reshape(x, (n, 1), order="C"), X],
    ])

    constraints = [
        Y >> 0,
        cp.diag(X) == x,
        cp.sum(x) == n_turbines,
        x >= 0,
        x <= 1,
        X >= 0,
        X <= 1,
    ]

    problem = cp.Problem(
        cp.Minimize(cp.trace(Q @ X)),
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
    max_iters= 20_000
):
    L = np.asarray(wake_loss_matrix, dtype=float)
    Q = L / 2

    n = Q.shape[0]

    x = cp.Variable(n)
    X = cp.Variable((n, n), symmetric=True)

    Y = cp.bmat([
        [np.ones((1, 1)), cp.reshape(x, (1, n), order="C")],
        [cp.reshape(x, (n, 1), order="C"), X],
    ])

    # Almeno una coppia invalida selezionata
    spacing_violations = cp.sum([
        X[int(i), int(j)]
        for i, j in invalid_pairs
    ])

    constraints = [
        Y >> 0,
        cp.diag(X) == x,

        # Numero corretto di turbine
        cp.sum(x) == n_turbines,

        # Forza almeno una spacing violation
        spacing_violations >= 1,

        x >= 0,
        x <= 1,
        X >= 0,
        X <= 1,
    ]

    problem = cp.Problem(
        cp.Minimize(cp.trace(Q @ X)),
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


def get_penalties(grid_resolution, max_iters):

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
    yaml_path=REFERENCE_LAYOUT
    )

    print(f"Candidati totali   : {len(candidate_locations)}")
    print(f"Turbine selezionate: {z_reference_up.sum()}")

    # Step 2: compute lower bound using a relaxed solution
    wake_loss = load_wake_loss_data(f"../results/precomputed/wake_loss_{grid_resolution}m.npz")
    wake_loss_matrix = wake_loss["wake_loss_matrix"]

    LB_cardinality = sdp_fixed_cardinality(
    wake_loss_matrix,
    n_turbines=80,
    max_iters=max_iters
    )

    LB_spacing = sdp_spacing_violation(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    max_iters=max_iters
    )

    # Step 3: compute lambda
    z_reference = layout_yaml_to_z(yaml_path=REFERENCE_LAYOUT)[0]
    wake_loss_up = load_wake_loss_data(REFERENCE_WAKE_MATRIX)
    wake_loss_matrix_up = wake_loss_up["wake_loss_matrix"]

    lambda_cardinality = compute_lambda_from_lb(wake_loss_matrix=wake_loss_matrix_up, z_reference=z_reference_up, lower_bound=LB_cardinality)
    lambda_spacing = compute_lambda_from_lb(wake_loss_matrix=wake_loss_matrix_up, z_reference=z_reference_up, lower_bound=LB_spacing)
    print(f"Lambda cardinality = {lambda_cardinality}")
    print(f"Lambda spacing = {lambda_spacing}")

    return lambda_cardinality, lambda_spacing

if __name__ == "__main__":
    from src.solvers.utils import get_invalid_pairs
    from src.qubo_windfarm_layout.evaluation import load_layout_coordinates
    from src.qubo_windfarm_layout.model import get_farm_area, get_mask, load_wake_loss_data
    
    MAX_ITERS = 500
    GRID_RESOLUTION = 300
    
    get_penalties(grid_resolution=GRID_RESOLUTION, max_iters=MAX_ITERS)
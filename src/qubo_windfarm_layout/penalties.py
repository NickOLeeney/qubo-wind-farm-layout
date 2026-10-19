import gc
import sys
import scs
import numpy as np
import cvxpy as cp

from scipy.spatial import cKDTree
from shapely.geometry import Point
sys.path.append("..")



from src.qubo_windfarm_layout.model import (
    compute_aep_from_coords,
    get_farm_area,
    get_mask,
    load_wake_loss_data,
    build_local_wake_loss_matrix_optimized,

)

from src.qubo_windfarm_layout.evaluation import (
    load_layout_coordinates,
)

from src.solvers.utils import (
    get_invalid_pairs,
)

from src.common.utils import (
    safe_cast_uint16,
    print_ram,
)


# ============================================================
# Reference-layout methods
# ============================================================

def _compute_reference_wake_matrix(
    layout_path,
):
    """
    Costruisce la pairwise wake-loss matrix L_ref
    per un qualsiasi layout YAML compatibile con
    il formato benchmark.
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
    lambda_N =
        safety_factor *
        max_i sum_{j != i} L_ij
    """

    locations, L_ref, _ = (
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
    lambda_S =
        safety_factor *
        sum_{i<j} L_ij
    """

    locations, L_ref, _ = (
        _compute_reference_wake_matrix(
            layout_path
        )
    )

    # Matrice piccola (81x81), quindi qui np.triu
    # non è un problema di memoria.
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


# ============================================================
# Layout -> binary vector
# ============================================================

def layout_yaml_to_z(
    yaml_path,
    candidate_locations,
    tol=1.0,
):
    """
    Ricostruisce il vettore binario z dato un layout YAML
    e la griglia candidati già calcolata.

    Questo evita di ricostruire farm_area, X, Y e mask
    una seconda volta.
    """

    selected = np.asarray(
        load_layout_coordinates(yaml_path),
        dtype=float,
    )

    tree = cKDTree(
        candidate_locations
    )

    dists, indices = tree.query(
        selected,
        k=1,
    )

    if np.any(dists > tol):

        bad = np.where(
            dists > tol
        )[0]

        raise ValueError(
            f"{len(bad)} posizioni del YAML non trovate "
            f"nella griglia "
            f"(distanza max = {dists.max():.2f} m "
            f"> tol={tol} m). "
            "Il reference layout deve essere compatibile "
            "con la grid resolution corrente."
        )

    # uint8 è più che sufficiente per un vettore binario
    z = np.zeros(
        len(candidate_locations),
        dtype=np.uint8,
    )

    z[indices] = 1

    return z


# ============================================================
# Upper bound
# ============================================================

def compute_upper_bound(
    wake_loss_matrix,
    z_reference,
):
    """
    H_wake = 0.5 * z^T L z

    Per risparmiare memoria viene estratta solamente
    la sott matrice relativa alle turbine selezionate.
    """

    L = np.asarray(
        wake_loss_matrix
    )

    selected = np.flatnonzero(
        z_reference
    )

    L_selected = L[
        np.ix_(
            selected,
            selected,
        )
    ]

    upper_bound = float(
        0.5
        * L_selected.sum(
            dtype=np.float64
        )
    )

    return upper_bound


def compute_lambda_from_lb(
    wake_loss_matrix,
    z_reference,
    lower_bound,
    delta=1e-6,
):
    """
    Mantiene la funzione per compatibilità con
    eventuale altro codice.
    """

    upper_bound = compute_upper_bound(
        wake_loss_matrix,
        z_reference,
    )

    print(
        f"Upper Bound: {upper_bound}"
    )

    lambda_ = (
        upper_bound
        - lower_bound
        + delta
    )

    return lambda_


# ============================================================
# SDP - Cardinality violation
# ============================================================

def sdp_fixed_cardinality(
    wake_loss_matrix,
    n_turbines=80,
    max_iters=20_000,
    strong=True,
):
    """
    SDP relaxation con cardinalità fissata.

    strong=True:
        aggiunge X >= 0.
        Relaxation più forte ma maggiore consumo RAM.

    strong=False:
        relaxation più leggera e scalabile.
    """

    L = np.asarray(
        wake_loss_matrix
    )

    n = L.shape[0]

    if L.ndim != 2 or L.shape[1] != n:
        raise ValueError(
            "wake_loss_matrix deve essere quadrata."
        )

    scale = float(
        L.max()
    )

    if scale <= 0:
        raise ValueError(
            "Wake-loss matrix non valida: max <= 0."
        )

    # --------------------------------------------------
    # Lifted PSD variable
    #
    # Y = [1   x.T]
    #     [x    X ]
    # --------------------------------------------------

    Y = cp.Variable(
        (n + 1, n + 1),
        PSD=True,
    )

    x = Y[
        0,
        1:
    ]

    X = Y[
        1:,
        1:
    ]

    constraints = [
        Y[0, 0] == 1,

        # binary lifting
        cp.diag(X) == x,

        # fixed cardinality
        cp.sum(x) == n_turbines,

        # x >= 0 already implied by PSD + diag(X)=x
        x <= 1,
    ]

    # Optional tightening.
    # Costs O(N^2) scalar inequalities.
    if strong:
        constraints.append(
            X >= 0
        )

    # --------------------------------------------------
    # Objective
    #
    # Instead of:
    #
    # L_scaled = L / scale
    #
    # use:
    #
    # (1 / scale) * objective(L)
    #
    # to avoid allocating another NxN matrix.
    # --------------------------------------------------

    objective = cp.Minimize(
        (0.5 / scale)
        * cp.sum(
            cp.multiply(
                L,
                X,
            )
        )
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    gc.collect()

    print(
        "RAM before cardinality SDP:"
    )
    print_ram()

    problem.solve(
        solver=cp.SCS,
        linear_solver=scs.LinearSolver.CPU_INDIRECT,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iters=max_iters,
        verbose=True,
    )

    print(
        "RAM after cardinality SDP:"
    )
    print_ram()

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"SDP failed: {problem.status}"
        )

    # Return objective in original wake-loss scale
    return float(
        problem.value * scale
    )
# ============================================================
# SDP - Spacing violation
# ============================================================

def sdp_spacing_violation(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    max_iters=20_000,
    strong=True,
):
    """
    SDP relaxation restricted to configurations
    containing at least one spacing violation.

    strong=True:
        adds X >= 0.

    strong=False:
        lighter relaxation.
    """

    L = np.asarray(
        wake_loss_matrix
    )

    n = L.shape[0]

    if L.ndim != 2 or L.shape[1] != n:
        raise ValueError(
            "wake_loss_matrix deve essere quadrata."
        )

    scale = float(
        L.max()
    )

    if scale <= 0:
        raise ValueError(
            "Wake-loss matrix non valida: max <= 0."
        )

    invalid_pairs = np.asarray(
        invalid_pairs,
        dtype=np.int32,
    )

    if (
        invalid_pairs.ndim != 2
        or invalid_pairs.shape[1] != 2
    ):
        raise ValueError(
            "invalid_pairs deve avere shape (m, 2)."
        )

    if len(invalid_pairs) == 0:
        raise ValueError(
            "Nessuna invalid pair disponibile."
        )

    if (
        invalid_pairs.min() < 0
        or invalid_pairs.max() >= n
    ):
        raise ValueError(
            "invalid_pairs contiene indici fuori "
            "dal range della wake-loss matrix."
        )

    # --------------------------------------------------
    # Lifted variable
    # --------------------------------------------------

    Y = cp.Variable(
        (n + 1, n + 1),
        PSD=True,
    )

    x = Y[
        0,
        1:
    ]

    X = Y[
        1:,
        1:
    ]

    ii = invalid_pairs[:, 0]
    jj = invalid_pairs[:, 1]

    constraints = [
        Y[0, 0] == 1,

        cp.diag(X) == x,

        cp.sum(x) == n_turbines,

        x <= 1,

        # At least one spacing violation
        cp.sum(
            X[ii, jj]
        ) >= 1,
    ]

    if strong:
        constraints.append(
            X >= 0
        )

    objective = cp.Minimize(
        (0.5 / scale)
        * cp.sum(
            cp.multiply(
                L,
                X,
            )
        )
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    gc.collect()

    print(
        "RAM before spacing SDP:"
    )
    print_ram()

    problem.solve(
        solver=cp.SCS,
        linear_solver=scs.LinearSolver.CPU_INDIRECT,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iters=max_iters,
        verbose=True,
    )

    print(
        "RAM after spacing SDP:"
    )
    print_ram()

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"SDP failed: {problem.status}"
        )

    return float(
        problem.value * scale
    )
# ============================================================
# Penalty computation
# ============================================================

def get_penalties(
    grid_resolution,
    max_iters,
    penalty_type,
    reference_layout,
    save_memory=True,
    strong=True,
    delta=1e-6,
):
    """
    Compute a penalty coefficient using:

        lambda = UB - LB + delta

    IMPORTANT:
    UB and LB are both evaluated on the SAME
    candidate grid and SAME wake-loss matrix.
    """

    MIN_DISTANCE = 396

    # ========================================================
    # Step 1: candidate grid
    # ========================================================

    _, farm_area = get_farm_area()

    X_grid, Y_grid, mask = get_mask(
        farm_area=farm_area,
        grid_resolution=grid_resolution,
    )

    candidate_locations = np.column_stack([
        X_grid[mask],
        Y_grid[mask],
    ])

    n_candidates = len(
        candidate_locations
    )

    print(
        f"Candidati totali: {n_candidates}"
    )

    # ========================================================
    # Step 2: map feasible reference layout to current grid
    # ========================================================

    z_reference = layout_yaml_to_z(
        yaml_path=reference_layout,
        candidate_locations=candidate_locations,
    )

    print(
        f"Turbine selezionate: "
        f"{int(z_reference.sum())}"
    )

    # ========================================================
    # Step 3: spacing pairs
    #
    # Only needed for spacing relaxation.
    # ========================================================

    invalid_pairs = None

    if penalty_type == "spacing":

        invalid_pairs = np.asarray(
            get_invalid_pairs(
                candidate_locations=candidate_locations,
                min_distance=MIN_DISTANCE,
            ),
            dtype=np.int32,
        )

        print(
            f"Coppie non valide: "
            f"{len(invalid_pairs)}"
        )

    # ========================================================
    # Step 4: load ONE wake-loss matrix
    # ========================================================

    wake_loss = load_wake_loss_data(
        f"../results/precomputed/"
        f"wake_loss_{grid_resolution}m.npz"
    )

    wake_loss_matrix = wake_loss.pop(
        "wake_loss_matrix"
    )

    del wake_loss

    # Optional quantization.
    #
    # Important:
    # if enabled, BOTH UB and LB use the quantized matrix,
    # so they remain internally coherent.
    if save_memory:

        wake_loss_matrix = safe_cast_uint16(
            wake_loss_matrix
        )

    if wake_loss_matrix.shape != (
        n_candidates,
        n_candidates,
    ):
        raise ValueError(
            "Wake-loss matrix e candidate grid "
            "hanno dimensioni incompatibili: "
            f"L={wake_loss_matrix.shape}, "
            f"candidates={n_candidates}"
        )

    # ========================================================
    # Step 5: compute UB using SAME matrix as SDP
    # ========================================================

    UB = compute_upper_bound(
        wake_loss_matrix,
        z_reference,
    )

    print(
        f"Upper Bound: {UB}"
    )

    # Geometry no longer needed.
    # z_reference also no longer needed because UB is known.

    del (
        farm_area,
        X_grid,
        Y_grid,
        mask,
        candidate_locations,
        z_reference,
    )

    gc.collect()

    print(
        "RAM before SDP:"
    )
    print_ram()

    # ========================================================
    # Step 6: compute lower bound
    # ========================================================

    if penalty_type == "cardinality":

        LB = sdp_fixed_cardinality(
            wake_loss_matrix=wake_loss_matrix,
            n_turbines=80,
            max_iters=max_iters,
            strong=strong,
        )

    elif penalty_type == "spacing":

        LB = sdp_spacing_violation(
            wake_loss_matrix=wake_loss_matrix,
            invalid_pairs=invalid_pairs,
            n_turbines=81,
            max_iters=max_iters,
            strong=strong,
        )

    else:
        raise ValueError(
            f"Unknown penalty type: "
            f"{penalty_type}"
        )

    print(
        f"Lower Bound ({penalty_type}): "
        f"{LB}"
    )

    # ========================================================
    # Step 7: lambda
    # ========================================================

    lambda_raw = (
        UB
        - LB
        + delta
    )

    # A negative cardinality lambda is suspicious:
    # LB80 should not exceed a valid UB81 when L >= 0.
    if (
        penalty_type == "cardinality"
        and lambda_raw < 0
    ):
        raise ValueError(
            "Negative cardinality lambda: "
            f"UB={UB}, LB={LB}. "
            "Check SDP validity, grid consistency "
            "and objective scaling."
        )

    # For spacing, UB < LB can in principle mean that
    # the violating class is already worse than the
    # feasible reference even without a positive penalty.
    if (
        penalty_type == "spacing"
        and lambda_raw < 0
    ):
        print(
            "WARNING: spacing UB < LB. "
            "Setting lambda_spacing to 0."
        )

        lambda_ = 0.0

    else:
        lambda_ = float(
            lambda_raw
        )

    print(
        f"Lambda {penalty_type}: "
        f"{lambda_}"
    )

    # ========================================================
    # Cleanup
    # ========================================================

    del wake_loss_matrix

    if invalid_pairs is not None:
        del invalid_pairs

    gc.collect()

    print(
        "RAM after SDP cleanup:"
    )
    print_ram()

    return lambda_


def get_local_penalties(
    grid_resolution,
    max_iters,
    penalty_type,
    reference_layout,
    wake_loss_path,
    save_memory=True,
    strong=True,
    delta=1e-6,
    match_tol=1e-6,
):
    """
    Compute a Big-M penalty for a LOCAL WFLO problem:

        lambda = UB - LB + delta

    The local candidate set is generated around the turbines of a
    feasible reference layout.

    IMPORTANT:
    UB, LB and the final SQBM+ QUBO must use the SAME:
        - candidate locations
        - wake-loss matrix
        - objective scaling
    """

    MIN_DISTANCE = 396.0
    N_TURBINES = 81
    search_radius = 2*grid_resolution
    # ========================================================
    # Step 1: reference feasible layout
    # ========================================================

    reference_positions = np.asarray(
        load_layout_coordinates(reference_layout),
        dtype=float,
    )

    if len(reference_positions) != N_TURBINES:
        raise ValueError(
            f"Expected {N_TURBINES} turbines, "
            f"found {len(reference_positions)}."
        )

    _, farm_area = get_farm_area()

    # ========================================================
    # Step 2: local candidate grid
    # ========================================================

    n_steps = int(
        np.floor(
            search_radius / grid_resolution
        )
    )

    offsets_1d = (
        np.arange(
            -n_steps,
            n_steps + 1,
        )
        * grid_resolution
    )

    dx, dy = np.meshgrid(
        offsets_1d,
        offsets_1d,
    )

    local_offsets = np.column_stack(
        [
            dx.ravel(),
            dy.ravel(),
        ]
    )

    # Keep only points inside the circular neighborhood
    local_offsets = local_offsets[
        np.linalg.norm(
            local_offsets,
            axis=1,
        )
        <= search_radius + 1e-9
    ]

    # Generate one local grid around each current turbine
    candidate_blocks = [
        position + local_offsets
        for position in reference_positions
    ]

    candidate_locations = np.vstack(
        candidate_blocks
    )

    # Explicitly include incumbent positions.
    # This guarantees that the feasible reference layout belongs
    # to the local search space.
    candidate_locations = np.vstack(
        [
            candidate_locations,
            reference_positions,
        ]
    )

    # Keep only points inside/on the farm boundary
    inside_mask = np.array(
        [
            farm_area.covers(
                Point(x, y)
            )
            for x, y in candidate_locations
        ],
        dtype=bool,
    )

    candidate_locations = (
        candidate_locations[
            inside_mask
        ]
    )

    # Remove duplicates caused by overlapping neighborhoods
    candidate_locations = np.unique(
        np.round(
            candidate_locations,
            6,
        ),
        axis=0,
    )

    n_candidates = len(
        candidate_locations
    )

    print(
        f"Local candidates: {n_candidates}"
    )

    # ========================================================
    # Step 3: map reference layout -> local binary vector
    # ========================================================

    candidate_tree = cKDTree(
        candidate_locations
    )

    distances, selected_indices = (
        candidate_tree.query(
            reference_positions,
            k=1,
        )
    )

    if np.any(
        distances > match_tol
    ):
        raise ValueError(
            "Reference layout is not contained in the "
            "local candidate set. "
            f"Max distance = {distances.max():.6f} m."
        )

    if (
        len(np.unique(selected_indices))
        != N_TURBINES
    ):
        raise ValueError(
            "Multiple reference turbines mapped to "
            "the same local candidate."
        )

    z_reference = np.zeros(
        n_candidates,
        dtype=np.int8,
    )

    z_reference[
        selected_indices
    ] = 1

    print(
        f"Reference turbines: "
        f"{int(z_reference.sum())}"
    )

    # ========================================================
    # Step 4: spacing pairs
    # ========================================================

    invalid_pairs = None

    if penalty_type == "spacing":

        invalid_pairs = np.asarray(
            get_invalid_pairs(
                candidate_locations=
                    candidate_locations,
                min_distance=
                    MIN_DISTANCE,
            ),
            dtype=np.int32,
        )

        print(
            f"Invalid pairs: "
            f"{len(invalid_pairs)}"
        )

    # ========================================================
    # Step 5: LOCAL wake-loss matrix
    # ========================================================

    

    

    # wake_result = (
    #     build_wake_loss_matrix_optimized(
    #         candidate_locations,
    #         grid_resolution,
    #     )
    # )
    import os
    if not os.path.exists(wake_loss_path):
        wake_loss_matrix, A0 = build_local_wake_loss_matrix_optimized(
        candidate_locations,
        output_path=wake_loss_path)
        print("Building local wake-loss matrix...")
    else:
        print("Found existing wake-loss matrix")
        wake_loss_matrix = np.load(wake_loss_path)["wake_loss_matrix"]


    # Your current implementation returns
    # (wake_loss_matrix, A0)
    # wake_loss_matrix = wake_result[0]

    if wake_loss_matrix.shape != (
        n_candidates,
        n_candidates,
    ):
        raise ValueError(
            "Wake-loss matrix and local candidate "
            "set are incompatible: "
            f"L={wake_loss_matrix.shape}, "
            f"candidates={n_candidates}."
        )

    # ========================================================
    # Step 6: optional compression
    # ========================================================

    if save_memory:

        wake_loss_matrix = (
            safe_cast_uint16(
                wake_loss_matrix
            )
        )

    # ========================================================
    # Step 7: UB from feasible incumbent
    # ========================================================

    UB = compute_upper_bound(
        wake_loss_matrix,
        z_reference,
    )

    print(
        f"Upper Bound: {UB}"
    )

    print(
        "RAM before SDP:"
    )

    print_ram()

    # ========================================================
    # Step 8: SDP lower bound
    # ========================================================

    if penalty_type == "cardinality":

        # Lowest objective potentially reachable with one
        # turbine fewer than the required 81.
        LB = sdp_fixed_cardinality(
            wake_loss_matrix=
                wake_loss_matrix,
            n_turbines=
                N_TURBINES - 1,
            max_iters=
                max_iters,
            strong=
                strong,
        )

    elif penalty_type == "spacing":

        LB = sdp_spacing_violation(
            wake_loss_matrix=
                wake_loss_matrix,
            invalid_pairs=
                invalid_pairs,
            n_turbines=
                N_TURBINES,
            max_iters=
                max_iters,
            strong=
                strong,
        )

    else:

        raise ValueError(
            f"Unknown penalty type: "
            f"{penalty_type}"
        )

    print(
        f"Lower Bound ({penalty_type}): "
        f"{LB}"
    )

    # ========================================================
    # Step 9: lambda
    # ========================================================

    lambda_raw = (
        UB
        - LB
        + delta
    )

    if (
        penalty_type == "cardinality"
        and lambda_raw < 0
    ):

        raise ValueError(
            "Negative cardinality lambda: "
            f"UB={UB}, LB={LB}. "
            "Check SDP validity, local candidate "
            "consistency and objective scaling."
        )

    if (
        penalty_type == "spacing"
        and lambda_raw < 0
    ):

        print(
            "WARNING: spacing UB < LB. "
            "Setting lambda_spacing to 0."
        )

        lambda_ = 0.0

    else:

        lambda_ = float(
            lambda_raw
        )

    print(
        f"Lambda {penalty_type}: "
        f"{lambda_}"
    )

    print(f"lambda: {lambda_}, \n UB: {float(UB)}, \n LB: {float(LB)}")

    # ========================================================
    # Return local problem too
    # ========================================================

    return {
        "lambda": lambda_,
        "UB": float(UB),
        "LB": float(LB),

        "candidate_locations":
            candidate_locations,

        "wake_loss_matrix":
            wake_loss_matrix,

        "z_reference":
            z_reference,

        "selected_indices":
            selected_indices,

        "invalid_pairs":
            invalid_pairs,

        "grid_resolution":
            grid_resolution,

        "search_radius":
            search_radius,
    }
# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute QUBO penalty coefficients "
            "using SDP relaxation."
        )
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=10000,
        help="Maximum SCS iterations.",
    )

    parser.add_argument(
        "--grid-resolution",
        type=int,
        required=True,
        help="Grid resolution in metres.",
    )

    parser.add_argument(
        "--penalty-type",
        type=str,
        choices=[
            "cardinality",
            "spacing",
        ],
        required=True,
        help="Penalty type to compute.",
    )

    parser.add_argument(
        "--reference-layout",
        type=str,
        required=True,
        help=(
            "Feasible reference layout YAML generated "
            "on the SAME candidate grid."
        ),
    )

    parser.add_argument(
        "--save-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Quantize wake-loss matrix to uint16. "
            "Use --no-save-memory to keep original dtype."
        ),
    )

    parser.add_argument(
            "--wake-loss-path",
            type=str,
            default=None,
            help=(
                "Search radius for local SDP"
            ),
        )

    parser.add_argument(
        "--strong",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add X >= 0 tightening. "
            "Use --no-strong for lower RAM usage."
        ),
    )

    args = parser.parse_args()

    if args.wake_loss_path:
        get_local_penalties(
            grid_resolution=args.grid_resolution,
            max_iters=args.max_iters,
            penalty_type=args.penalty_type,
            reference_layout=args.reference_layout,
            wake_loss_path=args.wake_loss_path,
)
    else:
        get_penalties(
                    grid_resolution=args.grid_resolution,
                    max_iters=args.max_iters,
                    penalty_type=args.penalty_type,
                    reference_layout=args.reference_layout,
                    save_memory=args.save_memory,
                    strong=args.strong,
                )

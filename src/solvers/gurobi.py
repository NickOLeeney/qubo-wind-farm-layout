import os
import numpy as np
import gurobipy as gp
import scipy.sparse as sp

from gurobipy import GRB
from dotenv import load_dotenv

load_dotenv()

env = gp.Env(
    params={
        "WLSACCESSID": os.getenv("WLSACCESSID"),
        "WLSSECRET": os.getenv("WLSSECRET"),
        "LICENSEID": int(os.getenv("LICENSEID")),
    }
)


def solve_wflo_gurobi(
    wake_loss_matrix,
    invalid_pairs,
    n_turbines=81,
    warm_start=None,
    time_limit=None,
    mip_gap=1e-3,
    verbose=True,
):
    L = np.asarray(wake_loss_matrix, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("wake_loss_matrix deve essere quadrata.")

    n = L.shape[0]

    invalid_pairs = np.asarray(invalid_pairs, dtype=int)

    model = gp.Model("wflo", env=env)

    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.MIPGap = mip_gap
    model.Params.NonConvex = 2

    model.Params.Threads = 1
    model.Params.NodefileStart = 0.5
    model.Params.NodefileDir = "/tmp"
    model.Params.SoftMemLimit = 28

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------

    z = model.addMVar(
        shape=n,
        vtype=GRB.BINARY,
        name="z",
    )

    # --------------------------------------------------
    # Warm start
    # --------------------------------------------------

    if warm_start is not None:
        warm_start = np.asarray(
            warm_start,
            dtype=float,
        )

        if warm_start.shape != (n,):
            raise ValueError(
                f"warm_start deve avere shape ({n},), "
                f"ricevuta {warm_start.shape}."
            )

        if not np.all(
            np.isclose(warm_start, 0)
            | np.isclose(warm_start, 1)
        ):
            raise ValueError(
                "warm_start deve essere binario."
            )

        z.Start = warm_start

    # --------------------------------------------------
    # Quadratic objective
    #
    # H_wake = sum_{i<j} L_ij z_i z_j
    # --------------------------------------------------

    Q = sp.csr_matrix(
        np.triu(L, k=1)
    )

    model.setObjective(
        z @ Q @ z,
        GRB.MINIMIZE,
    )

    # --------------------------------------------------
    # Cardinality
    # --------------------------------------------------

    model.addConstr(
        z.sum() == n_turbines,
        name="cardinality",
    )

    # --------------------------------------------------
    # Spacing constraints
    #
    # z_i + z_j <= 1
    # --------------------------------------------------

    if len(invalid_pairs) > 0:

        m = len(invalid_pairs)

        rows = np.repeat(
            np.arange(m),
            2,
        )

        cols = invalid_pairs.reshape(-1)

        data = np.ones(
            2 * m,
            dtype=float,
        )

        A_spacing = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(m, n),
        )

        model.addMConstr(
            A_spacing,
            z,
            "<",
            np.ones(m),
            name="spacing",
        )

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(
            f"Gurobi non ha trovato soluzioni. "
            f"Status: {model.Status}"
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    z_solution = np.rint(
        z.X
    ).astype(int)

    selected_indices = np.flatnonzero(
        z_solution
    )

    return {
        "z": z_solution,
        "selected_indices": selected_indices,
        "objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
        "runtime": float(model.Runtime),
        "status": model.Status,
    }


if __name__ == "__main__":
    import sys
    import argparse

    sys.path.append("..")
    from src.qubo_windfarm_layout.model import load_wake_loss_data, get_farm_area, get_mask
    from src.solvers.utils import get_invalid_pairs, save_benchmark_layout

    MIN_DISTANCE = 396
    WARMSTART = None

    parser = argparse.ArgumentParser(description="Gurobi Solver for WFLO.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum solver runtime [s].",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        help="Grid resolution in metres.",
        required=True
    )

    args = parser.parse_args()

    # Get Wake Loss Matrix
    wake_loss = load_wake_loss_data(f"../results/precomputed/wake_loss_{args.grid_resolution}m.npz")
    wake_loss_matrix = wake_loss["wake_loss_matrix"]

    # Get candidates
    polygons, farm_area = get_farm_area()
    X, Y, mask = get_mask(farm_area=farm_area, grid_resolution=args.grid_resolution)

    candidate_locations = np.column_stack([
        X[mask],
        Y[mask],
    ])

    # Get invalid pairs
    invalid_pairs = get_invalid_pairs(candidate_locations=candidate_locations, min_distance=MIN_DISTANCE)

    result = solve_wflo_gurobi(
        wake_loss_matrix=wake_loss_matrix,
        invalid_pairs=invalid_pairs,
        n_turbines=81,
        warm_start=WARMSTART,
        time_limit=args.timeout,
        mip_gap=0.01,
        )

    print("Objective:", result["objective"])
    print("Runtime:", result["runtime"])
    print("MIP gap:", result["mip_gap"])
    print("Best bound:", result["best_bound"])
    print("N turbines:", result["z"].sum())

    selected_indices = result["selected_indices"]
    selected_locations_gurobi = candidate_locations[
        selected_indices
    ]

    save_benchmark_layout(
        selected_locations=selected_locations_gurobi,
        solver_name="gurobi",
        filename=f"gurobi_{args.grid_resolution}_{args.timeout}s_warmstart_{WARMSTART}.yaml",
        grid_resolution=args.grid_resolution,
        )
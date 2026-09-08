import sys
import argparse
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.qubo_windfarm_layout.model import get_farm_area, get_mask, build_qubo_from_wake_matrix_optimized, build_wake_loss_matrix_optimized, load_wake_loss_data
from src.solvers.utils import get_invalid_pairs, save_benchmark_layout
from src.solvers.ortools import solve_wflo_ortools

# Variables
GRID_RESOLUTION = 150 # m
MIN_DISTANCE = 396  # 2 * 198 m

def solve(grid_resolution: int = GRID_RESOLUTION, time_limit_s: int = 10):
    # Get candidates
    polygons, farm_area = get_farm_area()
    X, Y, mask = get_mask(farm_area=farm_area, grid_resolution=grid_resolution)

    candidate_locations = np.column_stack([
        X[mask],
        Y[mask],
    ])

    print(f"Filtered candidates shape {candidate_locations.shape}")

    # Invalid pairs constraint
    invalid_pairs = get_invalid_pairs(candidate_locations=candidate_locations, min_distance=MIN_DISTANCE)


    # Matrix definition
    build_wake_loss_matrix_optimized(
        candidate_locations,
        grid_resolution=GRID_RESOLUTION,
        output_path=(_PROJECT_ROOT / "results" / "precomputed" / f"wake_loss_{grid_resolution}m.npz")
    )

    wake_loss = load_wake_loss_data(_PROJECT_ROOT / "results" / "precomputed" / f"wake_loss_{grid_resolution}m.npz")
    wake_loss_matrix = wake_loss["wake_loss_matrix"]
    A0 = wake_loss["A0"]

    Q, offset = build_qubo_from_wake_matrix_optimized(
        wake_loss_matrix,
        invalid_pairs,
        A0,
        n_turbines=81,
    )

    z_solution, solver = solve_wflo_ortools(
        wake_loss_matrix=wake_loss_matrix,
        invalid_pairs=invalid_pairs,
        n_turbines=81,
        time_limit_s=time_limit_s,
        num_workers=8,
    )

    selected_locations = candidate_locations[
        z_solution.astype(bool)
    ]

    save_benchmark_layout(
        selected_locations=selected_locations,
        solver_name="ortools",
        filename=f"ortools_{grid_resolution}_{time_limit_s}s.yaml",
        grid_resolution=grid_resolution,
        time_limit_s=time_limit_s
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=GRID_RESOLUTION,
        help=f"Grid resolution in meters (default: {GRID_RESOLUTION})",
    )
    parser.add_argument(
            "--timeout",
            type=int,
            default=10,
            help=f"Runtime timeout in seconds.",
        )
    args = parser.parse_args()
    solve(grid_resolution=args.grid_resolution)

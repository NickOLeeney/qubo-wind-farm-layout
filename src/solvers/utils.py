import os
import sys
import yaml

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

sys.path.append("..")
from src.qubo_windfarm_layout.model import compute_aep_from_coords

_PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = _PROJECT_ROOT / "results" / "layouts"

def save_benchmark_layout(
    selected_locations,
    solver_name,
    filename=None,
    output_dir=None,
    grid_resolution=None,
    time_limit_s=None,
    candidate_locations=None,
    wake_loss_matrix=None,
    metrics=None,
    match_tol=1e-6,
):
    """
    Salva un layout nello stesso schema utilizzato dai file
    del benchmark Thomas et al.

    Calcola e salva in metadata:
        - h_qubo: pairwise QUBO objective
        - aep_gwh: full-physics AEP

    Il file risultante è compatibile con getTurbLocYAML()
    e calculate_aep() del provided_model.py.
    """

    selected_locations = np.asarray(
        selected_locations,
        dtype=float,
    )

    if selected_locations.ndim != 2 or selected_locations.shape[1] != 2:
        raise ValueError(
            "selected_locations deve avere shape (N, 2)"
        )

    # --------------------------------------------------
    # Full-physics AEP
    # compute_aep_from_coords restituisce MWh
    # --------------------------------------------------

    aep_gwh = float(
        compute_aep_from_coords(selected_locations) / 1000.0
    )

    # --------------------------------------------------
    # Pairwise QUBO objective
    # H = sum_{i<j} L_ij z_i z_j
    # --------------------------------------------------

    h_qubo = None

    if candidate_locations is not None and wake_loss_matrix is not None:

        candidate_locations = np.asarray(
            candidate_locations,
            dtype=float,
        )

        wake_loss_matrix = np.asarray(
            wake_loss_matrix,
            dtype=float,
        )

        tree = cKDTree(candidate_locations)

        distances, selected_indices = tree.query(
            selected_locations,
            k=1,
        )

        if np.any(distances > match_tol):
            raise ValueError(
                "Alcune selected_locations non sono presenti "
                "in candidate_locations. "
                f"Max distance = {distances.max():.6f} m"
            )

        if len(np.unique(selected_indices)) != len(selected_locations):
            raise ValueError(
                "Più turbine sono state mappate sulla stessa candidate."
            )

        L_selected = wake_loss_matrix[
            np.ix_(
                selected_indices,
                selected_indices,
            )
        ]

        h_qubo = float(
            np.triu(
                L_selected,
                k=1,
            ).sum()
        )

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------

    if output_dir:
        output_dir = Path(
            os.path.join(
                _PROJECT_ROOT,
                output_dir,
            )
        )
    else:
        output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if filename is None:
        filename = f"{solver_name}.yaml"

    output_path = output_dir / filename

    # --------------------------------------------------
    # YAML
    # --------------------------------------------------

    metadata = {
        "solver": solver_name,
        "grid_resolution": grid_resolution,
        "n_turbines": len(selected_locations),
        "time_limit_s": time_limit_s,
        "h_qubo": h_qubo,
        "aep_gwh": aep_gwh,
    }

    if metrics is not None:
        metadata["cached_metrics"] = metrics

    data = {
        "title": (
            f"IEA Wind Task 37 case study 4 - {solver_name}"
        ),
        "description": (
            f"Layout generated using {solver_name}"
        ),

        "metadata": metadata,

        "definitions": {

            "wind_plant": {
                "type": "object",
                "description": "wind plant design",
                "properties": {
                    "turbine": {
                        "type": "array",
                        "items": [
                            {
                                "$ref": "iea37-10mw.yaml"
                            }
                        ],
                    }
                },
            },

            "position": {
                "description": (
                    "Turbine positions in Cartesian coordinates"
                ),
                "units": "m",
                "items": selected_locations.tolist(),
            },

            "plant_energy": {
                "description": "energy production data",
                "properties": {

                    "wake_model": {
                        "description": (
                            "wake model used to calculate AEP"
                        ),
                        "items": [
                            {
                                "$ref": "iea37-aepcalc.py"
                            }
                        ],
                    },

                    "wind_resource": {
                        "description": (
                            "wind resource used to calculate AEP"
                        ),
                        "properties": {
                            "items": [
                                {
                                    "$ref": "iea37-windrose-cs4.yaml"
                                }
                            ]
                        },
                    },
                },
            },
        },
    }

    with open(output_path, "w") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
        )

    print(f"Layout saved to: {output_path}")
    print(f"H_QUBO: {h_qubo}")
    print(f"AEP: {aep_gwh:.6f} GWh")

    return output_path


def get_invalid_pairs(candidate_locations, min_distance):
    distances = cdist(
        candidate_locations,
        candidate_locations
    )

    invalid_pairs = np.argwhere(
        (distances < min_distance)
        & (distances > 0)
    )

    invalid_pairs = np.argwhere(
        np.triu(
            (distances < min_distance)
            & (distances > 0),
            k=1
        )
    )
    return invalid_pairs


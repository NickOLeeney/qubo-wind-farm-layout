import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

_PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = _PROJECT_ROOT / "results" / "layouts"

def save_benchmark_layout(
    selected_locations,
    solver_name,
    filename=None,
    grid_resolution=None,
    time_limit_s=None
):
    """
    Salva un layout nello stesso schema utilizzato dai file
    del benchmark Thomas et al.

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

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{solver_name}.yaml"

    output_path = output_dir / filename

    data = {
        "title": f"IEA Wind Task 37 case study 4 - {solver_name}",
        "description": (
            f"Layout generated using {solver_name}"
        ),

        # Metadata nostri.
        # Il parser originale li ignora.
        "metadata": {
            "solver": solver_name,
            "grid_resolution": grid_resolution,
            "n_turbines": len(selected_locations),
            "time_limit_s": time_limit_s
        },

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
                        "description": "wake model used to calculate AEP",
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

                    # Volutamente non inseriamo l'AEP:
                    # verrà ricalcolata nel notebook di evaluation.
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


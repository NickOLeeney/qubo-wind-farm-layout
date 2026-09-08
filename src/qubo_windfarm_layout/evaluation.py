import sys
import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from shapely.geometry import Point
from scipy.spatial.distance import pdist

from src.qubo_windfarm_layout.model import compute_aep_from_coords, get_farm_area

ROOT = (
    Path.cwd().parent
    if Path.cwd().name == "notebooks"
    else Path.cwd()
)

BENCHMARK_ROOT = (
    ROOT
    / "external"
    / "thomas-wflo-benchmark"
)

PHYSICS_DIR = (
    BENCHMARK_ROOT
    / "src"
    / "physics-models"
)

OUR_LAYOUTS_DIR = (
    ROOT
    / "results"
    / "layouts"
)

REFERENCE_RESULTS_FILE = (
    ROOT
    / "results"
    / "reference_benchmark"
    / "thomas2023_results.csv"
)


if str(PHYSICS_DIR) not in sys.path:
    sys.path.insert(0, str(PHYSICS_DIR))

from provided_model import getTurbLocYAML


def load_layout_coordinates(filepath):
    """
    Carica le coordinate da uno YAML compatibile
    con il formato del benchmark Thomas et al.
    """

    coords, _, _ = getTurbLocYAML(
        str(filepath)
    )

    return np.asarray(
        coords,
        dtype=float,
    )


def get_solver_name(filepath):
    """
    Recupera il nome del solver dai metadata dello YAML.
    Se non presente, usa il nome del file.
    """

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    return (
        data
        .get("metadata", {})
        .get("solver", filepath.stem)
    )


def evaluate_layout(
    coords,
    single_turbine_aep,
    farm_area,
    expected_turbines=81,
    min_spacing=396.0,
):
    """
    Valuta un layout con il physics model ufficiale
    e verifica i vincoli del benchmark.
    """

    coords = np.asarray(
        coords,
        dtype=float,
    )

    n_turbines = len(coords)

    # --------------------------------------------------------
    # Official AEP
    # --------------------------------------------------------

    aep_mwh = compute_aep_from_coords(
        coords
    )

    aep_gwh = aep_mwh / 1000.0

    # --------------------------------------------------------
    # Wake loss
    # --------------------------------------------------------

    ideal_aep_mwh = (
        n_turbines
        * single_turbine_aep
    )

    wake_loss_pct = (
        100.0
        * (
            1.0
            - aep_mwh / ideal_aep_mwh
        )
    )

    # --------------------------------------------------------
    # Minimum turbine distance
    # --------------------------------------------------------

    min_distance_m = (
        float(pdist(coords).min())
        if n_turbines > 1
        else np.inf
    )

    # valid_spacing = (
    #     min_distance_m
    #     >= min_spacing - 1e-6
    # )

    spacing_tolerance = 1.0  # m

    valid_spacing = (
    min_distance_m
    >= min_spacing - spacing_tolerance
    )

    # --------------------------------------------------------
    # Cardinality
    # --------------------------------------------------------

    valid_cardinality = (
        n_turbines
        == expected_turbines
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    valid_boundary = all(
        farm_area.covers(
            Point(x, y)
        )
        for x, y in coords
    )

    valid = (
        valid_cardinality
        and valid_spacing
        and valid_boundary
    )

    return {
        "n_turbines": n_turbines,
        "aep_gwh": aep_gwh,
        "wake_loss_pct": wake_loss_pct,
        "min_distance_m": min_distance_m,
        "valid_cardinality": valid_cardinality,
        "valid_spacing": valid_spacing,
        "valid_boundary": valid_boundary,
        "valid": valid,
    }


def get_benchmark_comparison():
    # ============================================================
    # Load precomputed Thomas et al. reference results
    # ============================================================

    df_reference = pd.read_csv(
        REFERENCE_RESULTS_FILE
    )

    print(
        f"Loaded {len(df_reference)} "
        f"reference layouts."
    )


    # ============================================================
    # Benchmark constants
    # ============================================================

    _, farm_area = get_farm_area()

    single_turbine_aep = compute_aep_from_coords(
        np.array([
            [0.0, 0.0]
        ])
    )

    print(
        f"Single turbine AEP: "
        f"{single_turbine_aep / 1000:.3f} GWh/year"
    )


    # ============================================================
    # Evaluate our layouts
    # ============================================================

    our_results = []

    layout_files = sorted(
        OUR_LAYOUTS_DIR.glob("*.yaml")
    )

    for filepath in layout_files:

        solver_name = get_solver_name( 
            filepath
        )

        coords = load_layout_coordinates(
            filepath
        )

        metrics = evaluate_layout(
            coords=coords,
            single_turbine_aep=single_turbine_aep,
            farm_area=farm_area,
        )

        our_results.append({
            "method": solver_name,
            "source": "Our solver",
            "layout_file": filepath.name,
            **metrics,
        })

        print(
            f"{solver_name:<25} | "
            f"AEP {metrics['aep_gwh']:.3f} GWh | "
            f"wake loss {metrics['wake_loss_pct']:.3f}% | "
            f"valid={metrics['valid']}"
        )


    df_ours = pd.DataFrame(
        our_results
    )


    # ============================================================
    # Combine reference + our results
    # ============================================================

    df_results = pd.concat(
        [
            df_reference,
            df_ours,
        ],
        ignore_index=True,
    )


    # ============================================================
    # Comparison metrics
    # ============================================================

    baseline_aep = df_results.loc[
        df_results["method"] == "Baseline",
        "aep_gwh",
    ].iloc[0]


    # Best solver from Thomas et al.
    best_reference_aep = (
        df_reference.loc[
            df_reference["method"] != "Baseline",
            "aep_gwh",
        ]
        .max()
    )


    df_results[
        "improvement_vs_baseline_pct"
    ] = (
        100.0
        * (
            df_results["aep_gwh"]
            - baseline_aep
        )
        / baseline_aep
    )


    df_results[
        "gap_vs_best_reference_pct"
    ] = (
        100.0
        * (
            best_reference_aep
            - df_results["aep_gwh"]
        )
        / best_reference_aep
    )


    # ============================================================
    # Sort by official AEP
    # ============================================================

    df_results = (
        df_results
        .sort_values(
            "aep_gwh",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    # ============================================================
    # Display
    # ============================================================

    columns = [
        "method",
        "source",
        "aep_gwh",
        "wake_loss_pct",
        "improvement_vs_baseline_pct",
        "gap_vs_best_reference_pct",
        "min_distance_m",
        "valid",
    ]

    return df_results
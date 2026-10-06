import sys
import yaml
import shapely
import numpy as np
from pathlib import Path
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon


sys.path.append("../..")

_PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "external" / "thomas-wflo-benchmark" / "src" / "physics-models"))
from provided_model import (
    calcAEPcs3,
    getWindRoseYAML,
    getTurbAtrbtYAML,
)

# Setup
BENCHMARK_ROOT = _PROJECT_ROOT / "external" / "thomas-wflo-benchmark"

BOUNDARY_FILE = (
    BENCHMARK_ROOT
    / "src"
    / "input-files"
    / "farms"
    / "iea37-boundary-cs4.yaml"
)

WIND_FILE = (
    BENCHMARK_ROOT
    / "src"
    / "input-files"
    / "wind"
    / "iea37-windrose-cs4.yaml"
)

TURBINE_FILE = (
    BENCHMARK_ROOT
    / "src"
    / "input-files"
    / "turbines"
    / "iea37-10mw.yaml"
)

wind_dir, wind_dir_freq, wind_speeds, wind_speed_probs, *_ = (
    getWindRoseYAML(WIND_FILE)
)

turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = (
    getTurbAtrbtYAML(TURBINE_FILE)
)

# --------------------------------------------------------------------------------------------- #


def get_farm_area():
    with open(BOUNDARY_FILE) as f:
        boundary_data = yaml.safe_load(f)

    region_names = boundary_data['boundaries'].keys()
    polygons = []

    for region_name in region_names:
        region_coords = boundary_data['boundaries'][region_name]
        polygon = Polygon(region_coords)
        polygons.append(polygon)

    # Defining the union of the 5 regions
    farm_area = unary_union(polygons) 
    return polygons, farm_area


def get_mask(farm_area: MultiPolygon, grid_resolution: int=200):
    """
    grid_resolution (int): resolution grid step in meters
    """
    # Define the rectangular area including all the polygons
    min_x, min_y, max_x, max_y = farm_area.bounds
    x_coords = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_coords = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Building the mask (i.e. force all the point outside the polygons to have value 0)
    points = shapely.points(X.ravel(), Y.ravel())
    inside = shapely.covers(farm_area, points)
    mask = inside.reshape(X.shape)

    return X, Y, mask


def build_wake_loss_matrix_optimized(
    candidate_locations,
    grid_resolution,
    output_path=None,
):
    candidate_locations = np.asarray(
        candidate_locations,
        dtype=float,
    )

    n = len(candidate_locations)

    # --------------------------------------------------
    # 1. AEP singola turbina
    # --------------------------------------------------

    A0 = compute_aep_from_coords(
        [candidate_locations[0]]
    )

    # --------------------------------------------------
    # 2. Tutte le coppie i < j, vettorialmente
    # --------------------------------------------------

    i_idx, j_idx = np.triu_indices(n, k=1)

    displacements = (
        candidate_locations[j_idx]
        - candidate_locations[i_idx]
    )

    grid_displacements = np.rint(
        displacements / grid_resolution
    ).astype(np.int32)

    # --------------------------------------------------
    # 3. Canonicalizzazione
    # --------------------------------------------------

    dx = grid_displacements[:, 0]
    dy = grid_displacements[:, 1]

    flip = (
        (dx < 0)
        | ((dx == 0) & (dy < 0))
    )

    grid_displacements[flip] *= -1

    # --------------------------------------------------
    # 4. Displacement unici
    # --------------------------------------------------

    unique_displacements, inverse = np.unique(
        grid_displacements,
        axis=0,
        return_inverse=True,
    )

    print(f"Numero candidate locations: {n}")
    print(f"Numero coppie totali: {len(i_idx):,}")
    print(
        f"Numero displacement unici: "
        f"{len(unique_displacements):,}"
    )

    # --------------------------------------------------
    # 5. Physics model
    # --------------------------------------------------

    unique_losses = np.empty(
        len(unique_displacements),
        dtype=float,
    )

    origin = np.array([0.0, 0.0])

    for k, (dx_grid, dy_grid) in enumerate(
        unique_displacements
    ):

        second_turbine = np.array([
            dx_grid * grid_resolution,
            dy_grid * grid_resolution,
        ])

        aep_pair = compute_aep_from_coords([
            origin,
            second_turbine,
        ])

        unique_losses[k] = max(
            0.0,
            2 * A0 - aep_pair,
        )

    # --------------------------------------------------
    # 6. Mapping displacement -> coppie
    # --------------------------------------------------

    pair_losses = unique_losses[inverse]

    # --------------------------------------------------
    # 7. Costruzione wake-loss matrix
    # --------------------------------------------------

    L = np.zeros(
        (n, n),
        dtype=float,
    )

    L[i_idx, j_idx] = pair_losses
    L[j_idx, i_idx] = pair_losses

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            output_path,
            wake_loss_matrix=L,
            A0=A0,
            candidate_locations=candidate_locations,
            grid_resolution=grid_resolution,
        )

        print(
            f"Wake-loss data saved to: "
            f"{output_path}"
        )

    return None


def load_wake_loss_data(filepath):

    data = np.load(filepath)

    return {
        "wake_loss_matrix": data["wake_loss_matrix"],
        "A0": float(data["A0"]),
        "candidate_locations": data["candidate_locations"],
        "grid_resolution": int(data["grid_resolution"]),
    }


def build_qubo_from_wake_matrix_optimized(
    wake_loss_matrix,
    invalid_pairs,
    A0,
    n_turbines=81,
    lambda_cardinality=1e6,
    lambda_spacing=1e6,
    include_single_turbine_term=False,
):
    L = np.asarray(
        wake_loss_matrix,
        dtype=float,
    )

    n = L.shape[0]

    # -----------------------------------------
    # Wake
    #
    # 1/2 perché Q è simmetrica:
    #
    # z^T Q z contiene
    # 2 Q_ij z_i z_j
    # -----------------------------------------

    Q = 0.5 * L.copy()

    # -----------------------------------------
    # Produzione singola turbina
    # -----------------------------------------

    if include_single_turbine_term:
        diag = np.diag_indices(n)
        Q[diag] -= A0

    # -----------------------------------------
    # Spacing penalty
    # -----------------------------------------

    invalid_pairs = np.asarray(
        invalid_pairs,
        dtype=int,
    )

    i_invalid = invalid_pairs[:, 0]
    j_invalid = invalid_pairs[:, 1]

    Q[i_invalid, j_invalid] += (
        lambda_spacing / 2
    )

    Q[j_invalid, i_invalid] += (
        lambda_spacing / 2
    )

    # -----------------------------------------
    # Cardinality penalty
    #
    # off diagonal: lambda_N
    # diagonal:
    # lambda_N * (1 - 2K)
    # -----------------------------------------

    Q += lambda_cardinality

    diag = np.diag_indices(n)

    # Dopo Q += lambda:
    # diagonale = lambda.
    #
    # Sottraiamo 2*K*lambda:
    #
    # lambda - 2*K*lambda
    # = lambda(1 - 2K)
    Q[diag] -= (
        2
        * n_turbines
        * lambda_cardinality
    )

    offset = (
        lambda_cardinality
        * n_turbines**2
    )

    return Q, offset


def compute_aep_from_coords(coords):
    aep_by_direction = calcAEPcs3(
        turb_coords=np.asarray(coords),
        wind_freq=wind_dir_freq,
        wind_speeds=wind_speeds,
        wind_speed_probs=wind_speed_probs,
        wind_dir=wind_dir,
        turb_diam=turb_diam,
        turb_ci=turb_ci,
        turb_co=turb_co,
        rated_ws=rated_ws,
        rated_pwr=rated_pwr,
    )
    return np.sum(aep_by_direction)


def build_local_candidates(
    gurobi_positions,
    farm_area,
    radius_m,
    step_m,
    min_distance=396.0,
):
    gurobi_positions = np.asarray(gurobi_positions, dtype=float)

    # offset locali
    offsets_1d = np.arange(
        -radius_m,
        radius_m + step_m * 0.5,
        step_m,
    )

    dx, dy = np.meshgrid(
        offsets_1d,
        offsets_1d,
        indexing="xy",
    )

    offsets = np.column_stack([
        dx.ravel(),
        dy.ravel(),
    ])

    # tengo solo punti dentro il raggio
    offsets = offsets[
        np.linalg.norm(offsets, axis=1) <= radius_m
    ]

    # genero candidati attorno a ogni turbina Gurobi
    candidate_locations = (
        gurobi_positions[:, None, :]
        + offsets[None, :, :]
    ).reshape(-1, 2)

    # aggiungo esplicitamente le posizioni Gurobi originali
    candidate_locations = np.vstack([
        candidate_locations,
        gurobi_positions,
    ])

    # tengo solo punti dentro il farm boundary
    mask = np.array([
        farm_area.covers(Point(x, y))
        for x, y in candidate_locations
    ])

    candidate_locations = candidate_locations[mask]

    # rimuovo duplicati
    candidate_locations = np.unique(
        np.round(candidate_locations, 6),
        axis=0,
    )

    # invalid pairs
    tree = cKDTree(candidate_locations)

    invalid_pairs = tree.query_pairs(
        r=min_distance,
        output_type="ndarray",
    )

    # query_pairs include anche distanza == min_distance:
    # noi vogliamo solo < min_distance
    if len(invalid_pairs) > 0:
        d = np.linalg.norm(
            candidate_locations[invalid_pairs[:, 0]]
            - candidate_locations[invalid_pairs[:, 1]],
            axis=1,
        )

        invalid_pairs = invalid_pairs[
            d < min_distance
        ]

    return (
        candidate_locations,
        invalid_pairs.astype(np.int32),
    )


def build_local_wake_loss_matrix_optimized(
    candidate_locations,
    output_path,
    decimals=6,
):
    candidate_locations = np.asarray(
        candidate_locations,
        dtype=float,
    )

    n = len(candidate_locations)

    # 1. AEP singola turbina
    A0 = compute_aep_from_coords(
        [candidate_locations[0]]
    )

    # 2. Tutte le coppie i < j
    i_idx, j_idx = np.triu_indices(n, k=1)

    displacements = (
        candidate_locations[j_idx]
        - candidate_locations[i_idx]
    )

    # evita differenze floating tipo
    # 25.0000000001 vs 25.0
    displacements = np.round(
        displacements,
        decimals=decimals,
    )

    # 3. Canonicalizzazione:
    # (dx, dy) e (-dx, -dy) sono equivalenti
    dx = displacements[:, 0]
    dy = displacements[:, 1]

    flip = (
        (dx < 0)
        | ((dx == 0) & (dy < 0))
    )

    displacements[flip] *= -1

    # 4. Displacement unici
    unique_displacements, inverse = np.unique(
        displacements,
        axis=0,
        return_inverse=True,
    )

    print(f"Numero candidate locations: {n}")
    print(f"Numero coppie totali: {len(i_idx):,}")
    print(
        f"Numero displacement unici: "
        f"{len(unique_displacements):,}"
    )

    # 5. Physics model:
    # una sola chiamata per displacement unico
    unique_losses = np.empty(
        len(unique_displacements),
        dtype=float,
    )

    origin = np.array([0.0, 0.0])

    for k, displacement in enumerate(
        unique_displacements
    ):
        aep_pair = compute_aep_from_coords([
            origin,
            displacement,
        ])

        unique_losses[k] = max(
            0.0,
            2 * A0 - aep_pair,
        )

    # 6. Mapping verso tutte le coppie
    pair_losses = unique_losses[inverse]

    # 7. Wake-loss matrix
    L = np.zeros(
        (n, n),
        dtype=float,
    )

    L[i_idx, j_idx] = pair_losses
    L[j_idx, i_idx] = pair_losses

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
        )

        np.savez_compressed(
        output_path,
        wake_loss_matrix=L,
        A0=A0,
        candidate_locations=candidate_locations,
        )

        print(
            f"Wake-loss data saved to: "
            f"{output_path}"
        )

    return L, A0


def get_warmstart(candidate_locations, starting_layout):
    from src.qubo_windfarm_layout.evaluation import load_layout_coordinates
    warmstart_positions = np.asarray(
        load_layout_coordinates(
            starting_layout
        ),
        dtype=float,
    )

    tree = cKDTree(candidate_locations)

    distances, indices = tree.query(
        warmstart_positions,
        k=1,
    )

    # Verifica che le posizioni del warm start esistano
    # effettivamente nella candidate grid corrente
    if np.any(distances > 1e-6):
        raise ValueError(
            f"Alcune turbine del warm start non sono nella candidate grid. "
            f"Max distance = {distances.max():.6f} m"
        )

    warm_start = np.zeros(
        len(candidate_locations),
        dtype=int,
    )

    warm_start[indices] = 1
    return warm_start

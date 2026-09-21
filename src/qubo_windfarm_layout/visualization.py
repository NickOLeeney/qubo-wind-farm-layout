import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.qubo_windfarm_layout.model import get_farm_area

def plot_selected_layout(
    selected_locations,
    rotor_diameter=198.0,
    figsize=(10, 10),
):
    selected_locations = np.asarray(
        selected_locations,
        dtype=float,
    )

    polygons, _ = get_farm_area()

    _, ax = plt.subplots(figsize=figsize)

    # Boundary
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, linewidth=1.5)

    # Turbine
    rotor_radius = rotor_diameter / 2

    for x, y in selected_locations:
        turbine = Circle(
            (x, y),
            rotor_radius,
            fill=False,
            linewidth=1,
        )

        ax.add_patch(turbine)

        ax.scatter(
            x,
            y,
            s=8,
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"Optimized wind farm layout\n"
        f"{len(selected_locations)} turbines — "
    )

    plt.show()

def plot_gurobi_refinement(
    gurobi_positions,
    candidate_locations,
    radius_m,
    rotor_diameter=198.0,
    figsize=(10, 10),
):
    gurobi_positions = np.asarray(
        gurobi_positions,
        dtype=float,
    )

    candidate_locations = np.asarray(
        candidate_locations,
        dtype=float,
    )

    polygons, _ = get_farm_area()

    _, ax = plt.subplots(figsize=figsize)

    # Boundary
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, linewidth=1.5)

    # --------------------------------------------------
    # Identify refinement-only candidates
    # --------------------------------------------------

    gurobi_set = {
        tuple(np.round(p, 6))
        for p in gurobi_positions
    }

    refinement_mask = np.array([
        tuple(np.round(p, 6)) not in gurobi_set
        for p in candidate_locations
    ])

    refinement_locations = candidate_locations[
        refinement_mask
    ]

    # --------------------------------------------------
    # New candidate positions
    # --------------------------------------------------

    ax.scatter(
        refinement_locations[:, 0],
        refinement_locations[:, 1],
        s=12,
        alpha=0.5,
        label="Refinement candidates",
        zorder=2,
    )

    # --------------------------------------------------
    # Original Gurobi positions
    # --------------------------------------------------

    ax.scatter(
        gurobi_positions[:, 0],
        gurobi_positions[:, 1],
        s=30,
        marker="x",
        label="Gurobi solution",
        zorder=4,
    )

    rotor_radius = rotor_diameter / 2

    for x, y in gurobi_positions:

        # Search radius
        search_area = Circle(
            (x, y),
            radius_m,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.add_patch(search_area)

        # Original turbine rotor
        turbine = Circle(
            (x, y),
            rotor_radius,
            fill=False,
            linewidth=1,
        )
        ax.add_patch(turbine)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"Gurobi solution + local refinement\n"
        f"radius = {radius_m} m — "
        f"{len(refinement_locations)} new candidates"
    )

    ax.legend()

    plt.show()
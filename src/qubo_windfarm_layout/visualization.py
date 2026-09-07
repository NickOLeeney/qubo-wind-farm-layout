import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.qubo_windfarm_layout.model import get_farm_area, _compute_aep_from_coords

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

    # AEP totale del layout
    total_aep = _compute_aep_from_coords(
        selected_locations
    )

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
        f"AEP: {total_aep / 1000:.3f} GWh/year"
    )

    plt.show()

 
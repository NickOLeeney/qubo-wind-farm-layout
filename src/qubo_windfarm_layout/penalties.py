import sys
import numpy as np

sys.path.append("..")
from src.qubo_windfarm_layout.model import compute_aep_from_coords
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
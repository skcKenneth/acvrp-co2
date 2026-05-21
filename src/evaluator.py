"""
evaluator.py
============

The crux of the experimental design lives here.

A solution found on matrix X (SE / SM / SR / AR) is a *plan* -- a set
of route orderings. The plan's *real-world cost* depends on the
distance the vehicle actually has to travel, which is given by the
asymmetric road matrix AR. So to compare variants fairly we:

    1. Solve the CVRP on each variant's matrix X to obtain routes R_X.
    2. Re-evaluate R_X *as if it were driven on the road*, i.e.
       compute its distance, fuel, and CO2 using AR.
    3. Report the gap of each (X != AR) against AR as the
       "asymmetry penalty".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np

from .emissions_model import EmissionsParams, routes_to_metrics


@dataclass
class EvaluatedSolution:
    """A solution evaluated under the AR ground-truth matrix."""

    variant: str           # SE, SM, SR, AR
    routes: List[List[int]]
    distance_m: float      # Re-evaluated on AR (finite legs only)
    fuel_l: float
    co2_kg: float
    ar_feasible: bool = True
    infeasible_legs: int = 0

    def as_row(self) -> Dict[str, float | str | bool | int]:
        return asdict(self)


def reevaluate_on_ground_truth(
    variant: str,
    routes: List[List[int]],
    ar_matrix: np.ndarray,
    demands: Sequence[int],
    params: EmissionsParams,
) -> EvaluatedSolution:
    """
    Re-score a plan on the asymmetric road matrix and return the
    distance / fuel / CO2 it would actually generate.
    """
    metrics = routes_to_metrics(routes, ar_matrix, demands, params)
    return EvaluatedSolution(
        variant=variant,
        routes=routes,
        distance_m=float(metrics["distance_m"]),
        fuel_l=float(metrics["fuel_l"]),
        co2_kg=float(metrics["co2_kg"]),
        ar_feasible=bool(metrics["ar_feasible"]),
        infeasible_legs=int(metrics["infeasible_legs"]),
    )


def compute_penalties(
    evaluations: Dict[str, EvaluatedSolution],
    baseline: str = "AR",
) -> Dict[str, Dict[str, float]]:
    """
    Compute relative penalties (in %) for distance and CO2, taking the
    `baseline` variant as the reference.

    A positive penalty means the variant produces *more* of that
    quantity than the baseline does.
    """
    if baseline not in evaluations:
        raise KeyError(f"Baseline variant {baseline!r} not in evaluations.")

    ref = evaluations[baseline]
    if not ref.ar_feasible or ref.distance_m <= 0:
        return {}

    penalties: Dict[str, Dict[str, float]] = {}
    for code, ev in evaluations.items():
        if code == baseline:
            continue
        entry: Dict[str, float] = {
            "distance_pct": 100.0 * (ev.distance_m - ref.distance_m) / ref.distance_m,
        }
        if ev.ar_feasible and ref.ar_feasible and np.isfinite(ev.co2_kg) and np.isfinite(ref.co2_kg) and ref.co2_kg > 0:
            entry["co2_pct"] = 100.0 * (ev.co2_kg - ref.co2_kg) / ref.co2_kg
        if ev.ar_feasible and ref.ar_feasible and np.isfinite(ev.fuel_l) and np.isfinite(ref.fuel_l) and ref.fuel_l > 0:
            entry["fuel_pct"] = 100.0 * (ev.fuel_l - ref.fuel_l) / ref.fuel_l
        penalties[code] = entry
    return penalties

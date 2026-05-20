"""
emissions_model.py
==================

Linear fuel-consumption and CO2 emissions model used as the secondary
objective in the multi-objective CVRP.

Following the simplification used by Bektaş & Laporte (2011) and applied
to last-mile delivery vans, we express the litres of diesel burned on
a single arc (i -> j) as a linear function of three explanatory
variables:

    Y_ij = C1 * d_ij + C2 * t_ij + C3 * M_ij * d_ij

where
    d_ij is the arc length in metres,
    t_ij is the time spent on the arc in seconds,
    M_ij is the current payload mass in kg while the vehicle traverses
         the arc (decreases as deliveries are completed).

The total fuel for a route is the sum of Y_ij over all consecutive
arcs. CO2 emissions are obtained by multiplying total fuel by a
diesel-specific emission factor (about 2.68 kg of CO2 per litre).

This is a deliberate, well-known simplification: it captures the three
dominant cost drivers (distance, time, payload) while remaining linear
and therefore amenable to MILP solvers. Higher-order effects (road
gradient, speed-cubed aerodynamic drag) are deferred to future work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class EmissionsParams:
    """Parameters of the linear fuel model. Units in module docstring."""

    C1_distance: float
    C2_time: float
    C3_mass: float
    empty_mass_kg: float
    co2_per_litre: float
    avg_speed_kmh: float

    @property
    def avg_speed_mps(self) -> float:
        return self.avg_speed_kmh * 1000.0 / 3600.0


def arc_fuel_litres(
    distance_m: float,
    payload_kg: float,
    params: EmissionsParams,
) -> float:
    """
    Fuel burned on a single arc, in litres.

    Time on the arc is derived from distance and average speed:
        t = d / v.
    """
    speed = params.avg_speed_mps
    time_s = distance_m / speed if speed > 0 else 0.0
    total_mass = params.empty_mass_kg + payload_kg
    return (
        params.C1_distance * distance_m
        + params.C2_time * time_s
        + params.C3_mass * total_mass * distance_m
    )


def route_fuel_litres(
    route: Sequence[int],
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    params: EmissionsParams,
) -> float:
    """
    Total fuel burned by a vehicle traversing a single route.

    The route is a list of customer indices that *starts and ends at
    the depot* (index 0). The payload at departure equals the sum of
    customer demands on the route; it drops by demand[k] after visiting
    customer k.

    Demand units in this project are treated as kilograms for the
    purpose of payload computation; if a different unit scaling is
    desired, multiply demand by a kg-per-unit conversion before
    calling this function.
    """
    if len(route) < 2:
        return 0.0

    # Initial payload: sum of demands of all customers on the route.
    # The depot is at the start/end of the route with demand 0.
    payload = float(sum(demands[node] for node in route))

    total_fuel = 0.0
    for k in range(len(route) - 1):
        u, v = route[k], route[k + 1]
        leg = float(distance_matrix[u, v])
        if not np.isfinite(leg):
            # Unreachable arc -> treat as infinite fuel so any solver
            # that constructs this leg is penalised.
            return float("inf")
        total_fuel += arc_fuel_litres(leg, payload, params)
        # After arriving at v, that customer's load is dropped off.
        payload -= demands[v]
        if payload < 0:
            payload = 0.0
    return total_fuel


def fuel_to_co2_kg(litres: float, params: EmissionsParams) -> float:
    """Convert a fuel amount (litres of diesel) to kg of CO2 emitted."""
    return litres * params.co2_per_litre


def routes_to_metrics(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    params: EmissionsParams,
) -> dict[str, float]:
    """
    Aggregate fuel, distance, and CO2 across a *set* of routes (one
    route per vehicle).
    """
    total_distance = 0.0
    total_fuel = 0.0
    for route in routes:
        for k in range(len(route) - 1):
            d = float(distance_matrix[route[k], route[k + 1]])
            if np.isfinite(d):
                total_distance += d
        total_fuel += route_fuel_litres(route, distance_matrix, demands, params)
    return {
        "distance_m": total_distance,
        "fuel_l": total_fuel,
        "co2_kg": fuel_to_co2_kg(total_fuel, params),
    }

"""
baselines/solver_pyvrp.py
=========================

Wrapper around PyVRP (https://github.com/PyVRP/PyVRP), an open-source
implementation of Hybrid Genetic Search for CVRP (HGS-CVRP) — currently
the state-of-the-art classical CVRP solver on the standard benchmarks.

Used as a strong upper-bound baseline against which the neural policy
is compared.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class PyVRPSolution:
    routes: List[List[int]]
    cost: float


def solve_cvrp_pyvrp(
    distance_matrix: np.ndarray,     # (n, n) float metres
    demands: Sequence[int],
    vehicle_capacity: int,
    num_vehicles: int,
    depot: int = 0,
    time_limit_s: int = 30,
) -> PyVRPSolution:
    """
    Solve a CVRP instance with PyVRP's HGS engine and return the routes.

    PyVRP works in integer distance units internally; we scale our
    floating-point metre distances by 100 to keep two decimal places
    of precision and rescale the final objective back to metres.
    """
    # Lazy import so the rest of the project can still be imported on
    # machines that don't have PyVRP installed.
    from pyvrp import (
        Client,
        Depot,
        ProblemData,
        Solution,
        VehicleType,
    )
    from pyvrp.solve import solve
    from pyvrp.stop import MaxRuntime

    n = distance_matrix.shape[0]
    SCALE = 100  # metres * 100 -> centimetres-ish, integer

    int_matrix = np.round(distance_matrix * SCALE).astype(np.int64)

    clients = []
    depots = []
    for i in range(n):
        if i == depot:
            depots.append(Depot(x=0, y=0))    # coords unused; arc-cost is from matrix
        else:
            clients.append(Client(x=0, y=0, delivery=[int(demands[i])]))

    vehicle_types = [
        VehicleType(
            num_available=num_vehicles,
            capacity=[int(vehicle_capacity)],
            start_depot=0,
            end_depot=0,
        )
    ]

    data = ProblemData(
        clients=clients,
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrices=[int_matrix],
        duration_matrices=[int_matrix],  # PyVRP requires duration too; we reuse distance
    )

    result = solve(
        data,
        stop=MaxRuntime(time_limit_s),
        display=False,
    )
    sol: Solution = result.best
    cost = float(sol.distance()) / SCALE

    # Convert PyVRP's per-route client lists back to global node indices.
    # In our convention the depot has index 0 and customers are 1..n-1,
    # which matches PyVRP's internal numbering (clients are indexed
    # 1..n-1, depot is 0).
    routes_out = []
    for route in sol.routes():
        visits = list(route.visits())
        # PyVRP omits the depot; we add it back at both ends to match
        # the rest of the codebase's route format.
        full = [depot] + visits + [depot]
        routes_out.append(full)
    return PyVRPSolution(routes=routes_out, cost=cost)

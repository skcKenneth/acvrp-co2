"""
solver_ortools.py
=================

Wraps Google OR-Tools' constraint-programming router to solve a
Capacitated Vehicle Routing Problem (CVRP) given an arbitrary distance
matrix.

OR-Tools handles asymmetric matrices natively, which makes it the
obvious choice for the AR variant. For SE/SM/SR we feed it the same
solver with a symmetric matrix; the solver does not need to know
whether the input is symmetric.

The arc cost used by the solver is a weighted combination of distance
and an *approximate* per-arc emissions term. Because OR-Tools requires
integer costs, we scale the floating-point combined cost by
COST_SCALE and round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .emissions_model import EmissionsParams, arc_fuel_litres


COST_SCALE = 1000  # Integer scaling for arc costs


@dataclass
class CVRPSolution:
    """Container for the result of a single CVRP solve."""

    routes: List[List[int]]            # One list of node indices per vehicle
    objective_value: int               # Solver-reported integer objective
    raw_distance_matrix: np.ndarray    # Stored for downstream evaluation


def _make_arc_cost_callback(
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    params: EmissionsParams,
    alpha_distance: float,
    alpha_emissions: float,
):
    """
    Build a callback that returns an integer cost for traversing arc
    (i -> j). Cost is alpha_d * d + alpha_e * fuel(d, payload_approx),
    where payload_approx is approximated as average half-load because
    the true post-visit payload is route-dependent and unknown at
    callback time.
    """
    avg_payload = 0.5 * sum(demands)  # Coarse approximation

    def cost_callback(from_index: int, to_index: int) -> int:
        d = float(distance_matrix[from_index, to_index])
        if not np.isfinite(d):
            # Make impossible arcs prohibitively expensive
            return 10 ** 12
        fuel = arc_fuel_litres(d, avg_payload, params)
        combined = alpha_distance * d + alpha_emissions * fuel * 1000.0
        # Multiplying fuel by 1000 brings it into a roughly comparable
        # order of magnitude with distance in metres.
        return int(round(combined * COST_SCALE))

    return cost_callback


def solve_cvrp(
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    vehicle_capacity: int,
    num_vehicles: int,
    depot: int,
    params: EmissionsParams,
    alpha_distance: float = 0.5,
    alpha_emissions: float = 0.5,
    time_limit_s: int = 60,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> CVRPSolution:
    """
    Solve a CVRP and return a CVRPSolution.

    Parameters
    ----------
    distance_matrix : (n, n) array, metres. May be asymmetric.
    demands         : length-n sequence of integer demands; depot = 0.
    vehicle_capacity, num_vehicles, depot : standard CVRP inputs.
    params          : EmissionsParams for the secondary objective.
    alpha_*         : scalarisation weights of the multi-objective.
    """
    n = distance_matrix.shape[0]

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Translate matrix indices through the manager
    def make_callback(matrix):
        def callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return matrix(i, j)
        return callback

    raw_cost = _make_arc_cost_callback(
        distance_matrix, demands, params, alpha_distance, alpha_emissions
    )
    cost_idx = routing.RegisterTransitCallback(make_callback(raw_cost))
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

    # Capacity dimension
    def demand_callback(from_index):
        return int(demands[manager.IndexToNode(from_index)])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,                                  # null capacity slack
        [vehicle_capacity] * num_vehicles,  # per-vehicle capacity
        True,                               # start cumul to zero
        "Capacity",
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
    )
    search_params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
    )
    search_params.time_limit.seconds = time_limit_s

    assignment = routing.SolveWithParameters(search_params)
    if assignment is None:
        raise RuntimeError(
            "OR-Tools failed to find a feasible solution. "
            "Try relaxing capacity or increasing the time limit."
        )

    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = [manager.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            idx = assignment.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        # Only keep routes that actually visit at least one customer.
        if len(route) > 2:
            routes.append(route)

    return CVRPSolution(
        routes=routes,
        objective_value=assignment.ObjectiveValue(),
        raw_distance_matrix=distance_matrix,
    )

"""
experiments_full.py
===================

Master experiment runner for the four-solver / four-matrix comparison
that anchors the paper.

Comparison grid
---------------

                  | SE  | SM  | SR  | AR
    --------------+-----+-----+-----+-----
    OR-Tools      |  ✓  |  ✓  |  ✓  |  ✓     (classical, distance only)
    GA            |  ✓  |  ✓  |  ✓  |  ✓     (heuristic, distance only)
    Vanilla-AM    |  ✓  |  ✓  |  ✓  |  ✓ †   (coord-only NCO, distance only)
    MatNet-CVRP   |  ✓  |  ✓  |  ✓  |  ✓     (this work, dist & CO2)

    †  Vanilla-AM is the same trained policy applied to all four matrices;
       it cannot adapt to asymmetry by construction. The point is to
       quantify the resulting penalty.

Every solver is run on each variant's distance matrix, then the produced
route plan is re-evaluated on the AR ground-truth matrix so distances /
CO2 across solvers are directly comparable.

Usage
-----
    python -m src.experiments_full \\
        --config config.yaml \\
        --matnet-checkpoint models/matnet_cvrp_best.pt \\
        --baseline-checkpoint models/baseline_am_best.pt

Either checkpoint argument is optional; if absent, the corresponding
NCO solver is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .data_loader import (
    Customer,
    download_road_graph,
    load_customers,
    snap_customers_to_nodes,
)
from .distance_matrix import asymmetry_index, build_all_matrices
from .emissions_model import EmissionsParams
from .evaluator import (
    EvaluatedSolution,
    compute_penalties,
    reevaluate_on_ground_truth,
)
from .solver_ga import solve_cvrp_ga
from .solver_ortools import solve_cvrp
from .visualization import make_comparison_bar, make_folium_map


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _emissions_params_from_config(cfg: dict) -> EmissionsParams:
    e = cfg["emissions"]
    return EmissionsParams(
        C1_distance=e["C1_distance"],
        C2_time=e["C2_time"],
        C3_mass=e["C3_mass"],
        empty_mass_kg=e["empty_mass_kg"],
        co2_per_litre=e["co2_per_litre"],
        avg_speed_kmh=e["avg_speed_kmh"],
    )


# ---------------------------------------------------------------------------
# Solver adapters
# ---------------------------------------------------------------------------

def _solve_ortools(matrix, demands, cfg, params):
    sol = solve_cvrp(
        distance_matrix=matrix,
        demands=demands,
        vehicle_capacity=cfg["fleet"]["vehicle_capacity"],
        num_vehicles=cfg["fleet"]["num_vehicles"],
        depot=cfg["data"]["depot_index"],
        params=params,
        alpha_distance=cfg["objective"]["alpha_distance"],
        alpha_emissions=cfg["objective"]["alpha_emissions"],
        time_limit_s=cfg["ortools"]["time_limit_seconds"],
        first_solution_strategy=cfg["ortools"]["first_solution_strategy"],
        local_search_metaheuristic=cfg["ortools"]["local_search_metaheuristic"],
    )
    return sol.routes


def _solve_ga(matrix, demands, num_customers, cfg, params):
    ga = cfg["genetic_algorithm"]
    sol = solve_cvrp_ga(
        distance_matrix=matrix,
        demands=demands,
        vehicle_capacity=cfg["fleet"]["vehicle_capacity"],
        num_customers=num_customers,
        params=params,
        alpha_distance=cfg["objective"]["alpha_distance"],
        alpha_emissions=cfg["objective"]["alpha_emissions"],
        population_size=ga["population_size"],
        generations=ga["generations"],
        crossover_prob=ga["crossover_prob"],
        mutation_prob=ga["mutation_prob"],
        tournament_size=ga["tournament_size"],
        random_seed=ga["random_seed"],
    )
    return sol.routes


def _solve_nco(checkpoint_path: str, policy_kind: str, instance):
    """
    Lazy wrapper around the trained neural policy so that torch is
    imported only when the NCO solvers are actually requested. This
    keeps the classical experiment runnable on machines without torch.
    """
    import torch  # lazy
    from .nco.inference import solve_with_policy
    from .nco.model import ACVRPPolicy
    from .nco.baseline_am import CoordOnlyACVRPPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if policy_kind == "matnet":
        policy = ACVRPPolicy()
    elif policy_kind == "baseline":
        policy = CoordOnlyACVRPPolicy()
    else:
        raise ValueError(f"Unknown policy_kind: {policy_kind!r}")

    state = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state)
    policy.to(device).eval()

    return solve_with_policy(policy, instance, device=device, mode="pomo", n_samples=32)


def _instance_from_components(
    customers: List[Customer],
    matrix: np.ndarray,
    params: EmissionsParams,
    capacity: int,
):
    """
    Build a CVRPInstance object so the NCO solver can consume the
    same input as the classical ones. This avoids the NCO module
    depending on customer CSV format.
    """
    from .nco.dataset import _build_instance_from_distance

    coords = np.array(
        [[c.lat, c.lon] for c in customers], dtype=np.float32
    )
    demands = np.array([c.demand for c in customers], dtype=np.int64)
    return _build_instance_from_distance(
        coords=coords,
        demands=demands,
        dist=matrix,
        capacity=capacity,
        params=params,
        city_name="from_csv",
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    cfg_path: str,
    matnet_ckpt: Optional[str] = None,
    baseline_ckpt: Optional[str] = None,
) -> None:
    cfg = _load_config(cfg_path)
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. Data --------------------------------------------------------
    customers = load_customers(cfg["data"]["customers_csv"])
    graph = download_road_graph(
        centre_lat=cfg["region"]["centre_lat"],
        centre_lon=cfg["region"]["centre_lon"],
        radius_m=cfg["region"]["radius_m"],
        network_type=cfg["region"]["network_type"],
    )
    node_ids = snap_customers_to_nodes(graph, customers)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Customers: {len(customers) - 1} (+1 depot)")

    matrices = build_all_matrices(customers, graph, node_ids)
    print(f"AR asymmetry index: {asymmetry_index(matrices['AR']):.4f}")

    params = _emissions_params_from_config(cfg)
    demands = [c.demand for c in customers]
    num_customers = len(customers) - 1
    variants = cfg["output"]["variants"]
    capacity = cfg["fleet"]["vehicle_capacity"]

    # ---- 2. Solvers -----------------------------------------------------
    # solvers: a dict mapping solver_name -> function(matrix) -> routes
    solvers = {
        "OR-Tools": lambda m: _solve_ortools(m, demands, cfg, params),
        "GA":       lambda m: _solve_ga(m, demands, num_customers, cfg, params),
    }
    if matnet_ckpt is not None:
        solvers["MatNet-CVRP"] = lambda m: _solve_nco(
            matnet_ckpt, "matnet",
            _instance_from_components(customers, m, params, capacity),
        )
    if baseline_ckpt is not None:
        solvers["Vanilla-AM"] = lambda m: _solve_nco(
            baseline_ckpt, "baseline",
            _instance_from_components(customers, m, params, capacity),
        )

    # ---- 3. Run solver x variant grid -----------------------------------
    # solutions[solver][variant] -> list of routes
    solutions: Dict[str, Dict[str, list]] = {s: {} for s in solvers}
    for solver_name, solver_fn in solvers.items():
        for variant in variants:
            print(f"\n--- {solver_name} on {variant} ---")
            try:
                routes = solver_fn(matrices[variant])
                solutions[solver_name][variant] = routes
                print(f"  -> {len(routes)} route(s) produced.")
            except Exception as err:
                print(f"  ! Failed: {err}")
                solutions[solver_name][variant] = []

    # ---- 4. Re-evaluate everything on AR ground truth -------------------
    ar = matrices["AR"]
    evaluations: Dict[str, Dict[str, EvaluatedSolution]] = {}
    for solver_name, variant_to_routes in solutions.items():
        evaluations[solver_name] = {}
        for variant, routes in variant_to_routes.items():
            if not routes:
                continue
            ev = reevaluate_on_ground_truth(
                variant=variant, routes=routes,
                ar_matrix=ar, demands=demands, params=params,
            )
            evaluations[solver_name][variant] = ev

    # ---- 5. Write CSV summary ------------------------------------------
    summary_path = results_dir / "summary_full.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["solver", "variant", "distance_m", "fuel_l", "co2_kg"],
        )
        writer.writeheader()
        for solver_name, var_dict in evaluations.items():
            for variant, ev in var_dict.items():
                writer.writerow({
                    "solver": solver_name,
                    "variant": variant,
                    "distance_m": round(ev.distance_m, 2),
                    "fuel_l": round(ev.fuel_l, 4),
                    "co2_kg": round(ev.co2_kg, 4),
                })
    print(f"\nWrote {summary_path}")

    # ---- 6. Per-solver asymmetry penalties (vs AR within each solver) --
    penalty_report = {}
    for solver_name, var_dict in evaluations.items():
        if "AR" in var_dict and len(var_dict) > 1:
            penalty_report[solver_name] = compute_penalties(var_dict, baseline="AR")
    with open(results_dir / "penalties_full.json", "w", encoding="utf-8") as f:
        json.dump(penalty_report, f, indent=2)

    # ---- 7. Human-readable summary -------------------------------------
    print("\n========== RESULTS (km, kg CO2) ==========")
    print(
        f"{'Solver':<14}{'Variant':<8}"
        f"{'Distance (km)':>16}{'CO2 (kg)':>12}"
    )
    for solver_name, var_dict in evaluations.items():
        for variant, ev in var_dict.items():
            print(
                f"{solver_name:<14}{variant:<8}"
                f"{ev.distance_m / 1000.0:>16.2f}"
                f"{ev.co2_kg:>12.3f}"
            )

    print(f"\nAll outputs written to: {results_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full four-solver / four-matrix comparison.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--matnet-checkpoint", default=None)
    parser.add_argument("--baseline-checkpoint", default=None)
    args = parser.parse_args()
    run(args.config, args.matnet_checkpoint, args.baseline_checkpoint)


if __name__ == "__main__":
    main()

"""
experiments.py
==============

Entry point that ties everything together.

Usage
-----
    python -m src.experiments --config config.yaml

The script:

1. Downloads (or loads cached) OSM road graph.
2. Loads customer coordinates from CSV.
3. Builds the four distance matrices (SE, SM, SR, AR).
4. Solves the CVRP on each matrix using both OR-Tools and the GA.
5. Re-evaluates every plan on the AR ground truth.
6. Writes summary.csv, comparison.png, routes_map.html into results/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
from .visualization import (
    make_comparison_bar,
    make_folium_map,
    plot_ga_convergence,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def emissions_params_from_config(cfg: dict) -> EmissionsParams:
    e = cfg["emissions"]
    return EmissionsParams(
        C1_distance=e["C1_distance"],
        C2_time=e["C2_time"],
        C3_mass=e["C3_mass"],
        empty_mass_kg=e["empty_mass_kg"],
        co2_per_litre=e["co2_per_litre"],
        avg_speed_kmh=e["avg_speed_kmh"],
    )


def run_experiment(cfg_path: str, solver: str = "ortools") -> None:
    cfg = load_config(cfg_path)
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)

    # 1. Data
    customers = load_customers(cfg["data"]["customers_csv"])
    graph = download_road_graph(
        centre_lat=cfg["region"]["centre_lat"],
        centre_lon=cfg["region"]["centre_lon"],
        radius_m=cfg["region"]["radius_m"],
        network_type=cfg["region"]["network_type"],
    )
    node_ids = snap_customers_to_nodes(
        graph, customers, depot_index=cfg["data"]["depot_index"]
    )
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Customers: {len(customers) - 1} (+1 depot)")

    # 2. Distance matrices
    matrices = build_all_matrices(customers, graph, node_ids)
    ar_index = asymmetry_index(matrices["AR"])
    print(f"AR asymmetry index: {ar_index:.4f}")

    # 3. Solve each variant
    params = emissions_params_from_config(cfg)
    demands = [c.demand for c in customers]
    variants = cfg["output"]["variants"]
    alpha_d = cfg["objective"]["alpha_distance"]
    alpha_e = cfg["objective"]["alpha_emissions"]

    plans = {}     # variant -> routes
    for variant in variants:
        print(f"\n--- Solving variant {variant} ({solver}) ---")
        matrix = matrices[variant]
        if solver == "ortools":
            sol = solve_cvrp(
                distance_matrix=matrix,
                demands=demands,
                vehicle_capacity=cfg["fleet"]["vehicle_capacity"],
                num_vehicles=cfg["fleet"]["num_vehicles"],
                depot=cfg["data"]["depot_index"],
                params=params,
                alpha_distance=alpha_d,
                alpha_emissions=alpha_e,
                time_limit_s=cfg["ortools"]["time_limit_seconds"],
                first_solution_strategy=cfg["ortools"]["first_solution_strategy"],
                local_search_metaheuristic=cfg["ortools"]["local_search_metaheuristic"],
            )
            plans[variant] = sol.routes
        elif solver == "ga":
            ga_cfg = cfg["genetic_algorithm"]
            sol = solve_cvrp_ga(
                distance_matrix=matrix,
                demands=demands,
                vehicle_capacity=cfg["fleet"]["vehicle_capacity"],
                num_customers=len(customers) - 1,
                params=params,
                alpha_distance=alpha_d,
                alpha_emissions=alpha_e,
                population_size=ga_cfg["population_size"],
                generations=ga_cfg["generations"],
                crossover_prob=ga_cfg["crossover_prob"],
                mutation_prob=ga_cfg["mutation_prob"],
                tournament_size=ga_cfg["tournament_size"],
                random_seed=ga_cfg["random_seed"],
            )
            plans[variant] = sol.routes
            plot_ga_convergence(sol.history, results_dir / f"ga_convergence_{variant}.png")
        else:
            raise ValueError(f"Unknown solver: {solver}")
        print(f"  Found {len(sol.routes)} routes.")

    # 4. Re-evaluate on AR ground truth
    ar = matrices["AR"]
    evaluations = {
        v: reevaluate_on_ground_truth(v, plans[v], ar, demands, params)
        for v in variants
    }

    # 5. Save outputs
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "distance_m", "fuel_l", "co2_kg"]
        )
        writer.writeheader()
        for v in variants:
            ev = evaluations[v]
            writer.writerow({
                "variant": ev.variant,
                "distance_m": round(ev.distance_m, 2),
                "fuel_l": round(ev.fuel_l, 4),
                "co2_kg": round(ev.co2_kg, 4),
            })

    penalties = compute_penalties(evaluations, baseline="AR")
    with open(results_dir / "penalties.json", "w", encoding="utf-8") as f:
        json.dump(penalties, f, indent=2)

    make_folium_map(
        customers, graph, node_ids, evaluations,
        output_path=results_dir / "routes_map.html",
    )
    make_comparison_bar(
        evaluations, output_path=results_dir / "comparison.png",
    )

    # 6. Print human-readable summary
    print("\n========== RESULTS ==========")
    print(f"{'Variant':<8}{'Distance (km)':>16}{'Fuel (L)':>12}{'CO2 (kg)':>12}")
    for v in variants:
        ev = evaluations[v]
        fuel_str = f"{ev.fuel_l:>12.3f}" if ev.ar_feasible else f"{'nan':>12}"
        co2_str = f"{ev.co2_kg:>12.3f}" if ev.ar_feasible else f"{'nan':>12}"
        print(
            f"{v:<8}"
            f"{ev.distance_m / 1000.0:>16.2f}"
            f"{fuel_str}"
            f"{co2_str}"
        )
        if ev.infeasible_legs:
            print(
                f"         ({ev.infeasible_legs} route leg(s) unreachable on AR — "
                f"fuel/CO2 not defined)"
            )
    print("\nAsymmetry penalties (vs. AR baseline):")
    for v, p in penalties.items():
        print(
            f"  {v}: distance {p['distance_pct']:+.2f}%, "
            f"CO2 {p['co2_pct']:+.2f}%"
        )
    print(f"\nOutputs written to: {results_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACVRP-CO2 experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--solver",
        choices=["ortools", "ga"],
        default="ortools",
        help="Which solver to use for each variant.",
    )
    args = parser.parse_args()
    run_experiment(args.config, solver=args.solver)


if __name__ == "__main__":
    main()

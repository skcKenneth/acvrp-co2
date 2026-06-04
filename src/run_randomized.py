"""
run_randomized.py
=================

Run the randomized ACVRP-CO2 robustness experiment with OR-Tools.

For each randomized city-instance pair, this script reconstructs the
depot/customer list from ``data/randomized/instances_<city>.csv``,
builds SE / SM / SR / AR distance matrices, solves CVRP under each
matrix variant with OR-Tools, then re-evaluates every route plan under
AR as the ground-truth road network.

Raw result CSV schema:
city,instance_id,solver,variant,reference_variant,num_customers,total_demand,
distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,
co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed

Usage:
    python -m src.run_randomized --config configs/randomized.yaml --dry-run
    python -m src.run_randomized --config configs/randomized.yaml --city macau --max-instances 1 --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from .data_loader import Customer
from .distance_matrix import build_all_matrices
from .emissions_model import EmissionsParams
from .evaluator import EvaluatedSolution, reevaluate_on_ground_truth


RAW_RESULT_FIELDNAMES = [
    "city",
    "instance_id",
    "solver",
    "variant",
    "reference_variant",
    "num_customers",
    "total_demand",
    "distance_m",
    "fuel_l",
    "co2_kg",
    "distance_penalty_pct",
    "fuel_penalty_pct",
    "co2_penalty_pct",
    "feasible",
    "infeasible_reason",
    "runtime_seconds",
    "seed",
]

REQUIRED_NODE_COLUMNS = {
    "city",
    "instance_id",
    "node_id",
    "role",
    "osmid",
    "lat",
    "lon",
    "x",
    "y",
    "demand",
    "sampling_seed",
}

REQUIRED_METADATA_COLUMNS = {
    "city",
    "instance_id",
    "num_customers",
    "total_demand",
    "vehicle_capacity",
    "num_vehicles",
    "sampling_seed",
    "status",
}

SOLVER_NAME = "OR-Tools"


@dataclass(frozen=True)
class InstanceData:
    """Validated randomized instance ready for matrix construction."""

    city: str
    instance_id: int
    customers: list[Customer]
    graph_node_ids: list[Any]
    demands: list[int]
    num_customers: int
    total_demand: int
    vehicle_capacity: int
    num_vehicles: int
    sampling_seed: int


@dataclass
class VariantOutcome:
    """Solve/evaluation output for one matrix variant."""

    variant: str
    routes: list[list[int]]
    route_loads: list[int]
    objective_value: int | None
    runtime_seconds: float
    feasible: bool
    infeasible_reason: str
    evaluation: EvaluatedSolution | None


def _project_path(raw_path: str | Path, project_root: Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the randomized experiment YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} did not parse to a mapping.")
    return cfg


def load_graphml(graph_path: Path) -> nx.MultiDiGraph:
    """Load a cached OSMnx GraphML road graph."""
    if not graph_path.exists():
        raise FileNotFoundError(
            f"GraphML file not found: {graph_path}. "
            "Check city.graph_cache in configs/randomized.yaml."
        )
    try:
        import osmnx as ox
    except ImportError as err:
        raise ImportError(
            "Missing dependency 'osmnx'. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from err
    return ox.load_graphml(graph_path)


def load_node_rows(path: Path) -> pd.DataFrame:
    """Load and validate one node-level randomized instance CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Randomized instance CSV not found: {path}. "
            "Run python -m src.random_instances first."
        )
    df = pd.read_csv(path, dtype={"city": str, "role": str, "osmid": str})
    missing = REQUIRED_NODE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and validate the randomized instance metadata CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Randomized metadata CSV not found: {path}. "
            "Run python -m src.random_instances first."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_METADATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def selected_city_names(cfg: dict[str, Any], requested_city: str) -> list[str]:
    """Return selected city names in config order."""
    all_cities = list(cfg.get("cities", {}).keys())
    if not all_cities:
        raise ValueError("Config does not define any cities.")
    if requested_city == "all":
        return all_cities
    if requested_city not in all_cities:
        raise ValueError(
            f"Unknown city {requested_city!r}. Available cities: "
            f"{', '.join(all_cities)}"
        )
    return [requested_city]


def instance_csv_path(cfg: dict[str, Any], project_root: Path, city: str) -> Path:
    """Return the node-level CSV path for one city."""
    instances_dir = _project_path(cfg["output"]["instances_dir"], project_root)
    return instances_dir / f"instances_{city}.csv"


def metadata_csv_path(cfg: dict[str, Any], project_root: Path) -> Path:
    """Return the metadata CSV path from config."""
    return _project_path(cfg["output"]["instance_metadata_csv"], project_root)


def output_paths(cfg: dict[str, Any], project_root: Path) -> tuple[Path, Path, Path]:
    """Return raw-results, routes, and matrices output paths."""
    output = cfg["output"]
    raw_results = _project_path(output["raw_results_csv"], project_root)
    routes_dir = _project_path(output["routes_dir"], project_root)
    matrices_dir = _project_path(output["matrices_dir"], project_root)
    return raw_results, routes_dir, matrices_dir


def route_json_path(
    routes_dir: Path,
    city: str,
    instance_id: int,
    variant: str,
) -> Path:
    """Return the JSON route-plan path for one solve."""
    return routes_dir / f"{city}_instance_{instance_id:03d}_ortools_{variant}.json"


def matrix_path(
    matrices_dir: Path,
    city: str,
    instance_id: int,
    variant: str,
) -> Path:
    """Return the matrix cache path for one matrix variant."""
    return matrices_dir / f"{city}_instance_{instance_id:03d}_{variant}.npy"


def coerce_graph_node_id(graph: nx.MultiDiGraph, osmid: Any) -> Any:
    """Map a CSV OSM ID value onto the node ID type used by the graph."""
    candidates: list[Any] = [osmid, str(osmid)]
    try:
        candidates.append(int(osmid))
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        if candidate in graph:
            return candidate
    raise KeyError(f"OSM node {osmid!r} from instance CSV is not in the graph.")


def validate_instance(
    *,
    city: str,
    instance_id: int,
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    graph: nx.MultiDiGraph,
    expected_customers: int,
) -> InstanceData:
    """Validate and reconstruct one randomized instance."""
    instance_rows = rows[rows["instance_id"] == instance_id].copy()
    if instance_rows.empty:
        raise ValueError(f"City {city!r} instance {instance_id} has no node rows.")

    metadata_rows = metadata[
        (metadata["city"] == city) & (metadata["instance_id"] == instance_id)
    ]
    if len(metadata_rows) != 1:
        raise ValueError(
            f"City {city!r} instance {instance_id} has {len(metadata_rows)} "
            "metadata rows; expected exactly 1."
        )
    meta = metadata_rows.iloc[0]

    depot_rows = instance_rows[instance_rows["role"] == "depot"]
    customer_rows = instance_rows[instance_rows["role"] == "customer"]
    if len(depot_rows) != 1:
        raise ValueError(
            f"City {city!r} instance {instance_id} has {len(depot_rows)} depot "
            "rows; expected exactly 1."
        )
    if len(customer_rows) != expected_customers:
        raise ValueError(
            f"City {city!r} instance {instance_id} has {len(customer_rows)} "
            f"customer rows; expected {expected_customers}."
        )
    if int(meta["num_customers"]) != expected_customers:
        raise ValueError(
            f"City {city!r} instance {instance_id} metadata num_customers="
            f"{meta['num_customers']}; expected {expected_customers}."
        )

    instance_rows["node_id"] = instance_rows["node_id"].astype(int)
    instance_rows = instance_rows.sort_values("node_id")
    expected_node_ids = list(range(expected_customers + 1))
    actual_node_ids = instance_rows["node_id"].tolist()
    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"City {city!r} instance {instance_id} node_id sequence is "
            f"{actual_node_ids}; expected {expected_node_ids}."
        )
    if instance_rows.iloc[0]["role"] != "depot":
        raise ValueError(
            f"City {city!r} instance {instance_id} node_id=0 is not the depot."
        )

    customers: list[Customer] = []
    graph_node_ids: list[Any] = []
    for row in instance_rows.itertuples(index=False):
        node_id = int(row.node_id)
        demand = int(row.demand)
        if node_id == 0 and demand != int(meta.get("depot_demand", demand)):
            # Older metadata does not include depot_demand; the node row is authoritative.
            pass
        customers.append(
            Customer(
                index=node_id,
                name=f"{city}_{instance_id}_{row.role}_{node_id}",
                lat=float(row.lat),
                lon=float(row.lon),
                demand=demand,
            )
        )
        graph_node_ids.append(coerce_graph_node_id(graph, row.osmid))

    demands = [customer.demand for customer in customers]
    if demands[0] != 0:
        raise ValueError(
            f"City {city!r} instance {instance_id} depot demand is {demands[0]}; "
            "expected 0."
        )
    total_demand = sum(demands)
    if total_demand != int(meta["total_demand"]):
        raise ValueError(
            f"City {city!r} instance {instance_id} total demand is "
            f"{total_demand}; metadata says {meta['total_demand']}."
        )

    return InstanceData(
        city=city,
        instance_id=instance_id,
        customers=customers,
        graph_node_ids=graph_node_ids,
        demands=demands,
        num_customers=expected_customers,
        total_demand=total_demand,
        vehicle_capacity=int(meta["vehicle_capacity"]),
        num_vehicles=int(meta["num_vehicles"]),
        sampling_seed=int(meta["sampling_seed"]),
    )


def city_instance_ids(
    rows: pd.DataFrame,
    metadata: pd.DataFrame,
    city: str,
    max_instances: int | None,
) -> list[int]:
    """Return selected instance IDs for a city."""
    row_ids = set(rows.loc[rows["city"] == city, "instance_id"].astype(int))
    meta_ids = set(metadata.loc[metadata["city"] == city, "instance_id"].astype(int))
    if row_ids != meta_ids:
        raise ValueError(
            f"City {city!r} instance IDs differ between node rows and metadata: "
            f"nodes={sorted(row_ids)}, metadata={sorted(meta_ids)}."
        )
    ids = sorted(row_ids)
    if max_instances is not None:
        if max_instances <= 0:
            raise ValueError("--max-instances must be a positive integer.")
        ids = ids[:max_instances]
    return ids


def emissions_params_for_city(cfg: dict[str, Any], city_cfg: dict[str, Any]) -> EmissionsParams:
    """Build EmissionsParams from randomized config, using city speed."""
    e = cfg["emissions"]
    return EmissionsParams(
        C1_distance=float(e.get("C1_distance", e.get("beta_distance"))),
        C2_time=float(e.get("C2_time", e.get("beta_time"))),
        C3_mass=float(e.get("C3_mass", e.get("beta_payload_distance"))),
        empty_mass_kg=float(e["empty_mass_kg"]),
        co2_per_litre=float(e["co2_per_litre"]),
        avg_speed_kmh=float(city_cfg["avg_speed_kmh"]),
    )


def route_loads(routes: list[list[int]], demands: list[int]) -> list[int]:
    """Compute delivered load for each route."""
    return [int(sum(demands[node] for node in route if node != 0)) for route in routes]


def nonfinite_reason(matrix: np.ndarray, variant: str) -> str:
    """Return a concise matrix non-finite warning, or an empty string."""
    bad = int(np.size(matrix) - np.isfinite(matrix).sum())
    if bad == 0:
        return ""
    return f"{variant} matrix has {bad} non-finite entries"


def solve_and_evaluate_variant(
    *,
    instance: InstanceData,
    variant: str,
    matrix: np.ndarray,
    ar_matrix: np.ndarray,
    params: EmissionsParams,
    cfg: dict[str, Any],
) -> VariantOutcome:
    """Solve one variant with OR-Tools and evaluate the route under AR."""
    objective = cfg.get("objective", {})
    ortools_cfg = cfg["ortools"]
    started = time.perf_counter()
    objective_value: int | None = None
    routes: list[list[int]] = []
    loads: list[int] = []
    evaluation: EvaluatedSolution | None = None
    feasible = False
    reason = ""

    matrix_warning = nonfinite_reason(matrix, variant)
    try:
        from .solver_ortools import solve_cvrp

        solution = solve_cvrp(
            distance_matrix=matrix,
            demands=instance.demands,
            vehicle_capacity=instance.vehicle_capacity,
            num_vehicles=instance.num_vehicles,
            depot=0,
            params=params,
            alpha_distance=float(objective.get("alpha_distance", 0.5)),
            alpha_emissions=float(objective.get("alpha_emissions", 0.5)),
            time_limit_s=int(ortools_cfg["time_limit_seconds"]),
            first_solution_strategy=ortools_cfg["first_solution_strategy"],
            local_search_metaheuristic=ortools_cfg["local_search_metaheuristic"],
        )
        routes = solution.routes
        loads = route_loads(routes, instance.demands)
        objective_value = int(solution.objective_value)
        evaluation = reevaluate_on_ground_truth(
            variant=variant,
            routes=routes,
            ar_matrix=ar_matrix,
            demands=instance.demands,
            params=params,
        )
        feasible = bool(evaluation.ar_feasible)
        if not feasible:
            reason = f"unreachable_ar_legs={evaluation.infeasible_legs}"
        elif matrix_warning:
            # Keep successful rows clean, but retain the warning in route JSON.
            reason = ""
    except Exception as err:
        feasible = False
        reason = str(err)
        if matrix_warning:
            reason = f"{matrix_warning}; {reason}"

    runtime_seconds = time.perf_counter() - started
    return VariantOutcome(
        variant=variant,
        routes=routes,
        route_loads=loads,
        objective_value=objective_value,
        runtime_seconds=runtime_seconds,
        feasible=feasible,
        infeasible_reason=reason,
        evaluation=evaluation,
    )


def penalty(
    value: float | None,
    reference: float | None,
) -> float | None:
    """Return percentage penalty against reference, or None if undefined."""
    if value is None or reference is None:
        return None
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return None
    return 100.0 * (value - reference) / reference


def raw_row(
    *,
    instance: InstanceData,
    outcome: VariantOutcome,
    reference: VariantOutcome | None,
    reference_variant: str,
) -> dict[str, Any]:
    """Convert a variant outcome to a raw-results CSV row."""
    ev = outcome.evaluation
    ref_ev = reference.evaluation if reference and reference.feasible else None
    distance_m = ev.distance_m if ev is not None else None
    fuel_l = ev.fuel_l if ev is not None and outcome.feasible else None
    co2_kg = ev.co2_kg if ev is not None and outcome.feasible else None
    ref_distance = ref_ev.distance_m if ref_ev is not None else None
    ref_fuel = ref_ev.fuel_l if ref_ev is not None else None
    ref_co2 = ref_ev.co2_kg if ref_ev is not None else None

    return {
        "city": instance.city,
        "instance_id": instance.instance_id,
        "solver": SOLVER_NAME,
        "variant": outcome.variant,
        "reference_variant": reference_variant,
        "num_customers": instance.num_customers,
        "total_demand": instance.total_demand,
        "distance_m": distance_m,
        "fuel_l": fuel_l,
        "co2_kg": co2_kg,
        "distance_penalty_pct": penalty(distance_m, ref_distance),
        "fuel_penalty_pct": penalty(fuel_l, ref_fuel),
        "co2_penalty_pct": penalty(co2_kg, ref_co2),
        "feasible": outcome.feasible,
        "infeasible_reason": outcome.infeasible_reason,
        "runtime_seconds": outcome.runtime_seconds,
        "seed": instance.sampling_seed,
    }


def write_route_json(
    path: Path,
    instance: InstanceData,
    outcome: VariantOutcome,
) -> None:
    """Write one route-plan JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "city": instance.city,
        "instance_id": instance.instance_id,
        "solver": SOLVER_NAME,
        "variant": outcome.variant,
        "routes": outcome.routes,
        "route_loads": outcome.route_loads,
        "objective_value": outcome.objective_value,
        "runtime_seconds": outcome.runtime_seconds,
        "feasible": outcome.feasible,
        "infeasible_reason": outcome.infeasible_reason,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_raw_results(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the raw long-format result CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def ensure_outputs_can_be_written(paths: Iterable[Path], overwrite: bool) -> None:
    """Fail before running if any intended output already exists."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output file(s) already exist. Re-run with --overwrite to replace:\n"
            f"  {joined}"
        )


def planned_output_files(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    selected: dict[str, list[int]],
) -> list[Path]:
    """List files this run would write."""
    raw_results, routes_dir, matrices_dir = output_paths(cfg, project_root)
    variants = list(cfg["experiment"]["variants"])
    paths = [raw_results]
    if bool(cfg["output"].get("save_routes", True)):
        for city, instance_ids in selected.items():
            for instance_id in instance_ids:
                for variant in variants:
                    paths.append(route_json_path(routes_dir, city, instance_id, variant))
    if bool(cfg["output"].get("save_matrices", False)):
        for city, instance_ids in selected.items():
            for instance_id in instance_ids:
                for variant in variants:
                    paths.append(matrix_path(matrices_dir, city, instance_id, variant))
    return paths


def save_matrices(
    matrices_dir: Path,
    city: str,
    instance_id: int,
    matrices: dict[str, np.ndarray],
) -> None:
    """Save matrices as NumPy arrays."""
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for variant, matrix in matrices.items():
        np.save(matrix_path(matrices_dir, city, instance_id, variant), matrix)


def print_dry_run_summary(
    cfg: dict[str, Any],
    project_root: Path,
    city_names: list[str],
    max_instances: int | None,
) -> None:
    """Load randomized CSVs and print available instance summaries."""
    metadata = load_metadata(metadata_csv_path(cfg, project_root))
    expected_customers = int(cfg["experiment"]["num_customers"])
    print("Dry run only; no routes were solved and no files were written.")
    for city in city_names:
        rows = load_node_rows(instance_csv_path(cfg, project_root, city))
        ids = city_instance_ids(rows, metadata, city, max_instances=None)
        selected_ids = ids[:max_instances] if max_instances is not None else ids
        print(f"\nCity: {city}")
        print(f"  Available instances: {len(ids)}")
        print(f"  Selected instances for this command: {len(selected_ids)}")
        if not ids:
            raise ValueError(f"City {city!r} has no available instances.")
        first_id = selected_ids[0] if selected_ids else ids[0]
        first_rows = rows[rows["instance_id"] == first_id].sort_values("node_id")
        depot_count = int((first_rows["role"] == "depot").sum())
        customer_count = int((first_rows["role"] == "customer").sum())
        meta = metadata[
            (metadata["city"] == city) & (metadata["instance_id"] == first_id)
        ].iloc[0]
        print(
            f"  First selected instance: {first_id} "
            f"({depot_count} depot, {customer_count} customers)"
        )
        print(
            f"  Metadata: total_demand={int(meta['total_demand'])}, "
            f"sampling_seed={int(meta['sampling_seed'])}, "
            f"vehicle_capacity={int(meta['vehicle_capacity'])}, "
            f"num_vehicles={int(meta['num_vehicles'])}"
        )
        if depot_count != 1 or customer_count != expected_customers:
            raise ValueError(
                f"City {city!r} first selected instance has {depot_count} depot "
                f"rows and {customer_count} customer rows; expected 1 and "
                f"{expected_customers}."
            )


def run(
    *,
    config_path: str | Path,
    city: str,
    max_instances: int | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Run the randomized OR-Tools experiment."""
    config_path = Path(config_path)
    project_root = config_path.resolve().parent.parent
    cfg = load_config(config_path)
    city_names = selected_city_names(cfg, city)

    if dry_run:
        print_dry_run_summary(cfg, project_root, city_names, max_instances)
        return

    metadata = load_metadata(metadata_csv_path(cfg, project_root))
    city_rows = {
        city_name: load_node_rows(instance_csv_path(cfg, project_root, city_name))
        for city_name in city_names
    }
    selected = {
        city_name: city_instance_ids(city_rows[city_name], metadata, city_name, max_instances)
        for city_name in city_names
    }
    ensure_outputs_can_be_written(
        planned_output_files(cfg=cfg, project_root=project_root, selected=selected),
        overwrite=overwrite,
    )

    raw_results, routes_dir, matrices_dir = output_paths(cfg, project_root)
    variants = list(cfg["experiment"]["variants"])
    reference_variant = str(cfg["experiment"].get("reference_variant", "AR"))
    expected_customers = int(cfg["experiment"]["num_customers"])
    save_route_files = bool(cfg["output"].get("save_routes", True))
    save_matrix_files = bool(cfg["output"].get("save_matrices", False))
    all_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for city_name in city_names:
        city_cfg = cfg["cities"][city_name]
        graph = load_graphml(_project_path(city_cfg["graph_cache"], project_root))
        params = emissions_params_for_city(cfg, city_cfg)

        for instance_id in selected[city_name]:
            instance = validate_instance(
                city=city_name,
                instance_id=instance_id,
                rows=city_rows[city_name],
                metadata=metadata,
                graph=graph,
                expected_customers=expected_customers,
            )
            print(f"Running {city_name} instance {instance_id}...")
            matrices = build_all_matrices(
                instance.customers,
                graph,
                instance.graph_node_ids,
            )
            for variant, matrix in matrices.items():
                reason = nonfinite_reason(matrix, variant)
                if reason:
                    warnings.append(
                        f"{city_name} instance {instance_id}: {reason}"
                    )
            if save_matrix_files:
                save_matrices(matrices_dir, city_name, instance_id, matrices)

            outcomes: dict[str, VariantOutcome] = {}
            for variant in variants:
                outcome = solve_and_evaluate_variant(
                    instance=instance,
                    variant=variant,
                    matrix=matrices[variant],
                    ar_matrix=matrices["AR"],
                    params=params,
                    cfg=cfg,
                )
                outcomes[variant] = outcome
                if save_route_files:
                    write_route_json(
                        route_json_path(routes_dir, city_name, instance_id, variant),
                        instance,
                        outcome,
                    )

            reference = outcomes.get(reference_variant)
            if reference is None:
                raise ValueError(
                    f"Reference variant {reference_variant!r} was not run for "
                    f"{city_name} instance {instance_id}."
                )
            if not reference.feasible:
                warnings.append(
                    f"{city_name} instance {instance_id}: AR reference infeasible "
                    f"({reference.infeasible_reason})"
                )

            for variant in variants:
                all_rows.append(
                    raw_row(
                        instance=instance,
                        outcome=outcomes[variant],
                        reference=reference,
                        reference_variant=reference_variant,
                    )
                )

    write_raw_results(raw_results, all_rows)
    print(f"Wrote {len(all_rows)} raw result rows to {raw_results}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run randomized ACVRP-CO2 OR-Tools experiments.",
    )
    parser.add_argument(
        "--config",
        default="configs/randomized.yaml",
        help="Path to the randomized experiment YAML config.",
    )
    parser.add_argument(
        "--city",
        choices=["macau", "hongkong", "all"],
        default="all",
        help="City to run. Defaults to all cities.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit to the first K instances per selected city.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config/CSVs and print summaries without solving or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of raw results, route JSON, and matrix files.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        city=args.city,
        max_instances=args.max_instances,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

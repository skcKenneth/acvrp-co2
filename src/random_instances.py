"""
random_instances.py
===================

Generate reproducible randomized customer sets for the ACVRP-CO2
robustness experiment.

The script reads ``configs/randomized.yaml``, loads each city's cached
directed OSMnx GraphML road graph, snaps the fixed depot coordinates to
the nearest graph node, samples customer nodes that are reachable from
the depot and can also reach the depot, and writes:

* data/randomized/instances_macau.csv
* data/randomized/instances_hongkong.csv
* data/randomized/instance_metadata.csv

Node-level CSV schema:
city,instance_id,node_id,role,osmid,lat,lon,x,y,demand,sampling_seed

Metadata CSV schema:
city,instance_id,num_customers,total_demand,vehicle_capacity,num_vehicles,sampling_seed,status

Usage:
    python -m src.random_instances --config configs/randomized.yaml
    python -m src.random_instances --config configs/randomized.yaml --dry-run
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import yaml


NODE_FIELDNAMES = [
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
]

METADATA_FIELDNAMES = [
    "city",
    "instance_id",
    "num_customers",
    "total_demand",
    "vehicle_capacity",
    "num_vehicles",
    "sampling_seed",
    "status",
]


@dataclass(frozen=True)
class CityGenerationResult:
    """Generated rows and sampling diagnostics for one city."""

    city: str
    graph_nodes: int
    graph_edges: int
    depot_osmid: Any
    candidate_count: int
    node_rows: list[dict[str, Any]]
    metadata_rows: list[dict[str, Any]]


def _project_path(raw_path: str | Path, project_root: Path) -> Path:
    """Resolve a config path that is expected to be relative to repo root."""
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
    """Load an OSMnx GraphML road graph, failing clearly if unavailable."""
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


def snap_depot_node(graph: nx.MultiDiGraph, city_cfg: dict[str, Any]) -> Any:
    """Snap the configured fixed depot coordinates to the nearest graph node."""
    depot = city_cfg.get("depot", {})
    try:
        depot_lat = float(depot["latitude"])
        depot_lon = float(depot["longitude"])
    except KeyError as err:
        raise ValueError(f"City depot config is missing required key: {err}") from err

    best_node = None
    best_score = float("inf")
    for node_id, node_data in graph.nodes(data=True):
        if "x" not in node_data or "y" not in node_data:
            raise ValueError(f"Graph node {node_id!r} is missing x/y coordinates.")
        dx = float(node_data["x"]) - depot_lon
        dy = float(node_data["y"]) - depot_lat
        score = dx * dx + dy * dy
        if score < best_score:
            best_node = node_id
            best_score = score

    if best_node is None:
        raise ValueError("Cannot snap depot: graph contains no nodes.")
    return best_node


def reachable_candidate_nodes(
    graph: nx.MultiDiGraph,
    depot_node: Any,
    require_from_depot: bool,
    require_to_depot: bool,
) -> list[Any]:
    """
    Return graph nodes eligible for customer sampling under reachability rules.

    With the default randomized config, this is exactly the depot's directed
    strongly connected component excluding the depot.
    """
    all_nodes = set(graph.nodes)
    candidates = all_nodes

    if require_from_depot:
        from_depot = set(
            nx.single_source_dijkstra_path_length(
                graph,
                depot_node,
                weight="length",
            ).keys()
        )
        candidates &= from_depot

    if require_to_depot:
        reverse_graph = graph.reverse(copy=False)
        to_depot = set(
            nx.single_source_dijkstra_path_length(
                reverse_graph,
                depot_node,
                weight="length",
            ).keys()
        )
        candidates &= to_depot

    candidates.discard(depot_node)
    return sorted(candidates, key=lambda node: str(node))


def _node_coordinates(graph: nx.MultiDiGraph, osmid: Any) -> tuple[float, float]:
    """Return ``(x, y)`` coordinates for a graph node or fail clearly."""
    node_data = graph.nodes[osmid]
    if "x" not in node_data or "y" not in node_data:
        raise ValueError(f"Graph node {osmid!r} is missing x/y coordinates.")
    return float(node_data["x"]), float(node_data["y"])


def _node_row(
    *,
    city: str,
    instance_id: int,
    node_id: int,
    role: str,
    osmid: Any,
    graph: nx.MultiDiGraph,
    demand: int,
    sampling_seed: int,
) -> dict[str, Any]:
    """Build one node-level CSV row."""
    x, y = _node_coordinates(graph, osmid)
    return {
        "city": city,
        "instance_id": instance_id,
        "node_id": node_id,
        "role": role,
        "osmid": str(osmid),
        "lat": y,
        "lon": x,
        "x": x,
        "y": y,
        "demand": demand,
        "sampling_seed": sampling_seed,
    }


def _instance_seed(base_seed: int, city_index: int, instance_id: int) -> int:
    """Derive a stable per-instance seed from the base experiment seed."""
    return int(base_seed + city_index * 100_000 + instance_id)


def generate_city_instances(
    *,
    city: str,
    city_index: int,
    city_cfg: dict[str, Any],
    cfg: dict[str, Any],
    project_root: Path,
    dry_run: bool,
) -> CityGenerationResult:
    """Generate all randomized instance rows for one city."""
    graph_path = _project_path(city_cfg["graph_cache"], project_root)
    graph = load_graphml(graph_path)
    depot_node = snap_depot_node(graph, city_cfg)

    sampling_cfg = cfg.get("sampling", {})
    require_from_depot = bool(
        sampling_cfg.get("require_depot_to_customer_reachable", True)
    )
    require_to_depot = bool(
        sampling_cfg.get("require_customer_to_depot_reachable", True)
    )
    candidates = reachable_candidate_nodes(
        graph,
        depot_node,
        require_from_depot=require_from_depot,
        require_to_depot=require_to_depot,
    )

    experiment = cfg["experiment"]
    num_customers = int(experiment["num_customers"])
    instances_per_city = 1 if dry_run else int(experiment["instances_per_city"])
    if len(candidates) < num_customers:
        raise ValueError(
            f"City {city!r} has only {len(candidates)} reachable candidate "
            f"nodes, but num_customers={num_customers}. Check graph coverage "
            "or sampling reachability settings."
        )

    demand_cfg = cfg["demand"]
    demand_low = int(demand_cfg["low"])
    demand_high = int(demand_cfg["high"])
    depot_demand = int(demand_cfg.get("depot_demand", 0))
    if demand_low > demand_high:
        raise ValueError(
            f"Invalid demand range: low={demand_low} > high={demand_high}."
        )

    fleet_cfg = cfg["fleet"]
    node_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    base_seed = int(experiment["seed"])

    for instance_id in range(instances_per_city):
        sampling_seed = _instance_seed(base_seed, city_index, instance_id)
        rng = np.random.default_rng(sampling_seed)
        sampled_indices = rng.choice(
            len(candidates),
            size=num_customers,
            replace=False,
        )
        sampled_nodes = [candidates[int(index)] for index in sampled_indices]
        demands = rng.integers(
            demand_low,
            demand_high + 1,
            size=num_customers,
        ).astype(int)

        node_rows.append(
            _node_row(
                city=city,
                instance_id=instance_id,
                node_id=0,
                role="depot",
                osmid=depot_node,
                graph=graph,
                demand=depot_demand,
                sampling_seed=sampling_seed,
            )
        )
        for offset, (osmid, demand) in enumerate(
            zip(sampled_nodes, demands.tolist()),
            start=1,
        ):
            node_rows.append(
                _node_row(
                    city=city,
                    instance_id=instance_id,
                    node_id=offset,
                    role="customer",
                    osmid=osmid,
                    graph=graph,
                    demand=int(demand),
                    sampling_seed=sampling_seed,
                )
            )

        metadata_rows.append(
            {
                "city": city,
                "instance_id": instance_id,
                "num_customers": num_customers,
                "total_demand": int(demands.sum()),
                "vehicle_capacity": int(fleet_cfg["vehicle_capacity"]),
                "num_vehicles": int(fleet_cfg["num_vehicles"]),
                "sampling_seed": sampling_seed,
                "status": "generated",
            }
        )

    return CityGenerationResult(
        city=city,
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        depot_osmid=depot_node,
        candidate_count=len(candidates),
        node_rows=node_rows,
        metadata_rows=metadata_rows,
    )


def selected_city_names(cfg: dict[str, Any], requested_city: str) -> list[str]:
    """Return city names selected by the CLI, preserving config order."""
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


def output_paths_for(
    cfg: dict[str, Any],
    project_root: Path,
    cities: Iterable[str],
) -> tuple[dict[str, Path], Path]:
    """Return city instance CSV paths and combined metadata CSV path."""
    output_cfg = cfg["output"]
    instances_dir = _project_path(output_cfg["instances_dir"], project_root)
    instance_paths = {
        city: instances_dir / f"instances_{city}.csv"
        for city in cities
    }
    metadata_path = _project_path(output_cfg["instance_metadata_csv"], project_root)
    return instance_paths, metadata_path


def ensure_outputs_can_be_written(paths: Iterable[Path], overwrite: bool) -> None:
    """Stop before writing if any output exists and overwrite is not enabled."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output file(s) already exist. Re-run with --overwrite to replace:\n"
            f"  {joined}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with a fixed schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_dry_run_report(results: list[CityGenerationResult]) -> None:
    """Print candidate counts and sample output rows without writing files."""
    print("Dry run only; no files were written.")
    for result in results:
        print(f"\nCity: {result.city}")
        print(
            f"  Graph: {result.graph_nodes} nodes, "
            f"{result.graph_edges} edges"
        )
        print(f"  Depot OSM node: {result.depot_osmid}")
        print(f"  Reachable candidate customer nodes: {result.candidate_count}")
        print(f"  Generated node rows: {len(result.node_rows)}")
        print(f"  Generated metadata rows: {len(result.metadata_rows)}")
        print("  Sample metadata row:")
        print(f"    {result.metadata_rows[0]}")
        print("  Sample node rows:")
        for row in result.node_rows[:5]:
            print(f"    {row}")


def run(config_path: str | Path, city: str, dry_run: bool, overwrite: bool) -> None:
    """Generate randomized instances according to the YAML config."""
    config_path = Path(config_path)
    project_root = config_path.resolve().parent.parent
    cfg = load_config(config_path)

    city_names = selected_city_names(cfg, city)
    all_city_names = list(cfg["cities"].keys())
    instance_paths, metadata_path = output_paths_for(cfg, project_root, city_names)

    if not dry_run:
        ensure_outputs_can_be_written(
            list(instance_paths.values()) + [metadata_path],
            overwrite=overwrite,
        )

    results: list[CityGenerationResult] = []
    for city_name in city_names:
        city_index = all_city_names.index(city_name)
        results.append(
            generate_city_instances(
                city=city_name,
                city_index=city_index,
                city_cfg=cfg["cities"][city_name],
                cfg=cfg,
                project_root=project_root,
                dry_run=dry_run,
            )
        )

    if dry_run:
        print_dry_run_report(results)
        return

    all_metadata_rows: list[dict[str, Any]] = []
    for result in results:
        write_csv(instance_paths[result.city], result.node_rows, NODE_FIELDNAMES)
        all_metadata_rows.extend(result.metadata_rows)
    write_csv(metadata_path, all_metadata_rows, METADATA_FIELDNAMES)

    for result in results:
        print(
            f"Wrote {len(result.node_rows)} node rows for {result.city} to "
            f"{instance_paths[result.city]}"
        )
    print(f"Wrote {len(all_metadata_rows)} metadata rows to {metadata_path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate randomized ACVRP-CO2 customer instances.",
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
        help="City to generate. Defaults to all cities.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate one preview instance per selected city without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output CSV files.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        city=args.city,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

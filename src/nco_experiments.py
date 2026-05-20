"""
nco_experiments.py
==================

End-to-end script that trains the asymmetric NCO policy and evaluates
it against the classical baselines on the same set of OSM-derived
ACVRP instances.

Typical usage
-------------
    # 1) Train on synthetic asymmetric instances
    python -m src.nco_experiments --mode train --config configs/nco_config.yaml

    # 2) Evaluate the trained model on OSM held-out instances
    python -m src.nco_experiments --mode eval  --config configs/nco_config.yaml \
            --checkpoint models/acvrp_policy_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baselines.solver_pyvrp import solve_cvrp_pyvrp
from .data_loader import download_road_graph
from .emissions_model import EmissionsParams, routes_to_metrics
from .experiments import emissions_params_from_config
from .nco.dataset import (
    StreamingSyntheticDataset,
    SyntheticConfig,
    osm_dataset,
    synthetic_dataset,
)
from .nco.inference import solve_with_policy
from .nco.model import ACVRPPolicy
from .nco.trainer import TrainingConfig, evaluate, train
from .solver_ortools import solve_cvrp


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_policy(cfg: dict) -> ACVRPPolicy:
    m = cfg["model"]
    return ACVRPPolicy(
        node_feature_dim=3,
        edge_feature_dim=4,
        embed_dim=m["embed_dim"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_dim=m["ffn_dim"],
        tanh_clip=m["tanh_clip"],
    )


def run_training(cfg: dict) -> None:
    syn_cfg = SyntheticConfig(
        num_customers=cfg["data"]["num_customers"],
        capacity=cfg["data"]["capacity"],
        demand_low=cfg["data"]["demand_low"],
        demand_high=cfg["data"]["demand_high"],
        asymmetry_factor_max=cfg["data"]["asymmetry_factor_max"],
    )
    params = emissions_params_from_config(cfg)
    train_stream = StreamingSyntheticDataset(syn_cfg, params, seed=cfg["seed"])

    # Held-out synthetic validation set with a different seed
    val_set = synthetic_dataset(
        syn_cfg, params, num_instances=cfg["training"]["val_size"], seed=cfg["seed"] + 1,
    )

    policy = build_policy(cfg)
    train_cfg = TrainingConfig(
        batch_size=cfg["training"]["batch_size"],
        n_starts=cfg["training"]["n_starts"],
        epochs=cfg["training"]["epochs"],
        steps_per_epoch=cfg["training"]["steps_per_epoch"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        device=cfg["training"]["device"],
        save_prefix=cfg["training"]["save_prefix"],
    )
    log = train(policy, train_stream, train_cfg, eval_set=val_set)

    # Dump training history
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    with open(
        f"{cfg['training']['checkpoint_dir']}/{cfg['training']['save_prefix']}_log.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "epochs": log.epochs,
                "train_costs": log.train_costs,
                "train_losses": log.train_losses,
                "val_costs": log.val_costs,
            },
            f,
            indent=2,
        )


def run_evaluation(cfg: dict, checkpoint_path: str) -> None:
    """
    Evaluate three solvers (OR-Tools, PyVRP, our policy) on the same
    OSM-derived held-out instances and report mean cost per solver.
    """
    params = emissions_params_from_config(cfg)
    device = cfg["training"]["device"]

    policy = build_policy(cfg)
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy.to(device)
    policy.eval()

    # Load each city in turn and evaluate the held-out OSM instance set.
    results = []
    for city in cfg["evaluation"]["cities"]:
        graph = download_road_graph(
            centre_lat=city["centre_lat"],
            centre_lon=city["centre_lon"],
            radius_m=city["radius_m"],
            network_type="drive",
        )
        instances = osm_dataset(
            graph=graph,
            num_customers=cfg["data"]["num_customers"],
            capacity=cfg["data"]["capacity"],
            params=params,
            num_instances=cfg["evaluation"]["instances_per_city"],
            city_name=city["name"],
            seed=cfg["seed"] + 100,
        )

        for inst_idx, inst in enumerate(instances):
            row = {"city": city["name"], "instance": inst_idx}

            # 1) Neural policy (greedy and POMO sampling)
            policy_routes_greedy = solve_with_policy(
                policy, inst, device=device, mode="greedy"
            )
            row["nco_greedy_distance"] = routes_to_metrics(
                policy_routes_greedy, inst.distance, inst.demands.tolist(), params
            )["distance_m"]

            policy_routes_pomo = solve_with_policy(
                policy, inst, device=device, mode="pomo",
                n_samples=cfg["evaluation"]["pomo_samples"],
            )
            row["nco_pomo_distance"] = routes_to_metrics(
                policy_routes_pomo, inst.distance, inst.demands.tolist(), params
            )["distance_m"]

            # 2) OR-Tools
            ortools_sol = solve_cvrp(
                distance_matrix=inst.distance,
                demands=inst.demands.tolist(),
                vehicle_capacity=inst.capacity,
                num_vehicles=cfg["evaluation"]["num_vehicles"],
                depot=inst.depot_index,
                params=params,
                time_limit_s=cfg["evaluation"]["ortools_time_s"],
            )
            row["ortools_distance"] = routes_to_metrics(
                ortools_sol.routes, inst.distance, inst.demands.tolist(), params
            )["distance_m"]

            # 3) PyVRP (HGS)
            pyvrp_sol = solve_cvrp_pyvrp(
                distance_matrix=inst.distance,
                demands=inst.demands.tolist(),
                vehicle_capacity=inst.capacity,
                num_vehicles=cfg["evaluation"]["num_vehicles"],
                depot=inst.depot_index,
                time_limit_s=cfg["evaluation"]["pyvrp_time_s"],
            )
            row["pyvrp_distance"] = routes_to_metrics(
                pyvrp_sol.routes, inst.distance, inst.demands.tolist(), params
            )["distance_m"]

            results.append(row)
            print(
                f"{city['name']:>16} #{inst_idx:02d} | "
                f"NCO-greedy {row['nco_greedy_distance']/1000:6.2f} km | "
                f"NCO-POMO   {row['nco_pomo_distance']/1000:6.2f} km | "
                f"OR-Tools   {row['ortools_distance']/1000:6.2f} km | "
                f"PyVRP      {row['pyvrp_distance']/1000:6.2f} km"
            )

    out_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nco_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Compute per-solver mean and optimality gap relative to PyVRP.
    arr = {k: np.array([r[k] for r in results]) for k in [
        "nco_greedy_distance", "nco_pomo_distance",
        "ortools_distance", "pyvrp_distance",
    ]}
    print("\n--- Aggregate results (all cities) ---")
    for k, v in arr.items():
        gap = 100.0 * (v.mean() - arr["pyvrp_distance"].mean()) / arr["pyvrp_distance"].mean()
        print(f"  {k:30s}  mean = {v.mean() / 1000:6.2f} km  (gap vs PyVRP {gap:+.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ACVRP NCO training and evaluation.")
    parser.add_argument("--config", default="configs/nco_config.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a trained .pt file (for eval mode).")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        run_training(cfg)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required in eval mode.")
        run_evaluation(cfg, args.checkpoint)


if __name__ == "__main__":
    main()

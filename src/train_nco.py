"""
train_nco.py
============

Command-line entry point for training the two neural policies:

  matnet    : the bidirectional edge-attention encoder + capacity-aware
              decoder (our main contribution, "MatNet-CVRP")
  baseline  : a coord-only Attention Model in the style of Kool 2019,
              kept for like-for-like comparison

Usage
-----
    python -m src.train_nco --policy matnet --config configs/train.yaml
    python -m src.train_nco --policy baseline --config configs/train.yaml

The script reads training hyperparameters from a YAML file, builds the
streaming synthetic dataset, optionally adds a small fixed OSM-derived
evaluation set, and writes checkpoints to `models/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

from .emissions_model import EmissionsParams


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_params(cfg: dict) -> EmissionsParams:
    e = cfg["emissions"]
    return EmissionsParams(
        C1_distance=e["C1_distance"],
        C2_time=e["C2_time"],
        C3_mass=e["C3_mass"],
        empty_mass_kg=e["empty_mass_kg"],
        co2_per_litre=e["co2_per_litre"],
        avg_speed_kmh=e["avg_speed_kmh"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACVRP NCO policy.")
    parser.add_argument(
        "--policy", choices=["matnet", "baseline"], required=True,
        help="Which architecture to train.",
    )
    parser.add_argument(
        "--config", default="configs/train.yaml",
        help="Path to the training YAML.",
    )
    parser.add_argument(
        "--osm-eval", action="store_true",
        help="If set, also build a small OSM-derived evaluation set "
             "from the region defined in the config.",
    )
    args = parser.parse_args()

    # Lazy imports so that the help message works even without torch.
    import torch  # noqa: F401
    from .nco.dataset import (
        StreamingSyntheticDataset,
        SyntheticConfig,
        osm_dataset,
    )
    from .nco.model import ACVRPPolicy
    from .nco.baseline_am import CoordOnlyACVRPPolicy
    from .nco.trainer import TrainingConfig, train

    cfg = _load_yaml(args.config)
    params = _make_params(cfg)

    # ---- Synthetic streaming dataset -----------------------------------
    syn_cfg = SyntheticConfig(**cfg["synthetic"])
    stream = StreamingSyntheticDataset(syn_cfg, params, seed=cfg.get("seed", 0))

    # ---- Optional OSM eval set -----------------------------------------
    eval_set: List = []
    if args.osm_eval and "osm_eval" in cfg:
        from .data_loader import download_road_graph
        eo = cfg["osm_eval"]
        print(f"Downloading OSM graph for evaluation set "
              f"@({eo['centre_lat']}, {eo['centre_lon']}), r={eo['radius_m']} m...")
        g = download_road_graph(
            centre_lat=eo["centre_lat"],
            centre_lon=eo["centre_lon"],
            radius_m=eo["radius_m"],
            network_type=eo.get("network_type", "drive"),
        )
        eval_set = osm_dataset(
            graph=g,
            num_customers=eo["num_customers"],
            capacity=eo["capacity"],
            params=params,
            num_instances=eo["num_instances"],
            city_name=eo.get("city_name", "osm_eval"),
            seed=cfg.get("seed", 0) + 1,
        )
        print(f"Built {len(eval_set)} OSM evaluation instances.")

    # ---- Policy --------------------------------------------------------
    model_cfg = cfg["model"]
    n_customers = cfg["synthetic"]["num_customers"]

    if args.policy == "matnet":
        policy = ACVRPPolicy(**model_cfg)
        default_prefix = f"matnet_cvrp_n{n_customers}"
    else:
        # CoordOnlyACVRPPolicy does not accept edge_feature_dim.
        coord_cfg = {k: v for k, v in model_cfg.items() if k != "edge_feature_dim"}
        policy = CoordOnlyACVRPPolicy(**coord_cfg)
        default_prefix = f"baseline_am_n{n_customers}"

    # Allow the YAML to override the default checkpoint prefix.
    # This lets two different configs (e.g. N=20 vs N=50) coexist in
    # the same models/ directory without overwriting each other.
    save_prefix = cfg.get("training", {}).get("save_prefix", default_prefix)

    # ---- Trainer config ------------------------------------------------
    tcfg_dict = dict(cfg["training"])
    tcfg_dict["save_prefix"] = save_prefix
    tcfg = TrainingConfig(**tcfg_dict)
    Path(tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training {args.policy} policy ({sum(p.numel() for p in policy.parameters()):,} params)")
    log = train(
        policy=policy,
        train_stream=stream,
        cfg=tcfg,
        eval_set=eval_set if eval_set else None,
    )

    # Save the run log as JSON so that downstream plotting tools can use it.
    import json
    log_path = Path(tcfg.checkpoint_dir) / f"{save_prefix}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": log.epochs,
            "train_costs": log.train_costs,
            "train_losses": log.train_losses,
            "val_costs": log.val_costs,
        }, f, indent=2)
    print(f"Wrote training log to {log_path}")


if __name__ == "__main__":
    main()

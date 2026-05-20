"""
sweep_pareto.py
===============

Train one model per CO2 weight to produce a Pareto front of
distance vs CO2 trade-offs. This is the experimental backbone of
research question RQ5.

For each weight w in {0.0, 0.25, 0.50, 0.75, 1.0}:
  - Load the configured training YAML
  - Override model.cost_mode = "blend" and model.co2_weight = w
  - Train the policy with save_prefix = f"matnet_co2w{int(w*100):03d}"
  - Save final and best checkpoints

Usage
-----
    python -m src.sweep_pareto --config configs/train_n50.yaml

Total wall-clock on an RTX 5070 Ti: roughly 5 * (single-model time) = 15-25 h.
You may want to break this into chunks: run 1-2 weights overnight at a time.

After all 5 models have trained, run:
    python -m src.experiments_full ... per checkpoint ...
and plot the resulting (distance_m, co2_kg) pairs as the Pareto front.
"""
from __future__ import annotations

import argparse
import copy
import yaml

from .train_nco import _load_yaml, _make_params  # reuse helpers


CO2_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto-front sweep over CO2 weights.")
    parser.add_argument("--config", default="configs/train_n50.yaml")
    parser.add_argument(
        "--weights", nargs="+", type=float, default=CO2_WEIGHTS,
        help=(
            "Override the default weight list. e.g. --weights 0.25 0.75 "
            "to skip the endpoints. Order doesn't matter."
        ),
    )
    parser.add_argument(
        "--osm-eval", action="store_true",
        help="Pass through to the trainer's --osm-eval flag.",
    )
    args = parser.parse_args()

    import torch  # noqa: F401
    from .nco.dataset import StreamingSyntheticDataset, SyntheticConfig
    from .nco.model import ACVRPPolicy
    from .nco.trainer import TrainingConfig, train

    base_cfg = _load_yaml(args.config)
    params = _make_params(base_cfg)

    syn_cfg = SyntheticConfig(**base_cfg["synthetic"])
    stream = StreamingSyntheticDataset(syn_cfg, params, seed=base_cfg.get("seed", 0))

    eval_set = None
    if args.osm_eval and "osm_eval" in base_cfg:
        from .data_loader import download_road_graph
        from .nco.dataset import osm_dataset
        eo = base_cfg["osm_eval"]
        g = download_road_graph(
            centre_lat=eo["centre_lat"], centre_lon=eo["centre_lon"],
            radius_m=eo["radius_m"], network_type=eo.get("network_type", "drive"),
        )
        eval_set = osm_dataset(
            graph=g, num_customers=eo["num_customers"], capacity=eo["capacity"],
            params=params, num_instances=eo["num_instances"],
            city_name=eo.get("city_name", "osm_eval"),
            seed=base_cfg.get("seed", 0) + 1,
        )
        print(f"Built {len(eval_set)} OSM eval instances.")

    for w in sorted(args.weights):
        print(f"\n========== Training with co2_weight = {w:.2f} ==========")
        model_cfg = copy.deepcopy(base_cfg["model"])
        model_cfg["cost_mode"] = "blend"
        model_cfg["co2_weight"] = float(w)
        policy = ACVRPPolicy(**model_cfg)

        tcfg_dict = dict(base_cfg["training"])
        tcfg_dict["save_prefix"] = f"matnet_co2w{int(w * 100):03d}"
        tcfg = TrainingConfig(**tcfg_dict)

        train(
            policy=policy,
            train_stream=stream,
            cfg=tcfg,
            eval_set=eval_set,
        )

    print("\nSweep complete. Saved checkpoints:")
    for w in sorted(args.weights):
        print(f"  models/matnet_co2w{int(w * 100):03d}_best.pt")


if __name__ == "__main__":
    main()

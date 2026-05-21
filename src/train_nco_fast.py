"""
train_nco_fast.py
=================

Faster variant of train_nco.py. Uses the optimisations in
`nco.trainer_fast`: AMP, TF32, larger batch sizes, skipped early-epoch
evaluation, and optional torch.compile.

Usage examples
--------------
    # Train MatNet with the default speed settings (AMP only, no compile)
    python -m src.train_nco_fast --policy matnet --config configs/train.yaml --osm-eval

    # Aggressive: also enable torch.compile (try this if AMP alone works)
    python -m src.train_nco_fast --policy matnet --config configs/train.yaml \\
        --osm-eval --compile

    # Conservative: AMP off (debug if AMP causes NaNs on your GPU)
    python -m src.train_nco_fast --policy baseline --config configs/train.yaml \\
        --osm-eval --no-amp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .emissions_model import EmissionsParams


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_params(cfg: dict) -> EmissionsParams:
    e = cfg["emissions"]
    return EmissionsParams(
        C1_distance=e["C1_distance"], C2_time=e["C2_time"], C3_mass=e["C3_mass"],
        empty_mass_kg=e["empty_mass_kg"], co2_per_litre=e["co2_per_litre"],
        avg_speed_kmh=e["avg_speed_kmh"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACVRP NCO policy (fast).")
    parser.add_argument("--policy", choices=["matnet", "baseline"], required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--osm-eval", action="store_true")

    # Speedup knobs
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision (slower but most stable).")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                        help="AMP precision. bfloat16 is recommended on Ampere+.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (extra 20-50%% speedup, occasionally unstable).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config. AMP gives memory headroom; "
                             "128 or 192 is a good starting point for N=50.")
    parser.add_argument("--skip-eval-until", type=int, default=20,
                        help="Skip eval for the first N epochs (saves time during warmup).")
    args = parser.parse_args()

    import torch  # noqa: F401
    from .nco.dataset import StreamingSyntheticDataset, SyntheticConfig, osm_dataset
    from .nco.model import ACVRPPolicy
    from .nco.baseline_am import CoordOnlyACVRPPolicy
    from .nco.trainer_fast import TrainingConfigFast, train_fast

    cfg = _load_yaml(args.config)
    params = _make_params(cfg)

    syn_cfg = SyntheticConfig(**cfg["synthetic"])
    stream = StreamingSyntheticDataset(syn_cfg, params, seed=cfg.get("seed", 0))

    eval_set = []
    if args.osm_eval and "osm_eval" in cfg:
        from .data_loader import download_road_graph
        eo = cfg["osm_eval"]
        g = download_road_graph(
            centre_lat=eo["centre_lat"], centre_lon=eo["centre_lon"],
            radius_m=eo["radius_m"], network_type=eo.get("network_type", "drive"),
        )
        eval_set = osm_dataset(
            graph=g, num_customers=eo["num_customers"], capacity=eo["capacity"],
            params=params, num_instances=eo["num_instances"],
            city_name=eo.get("city_name", "osm_eval"),
            seed=cfg.get("seed", 0) + 1,
        )
        print(f"Built {len(eval_set)} OSM eval instances.")

    model_cfg = cfg["model"]
    if args.policy == "matnet":
        policy = ACVRPPolicy(**model_cfg)
        save_prefix = "matnet_cvrp"
    else:
        coord_cfg = {k: v for k, v in model_cfg.items() if k != "edge_feature_dim"}
        policy = CoordOnlyACVRPPolicy(**coord_cfg)
        save_prefix = "baseline_am"

    # Build the fast training config from the YAML's `training` block,
    # then apply CLI overrides.
    tcfg_dict = dict(cfg["training"])
    tcfg_dict["save_prefix"] = save_prefix
    if args.batch_size is not None:
        tcfg_dict["batch_size"] = args.batch_size

    tcfg = TrainingConfigFast(
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        use_compile=args.compile,
        skip_eval_until_epoch=args.skip_eval_until,
        **tcfg_dict,
    )
    Path(tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Training {args.policy} policy "
        f"({sum(p.numel() for p in policy.parameters()):,} params)"
    )
    log = train_fast(policy, stream, tcfg, eval_set if eval_set else None)

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

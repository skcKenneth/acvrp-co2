"""
nco/trainer.py
==============

Training loop for the ACVRP policy.

Uses REINFORCE with the POMO multi-start baseline:

    grad = E[(L - L_bar) * grad log pi]

where L_bar is the average cost across the n_starts parallel rollouts
launched from the same instance. This gives a low-variance, on-policy
baseline that doesn't depend on geometric symmetry, so it works on
asymmetric road data.

The training set is generated on-the-fly via `StreamingSyntheticDataset`
so the model never sees the same instance twice; this is the standard
NCO recipe to avoid overfitting.

Validation is run on a fixed held-out set, ideally including real OSM
instances, to give a meaningful indication of generalisation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import StreamingSyntheticDataset
from .instance import CVRPInstance, collate_instances
from .model import ACVRPPolicy


# --- Helpers ---------------------------------------------------------------

def _extract_arch_hparams(policy: torch.nn.Module) -> dict:
    """
    Best-effort recovery of the constructor hyperparameters that built
    `policy`. We rely on the convention that ACVRPPolicy and
    CoordOnlyACVRPPolicy store all relevant args as attributes after
    __init__. If a field is missing we fall back to a sensible default.
    """
    defaults = {
        "node_feature_dim": 3,
        "edge_feature_dim": 4,
        "embed_dim": 128,
        "n_heads": 8,
        "n_layers": 3,
        "ffn_dim": 256,
        "tanh_clip": 10.0,
        "cost_mode": "distance",
        "co2_weight": 0.5,
        "co2_scale": 1000.0,
    }
    out = {}
    for k, v in defaults.items():
        out[k] = getattr(policy, k, v)
    # Pull architecture sizes from the actual encoder if attrs are absent.
    enc = getattr(policy, "encoder", None)
    if enc is not None:
        # Number of transformer blocks
        blocks = getattr(enc, "blocks", None)
        if blocks is not None and hasattr(blocks, "__len__"):
            out["n_layers"] = len(blocks)
        # Embedding dim
        proj = getattr(enc, "node_proj", None) or getattr(enc, "input_proj", None)
        if proj is not None and hasattr(proj, "out_features"):
            out["embed_dim"] = proj.out_features
    return out


def save_checkpoint(
    policy: torch.nn.Module,
    path: str,
    policy_kind: str = "matnet",
) -> None:
    """
    Save a self-describing checkpoint: state_dict + architecture metadata.

    The metadata lets `load_checkpoint` rebuild the correct architecture
    even if defaults in the model class change later, and lets the
    grid-comparison runner load checkpoints trained with N=20 or N=50
    interchangeably.
    """
    torch.save(
        {
            "version": 2,
            "policy_kind": policy_kind,
            "state_dict": policy.state_dict(),
            "arch": _extract_arch_hparams(policy),
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load a checkpoint saved by `save_checkpoint` and return the
    correctly-constructed policy with weights already loaded.

    Backwards-compatible: if the file is a raw state_dict (the old
    format used before this change), we attempt to load it into the
    default ACVRPPolicy architecture and surface a clear error message
    if the shapes don't match.
    """
    from .model import ACVRPPolicy
    from .baseline_am import CoordOnlyACVRPPolicy

    blob = torch.load(path, map_location=device)

    # New self-describing format
    if isinstance(blob, dict) and blob.get("version", 0) >= 2:
        arch = blob["arch"]
        if blob.get("policy_kind") == "baseline":
            arch = {k: v for k, v in arch.items() if k != "edge_feature_dim"}
            policy = CoordOnlyACVRPPolicy(**arch)
        else:
            policy = ACVRPPolicy(**arch)
        policy.load_state_dict(blob["state_dict"])
        policy.to(device).eval()
        return policy

    # Legacy format: bare state_dict. Try to infer architecture from the
    # parameter shapes.
    state = blob if isinstance(blob, dict) and "state_dict" not in blob else blob.get("state_dict", blob)

    # Infer embed_dim and n_layers from the keys.
    embed_dim = None
    n_layers = 0
    has_edge_proj = False
    for k, v in state.items():
        if k.endswith("node_proj.weight") or k.endswith("input_proj.weight"):
            embed_dim = v.shape[0]
        if k.startswith("encoder.blocks.") and ".attn.W_q.weight" in k:
            block_idx = int(k.split(".")[2])
            n_layers = max(n_layers, block_idx + 1)
        if "W_e" in k or "edge_proj" in k:
            has_edge_proj = True

    if embed_dim is None or n_layers == 0:
        raise RuntimeError(
            f"Could not infer architecture from legacy checkpoint at {path}. "
            "Re-save it with the new save_checkpoint() helper."
        )

    arch = {
        "embed_dim": embed_dim,
        "n_layers": n_layers,
        "ffn_dim": 2 * embed_dim,    # Convention used in this project
    }
    if has_edge_proj:
        policy = ACVRPPolicy(**arch)
    else:
        # Coord-only baseline has no edge attention
        policy = CoordOnlyACVRPPolicy(**arch)
    policy.load_state_dict(state)
    policy.to(device).eval()
    return policy


@dataclass
class TrainingConfig:
    batch_size: int = 64
    n_starts: int = 8                # POMO multi-start
    epochs: int = 50
    steps_per_epoch: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 2.0
    eval_every: int = 1              # epochs between evaluations
    checkpoint_dir: str = "models"
    device: str = "cuda"             # "cuda" or "cpu"
    save_prefix: str = "acvrp_policy"
    # Early stopping: halt if val cost hasn't improved for `patience`
    # consecutive evaluations. Set to None to disable.
    early_stopping_patience: Optional[int] = 20
    # POMO multi-start width used during evaluation. None -> same as n_starts.
    eval_n_starts: Optional[int] = None


@dataclass
class TrainingLog:
    train_costs: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_costs: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)


def evaluate(
    policy: ACVRPPolicy,
    eval_set: List[CVRPInstance],
    device: str,
    batch_size: int = 32,
    n_starts: int = 8,
) -> float:
    """
    Mean tour cost on the evaluation set, evaluated with POMO
    multi-start and *minimum* over starts per instance (the standard
    NCO inference protocol).

    Using POMO at evaluation time matches the training-time distribution
    and typically tightens val cost by 2-5% versus single greedy rollout.
    The reported value is therefore directly comparable to the best
    val cost a fully-converged classical solver would obtain.
    """
    policy.eval()
    costs = []
    with torch.no_grad():
        for i in range(0, len(eval_set), batch_size):
            batch_instances = eval_set[i:i + batch_size]
            batch = collate_instances(batch_instances).to(device)
            B = batch.batch_size
            # POMO sample with n_starts; cost shape is (B * n_starts,).
            rollout = policy.pomo_sample(batch, n_starts=n_starts)
            cost_per_start = rollout.cost.view(B, n_starts)
            # Take the best (minimum-cost) start for each instance.
            best_cost = cost_per_start.min(dim=1).values
            costs.append(best_cost.cpu())
    policy.train()
    return float(torch.cat(costs).mean().item())


def train(
    policy: ACVRPPolicy,
    train_stream: StreamingSyntheticDataset,
    cfg: TrainingConfig,
    eval_set: Optional[List[CVRPInstance]] = None,
) -> TrainingLog:
    """
    Train `policy` in place. Returns a log of per-epoch metrics.

    The training proceeds in fixed-size epochs (configured via
    `steps_per_epoch * batch_size` instances per epoch). Each step:
      1. Sample a batch from the streaming generator.
      2. Run `n_starts` POMO rollouts in parallel.
      3. Compute the REINFORCE loss with POMO baseline.
      4. Backprop, clip gradients, step.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    policy.to(device)
    policy.train()

    optim = AdamW(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=cfg.epochs * cfg.steps_per_epoch)

    log = TrainingLog()
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience_counter = 0     # epochs since last val improvement
    eval_starts = cfg.eval_n_starts if cfg.eval_n_starts is not None else cfg.n_starts

    for epoch in range(cfg.epochs):
        t0 = time.time()
        epoch_costs = []
        epoch_losses = []
        for step in range(cfg.steps_per_epoch):
            instances = train_stream.sample_batch(cfg.batch_size)
            batch = collate_instances(instances).to(device)
            rollout = policy.pomo_sample(batch, n_starts=cfg.n_starts)

            # Reshape (B * n_starts,) -> (B, n_starts) to compute baseline.
            costs = rollout.cost.view(cfg.batch_size, cfg.n_starts)
            log_probs = rollout.log_probs.view(cfg.batch_size, cfg.n_starts)
            baseline = costs.mean(dim=1, keepdim=True)
            advantage = (costs - baseline).detach()

            loss = (advantage * log_probs).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()

            epoch_costs.append(costs.mean().item())
            epoch_losses.append(loss.item())

        train_cost = sum(epoch_costs) / len(epoch_costs)
        train_loss = sum(epoch_losses) / len(epoch_losses)
        log.train_costs.append(train_cost)
        log.train_losses.append(train_loss)
        log.epochs.append(epoch)

        if eval_set is not None and (epoch + 1) % cfg.eval_every == 0:
            val_cost = evaluate(
                policy, eval_set, device,
                batch_size=cfg.batch_size, n_starts=eval_starts,
            )
            log.val_costs.append(val_cost)
            if val_cost < best_val:
                best_val = val_cost
                patience_counter = 0
                save_checkpoint(
                    policy,
                    f"{cfg.checkpoint_dir}/{cfg.save_prefix}_best.pt",
                    policy_kind=("baseline" if "baseline" in cfg.save_prefix else "matnet"),
                )
            else:
                patience_counter += 1
        else:
            val_cost = float("nan")

        elapsed = time.time() - t0
        print(
            f"epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"train cost {train_cost:8.1f} | loss {train_loss:+.4f} | "
            f"val cost {val_cost:8.1f} | patience {patience_counter:2d} | "
            f"{elapsed:5.1f}s"
        )

        # Early stopping check
        if (
            cfg.early_stopping_patience is not None
            and patience_counter >= cfg.early_stopping_patience
        ):
            print(
                f"Early stopping: no val improvement for "
                f"{cfg.early_stopping_patience} consecutive evals "
                f"(best val={best_val:.1f}). Halting training."
            )
            break

    # Always save the final model in addition to the best.
    save_checkpoint(
        policy,
        f"{cfg.checkpoint_dir}/{cfg.save_prefix}_final.pt",
        policy_kind=("baseline" if "baseline" in cfg.save_prefix else "matnet"),
    )
    return log

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


def _infer_legacy_arch(state: dict) -> dict:
    """
    Recover architecture hyperparameters from a bare state_dict.

    Looks at parameter tensor shapes to infer embed_dim, n_layers,
    ffn_dim, n_heads, and whether the encoder has edge-feature
    projections (i.e. MatNet vs Vanilla-AM).

    Raises RuntimeError if any required shape cannot be located.
    """
    embed_dim = None
    n_layers = 0
    ffn_dim = None
    n_heads = None
    has_edge_proj = False

    for k, v in state.items():
        # embed_dim: from the input projection (node_proj for MatNet,
        # input_proj for Vanilla-AM).
        if k.endswith("node_proj.weight") or k.endswith("input_proj.weight"):
            embed_dim = v.shape[0]
        # n_layers: highest block index encountered.
        if k.startswith("encoder.blocks.") and ".attn.W_q.weight" in k:
            block_idx = int(k.split(".")[2])
            n_layers = max(n_layers, block_idx + 1)
        # ffn_dim: from the first FFN linear's output dimension.
        # PyTorch stores nn.Linear weights as (out_features, in_features),
        # so blocks[0].ffn.0.weight has shape (ffn_dim, embed_dim).
        if k.endswith("blocks.0.ffn.0.weight"):
            ffn_dim = v.shape[0]
        # MatNet has an edge-feature projection W_e in EdgeAwareAttention;
        # Vanilla-AM uses nn.MultiheadAttention (no W_e key).
        if "W_e" in k or "edge_proj" in k:
            has_edge_proj = True

    if embed_dim is None or n_layers == 0:
        raise RuntimeError(
            "Could not infer architecture from legacy checkpoint: "
            "missing input projection or transformer blocks."
        )

    # n_heads cannot be uniquely recovered from weight shapes alone
    # (W_q.weight is (embed_dim, embed_dim) regardless of head count),
    # so we fall back to the project-wide convention. All YAML configs
    # in this project use n_heads=8.
    if n_heads is None:
        n_heads = 8

    if ffn_dim is None:
        # Fallback if the FFN naming changed; keep the project's
        # historical convention but flag it.
        ffn_dim = 2 * embed_dim

    return {
        "embed_dim": embed_dim,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "n_heads": n_heads,
        "_has_edge_proj": has_edge_proj,
    }


def load_checkpoint(path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load a checkpoint saved by `save_checkpoint` and return the
    correctly-constructed policy with weights already loaded.

    Backwards-compatible: if the file is a raw state_dict (the old
    format used before this change), we attempt to infer the
    architecture from the parameter shapes and emit a warning so the
    user knows they should retrain.
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

    # Legacy format: bare state_dict. Infer architecture from shapes.
    state = (
        blob
        if isinstance(blob, dict) and "state_dict" not in blob
        else blob.get("state_dict", blob)
    )

    import warnings
    warnings.warn(
        f"Loading legacy bare-state_dict checkpoint at {path!r}. "
        "Inferring architecture from parameter shapes; consider retraining "
        "to produce a v2 self-describing checkpoint.",
        DeprecationWarning,
        stacklevel=2,
    )

    inferred = _infer_legacy_arch(state)
    has_edge_proj = inferred.pop("_has_edge_proj")

    if has_edge_proj:
        policy = ACVRPPolicy(**inferred)
    else:
        # Coord-only baseline has no edge attention
        policy = CoordOnlyACVRPPolicy(**inferred)

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

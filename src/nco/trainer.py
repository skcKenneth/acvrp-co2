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
) -> float:
    """Greedy mean cost on the evaluation set."""
    policy.eval()
    costs = []
    with torch.no_grad():
        for i in range(0, len(eval_set), batch_size):
            batch = collate_instances(eval_set[i:i + batch_size]).to(device)
            rollout = policy.greedy(batch)
            costs.append(rollout.cost.cpu())
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
            val_cost = evaluate(policy, eval_set, device, batch_size=cfg.batch_size)
            log.val_costs.append(val_cost)
            if val_cost < best_val:
                best_val = val_cost
                torch.save(
                    policy.state_dict(),
                    f"{cfg.checkpoint_dir}/{cfg.save_prefix}_best.pt",
                )
        else:
            val_cost = float("nan")

        elapsed = time.time() - t0
        print(
            f"epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"train cost {train_cost:8.1f} | loss {train_loss:+.4f} | "
            f"val cost {val_cost:8.1f} | {elapsed:5.1f}s"
        )

    # Always save the final model in addition to the best.
    torch.save(
        policy.state_dict(),
        f"{cfg.checkpoint_dir}/{cfg.save_prefix}_final.pt",
    )
    return log

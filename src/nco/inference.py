"""
nco/inference.py
================

Helpers for running a trained ACVRPPolicy on a *single* CVRPInstance
and converting the rollout into the same route format used elsewhere
in the project (list of [0, ..., 0] subroutes).
"""
from __future__ import annotations

from typing import List

import torch

from .instance import CVRPInstance, collate_instances
from .model import ACVRPPolicy


def actions_to_routes(actions: List[int], depot_index: int = 0) -> List[List[int]]:
    """
    Split a flat sequence of node visits into one route per vehicle.

    A new route starts each time the vehicle leaves the depot, and
    ends when it returns to it. Empty subroutes (depot -> depot) are
    discarded.
    """
    routes: List[List[int]] = []
    current: List[int] = []
    for node in actions:
        if node == depot_index:
            if current and any(n != depot_index for n in current[1:]):
                # Close the current route
                if current[0] != depot_index:
                    current = [depot_index] + current
                if current[-1] != depot_index:
                    current.append(depot_index)
                routes.append(current)
            current = [depot_index]
        else:
            current.append(node)
    # Tail flush
    if current and any(n != depot_index for n in current[1:]):
        if current[-1] != depot_index:
            current.append(depot_index)
        routes.append(current)
    return routes


def solve_with_policy(
    policy: ACVRPPolicy,
    instance: CVRPInstance,
    device: str = "cuda",
    mode: str = "greedy",
    n_samples: int = 32,
) -> List[List[int]]:
    """
    Run a trained policy on a single instance and return the resulting
    route plan.

    Parameters
    ----------
    mode : "greedy" | "pomo"
        "greedy" performs a single argmax rollout.
        "pomo"   samples `n_samples` rollouts from different start
                 customers and returns the best (lowest-cost) one.
    """
    policy.eval()
    batch = collate_instances([instance])
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    batch = batch.to(device_t)

    with torch.no_grad():
        if mode == "greedy":
            rollout = policy.greedy(batch)
            best_actions = rollout.actions[0].cpu().tolist()
        elif mode == "pomo":
            rollout = policy.pomo_sample(batch, n_starts=n_samples)
            # `policy.pomo_sample` tiles batch by n_samples, so for a single
            # input instance the rollout has shape (n_samples, T) effectively
            best_idx = rollout.cost.argmin().item()
            best_actions = rollout.actions[best_idx].cpu().tolist()
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    return actions_to_routes(best_actions, depot_index=instance.depot_index)

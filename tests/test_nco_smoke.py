"""
Smoke tests for the NCO subpackage.

These tests check tensor shapes and termination behaviour without
requiring GPU training. They run on CPU in a few seconds. The
mathematical correctness of the underlying attention layers is assumed
from the well-tested PyTorch primitives; what we verify here is that
*our gluing code* is internally consistent.

Skipped automatically if torch is not installed.

Run with one of:
    python -m pytest tests/test_nco_smoke.py -v          (recommended)
    pytest tests/test_nco_smoke.py -v
    python tests/test_nco_smoke.py                        (also works,
                                                          see bootstrap
                                                          below)
"""
# --- sys.path bootstrap ----------------------------------------------------
# So that `python tests/test_nco_smoke.py` works even when the project
# is not installed as a package. When pytest is the runner, the
# project-root conftest.py handles this and the block below is a
# harmless no-op.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# --------------------------------------------------------------------------

import math

import pytest

torch = pytest.importorskip("torch")  # skip the whole file if no torch.

import numpy as np

from src.emissions_model import EmissionsParams
from src.nco.dataset import SyntheticConfig, make_synthetic_instance
from src.nco.decoder import make_mask
from src.nco.instance import collate_instances


@pytest.fixture
def params() -> EmissionsParams:
    return EmissionsParams(
        C1_distance=0.000125, C2_time=0.00040, C3_mass=0.0000018,
        empty_mass_kg=2200, co2_per_litre=2.68, avg_speed_kmh=30,
    )


@pytest.fixture
def small_batch(params):
    rng = np.random.default_rng(0)
    cfg = SyntheticConfig(num_customers=6, capacity=20)
    instances = [make_synthetic_instance(cfg, params, rng) for _ in range(4)]
    return collate_instances(instances)


def test_synthetic_instance_basic_shape(params):
    rng = np.random.default_rng(0)
    cfg = SyntheticConfig(num_customers=5, capacity=10)
    inst = make_synthetic_instance(cfg, params, rng)
    assert inst.distance.shape == (6, 6)
    assert inst.demands.shape == (6,)
    assert inst.demands[0] == 0       # depot demand zero
    # Diagonal is zero
    assert (np.diag(inst.distance) == 0).all()


def test_synthetic_matrix_is_asymmetric(params):
    """With factor_max > 1, at least some pair (i, j) must differ from (j, i)."""
    rng = np.random.default_rng(0)
    cfg = SyntheticConfig(num_customers=10, capacity=20, asymmetry_factor_max=1.4)
    inst = make_synthetic_instance(cfg, params, rng)
    differences = np.abs(inst.distance - inst.distance.T)
    assert differences.sum() > 0


def test_collate_consistency(small_batch):
    """Collating preserves shapes and produces a torch.Tensor batch."""
    assert small_batch.batch_size == 4
    assert small_batch.num_nodes == 7   # 6 customers + 1 depot
    assert small_batch.locations.shape == (4, 7, 2)
    assert small_batch.distance.shape == (4, 7, 7)
    assert small_batch.edge_features.shape[-1] == 4
    assert small_batch.demands[:, 0].sum().item() == 0     # all depots demand 0


def test_make_mask_depot_failsafe():
    """If no customer is reachable AND vehicle is at depot, depot is forced on."""
    B, N = 1, 4
    visited = torch.tensor([[True, True, True, True]])         # all customers done
    demands = torch.tensor([[0, 5, 5, 5]])
    remaining = torch.tensor([0])
    current = torch.tensor([0])
    depot_index = torch.tensor([0])
    mask = make_mask(visited, demands, remaining, current, depot_index)
    # At least one slot must be on so the rollout can step.
    assert mask.any().item()
    # The only on entry should be the depot column (index 0).
    assert mask[0, 0].item()


def test_make_mask_capacity_filter():
    """Customers whose demand exceeds remaining capacity must be masked off."""
    B, N = 1, 4
    visited = torch.tensor([[True, False, False, False]])   # only depot 'visited'
    demands = torch.tensor([[0, 3, 8, 5]])
    remaining = torch.tensor([4])     # capacity 4 left
    current = torch.tensor([0])
    depot_index = torch.tensor([0])
    mask = make_mask(visited, demands, remaining, current, depot_index)
    # Customer 1 (demand 3) fits, customer 2 (8) does not, customer 3 (5) does not.
    assert mask[0, 1].item()
    assert not mask[0, 2].item()
    assert not mask[0, 3].item()


def test_policy_rollout_visits_all_customers(small_batch, params):
    """A greedy rollout must produce a tour that visits every customer."""
    from src.nco.model import ACVRPPolicy
    torch.manual_seed(0)
    policy = ACVRPPolicy(embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64)
    policy.eval()

    with torch.no_grad():
        rollout = policy.greedy(small_batch)

    # Every customer (index > 0) must appear in the action sequence
    for b in range(small_batch.batch_size):
        acts = set(rollout.actions[b].tolist())
        for cust in range(1, small_batch.num_nodes):
            assert cust in acts, f"Batch {b} missed customer {cust}"

    # Distance must be finite and positive
    assert (rollout.distance_m > 0).all().item()
    assert torch.isfinite(rollout.distance_m).all().item()
    # CO2 must be non-negative
    assert (rollout.co2_kg >= 0).all().item()


def test_pomo_sample_shapes(small_batch):
    """POMO sample tiles the batch by n_starts."""
    from src.nco.model import ACVRPPolicy
    torch.manual_seed(0)
    policy = ACVRPPolicy(embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64)
    policy.eval()

    with torch.no_grad():
        rollout = policy.pomo_sample(small_batch, n_starts=3)

    expected_B = small_batch.batch_size * 3
    assert rollout.cost.shape == (expected_B,)
    assert rollout.distance_m.shape == (expected_B,)
    assert rollout.actions.shape[0] == expected_B


def test_cost_mode_blend_combines_distance_and_co2(small_batch):
    """`cost_mode='blend'` must depend on both distance and CO2."""
    from src.nco.model import ACVRPPolicy
    torch.manual_seed(0)
    policy = ACVRPPolicy(
        embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64,
        cost_mode="blend", co2_weight=0.5, co2_scale=1000.0,
    )
    policy.eval()

    with torch.no_grad():
        rollout = policy.greedy(small_batch)

    expected = 0.5 * rollout.distance_m + 0.5 * 1000.0 * rollout.co2_kg
    assert torch.allclose(rollout.cost, expected, rtol=1e-4)

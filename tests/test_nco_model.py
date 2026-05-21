"""
Unit tests for the NCO instance/dataset modules.

Tests that don't require torch (those would crash imports if torch is
missing). Tests that need torch are marked with a runtime skip.
"""
import math

import numpy as np
import pytest

from src.emissions_model import EmissionsParams
from src.nco.dataset import (
    SyntheticConfig,
    make_synthetic_instance,
    synthetic_dataset,
)


@pytest.fixture
def params() -> EmissionsParams:
    return EmissionsParams(
        C1_distance=0.000125,
        C2_time=0.00040,
        C3_mass=0.0000018,
        empty_mass_kg=2200,
        co2_per_litre=2.68,
        avg_speed_kmh=30,
    )


def test_synthetic_instance_shapes(params):
    cfg = SyntheticConfig(num_customers=10, capacity=20)
    rng = np.random.default_rng(0)
    inst = make_synthetic_instance(cfg, params, rng)

    n = cfg.num_customers + 1
    assert inst.num_nodes == n
    assert inst.locations.shape == (n, 2)
    assert inst.demands.shape == (n,)
    assert inst.distance.shape == (n, n)
    assert inst.time.shape == (n, n)
    assert inst.fuel_per_arc.shape == (n, n)
    assert inst.co2_per_arc.shape == (n, n)


def test_synthetic_instance_has_asymmetry(params):
    """The injected asymmetry should make d[i,j] != d[j,i] for many pairs."""
    cfg = SyntheticConfig(num_customers=10, capacity=20, asymmetry_factor_max=1.4)
    rng = np.random.default_rng(0)
    inst = make_synthetic_instance(cfg, params, rng)
    n = inst.num_nodes
    asym_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(inst.distance[i, j], inst.distance[j, i], rel_tol=1e-6):
                asym_count += 1
    # With asymmetry_factor_max > 1, virtually every pair should differ.
    total_pairs = n * (n - 1) // 2
    assert asym_count > 0.8 * total_pairs, (
        f"Expected ~all pairs asymmetric; got {asym_count}/{total_pairs}"
    )


def test_synthetic_instance_depot_demand_zero(params):
    cfg = SyntheticConfig(num_customers=10, capacity=20)
    rng = np.random.default_rng(0)
    inst = make_synthetic_instance(cfg, params, rng)
    assert inst.demands[inst.depot_index] == 0


def test_sanitize_distance_matrix_replaces_inf():
    from src.nco.dataset import sanitize_distance_matrix

    dist = np.array([[0.0, 100.0], [math.inf, 0.0]], dtype=float)
    clean = sanitize_distance_matrix(dist)
    assert np.isfinite(clean).all()
    assert clean[0, 1] == pytest.approx(100.0)
    assert clean[1, 0] == pytest.approx(1000.0)


def test_synthetic_dataset_seed_reproducibility(params):
    """Same seed must produce identical datasets."""
    cfg = SyntheticConfig(num_customers=5, capacity=15)
    a = synthetic_dataset(cfg, params, num_instances=4, seed=42)
    b = synthetic_dataset(cfg, params, num_instances=4, seed=42)
    for inst_a, inst_b in zip(a, b):
        assert np.allclose(inst_a.distance, inst_b.distance)
        assert np.array_equal(inst_a.demands, inst_b.demands)


def test_emissions_consistency_at_zero_payload(params):
    """fuel_per_arc must be >= 0 and proportional to distance at empty load."""
    cfg = SyntheticConfig(num_customers=5, capacity=15)
    rng = np.random.default_rng(0)
    inst = make_synthetic_instance(cfg, params, rng)
    n = inst.num_nodes
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            assert inst.fuel_per_arc[i, j] >= 0
            # Distance scaled by linear coeffs; spot-check that arcs with
            # larger distance have larger fuel.
    # Larger arc => more fuel under linear model (monotonicity)
    flat_d = inst.distance.flatten()
    flat_f = inst.fuel_per_arc.flatten()
    # Sort by distance, check fuel is sorted too (within numerical tolerance)
    order = np.argsort(flat_d)
    sorted_f = flat_f[order]
    assert np.all(np.diff(sorted_f) >= -1e-9)


# ---------------------------------------------------------------------------
# Torch-dependent tests: only run if torch is importable.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed; skipping NCO model tests.")


def test_collate_instances_shapes(params):
    from src.nco.instance import collate_instances
    cfg = SyntheticConfig(num_customers=6, capacity=15)
    rng = np.random.default_rng(0)
    insts = [make_synthetic_instance(cfg, params, rng) for _ in range(3)]
    batch = collate_instances(insts)
    n = cfg.num_customers + 1
    assert batch.locations.shape == (3, n, 2)
    assert batch.demands.shape == (3, n)
    assert batch.distance.shape == (3, n, n)
    assert batch.edge_features.shape == (3, n, n, 4)
    assert batch.capacity.shape == (3,)


def test_acvrp_policy_forward_pass(params):
    from src.nco.instance import collate_instances
    from src.nco.model import ACVRPPolicy
    cfg = SyntheticConfig(num_customers=5, capacity=10)
    rng = np.random.default_rng(0)
    insts = [make_synthetic_instance(cfg, params, rng) for _ in range(2)]
    batch = collate_instances(insts)

    policy = ACVRPPolicy(
        node_feature_dim=3, edge_feature_dim=4,
        embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64,
    )
    rollout = policy.greedy(batch)

    # Sanity: actions shape (B, T) and cost shape (B,)
    assert rollout.cost.shape == (2,)
    # Cost should be positive (non-trivial tour)
    assert (rollout.cost > 0).all()
    # All customers must be visited in each rollout
    for b in range(2):
        visited = set(rollout.actions[b].tolist())
        for cust in range(1, cfg.num_customers + 1):
            assert cust in visited, f"Customer {cust} missing from rollout {b}"


def test_greedy_forward_finite_with_inf_distance_matrix(params):
    """Instances built from matrices with inf must produce finite rollouts."""
    from src.nco.dataset import _build_instance_from_distance
    from src.nco.instance import collate_instances
    from src.nco.model import ACVRPPolicy

    n = 6
    coords = np.random.default_rng(0).uniform(0, 5000, size=(n, 2)).astype(np.float32)
    demands = np.array([0, 3, 4, 2, 5, 1], dtype=np.int64)
    dist = np.full((n, n), 500.0, dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist[0, 3] = math.inf

    inst = _build_instance_from_distance(
        coords, demands, dist, capacity=20, params=params, city_name="test",
    )
    assert np.isfinite(inst.distance).all()

    batch = collate_instances([inst])
    assert np.isfinite(batch.edge_features.numpy()).all()

    policy = ACVRPPolicy(
        node_feature_dim=3, edge_feature_dim=4,
        embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64,
    )
    rollout = policy.greedy(batch)
    assert torch.isfinite(rollout.cost).all()


def test_pomo_sample_baseline_diversity(params):
    """POMO multi-start should produce different routes per start."""
    from src.nco.instance import collate_instances
    from src.nco.model import ACVRPPolicy
    cfg = SyntheticConfig(num_customers=5, capacity=10)
    rng = np.random.default_rng(0)
    insts = [make_synthetic_instance(cfg, params, rng)]
    batch = collate_instances(insts)

    policy = ACVRPPolicy(
        node_feature_dim=3, edge_feature_dim=4,
        embed_dim=32, n_heads=4, n_layers=2, ffn_dim=64,
    )
    rollout = policy.pomo_sample(batch, n_starts=4)
    # Expect (B * n_starts,) = (4,) outputs
    assert rollout.cost.shape == (4,)
    # Different starts should produce different action sequences
    unique_first_actions = set(rollout.actions[:, 1].tolist())
    assert len(unique_first_actions) >= 2, "POMO starts should differ"

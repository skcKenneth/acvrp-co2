"""
Unit tests for the linear emissions model.

Run with:  pytest tests/
"""
import math

import numpy as np
import pytest

from src.emissions_model import (
    EmissionsParams,
    arc_fuel_litres,
    fuel_to_co2_kg,
    route_fuel_litres,
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


def test_zero_distance_zero_fuel(params):
    """An arc of length zero burns no fuel."""
    assert arc_fuel_litres(0.0, payload_kg=500, params=params) == 0.0


def test_fuel_increases_with_distance(params):
    """Linear model: doubling distance doubles fuel (for fixed payload)."""
    f1 = arc_fuel_litres(1000.0, 500, params)
    f2 = arc_fuel_litres(2000.0, 500, params)
    assert math.isclose(2 * f1, f2, rel_tol=1e-9)


def test_fuel_increases_with_payload(params):
    """Heavier payloads burn more fuel for the same distance."""
    light = arc_fuel_litres(1000.0, 100, params)
    heavy = arc_fuel_litres(1000.0, 5000, params)
    assert heavy > light


def test_co2_conversion(params):
    """Fuel-to-CO2 conversion is linear with the given factor."""
    assert fuel_to_co2_kg(1.0, params) == pytest.approx(params.co2_per_litre)
    assert fuel_to_co2_kg(0.0, params) == 0.0


def test_route_fuel_with_simple_path(params):
    """
    A trivial three-node route (depot -> A -> depot) must match the
    sum of arc fuels with payload tracked correctly.
    """
    D = np.array([
        [0,    1000, 2000],
        [1000, 0,    1500],
        [2000, 1500, 0   ],
    ], dtype=float)
    demands = [0, 10, 0]   # depot, customer with demand 10, dummy
    route = [0, 1, 0]
    # Manual reference:
    # leg 1: depot->A with payload 10
    expected_leg1 = arc_fuel_litres(1000.0, 10.0, params)
    # leg 2: A->depot with payload 0 (just dropped off)
    expected_leg2 = arc_fuel_litres(1000.0, 0.0, params)
    expected = expected_leg1 + expected_leg2

    assert math.isclose(
        route_fuel_litres(route, D, demands, params),
        expected,
        rel_tol=1e-9,
    )


def test_routes_to_metrics_infeasible_on_ar(params):
    from src.emissions_model import routes_to_metrics

    D = np.array([[0, 1000], [math.inf, 0]], dtype=float)
    demands = [0, 5]
    metrics = routes_to_metrics([[0, 1, 0]], D, demands, params)
    assert metrics["infeasible_legs"] == 1
    assert metrics["ar_feasible"] is False
    assert math.isnan(metrics["co2_kg"])
    assert metrics["distance_m"] == pytest.approx(1000.0)


def test_unreachable_arc_returns_inf(params):
    """If a route requires an infinite-cost arc, fuel should be inf."""
    D = np.array([[0, math.inf], [math.inf, 0]], dtype=float)
    demands = [0, 5]
    assert math.isinf(route_fuel_litres([0, 1, 0], D, demands, params))

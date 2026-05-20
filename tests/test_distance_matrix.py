"""
Unit tests for the distance-matrix builders.
"""
import math

import numpy as np
import pytest

from src.data_loader import Customer
from src.distance_matrix import (
    asymmetry_index,
    build_euclidean_matrix,
    build_manhattan_matrix,
    haversine_distance_m,
    symmetrise_road_matrix,
)


def _make_customers():
    return [
        Customer(0, "Depot", 25.0143, 121.4672, 0),
        Customer(1, "A",     25.0170, 121.4640, 5),
        Customer(2, "B",     25.0100, 121.4700, 7),
    ]


def test_haversine_known_distance():
    """
    Haversine between two points one degree of latitude apart should
    be about 111 km. We allow a 1% tolerance because the Earth radius
    chosen is a mean value.
    """
    d = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d, 111_195.0, rel_tol=0.01)


def test_euclidean_matrix_is_symmetric():
    customers = _make_customers()
    D = build_euclidean_matrix(customers)
    assert np.allclose(D, D.T)
    # diagonal is zero
    assert np.allclose(np.diag(D), 0.0)


def test_manhattan_matrix_is_symmetric_and_nonzero_offdiag():
    customers = _make_customers()
    D = build_manhattan_matrix(customers)
    assert np.allclose(D, D.T)
    assert (D[np.triu_indices_from(D, k=1)] > 0).all()


def test_symmetrise_road_matrix_averages_directions():
    asym = np.array([
        [0.0, 100.0, 200.0],
        [120.0, 0.0, 150.0],
        [180.0, 170.0, 0.0],
    ])
    sym = symmetrise_road_matrix(asym)
    assert sym[0, 1] == pytest.approx(110.0)
    assert sym[0, 1] == sym[1, 0]
    assert sym[2, 0] == pytest.approx(190.0)


def test_asymmetry_index_bounds():
    """The index lies in [0, 1] and is 0 for a symmetric matrix."""
    sym = np.array([[0, 5], [5, 0]], dtype=float)
    asym = np.array([[0, 5], [50, 0]], dtype=float)
    assert asymmetry_index(sym) == pytest.approx(0.0)
    assert 0.0 < asymmetry_index(asym) <= 1.0

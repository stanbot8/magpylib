"""Test that non-finite observer positions are rejected early."""

import numpy as np
import pytest
import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput


@pytest.fixture()
def source():
    """A Circle source for observer input tests."""
    return magpy.current.Circle(current=100, diameter=0.1)


def test_nan_observer_raises(source):
    with pytest.raises(MagpylibBadUserInput, match="finite"):
        source.getB((0, 0, float("nan")))


def test_inf_observer_raises(source):
    with pytest.raises(MagpylibBadUserInput, match="finite"):
        source.getB((0, 0, float("inf")))


def test_neg_inf_observer_raises(source):
    with pytest.raises(MagpylibBadUserInput, match="finite"):
        source.getB((float("-inf"), 0, 0))


def test_nan_in_observer_array_raises(source):
    obs = np.array([[0, 0, 0], [0, 0, np.nan]])
    with pytest.raises(MagpylibBadUserInput, match="finite"):
        source.getB(obs)


def test_finite_observers_unchanged(source):
    """Finite inputs should work exactly as before."""
    B = source.getB((0, 0, 0.05))
    assert np.all(np.isfinite(B))

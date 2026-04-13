"""Edge case tests for field computation with boundary-value inputs.

Validates that ``getB`` handles degenerate, extreme, and boundary-value
observer positions and source parameters without crashing or returning
non-finite results.
"""

import numpy as np
import pytest
import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput


@pytest.fixture()
def circle_source():
    """A simple Circle current source for field tests."""
    return magpy.current.Circle(current=100, diameter=0.1)


class TestObserverBoundaryValues:
    """``getB`` with edge-case observer positions."""

    def test_at_origin(self, circle_source):
        """On-axis center of a loop should give finite Bz."""
        B = circle_source.getB((0, 0, 0))
        assert np.all(np.isfinite(B))
        assert B[2] > 0

    def test_on_current_loop(self, circle_source):
        """Observer on the wire itself should not raise."""
        B = circle_source.getB((0.05, 0, 0))
        assert B.shape == (3,)

    def test_far_field_decays(self, circle_source):
        """Field 1000 m away should be negligibly small."""
        B = circle_source.getB((0, 0, 1000))
        assert np.all(np.isfinite(B))
        assert np.linalg.norm(B) < 1e-6

    def test_multiple_observers_shape(self, circle_source):
        """Array input (n, 3) should return (n, 3) output."""
        obs = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 1.0]])
        B = circle_source.getB(obs)
        assert B.shape == (3, 3)

    def test_single_observer_shape(self, circle_source):
        """Tuple input (x, y, z) should return shape (3,)."""
        B = circle_source.getB((0, 0, 0.1))
        assert B.shape == (3,)


class TestSourceBoundaryValues:
    """Source creation and field with boundary parameter values."""

    def test_zero_current_gives_zero_field(self):
        """I=0 should produce exactly zero field everywhere."""
        src = magpy.current.Circle(current=0, diameter=0.1)
        B = src.getB((0, 0, 0.01))
        np.testing.assert_allclose(B, 0.0, atol=1e-20)

    def test_very_small_source(self):
        """Microscale source (1 um diameter) should give finite field."""
        src = magpy.current.Circle(current=100, diameter=1e-6)
        B = src.getB((0, 0, 1e-3))
        assert np.all(np.isfinite(B))

    def test_negative_current_flips_field(self):
        """Reversing current should negate the field exactly."""
        src_pos = magpy.current.Circle(current=100, diameter=0.1)
        src_neg = magpy.current.Circle(current=-100, diameter=0.1)
        B_pos = src_pos.getB((0, 0, 0.01))
        B_neg = src_neg.getB((0, 0, 0.01))
        np.testing.assert_allclose(B_pos, -B_neg, rtol=1e-10)


class TestCollectionBoundaryValues:
    """Collection behavior at boundaries."""

    def test_empty_collection_raises(self):
        """Calling getB on an empty Collection should raise."""
        coll = magpy.Collection()
        with pytest.raises(MagpylibBadUserInput):
            coll.getB((0, 0, 0))

    def test_single_source_matches_bare(self):
        """Collection(src).getB should equal src.getB."""
        src = magpy.current.Circle(current=100, diameter=0.1)
        coll = magpy.Collection(src)
        obs = (0, 0, 0.05)
        np.testing.assert_allclose(coll.getB(obs), src.getB(obs), rtol=1e-10)

    def test_superposition_linearity(self):
        """Two identical co-located sources should double the field."""
        src1 = magpy.current.Circle(current=100, diameter=0.1)
        src2 = magpy.current.Circle(current=100, diameter=0.1)
        coll = magpy.Collection(src1, src2)
        obs = (0, 0, 0.05)
        np.testing.assert_allclose(
            coll.getB(obs), 2 * src1.getB(obs), rtol=1e-10
        )

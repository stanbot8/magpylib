"""Tests for HTS tape material models."""

import numpy as np
import pytest
from magpylib.materials import HTSTape, SUPERPOWER_SCS4050, SUPERPOWER_SCS12050


class TestHTSTapeIcModel:
    """Validate Ic(T, B, theta) against known physical constraints."""

    def test_zero_above_tc(self):
        tape = SUPERPOWER_SCS4050
        assert tape.Ic(tape.Tc + 1, 0) == 0.0
        assert tape.Ic(100, 0) == 0.0

    def test_matches_77k_spec(self):
        tape = SUPERPOWER_SCS4050
        Ic_77 = tape.Ic(77.0, 0.0)
        assert abs(Ic_77 - tape.Ic_at_77K_sf) / tape.Ic_at_77K_sf < 0.05

    def test_decreases_with_temperature(self):
        tape = SUPERPOWER_SCS12050
        assert tape.Ic(50, 1.0) > tape.Ic(65, 1.0) > tape.Ic(77, 1.0) > 0

    def test_decreases_with_field(self):
        tape = SUPERPOWER_SCS12050
        assert tape.Ic(65, 0.0) > tape.Ic(65, 1.0) > tape.Ic(65, 5.0) > 0

    def test_anisotropy(self):
        """Perpendicular field (theta=0) gives lower Ic than parallel (theta=pi/2)."""
        tape = SUPERPOWER_SCS12050
        assert tape.Ic(65, 3.0, theta=np.pi / 2) > tape.Ic(65, 3.0, theta=0.0)

    def test_array_matches_scalar(self):
        tape = SUPERPOWER_SCS12050
        T = np.array([50, 65, 77])
        B = np.array([1.0, 2.0, 3.0])
        Ic_arr = tape.Ic_array(T, B)
        for i in range(3):
            assert abs(Ic_arr[i] - tape.Ic(T[i], B[i])) < 1e-10

    def test_12mm_roughly_3x_4mm(self):
        ratio = SUPERPOWER_SCS12050.Ic_at_77K_sf / SUPERPOWER_SCS4050.Ic_at_77K_sf
        assert 2.5 < ratio < 3.5

    def test_jc_positive(self):
        tape = SUPERPOWER_SCS12050
        assert tape.Jc(65, 1.0) > 0

    def test_custom_tape(self):
        """User can create a tape with custom parameters."""
        tape = HTSTape(width=6e-3, Ic_at_77K_sf=200.0, Tc=90.0)
        assert tape.Ic(77.0, 0.0) == pytest.approx(200.0, rel=0.05)
        assert tape.Ic(90.1, 0.0) == 0.0

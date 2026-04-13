"""Tests for solenoid section builder."""

import numpy as np
import pytest
from magpylib.builders import solenoid_collection


def _helmholtz_pair():
    """Helmholtz coil pair: two identical coils separated by their radius."""
    R = 0.5
    return [
        {"r_inner": R, "r_outer": R + 0.001, "z_center": R / 2,
         "length": 0.001, "n_turns": 1, "n_layers": 1},
        {"r_inner": R, "r_outer": R + 0.001, "z_center": -R / 2,
         "length": 0.001, "n_turns": 1, "n_layers": 1},
    ]


def test_returns_collection():
    sections = [{"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
                 "length": 0.1, "n_turns": 10, "n_layers": 5}]
    coll = solenoid_collection(sections, 100.0)
    assert hasattr(coll, "getB")


def test_field_at_center_positive():
    sections = [{"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
                 "length": 0.1, "n_turns": 10, "n_layers": 5}]
    coll = solenoid_collection(sections, 100.0)
    B = coll.getB((0, 0, 0))
    assert B[2] > 0  # Bz positive at center


def test_opposite_polarity_cancels():
    sections = [
        {"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
         "length": 0.1, "n_turns": 10, "n_layers": 5, "polarity": 1.0},
        {"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
         "length": 0.1, "n_turns": 10, "n_layers": 5, "polarity": -1.0},
    ]
    coll = solenoid_collection(sections, 100.0)
    B = coll.getB((0, 0, 0))
    assert abs(B[2]) < 1e-10


def test_helmholtz_uniform_region():
    """Helmholtz pair should produce roughly uniform field near center."""
    coll = solenoid_collection(_helmholtz_pair(), 100.0)
    B_center = coll.getB((0, 0, 0))
    B_offset = coll.getB((0, 0, 0.05))
    # Within 5cm of center, field should vary by < 5%
    assert abs(B_offset[2] - B_center[2]) / abs(B_center[2]) < 0.05


def test_field_falls_off():
    sections = [{"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
                 "length": 0.1, "n_turns": 10, "n_layers": 5}]
    coll = solenoid_collection(sections, 100.0)
    B_near = abs(coll.getB((0, 0, 0))[2])
    B_far = abs(coll.getB((0, 0, 2.0))[2])
    assert B_near > B_far


def test_more_turns_more_field():
    sec_10 = [{"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
               "length": 0.1, "n_turns": 10, "n_layers": 5}]
    sec_100 = [{"r_inner": 0.3, "r_outer": 0.31, "z_center": 0.0,
                "length": 0.1, "n_turns": 100, "n_layers": 50}]
    B_10 = solenoid_collection(sec_10, 100.0).getB((0, 0, 0))[2]
    B_100 = solenoid_collection(sec_100, 100.0).getB((0, 0, 0))[2]
    assert B_100 > B_10


def test_sampling_preserves_ampere_turns():
    """Field should be similar regardless of sampling density."""
    sections = [{"r_inner": 0.3, "r_outer": 0.35, "z_center": 0.0,
                 "length": 0.2, "n_turns": 100, "n_layers": 50}]
    B_coarse = solenoid_collection(sections, 100.0, max_samples_r=3, max_samples_z=4).getB((0, 0, 0))
    B_fine = solenoid_collection(sections, 100.0, max_samples_r=8, max_samples_z=12).getB((0, 0, 0))
    assert abs(B_coarse[2] - B_fine[2]) / abs(B_fine[2]) < 0.10

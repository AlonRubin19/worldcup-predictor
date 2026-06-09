"""Tests for xG calibration layer.

Calibration: soft compress (scale=0.65 around baseline=1.35) + hard clamp [0.2, 2.8].

Design properties:
  - Never produces extreme values
  - Preserves ordering (monotonic)
  - Mid-range values change minimally
  - Baseline xG is a fixed point
"""

import pytest
from src.models.xg_calibration import calibrate_xg, BASELINE_XG, SCALE, XG_FLOOR, XG_CEIL


# --- Fixed-point and range ---

def test_baseline_xg_is_unchanged():
    assert abs(calibrate_xg(BASELINE_XG) - BASELINE_XG) < 1e-9


def test_output_never_below_floor():
    for raw in [0.0, 0.05, 0.1, 0.15, 0.19, 0.2]:
        assert calibrate_xg(raw) >= XG_FLOOR


def test_output_never_above_ceiling():
    for raw in [2.8, 3.0, 3.5, 4.0, 4.5, 10.0]:
        assert calibrate_xg(raw) <= XG_CEIL


def test_extreme_high_xg_clamped_at_ceiling():
    # Raw 4.5 (old max) must come out at XG_CEIL
    assert calibrate_xg(4.5) == XG_CEIL


def test_floor_triggered_by_negative_input():
    # Soft compress of negative raw produces output < 0.2 → floor clamps it.
    # calibrate_xg(-1.0) = 1.35 + 0.65*(-1.0-1.35) = 1.35 - 1.5275 = -0.177 → clamped to 0.2
    assert calibrate_xg(-1.0) == XG_FLOOR


# --- Ordering preservation ---

def test_ordering_preserved_for_increasing_inputs():
    values = [0.3, 0.8, 1.0, 1.35, 1.6, 2.0, 2.5, 3.0, 3.5]
    calibrated = [calibrate_xg(v) for v in values]
    for i in range(len(calibrated) - 1):
        assert calibrated[i] <= calibrated[i + 1], (
            f"Ordering violated at index {i}: "
            f"calibrate({values[i]})={calibrated[i]} > calibrate({values[i+1]})={calibrated[i+1]}"
        )


def test_strictly_increasing_in_unclamped_range():
    values = [0.3, 0.8, 1.0, 1.35, 1.7, 2.0, 2.5]
    calibrated = [calibrate_xg(v) for v in values]
    for i in range(len(calibrated) - 1):
        assert calibrated[i] < calibrated[i + 1]


# --- Compression: mid-range changes minimally ---

def test_mid_range_xg_changes_by_less_than_15pct():
    # Values near baseline should move < 15%
    for raw in [1.0, 1.2, 1.35, 1.5, 1.8]:
        cal = calibrate_xg(raw)
        pct_change = abs(cal - raw) / raw
        assert pct_change < 0.15, (
            f"Mid-range xG {raw} changed by {pct_change:.1%} after calibration (expected < 15%)"
        )


def test_high_xg_compressed_significantly():
    # A raw 3.5 should come down noticeably from 3.5
    cal = calibrate_xg(3.5)
    assert cal < 3.0, f"Raw 3.5 should be compressed below 3.0, got {cal}"


def test_calibration_reduces_mean_of_extreme_distribution():
    extremes = [3.0, 3.5, 4.0, 4.5]
    raw_mean = sum(extremes) / len(extremes)
    cal_mean = sum(calibrate_xg(x) for x in extremes) / len(extremes)
    assert cal_mean < raw_mean


# --- Specific known values ---

def test_xg_1_0_compressed_toward_baseline():
    # 1.0 is below baseline → soft compress moves it up slightly toward 1.35
    cal = calibrate_xg(1.0)
    assert cal > 1.0 - 1e-9  # should be >= raw (pulled toward baseline)


def test_xg_2_0_compressed_below_2_0():
    # 2.0 is above baseline → soft compress moves it down toward 1.35
    cal = calibrate_xg(2.0)
    assert cal <= 2.0


def test_scale_and_constants_have_expected_values():
    assert BASELINE_XG == 1.35
    assert 0.5 <= SCALE <= 0.8
    assert XG_FLOOR == 0.2
    assert XG_CEIL == 2.8

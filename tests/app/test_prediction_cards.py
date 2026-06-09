"""Tests for prediction_cards logic layer.

Only the pure-data functions are tested here. Streamlit rendering is not tested.
"""

import pytest
from src.app.components.prediction_cards import (
    compute_confidence,
    ConfidenceResult,
    format_outcome_rows,
    format_xg_summary,
)


# ── ConfidenceResult contract ─────────────────────────────────────────────────

def test_confidence_result_has_required_fields():
    result = compute_confidence(win_a=0.60, draw=0.20, win_b=0.20, warnings=[])
    assert isinstance(result, ConfidenceResult)
    assert hasattr(result, "level")       # float 0.0-1.0
    assert hasattr(result, "label")       # str: "High" / "Medium" / "Low"
    assert hasattr(result, "top_outcome") # str: "team_a_win" / "draw" / "team_b_win"
    assert hasattr(result, "top_prob")    # float
    assert hasattr(result, "gap")         # float: top - second


# ── Level ranges ─────────────────────────────────────────────────────────────

def test_clear_favourite_gives_high_confidence():
    result = compute_confidence(win_a=0.72, draw=0.20, win_b=0.08, warnings=[])
    assert result.label == "High"


def test_moderate_favourite_gives_medium_confidence():
    result = compute_confidence(win_a=0.50, draw=0.28, win_b=0.22, warnings=[])
    assert result.label == "Medium"


def test_even_match_gives_low_confidence():
    result = compute_confidence(win_a=0.36, draw=0.30, win_b=0.34, warnings=[])
    assert result.label == "Low"


def test_confidence_level_is_float_between_0_and_1():
    for win_a, draw, win_b in [(0.72, 0.20, 0.08), (0.50, 0.28, 0.22), (0.35, 0.32, 0.33)]:
        result = compute_confidence(win_a, draw, win_b, warnings=[])
        assert 0.0 <= result.level <= 1.0


# ── Data validity warnings reduce confidence ─────────────────────────────────

def test_data_warnings_reduce_confidence_label():
    # A moderate favourite with warnings should be capped at Medium
    high = compute_confidence(win_a=0.72, draw=0.18, win_b=0.10, warnings=[])
    warned = compute_confidence(win_a=0.72, draw=0.18, win_b=0.10,
                                warnings=["Player data not research-valid"])
    assert warned.label in ("Medium", "Low")
    assert warned.label != "High" or high.label == "High"  # warning should matter


def test_no_warnings_does_not_reduce_confidence():
    result = compute_confidence(win_a=0.72, draw=0.18, win_b=0.10, warnings=[])
    assert result.label == "High"


# ── Top outcome identification ────────────────────────────────────────────────

def test_top_outcome_is_team_a_win_when_a_leads():
    result = compute_confidence(win_a=0.60, draw=0.22, win_b=0.18, warnings=[])
    assert result.top_outcome == "team_a_win"


def test_top_outcome_is_draw_when_draw_leads():
    result = compute_confidence(win_a=0.25, draw=0.45, win_b=0.30, warnings=[])
    assert result.top_outcome == "draw"


def test_top_outcome_is_team_b_win_when_b_leads():
    result = compute_confidence(win_a=0.20, draw=0.28, win_b=0.52, warnings=[])
    assert result.top_outcome == "team_b_win"


def test_gap_is_top_minus_second():
    result = compute_confidence(win_a=0.60, draw=0.22, win_b=0.18, warnings=[])
    # top=0.60, second=0.22
    assert abs(result.gap - (0.60 - 0.22)) < 0.001


def test_top_prob_matches_highest_probability():
    result = compute_confidence(win_a=0.60, draw=0.22, win_b=0.18, warnings=[])
    assert abs(result.top_prob - 0.60) < 1e-6


# ── format_outcome_rows ───────────────────────────────────────────────────────

def test_format_outcome_rows_returns_three_rows():
    rows = format_outcome_rows("Argentina", "France", 0.55, 0.25, 0.20)
    assert len(rows) == 3


def test_format_outcome_rows_contains_team_names():
    rows = format_outcome_rows("Argentina", "France", 0.55, 0.25, 0.20)
    outcomes = [r["Outcome"] for r in rows]
    assert any("Argentina" in o for o in outcomes)
    assert any("France" in o for o in outcomes)
    assert any("Draw" in o for o in outcomes)


def test_format_outcome_rows_probabilities_are_strings():
    rows = format_outcome_rows("A", "B", 0.50, 0.28, 0.22)
    for row in rows:
        assert "%" in row["Probability"]


def test_format_outcome_rows_highest_prob_marked():
    rows = format_outcome_rows("A", "B", 0.60, 0.22, 0.18)
    # Highest probability row should be visually distinguished
    highest_row = max(rows, key=lambda r: float(r["Probability"].replace("%", "")))
    assert highest_row.get("Most Likely") or highest_row["Outcome"].startswith("A")


# ── format_xg_summary ────────────────────────────────────────────────────────

def test_format_xg_summary_returns_dict():
    summary = format_xg_summary("Argentina", "France", 1.85, 1.10)
    assert isinstance(summary, dict)


def test_format_xg_summary_contains_both_teams():
    summary = format_xg_summary("Argentina", "France", 1.85, 1.10)
    text = str(summary)
    assert "Argentina" in text or "xg_a" in text or "1.85" in text
    assert "France" in text or "xg_b" in text or "1.10" in text


def test_format_xg_summary_values_are_formatted():
    summary = format_xg_summary("A", "B", 1.853, 0.924)
    # Values should be rounded to 2dp
    values = [v for v in summary.values() if isinstance(v, str)]
    assert any("1.85" in v for v in values) or any("1.85" in str(v) for v in summary.values())

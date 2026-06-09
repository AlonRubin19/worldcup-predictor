"""Tests for explanation_panel logic layer."""

import pytest
from src.app.components.explanation_panel import (
    format_driver_table,
    format_validity_badge,
    ValidityBadge,
)
from src.explainability.driver import DriverContribution


def _make_driver(name="ELO advantage", team="Argentina", direction="positive",
                 magnitude=0.05, description="ELO advantage: 200 pts") -> DriverContribution:
    return DriverContribution(
        name=name, team=team, direction=direction,
        magnitude=magnitude, description=description,
    )


# ── format_driver_table ───────────────────────────────────────────────────────

def test_format_driver_table_returns_list_of_dicts():
    drivers = [_make_driver(), _make_driver("Attack strength", magnitude=0.12)]
    rows = format_driver_table(drivers)
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)


def test_format_driver_table_one_row_per_driver():
    drivers = [_make_driver(), _make_driver("Attack strength"), _make_driver("Dixon-Coles adjustment")]
    rows = format_driver_table(drivers)
    assert len(rows) == 3


def test_format_driver_table_excludes_zero_magnitude_dc_driver():
    # Dixon-Coles driver has magnitude=0 and should be rendered differently (context only)
    dc = _make_driver("Dixon-Coles adjustment", magnitude=0.0, direction="neutral")
    other = _make_driver("ELO advantage", magnitude=0.05)
    rows = format_driver_table([other, dc])
    # DC row still present but should not show a numeric magnitude
    dc_row = next((r for r in rows if "Dixon" in r.get("Driver", "")), None)
    assert dc_row is not None
    assert dc_row.get("Magnitude") in ("—", "", None, "Context only")


def test_format_driver_table_direction_shown():
    drivers = [_make_driver(direction="positive"), _make_driver("Attack strength", direction="negative")]
    rows = format_driver_table(drivers)
    directions = [r.get("Direction", "") for r in rows]
    assert any("positive" in d or "+" in d or "up" in d.lower() for d in directions)
    assert any("negative" in d or "-" in d or "down" in d.lower() for d in directions)


def test_format_driver_table_team_shown():
    drivers = [_make_driver(team="Brazil"), _make_driver(team="Germany")]
    rows = format_driver_table(drivers)
    teams = [r.get("Team", "") for r in rows]
    assert "Brazil" in teams
    assert "Germany" in teams


def test_format_driver_table_empty_input_returns_empty_list():
    assert format_driver_table([]) == []


# ── format_validity_badge ─────────────────────────────────────────────────────

def test_validity_badge_has_required_fields():
    badge = format_validity_badge(is_research_valid=True, warnings=[])
    assert isinstance(badge, ValidityBadge)
    assert hasattr(badge, "label")
    assert hasattr(badge, "color")
    assert hasattr(badge, "tooltip")


def test_research_valid_badge_is_green():
    badge = format_validity_badge(is_research_valid=True, warnings=[])
    assert badge.color in ("green", "success", "#2ecc71") or "green" in badge.color.lower()


def test_not_research_valid_badge_is_orange_or_yellow():
    badge = format_validity_badge(is_research_valid=False, warnings=["not research-valid"])
    assert badge.color in ("orange", "warning", "yellow", "#f39c12") or \
           badge.color.lower() in ("orange", "warning", "yellow")


def test_research_valid_badge_label_indicates_validity():
    badge = format_validity_badge(is_research_valid=True, warnings=[])
    assert "research" in badge.label.lower() or "valid" in badge.label.lower()


def test_not_valid_badge_label_indicates_engineering():
    badge = format_validity_badge(is_research_valid=False, warnings=["eng only"])
    assert "engineering" in badge.label.lower() or "illustrative" in badge.label.lower() \
           or "not" in badge.label.lower()


def test_warnings_reflected_in_tooltip():
    warnings = ["Player data is engineering-valid only"]
    badge = format_validity_badge(is_research_valid=False, warnings=warnings)
    assert "player" in badge.tooltip.lower() or "engineering" in badge.tooltip.lower()


def test_no_warnings_gives_clean_tooltip():
    badge = format_validity_badge(is_research_valid=True, warnings=[])
    assert badge.tooltip  # not empty

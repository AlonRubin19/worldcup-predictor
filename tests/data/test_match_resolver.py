import pytest
import pandas as pd
from src.data.match_resolver import resolve_match, resolve_all_matches, ResolvedMatchStats


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_db_row(**kwargs):
    """Build a minimal match_results DataFrame row."""
    defaults = dict(
        date="2022-11-20", team_a="Ecuador", team_b="Qatar",
        team_a_elo_pre=1755.0, team_b_elo_pre=1634.0,
        team_a_goals_for_last_10=1.40, team_a_goals_against_last_10=0.90,
        team_b_goals_for_last_10=0.80, team_b_goals_against_last_10=1.60,
        team_a_points_per_game_last_10=2.10, team_b_points_per_game_last_10=0.90,
        team_a_matches_available=10, team_b_matches_available=10,
        team_a_goals=2, team_b_goals=0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ── Tests ────────────────────────────────────────────────────────────────────

def test_exact_order_match_found():
    """When historical order matches DB order, resolve succeeds."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")
    result = resolve_match("2022-11-20", "Ecuador", "Qatar", db)
    assert result is not None
    assert isinstance(result, ResolvedMatchStats)
    assert result.was_reversed is False


def test_reversed_order_match_found():
    """When historical teams are reversed vs DB, resolve still succeeds."""
    # DB has Ecuador as team_a, Qatar as team_b
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_elo_pre=1755.0, team_b_elo_pre=1634.0)
    # But historical_matches has Qatar first
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    assert result is not None
    assert result.was_reversed is True


def test_elo_correctly_swapped_on_reversal():
    """After reversal, team_a in result gets DB's team_b ELO."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_elo_pre=1755.0, team_b_elo_pre=1634.0)
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    # Qatar was team_b in DB (elo=1634), now team_a in result
    assert abs(result.team_a_elo_pre - 1634.0) < 1e-6
    assert abs(result.team_b_elo_pre - 1755.0) < 1e-6


def test_ppg_correctly_swapped_on_reversal():
    """After reversal, team_a PPG comes from DB's team_b PPG."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar",
                      team_a_points_per_game_last_10=2.10,
                      team_b_points_per_game_last_10=0.90)
    result = resolve_match("2022-11-20", "Qatar", "Ecuador", db)
    # Qatar was team_b in DB (ppg=0.90), now team_a in result
    assert abs(result.team_a_points_per_game_last_10 - 0.90) < 1e-6
    assert abs(result.team_b_points_per_game_last_10 - 2.10) < 1e-6


def test_no_match_returns_none():
    """When date/team pair doesn't exist in DB, returns None."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")
    result = resolve_match("2022-11-20", "France", "Brazil", db)
    assert result is None


def test_resolve_all_returns_unresolved_list():
    """resolve_all_matches correctly separates resolved from unresolved rows."""
    db = _make_db_row(team_a="Ecuador", team_b="Qatar")

    historical = pd.DataFrame([
        {"date": "2022-11-20", "team_a": "Ecuador", "team_b": "Qatar",
         "team_a_goals": 2, "team_b_goals": 0},
        {"date": "2022-11-21", "team_a": "France", "team_b": "Brazil",
         "team_a_goals": 1, "team_b_goals": 0},  # Not in DB
    ])
    resolved, unresolved = resolve_all_matches(historical, db)
    assert len(resolved) == 1
    assert len(unresolved) == 1
    assert unresolved[0]["team_a"] == "France"

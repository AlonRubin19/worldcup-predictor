"""Tests for the tournament simulator."""

import pytest
from src.tournament.simulator import (
    simulate_match,
    simulate_knockout_match,
    run_tournament,
    run_monte_carlo,
    TournamentResult,
    MonteCarloResult,
    MatchOutcome,
)
from src.models.research_valid_predictor import ResearchValidInput
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams

_STRONG  = TeamSnapshot(elo=2100.0, ppg=2.5)
_WEAK    = TeamSnapshot(elo=1600.0, ppg=0.8)
_PAR_STR = StrengthParams(alpha_attack=1.5, beta_defense=0.8, matches_used=100)
_PAR_WEK = StrengthParams(alpha_attack=0.6, beta_defense=1.5, matches_used=100)
_PAR_MED = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=100)
_MED     = TeamSnapshot(elo=1850.0, ppg=1.8)

_FIXTURE_PATH = None  # set in tests that need it


def _inp(snap_a=_STRONG, snap_b=_WEAK, par_a=_PAR_STR, par_b=_PAR_WEK) -> ResearchValidInput:
    return ResearchValidInput(
        team_a="Strong", team_b="Weak",
        snapshot_a=snap_a, snapshot_b=snap_b,
        params_a=par_a, params_b=par_b,
        rho=-0.30,
    )


# ── simulate_match ────────────────────────────────────────────────────────────

def test_simulate_match_returns_match_outcome():
    inp = _inp()
    outcome = simulate_match(inp, rng_seed=42)
    assert isinstance(outcome, MatchOutcome)


def test_match_outcome_has_required_fields():
    outcome = simulate_match(_inp(), rng_seed=42)
    assert hasattr(outcome, "goals_a")
    assert hasattr(outcome, "goals_b")
    assert hasattr(outcome, "winner")   # "team_a" | "draw" | "team_b"
    assert isinstance(outcome.goals_a, int)
    assert isinstance(outcome.goals_b, int)


def test_simulate_match_winner_consistent_with_goals():
    outcome = simulate_match(_inp(), rng_seed=42)
    if outcome.goals_a > outcome.goals_b:
        assert outcome.winner == "team_a"
    elif outcome.goals_b > outcome.goals_a:
        assert outcome.winner == "team_b"
    else:
        assert outcome.winner == "draw"


def test_simulate_match_deterministic_with_seed():
    outcome_1 = simulate_match(_inp(), rng_seed=99)
    outcome_2 = simulate_match(_inp(), rng_seed=99)
    assert outcome_1.goals_a == outcome_2.goals_a
    assert outcome_1.goals_b == outcome_2.goals_b


def test_simulate_match_differs_with_different_seeds():
    outcomes = [simulate_match(_inp(), rng_seed=i) for i in range(20)]
    results = [(o.goals_a, o.goals_b) for o in outcomes]
    assert len(set(results)) > 1  # not all identical


def test_stronger_team_wins_more_often():
    wins_strong = sum(
        simulate_match(_inp(), rng_seed=i).winner == "team_a"
        for i in range(200)
    )
    assert wins_strong > 100  # strong team wins majority


# ── simulate_knockout_match ───────────────────────────────────────────────────

def test_knockout_match_never_draws():
    for seed in range(50):
        outcome = simulate_knockout_match(_inp(), rng_seed=seed)
        assert outcome.winner in ("team_a", "team_b")
        assert outcome.winner != "draw"


def test_knockout_advances_stronger_team_more_often():
    wins = sum(
        simulate_knockout_match(_inp(), rng_seed=i).winner == "team_a"
        for i in range(200)
    )
    assert wins > 100


# ── run_tournament ────────────────────────────────────────────────────────────

def test_run_tournament_returns_result():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_tournament(fixture_path, snaps, params, rng_seed=0)
    assert isinstance(result, TournamentResult)


def test_run_tournament_has_champion():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_tournament(fixture_path, snaps, params, rng_seed=0)
    assert result.champion is not None
    assert isinstance(result.champion, str)
    assert len(result.champion) > 0


def test_run_tournament_tracks_advancement():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_tournament(fixture_path, snaps, params, rng_seed=0)
    # advancement dict: {team: furthest_stage_reached}
    assert isinstance(result.advancement, dict)
    assert result.champion in result.advancement
    assert result.advancement[result.champion] == "final"


# ── run_monte_carlo ───────────────────────────────────────────────────────────

def test_run_monte_carlo_returns_result():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_monte_carlo(fixture_path, snaps, params, n=50, rng_seed=0)
    assert isinstance(result, MonteCarloResult)


def test_monte_carlo_win_probabilities_sum_to_one():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_monte_carlo(fixture_path, snaps, params, n=100, rng_seed=0)
    total_wins = sum(result.win_tournament.values())
    assert abs(total_wins - 1.0) < 0.01


def test_monte_carlo_reach_r16_all_teams():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_monte_carlo(fixture_path, snaps, params, n=100, rng_seed=0)
    total_r16 = sum(result.reach_r16.values())
    # Exactly 16 teams qualify from groups per simulation × n sims = 16
    assert abs(total_r16 - 16.0) < 1.0


def test_monte_carlo_stronger_team_wins_more():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    result = run_monte_carlo(fixture_path, snaps, params, n=200, rng_seed=42)
    # Brazil or Argentina should be top-2 by win probability
    top2 = sorted(result.win_tournament.items(), key=lambda x: -x[1])[:2]
    top2_teams = {t for t, _ in top2}
    strong_teams = {"Brazil", "Argentina", "France", "England", "Spain"}
    assert top2_teams & strong_teams, f"Expected a strong team in top-2, got {top2_teams}"


def test_monte_carlo_deterministic_with_seed():
    from pathlib import Path
    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    snaps = load_team_snapshots()
    params = load_strength_params()
    r1 = run_monte_carlo(fixture_path, snaps, params, n=50, rng_seed=7)
    r2 = run_monte_carlo(fixture_path, snaps, params, n=50, rng_seed=7)
    assert r1.win_tournament == r2.win_tournament

from __future__ import annotations

from src.data.market_sources.supabase_betting_app import (
    BettingAppOddsRow,
    find_odds_for_match,
    _normalize,
)


def _row(team_a="Mexico", team_b="South Africa", home=1.44, draw=4.34, away=7.9):
    return BettingAppOddsRow(
        team_a=team_a, team_b=team_b, date="2026-06-17",
        home_odds=home, draw_odds=draw, away_odds=away,
        bookmaker="avg/36", updated_at="2026-06-01T08:24:36Z",
    )


def test_normalize_handles_known_aliases():
    assert _normalize("Türkiye") == "turkey"
    assert _normalize("USA") == "usa"
    assert _normalize("IR Iran") == "iran"


def test_find_odds_direct_order(monkeypatch):
    monkeypatch.setattr(
        "src.data.market_sources.supabase_betting_app.get_betting_app_odds",
        lambda: [_row()],
    )
    result = find_odds_for_match("Mexico", "South Africa")
    assert result is not None
    assert result.home_odds == 1.44
    assert result.away_odds == 7.9


def test_find_odds_swapped_order_swaps_odds(monkeypatch):
    monkeypatch.setattr(
        "src.data.market_sources.supabase_betting_app.get_betting_app_odds",
        lambda: [_row()],
    )
    result = find_odds_for_match("South Africa", "Mexico")
    assert result is not None
    # team_a is now South Africa, so "home_odds" should be the original away odds
    assert result.team_a == "South Africa"
    assert result.home_odds == 7.9
    assert result.away_odds == 1.44


def test_find_odds_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(
        "src.data.market_sources.supabase_betting_app.get_betting_app_odds",
        lambda: [_row()],
    )
    assert find_odds_for_match("Brazil", "Argentina") is None

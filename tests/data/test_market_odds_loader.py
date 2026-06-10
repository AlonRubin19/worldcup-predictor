from __future__ import annotations

import pandas as pd
import pytest

from src.data.market_odds_loader import get_market_odds_for_match


def _write_odds(tmp_path, rows):
    path = tmp_path / "odds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_placeholder_odds_are_not_research_valid(tmp_path):
    path = _write_odds(tmp_path, [{
        "match_id": "1", "date": "2026-06-01", "team_a": "Spain", "team_b": "France",
        "bookmaker": "placeholder", "opening_home_odds": 2.0, "opening_draw_odds": 3.0,
        "opening_away_odds": 4.0, "closing_home_odds": 2.0, "closing_draw_odds": 3.0,
        "closing_away_odds": 4.0, "source_type": "placeholder", "research_valid": "false",
    }])

    result = get_market_odds_for_match("Spain", "France", path)

    assert result.research_valid is False


def test_research_valid_odds_return_normalized_implied_probs(tmp_path):
    path = _write_odds(tmp_path, [{
        "match_id": "1", "date": "2026-06-01", "team_a": "Spain", "team_b": "France",
        "bookmaker": "bet365", "opening_home_odds": 2.0, "opening_draw_odds": 3.0,
        "opening_away_odds": 4.0, "closing_home_odds": 2.0, "closing_draw_odds": 3.5,
        "closing_away_odds": 4.0, "source_type": "the_odds_api", "research_valid": "true",
    }])

    result = get_market_odds_for_match("Spain", "France", path)

    assert result.research_valid is True
    assert result.win_a + result.draw + result.win_b == pytest.approx(1.0)
    # 1/2.0 is the largest implied prob -> Spain favoured
    assert result.win_a > result.win_b


def test_reversed_team_order_swaps_odds(tmp_path):
    path = _write_odds(tmp_path, [{
        "match_id": "1", "date": "2026-06-01", "team_a": "France", "team_b": "Spain",
        "bookmaker": "bet365", "opening_home_odds": 4.0, "opening_draw_odds": 3.5,
        "opening_away_odds": 2.0, "closing_home_odds": 4.0, "closing_draw_odds": 3.5,
        "closing_away_odds": 2.0, "source_type": "the_odds_api", "research_valid": "true",
    }])

    # Querying as (Spain, France) -- Spain is "team_b" in the file (odds 2.0)
    result = get_market_odds_for_match("Spain", "France", path)

    assert result.win_a > result.win_b


def test_no_match_found_returns_unavailable(tmp_path):
    path = _write_odds(tmp_path, [{
        "match_id": "1", "date": "2026-06-01", "team_a": "Argentina", "team_b": "Brazil",
        "bookmaker": "placeholder", "opening_home_odds": 2.0, "opening_draw_odds": 3.0,
        "opening_away_odds": 4.0, "closing_home_odds": 2.0, "closing_draw_odds": 3.0,
        "closing_away_odds": 4.0, "source_type": "placeholder", "research_valid": "false",
    }])

    result = get_market_odds_for_match("Spain", "France", path)

    assert result.research_valid is False
    assert result.win_a is None

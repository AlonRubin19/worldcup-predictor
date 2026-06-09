"""Tests for the live lineup loader.

TDD: all tests written RED-first before any production code exists.

Uses pure parsing functions — no HTTP calls.
"""

from __future__ import annotations

import pytest

from src.data.live_lineup_loader import parse_lineup_response
from src.data.lineup_loader import ExpectedLineupEntry


# ─────────────────────────────────────────────────────────────────────────────
# Sample API lineup response
# ─────────────────────────────────────────────────────────────────────────────

def _player(player_id: int, name: str, number: int = 1, pos: str = "G") -> dict:
    return {"player": {"id": player_id, "name": name, "number": number, "pos": pos}}


def _sample_lineup_response(
    home_team: str = "Brazil",
    home_team_id: int = 6,
    away_team: str = "Mexico",
    away_team_id: int = 24,
    home_starters: list[dict] | None = None,
    away_starters: list[dict] | None = None,
    home_subs: list[dict] | None = None,
    away_subs: list[dict] | None = None,
) -> dict:
    if home_starters is None:
        home_starters = [_player(i, f"Home P{i}", i, "G" if i == 1 else "M")
                         for i in range(1, 12)]
    if away_starters is None:
        away_starters = [_player(i + 100, f"Away P{i}", i, "G" if i == 1 else "M")
                         for i in range(1, 12)]
    if home_subs is None:
        home_subs = [_player(50, "Home Sub", 12, "G")]
    if away_subs is None:
        away_subs = [_player(150, "Away Sub", 12, "G")]

    return {
        "results": 2,
        "response": [
            {
                "team": {"id": home_team_id, "name": home_team},
                "formation": "4-3-3",
                "startXI": home_starters,
                "substitutes": home_subs,
            },
            {
                "team": {"id": away_team_id, "name": away_team},
                "formation": "4-4-2",
                "startXI": away_starters,
                "substitutes": away_subs,
            },
        ],
    }


def _empty_lineup_response() -> dict:
    return {"results": 0, "response": []}


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class TestParseLineupReturnType:
    def test_returns_list(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert isinstance(result, list)

    def test_returns_expected_lineup_entries(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(isinstance(e, ExpectedLineupEntry) for e in result)

    def test_empty_response_returns_empty(self):
        result = parse_lineup_response(_empty_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Starters
# ─────────────────────────────────────────────────────────────────────────────

class TestStartersParsing:
    def test_starters_have_expected_starter_true(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        starters = [e for e in result if e.lineup_status == "official"
                    and e.expected_starter]
        assert len(starters) > 0

    def test_11_starters_per_team(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        brazil_starters = [e for e in result if e.team == "Brazil" and e.expected_starter]
        mexico_starters = [e for e in result if e.team == "Mexico" and e.expected_starter]
        assert len(brazil_starters) == 11
        assert len(mexico_starters) == 11

    def test_starters_lineup_status_official(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        starters = [e for e in result if e.expected_starter]
        assert all(e.lineup_status == "official" for e in starters)


# ─────────────────────────────────────────────────────────────────────────────
# Substitutes
# ─────────────────────────────────────────────────────────────────────────────

class TestSubstitutesParsing:
    def test_substitutes_have_expected_starter_false(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        subs = [e for e in result if not e.expected_starter]
        assert len(subs) > 0
        assert all(not e.expected_starter for e in subs)

    def test_substitutes_lineup_status_bench(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        subs = [e for e in result if not e.expected_starter]
        assert all(e.lineup_status == "bench" for e in subs)


# ─────────────────────────────────────────────────────────────────────────────
# Source and research validity
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceAndResearchValidity:
    def test_source_type_is_official_lineup(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.source_type == "official_lineup" for e in result)

    def test_research_valid_true_for_all_entries(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.research_valid is True for e in result)

    def test_availability_status_fit(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.availability_status == "fit" for e in result)

    def test_availability_factor_one(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.availability_factor == pytest.approx(1.0) for e in result)

    def test_form_factor_one(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.form_factor == pytest.approx(1.0) for e in result)


# ─────────────────────────────────────────────────────────────────────────────
# Match metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchMetadata:
    def test_fixture_id_stored_as_match_id_string(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=855744, date="2026-06-11")
        assert all(e.match_id == "855744" for e in result)

    def test_date_stored_correctly(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.date == "2026-06-11" for e in result)

    def test_team_name_populated(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        teams = {e.team for e in result}
        assert "Brazil" in teams
        assert "Mexico" in teams


# ─────────────────────────────────────────────────────────────────────────────
# Player ID format
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerIdFormat:
    def test_player_id_prefixed_with_api(self):
        result = parse_lineup_response(_sample_lineup_response(),
                                       fixture_id=1, date="2026-06-11")
        assert all(e.player_id.startswith("api_") for e in result)

    def test_player_id_contains_api_player_id(self):
        starters = [_player(999, "TestPlayer")]
        response = _sample_lineup_response(home_starters=starters)
        result = parse_lineup_response(response, fixture_id=1, date="2026-06-11")
        home_entries = [e for e in result if e.team == "Brazil"]
        assert any("999" in e.player_id for e in home_entries)

    def test_player_name_stored(self):
        starters = [_player(1, "Alisson Becker", 1, "G")]
        response = _sample_lineup_response(home_starters=starters)
        result = parse_lineup_response(response, fixture_id=1, date="2026-06-11")
        home_entries = [e for e in result if e.team == "Brazil"]
        assert any(e.player_name == "Alisson Becker" for e in home_entries)

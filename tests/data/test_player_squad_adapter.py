"""Tests for player_squad_adapter.py — API-Football squad/player/injury parsing.

TDD: written RED first. Pure parsing functions only — no HTTP calls.
These tests use synthetic API-Football response shapes (subset of real
fields) so they do not depend on a live API key or network access.
"""

from __future__ import annotations

import pytest

from src.data.player_squad_adapter import (
    parse_squad_response,
    parse_player_statistics_response,
    parse_injuries_response,
    build_player_profile_row,
    SquadPlayer,
    PlayerSeasonStats,
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_squad_response — /players/squads
# ─────────────────────────────────────────────────────────────────────────────

_SQUAD_RESPONSE = {
    "response": [
        {
            "team": {"id": 6, "name": "Spain"},
            "players": [
                {"id": 999, "name": "Lamine Yamal", "age": 18, "number": 19, "position": "Attacker"},
                {"id": 998, "name": "Some Keeper", "age": 30, "number": 1, "position": "Goalkeeper"},
            ],
        }
    ]
}


class TestParseSquadResponse:
    def test_returns_list_of_squad_players(self):
        players = parse_squad_response(_SQUAD_RESPONSE, team_name="Spain")
        assert all(isinstance(p, SquadPlayer) for p in players)
        assert len(players) == 2

    def test_fields_mapped(self):
        players = parse_squad_response(_SQUAD_RESPONSE, team_name="Spain")
        yamal = next(p for p in players if p.player_name == "Lamine Yamal")
        assert yamal.player_id == "999"
        assert yamal.team == "Spain"
        assert yamal.position == "FW"

    def test_position_mapping(self):
        players = parse_squad_response(_SQUAD_RESPONSE, team_name="Spain")
        gk = next(p for p in players if p.player_name == "Some Keeper")
        assert gk.position == "GK"

    def test_empty_response_returns_empty_list(self):
        assert parse_squad_response({"response": []}, team_name="Spain") == []

    def test_missing_team_returns_empty_list(self):
        assert parse_squad_response({"response": []}, team_name="Norway") == []


# ─────────────────────────────────────────────────────────────────────────────
# parse_player_statistics_response — /players?team=X&season=Y
# ─────────────────────────────────────────────────────────────────────────────

_STATS_RESPONSE = {
    "response": [
        {
            "player": {"id": 999, "name": "Lamine Yamal"},
            "statistics": [
                {
                    "games": {"minutes": 1800, "appearences": 20, "position": "Attacker"},
                    "goals": {"total": 9, "assists": 8},
                    "shots": {"total": 60, "on": 30},
                    "penalty": {"won": 1, "scored": 1},
                }
            ],
        },
        {
            # Player with zero minutes — should not divide by zero
            "player": {"id": 997, "name": "Bench Player"},
            "statistics": [
                {"games": {"minutes": 0, "appearences": 0, "position": "Midfielder"},
                 "goals": {"total": 0, "assists": 0},
                 "shots": {"total": 0, "on": 0},
                 "penalty": {"won": 0, "scored": 0}}
            ],
        },
    ]
}


class TestParsePlayerStatisticsResponse:
    def test_returns_dict_keyed_by_player_id(self):
        stats = parse_player_statistics_response(_STATS_RESPONSE)
        assert "999" in stats
        assert isinstance(stats["999"], PlayerSeasonStats)

    def test_goals_per_90_computed(self):
        stats = parse_player_statistics_response(_STATS_RESPONSE)
        s = stats["999"]
        # 9 goals over 1800 minutes = 0.45 goals/90
        assert s.goals_per_90 == pytest.approx(9 / 1800 * 90)

    def test_xg_proxy_per_90_computed_from_shots_on_target(self):
        stats = parse_player_statistics_response(_STATS_RESPONSE)
        s = stats["999"]
        # proxy: shots_on_target * 0.3 conversion, per 90
        assert s.xg_per_90_proxy > 0

    def test_zero_minutes_does_not_crash(self):
        stats = parse_player_statistics_response(_STATS_RESPONSE)
        s = stats["997"]
        assert s.goals_per_90 == 0.0
        assert s.xg_per_90_proxy == 0.0

    def test_penalty_taker_flag(self):
        stats = parse_player_statistics_response(_STATS_RESPONSE)
        assert stats["999"].penalty_taker is True
        assert stats["997"].penalty_taker is False


# ─────────────────────────────────────────────────────────────────────────────
# parse_injuries_response — /injuries?team=X
# ─────────────────────────────────────────────────────────────────────────────

_INJURIES_RESPONSE = {
    "response": [
        {"player": {"id": 999, "name": "Lamine Yamal"}, "player_injury": {"type": "Hamstring"}},
    ]
}


class TestParseInjuriesResponse:
    def test_returns_set_of_injured_player_ids(self):
        injured = parse_injuries_response(_INJURIES_RESPONSE)
        assert injured == {"999"}

    def test_empty_response_returns_empty_set(self):
        assert parse_injuries_response({"response": []}) == set()


# ─────────────────────────────────────────────────────────────────────────────
# build_player_profile_row — combine squad + stats + injuries -> profile row
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPlayerProfileRow:
    def setup_method(self):
        self.squad_player = SquadPlayer(
            player_id="999", player_name="Lamine Yamal", team="Spain",
            position="FW",
        )
        self.stats = PlayerSeasonStats(
            player_id="999", minutes=1800, appearances=20,
            goals_per_90=0.45, assists_per_90=0.40,
            xg_per_90_proxy=0.50, xa_per_90_proxy=0.45,
            penalty_taker=True,
        )

    def test_research_valid_when_stats_present(self):
        row = build_player_profile_row(self.squad_player, self.stats, injured=False)
        assert row["source_type"] == "api_football"
        assert row["research_valid"] is True

    def test_not_research_valid_when_stats_missing(self):
        row = build_player_profile_row(self.squad_player, stats=None, injured=False)
        assert row["source_type"] == "api_football_squad_only"
        assert row["research_valid"] is False
        # Falls back to a small placeholder xG so the player can still appear
        assert row["xg_per_90"] >= 0.0

    def test_injured_player_has_zero_availability(self):
        row = build_player_profile_row(self.squad_player, self.stats, injured=True)
        assert row["availability_factor"] == 0.0

    def test_healthy_player_has_full_availability(self):
        row = build_player_profile_row(self.squad_player, self.stats, injured=False)
        assert row["availability_factor"] == 1.0

    def test_row_has_required_player_profile_columns(self):
        row = build_player_profile_row(self.squad_player, self.stats, injured=False)
        for col in (
            "player_id", "player_name", "team", "position",
            "goals_per_90", "xg_per_90", "assists_per_90", "xa_per_90",
            "international_caps", "base_impact_score",
            "source_type", "research_valid", "availability_factor",
        ):
            assert col in row

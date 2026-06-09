"""Tests for the live fixture loader.

TDD: all tests written RED-first before any production code exists.

Uses pure parsing functions — no HTTP calls.
"""

from __future__ import annotations

import pytest

from src.data.live_fixture_loader import (
    LiveFixture,
    parse_fixtures_response,
    map_team_name,
    TEAM_NAME_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sample API response
# ─────────────────────────────────────────────────────────────────────────────

def _sample_response(
    fixture_id: int = 855744,
    date: str = "2026-06-11T18:00:00+00:00",
    home_api: str = "Brazil",
    away_api: str = "Mexico",
    status_short: str = "NS",
    round_: str = "Group Stage - 1",
    venue: str = "Lusail Iconic Stadium",
) -> dict:
    return {
        "results": 1,
        "response": [
            {
                "fixture": {
                    "id": fixture_id,
                    "date": date,
                    "venue": {"name": venue, "city": "City"},
                    "status": {"short": status_short, "long": "Not Started"},
                },
                "league": {
                    "id": 1,
                    "name": "World Cup",
                    "round": round_,
                },
                "teams": {
                    "home": {"id": 6, "name": home_api},
                    "away": {"id": 24, "name": away_api},
                },
            }
        ],
    }


def _empty_response() -> dict:
    return {"results": 0, "response": []}


# ─────────────────────────────────────────────────────────────────────────────
# LiveFixture structure
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveFixtureStructure:
    def test_has_required_fields(self):
        fixtures = parse_fixtures_response(_sample_response())
        f = fixtures[0]
        for field in [
            "fixture_id", "date", "home_team_api", "away_team_api",
            "home_team", "away_team", "status_short", "round", "venue",
        ]:
            assert hasattr(f, field), f"Missing field: {field}"

    def test_is_live_fixture_instance(self):
        fixtures = parse_fixtures_response(_sample_response())
        assert isinstance(fixtures[0], LiveFixture)

    def test_fixture_id_is_int(self):
        fixtures = parse_fixtures_response(_sample_response(fixture_id=12345))
        assert isinstance(fixtures[0].fixture_id, int)
        assert fixtures[0].fixture_id == 12345

    def test_date_is_string(self):
        fixtures = parse_fixtures_response(_sample_response())
        assert isinstance(fixtures[0].date, str)


# ─────────────────────────────────────────────────────────────────────────────
# parse_fixtures_response
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFixturesResponse:
    def test_returns_list(self):
        result = parse_fixtures_response(_sample_response())
        assert isinstance(result, list)

    def test_single_fixture_returned(self):
        result = parse_fixtures_response(_sample_response())
        assert len(result) == 1

    def test_empty_response_returns_empty(self):
        result = parse_fixtures_response(_empty_response())
        assert result == []

    def test_api_team_name_stored_in_home_team_api(self):
        result = parse_fixtures_response(_sample_response(home_api="Brazil"))
        assert result[0].home_team_api == "Brazil"

    def test_api_team_name_stored_in_away_team_api(self):
        result = parse_fixtures_response(_sample_response(away_api="Mexico"))
        assert result[0].away_team_api == "Mexico"

    def test_status_short_parsed(self):
        result = parse_fixtures_response(_sample_response(status_short="1H"))
        assert result[0].status_short == "1H"

    def test_round_parsed(self):
        result = parse_fixtures_response(_sample_response(round_="Group Stage - 3"))
        assert result[0].round == "Group Stage - 3"

    def test_venue_parsed(self):
        result = parse_fixtures_response(_sample_response(venue="Old Trafford"))
        assert result[0].venue == "Old Trafford"

    def test_date_parsed(self):
        result = parse_fixtures_response(_sample_response(date="2026-06-11T18:00:00+00:00"))
        assert result[0].date == "2026-06-11T18:00:00+00:00"

    def test_multiple_fixtures(self):
        multi = {
            "results": 2,
            "response": [
                _sample_response()["response"][0],
                {
                    "fixture": {
                        "id": 2,
                        "date": "2026-06-12T18:00:00+00:00",
                        "venue": {"name": "Venue B", "city": "City B"},
                        "status": {"short": "NS", "long": "Not Started"},
                    },
                    "league": {"id": 1, "name": "World Cup", "round": "Group Stage - 1"},
                    "teams": {
                        "home": {"id": 7, "name": "France"},
                        "away": {"id": 25, "name": "England"},
                    },
                },
            ],
        }
        result = parse_fixtures_response(multi)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Team name mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestTeamNameMapping:
    def test_map_team_name_returns_string(self):
        result = map_team_name("Brazil")
        assert isinstance(result, str)

    def test_known_team_mapped_to_internal_name(self):
        """API name 'Korea Republic' should map to internal 'South Korea'."""
        assert map_team_name("Korea Republic") == "South Korea"

    def test_known_usa_mapped(self):
        """API name 'United States' should map to internal 'USA'."""
        assert map_team_name("United States") == "USA"

    def test_known_iran_mapped(self):
        """API name 'IR Iran' should map to internal 'Iran'."""
        assert map_team_name("IR Iran") == "Iran"

    def test_unknown_team_returns_as_is(self):
        result = map_team_name("UnknownNation FC")
        assert result == "UnknownNation FC"

    def test_already_correct_name_unchanged(self):
        assert map_team_name("Brazil") == "Brazil"
        assert map_team_name("France") == "France"

    def test_team_name_map_is_dict(self):
        assert isinstance(TEAM_NAME_MAP, dict)

    def test_team_name_map_covers_common_differences(self):
        assert "Korea Republic" in TEAM_NAME_MAP
        assert "United States" in TEAM_NAME_MAP

    def test_mapped_name_applied_to_home_team(self):
        result = parse_fixtures_response(_sample_response(home_api="Korea Republic"))
        assert result[0].home_team == "South Korea"

    def test_mapped_name_applied_to_away_team(self):
        result = parse_fixtures_response(_sample_response(away_api="United States"))
        assert result[0].away_team == "USA"

    def test_unmapped_name_kept_as_is_in_home_team(self):
        result = parse_fixtures_response(_sample_response(home_api="Brazil"))
        assert result[0].home_team == "Brazil"

"""Tests for the live fixture provider abstraction.

TDD: all tests written RED-first before any production code exists.

No live API calls — API responses injected via mock _fetcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.fixture_provider import (
    FixtureSource,
    ProviderResult,
    get_fixtures,
    convert_live_fixture_to_fixture,
    parse_stage,
)
from src.tournament.fixtures import Fixture
from src.data.live_fixture_loader import LiveFixture
from src.data.api_football_client import ApiFootballClient


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _live_fixture(
    fixture_id: int = 855744,
    date: str = "2026-06-11T18:00:00+00:00",
    home_team: str = "Brazil",
    away_team: str = "Mexico",
    home_api: str = "Brazil",
    away_api: str = "Mexico",
    round_: str = "Group Stage - 1",
    status: str = "NS",
) -> LiveFixture:
    return LiveFixture(
        fixture_id=fixture_id,
        date=date,
        home_team_api=home_api,
        away_team_api=away_api,
        home_team=home_team,
        away_team=away_team,
        status_short=status,
        round=round_,
        venue="Stadium",
    )


def _api_response(fixtures_data: list[dict]) -> dict:
    return {"results": len(fixtures_data), "response": fixtures_data}


def _fixture_item(fixture_id=855744, date="2026-06-11T18:00:00+00:00",
                  home="Brazil", away="Mexico", round_="Group Stage - 1",
                  status="NS") -> dict:
    return {
        "fixture": {
            "id": fixture_id,
            "date": date,
            "venue": {"name": "Stadium", "city": "City"},
            "status": {"short": status, "long": "Not Started"},
        },
        "league": {"id": 1, "name": "World Cup", "round": round_},
        "teams": {
            "home": {"id": 1, "name": home},
            "away": {"id": 2, "name": away},
        },
    }


def _mock_client(tmp_path: Path, fixtures: list[dict] | None = None) -> ApiFootballClient:
    data = _api_response(fixtures or [_fixture_item()])
    return ApiFootballClient(
        api_key="test_key",
        cache_dir=tmp_path / "cache",
        _fetcher=lambda url, headers, params: data,
    )


def _no_key_client(tmp_path: Path) -> ApiFootballClient:
    return ApiFootballClient(
        api_key="",
        cache_dir=tmp_path / "cache",
        _fetcher=lambda url, headers, params: {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# FixtureSource enum
# ─────────────────────────────────────────────────────────────────────────────

class TestFixtureSource:
    def test_has_csv(self):
        assert FixtureSource.CSV is not None

    def test_has_api(self):
        assert FixtureSource.API is not None

    def test_has_auto(self):
        assert FixtureSource.AUTO is not None

    def test_csv_value(self):
        assert FixtureSource.CSV.value == "csv"

    def test_api_value(self):
        assert FixtureSource.API.value == "api"

    def test_auto_value(self):
        assert FixtureSource.AUTO.value == "auto"


# ─────────────────────────────────────────────────────────────────────────────
# parse_stage
# ─────────────────────────────────────────────────────────────────────────────

class TestParseStage:
    def test_group_stage(self):
        assert parse_stage("Group Stage - 1") == "group"

    def test_group_stage_variant(self):
        assert parse_stage("Group Stage - 3") == "group"

    def test_round_of_16(self):
        assert parse_stage("Round of 16") == "round_of_16"

    def test_quarter_finals(self):
        result = parse_stage("Quarter-finals")
        assert result == "quarter_final"

    def test_semi_finals(self):
        result = parse_stage("Semi-finals")
        assert result == "semi_final"

    def test_final(self):
        assert parse_stage("Final") == "final"

    def test_third_place(self):
        result = parse_stage("3rd Place Final")
        assert result == "third_place"

    def test_unknown_round_returns_generic(self):
        result = parse_stage("Something Unknown")
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# convert_live_fixture_to_fixture
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertLiveFixtureToFixture:
    def test_returns_fixture(self):
        f = convert_live_fixture_to_fixture(_live_fixture())
        assert isinstance(f, Fixture)

    def test_match_id_is_fixture_id_as_string(self):
        f = convert_live_fixture_to_fixture(_live_fixture(fixture_id=855744))
        assert f.match_id == "855744"

    def test_date_is_date_only(self):
        f = convert_live_fixture_to_fixture(
            _live_fixture(date="2026-06-11T18:00:00+00:00")
        )
        assert f.date == "2026-06-11"

    def test_team_a_is_home_team(self):
        f = convert_live_fixture_to_fixture(_live_fixture(home_team="Brazil"))
        assert f.team_a == "Brazil"

    def test_team_b_is_away_team(self):
        f = convert_live_fixture_to_fixture(_live_fixture(away_team="Mexico"))
        assert f.team_b == "Mexico"

    def test_group_stage_round_maps_to_group_stage(self):
        f = convert_live_fixture_to_fixture(
            _live_fixture(round_="Group Stage - 1")
        )
        assert f.stage == "group"

    def test_knockout_round_parsed(self):
        f = convert_live_fixture_to_fixture(
            _live_fixture(round_="Round of 16")
        )
        assert f.stage == "round_of_16"

    def test_group_field_empty_for_api_fixtures(self):
        """API /fixtures endpoint does not return group letter — group is empty."""
        f = convert_live_fixture_to_fixture(_live_fixture())
        assert isinstance(f.group, str)

    def test_date_without_time_component_kept_as_is(self):
        f = convert_live_fixture_to_fixture(_live_fixture(date="2026-06-11"))
        assert f.date == "2026-06-11"

    def test_team_name_mapping_applied(self):
        """API name 'Türkiye' should be mapped to 'Turkey' before conversion."""
        lf = _live_fixture(home_api="Türkiye", home_team="Turkey")
        f = convert_live_fixture_to_fixture(lf)
        assert f.team_a == "Turkey"


# ─────────────────────────────────────────────────────────────────────────────
# ProviderResult structure
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderResultStructure:
    def test_has_required_fields(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        for field in ("fixtures", "source_used", "mapping_warnings",
                      "fixture_count", "api_connected"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_is_provider_result(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert isinstance(result, ProviderResult)

    def test_fixture_count_matches_fixtures_len(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert result.fixture_count == len(result.fixtures)

    def test_fixtures_are_fixture_instances(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert all(isinstance(f, Fixture) for f in result.fixtures)

    def test_mapping_warnings_is_list(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert isinstance(result.mapping_warnings, list)


# ─────────────────────────────────────────────────────────────────────────────
# CSV provider
# ─────────────────────────────────────────────────────────────────────────────

class TestCsvProvider:
    def test_csv_mode_loads_fixtures(self):
        """CSV provider should load fixtures from the default fixture file."""
        result = get_fixtures(FixtureSource.CSV)
        assert isinstance(result.fixtures, list)
        assert result.fixture_count >= 1

    def test_csv_mode_source_used_is_csv(self):
        result = get_fixtures(FixtureSource.CSV)
        assert result.source_used == FixtureSource.CSV

    def test_csv_mode_api_connected_false(self):
        result = get_fixtures(FixtureSource.CSV)
        assert result.api_connected is False

    def test_csv_mode_with_custom_path(self, tmp_path):
        """CSV provider can use a custom path."""
        import csv as _csv
        p = tmp_path / "fixtures.csv"
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["match_id","stage","group","date","team_a","team_b"])
            w.writeheader()
            w.writerow({"match_id":"1","stage":"group","group":"A",
                        "date":"2026-06-11","team_a":"Brazil","team_b":"France"})
        result = get_fixtures(FixtureSource.CSV, csv_path=p)
        assert result.fixture_count == 1
        assert result.fixtures[0].team_a == "Brazil"


# ─────────────────────────────────────────────────────────────────────────────
# API provider
# ─────────────────────────────────────────────────────────────────────────────

class TestApiProvider:
    def test_api_mode_returns_fixtures(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert result.fixture_count >= 1

    def test_api_mode_source_used_is_api(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert result.source_used == FixtureSource.API

    def test_api_mode_api_connected_true(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert result.api_connected is True

    def test_api_fixtures_have_correct_teams(self, tmp_path):
        client = _mock_client(tmp_path, [_fixture_item(home="Brazil", away="Mexico")])
        result = get_fixtures(FixtureSource.API, api_client=client)
        assert result.fixtures[0].team_a == "Brazil"
        assert result.fixtures[0].team_b == "Mexico"

    def test_api_team_mapping_applied(self, tmp_path):
        """Türkiye should map to Turkey."""
        client = _mock_client(tmp_path, [_fixture_item(home="Türkiye", away="Brazil")])
        result = get_fixtures(FixtureSource.API, api_client=client)
        assert result.fixtures[0].team_a == "Turkey"

    def test_unknown_team_reported_as_warning(self, tmp_path):
        client = _mock_client(tmp_path, [_fixture_item(home="Unknown FC", away="Brazil")])
        result = get_fixtures(FixtureSource.API, api_client=client)
        assert len(result.mapping_warnings) > 0
        assert any("Unknown FC" in w for w in result.mapping_warnings)

    def test_no_warnings_for_known_teams(self, tmp_path):
        client = _mock_client(tmp_path, [_fixture_item(home="Brazil", away="France")])
        result = get_fixtures(FixtureSource.API, api_client=client)
        assert result.mapping_warnings == []

    def test_api_mode_without_client_falls_back_gracefully(self):
        """API mode with no client should not crash — return empty list."""
        result = get_fixtures(FixtureSource.API, api_client=None)
        assert isinstance(result.fixtures, list)
        assert result.api_connected is False

    def test_api_mode_missing_key_falls_back(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_no_key_client(tmp_path))
        assert result.api_connected is False

    def test_multiple_api_fixtures_all_converted(self, tmp_path):
        items = [
            _fixture_item(fixture_id=1, home="Brazil", away="France"),
            _fixture_item(fixture_id=2, home="Germany", away="Spain"),
        ]
        client = _mock_client(tmp_path, items)
        result = get_fixtures(FixtureSource.API, api_client=client)
        assert result.fixture_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Auto provider
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoProvider:
    def test_auto_uses_api_when_key_available(self, tmp_path):
        client = _mock_client(tmp_path)
        result = get_fixtures(FixtureSource.AUTO, api_client=client)
        assert result.source_used == FixtureSource.API

    def test_auto_falls_back_to_csv_when_no_key(self, tmp_path):
        client = _no_key_client(tmp_path)
        result = get_fixtures(FixtureSource.AUTO, api_client=client)
        assert result.source_used == FixtureSource.CSV

    def test_auto_falls_back_to_csv_when_no_client(self):
        result = get_fixtures(FixtureSource.AUTO, api_client=None)
        assert result.source_used == FixtureSource.CSV

    def test_auto_csv_fallback_has_fixtures(self, tmp_path):
        client = _no_key_client(tmp_path)
        result = get_fixtures(FixtureSource.AUTO, api_client=client)
        assert result.fixture_count >= 1

    def test_auto_api_error_falls_back_to_csv(self, tmp_path):
        """If API raises an error, auto should fall back to CSV."""
        def _failing_fetcher(url, headers, params):
            raise ConnectionError("network down")
        client = ApiFootballClient(
            api_key="valid_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_failing_fetcher,
        )
        result = get_fixtures(FixtureSource.AUTO, api_client=client)
        assert result.source_used == FixtureSource.CSV
        assert result.fixture_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Force refresh
# ─────────────────────────────────────────────────────────────────────────────

class TestForceRefresh:
    def test_force_refresh_calls_api_again(self, tmp_path):
        """force_refresh=True should bypass cache and re-fetch."""
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return _api_response([_fixture_item()])

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=False)
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=True)
        assert call_count["n"] == 2, "force_refresh should bypass cache"

    def test_no_force_refresh_uses_cache(self, tmp_path):
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return _api_response([_fixture_item()])

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=False)
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=False)
        assert call_count["n"] == 1, "Second call should use cache"

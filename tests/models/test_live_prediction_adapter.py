"""Tests for the live prediction adapter.

TDD: all tests written RED-first before any production code exists.

No live API calls — all HTTP injected via mock _fetcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.live_prediction_adapter import (
    LiveDataStatus,
    get_live_data_status,
    load_live_lineups_for_match,
)
from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.lineup_loader import ExpectedLineupEntry


# ─────────────────────────────────────────────────────────────────────────────
# Sample lineup API response helpers
# ─────────────────────────────────────────────────────────────────────────────

def _player(pid: int, name: str, pos: str = "M") -> dict:
    return {"player": {"id": pid, "name": name, "number": pid, "pos": pos}}


def _lineup_response(home: str = "Brazil", away: str = "Mexico") -> dict:
    home_starters = [_player(i, f"H{i}") for i in range(1, 12)]
    away_starters = [_player(i + 100, f"A{i}") for i in range(1, 12)]
    return {
        "results": 2,
        "response": [
            {
                "team": {"id": 6, "name": home},
                "formation": "4-3-3",
                "startXI": home_starters,
                "substitutes": [],
            },
            {
                "team": {"id": 24, "name": away},
                "formation": "4-4-2",
                "startXI": away_starters,
                "substitutes": [],
            },
        ],
    }


def _empty_lineup_response() -> dict:
    return {"results": 0, "response": []}


def _make_client(tmp_path: Path, api_key: str = "test_key",
                 fetcher_data: dict | None = None) -> ApiFootballClient:
    data = fetcher_data or _empty_lineup_response()
    return ApiFootballClient(
        api_key=api_key,
        cache_dir=tmp_path / "cache",
        _fetcher=lambda url, headers, params: data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LiveDataStatus structure
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveDataStatusStructure:
    def test_has_required_fields(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        for field in [
            "api_connected", "last_refresh", "fixture_source",
            "lineup_source", "lineup_status_label",
        ]:
            assert hasattr(status, field), f"Missing field: {field}"

    def test_is_live_data_status_instance(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        assert isinstance(status, LiveDataStatus)

    def test_api_connected_is_bool(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        assert isinstance(status.api_connected, bool)


# ─────────────────────────────────────────────────────────────────────────────
# API connectivity
# ─────────────────────────────────────────────────────────────────────────────

class TestApiConnectivity:
    def test_api_connected_true_when_key_present(self, tmp_path):
        client = _make_client(tmp_path, api_key="valid_key")
        status = get_live_data_status(client)
        assert status.api_connected is True

    def test_api_connected_false_when_key_missing(self, tmp_path):
        client = ApiFootballClient(
            api_key="",
            cache_dir=tmp_path / "cache",
            _fetcher=lambda url, headers, params: {},
        )
        status = get_live_data_status(client)
        assert status.api_connected is False

    def test_api_connected_false_when_key_none(self, tmp_path):
        client = ApiFootballClient(
            api_key=None,
            cache_dir=tmp_path / "cache",
            _fetcher=lambda url, headers, params: {},
        )
        status = get_live_data_status(client)
        assert status.api_connected is False

    def test_last_refresh_is_string_or_none(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        assert status.last_refresh is None or isinstance(status.last_refresh, str)


# ─────────────────────────────────────────────────────────────────────────────
# load_live_lineups_for_match
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLiveLineupsForMatch:
    def test_returns_tuple_of_two_lists(self, tmp_path):
        client = _make_client(tmp_path,
                               fetcher_data=_lineup_response("Brazil", "Mexico"))
        result = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        entries_a, entries_b = result
        assert isinstance(entries_a, list)
        assert isinstance(entries_b, list)

    def test_entries_are_expected_lineup_entry_type(self, tmp_path):
        client = _make_client(tmp_path,
                               fetcher_data=_lineup_response("Brazil", "Mexico"))
        entries_a, entries_b = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert all(isinstance(e, ExpectedLineupEntry) for e in entries_a + entries_b)

    def test_team_a_entries_belong_to_team_a(self, tmp_path):
        client = _make_client(tmp_path,
                               fetcher_data=_lineup_response("Brazil", "Mexico"))
        entries_a, _ = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert all(e.team == "Brazil" for e in entries_a)

    def test_team_b_entries_belong_to_team_b(self, tmp_path):
        client = _make_client(tmp_path,
                               fetcher_data=_lineup_response("Brazil", "Mexico"))
        _, entries_b = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert all(e.team == "Mexico" for e in entries_b)

    def test_empty_api_response_returns_empty_lists(self, tmp_path):
        client = _make_client(tmp_path, fetcher_data=_empty_lineup_response())
        entries_a, entries_b = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert entries_a == []
        assert entries_b == []

    def test_missing_api_key_returns_empty_lists(self, tmp_path):
        """ApiKeyMissingError must be caught — never crash the app."""
        client = ApiFootballClient(
            api_key="",
            cache_dir=tmp_path / "cache",
            _fetcher=lambda url, headers, params: {},
        )
        entries_a, entries_b = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert entries_a == []
        assert entries_b == []

    def test_api_error_returns_empty_lists_not_crash(self, tmp_path):
        """Any exception from the API should be caught gracefully."""
        def _raising_fetcher(url, headers, params):
            raise ConnectionError("network down")

        client = ApiFootballClient(
            api_key="valid_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_raising_fetcher,
        )
        # Should not raise — must return empty lists
        entries_a, entries_b = load_live_lineups_for_match(
            client, fixture_id=1, team_a="Brazil", team_b="Mexico"
        )
        assert entries_a == []
        assert entries_b == []


# ─────────────────────────────────────────────────────────────────────────────
# Lineup status label
# ─────────────────────────────────────────────────────────────────────────────

class TestLineupStatusLabel:
    def test_official_label_when_lineups_available(self, tmp_path):
        client = _make_client(tmp_path,
                               fetcher_data=_lineup_response("Brazil", "Mexico"))
        status = get_live_data_status(client, fixture_id=1,
                                      team_a="Brazil", team_b="Mexico")
        assert "official" in status.lineup_status_label.lower()

    def test_unavailable_label_when_no_lineups(self, tmp_path):
        client = _make_client(tmp_path, fetcher_data=_empty_lineup_response())
        status = get_live_data_status(client, fixture_id=1,
                                      team_a="Brazil", team_b="Mexico")
        assert any(word in status.lineup_status_label.lower()
                   for word in ("unavailable", "not available", "no lineup", "pending"))

    def test_not_connected_label_when_no_key(self, tmp_path):
        client = ApiFootballClient(
            api_key="",
            cache_dir=tmp_path / "cache",
            _fetcher=lambda url, headers, params: {},
        )
        status = get_live_data_status(client)
        assert any(word in status.lineup_status_label.lower()
                   for word in ("not connected", "no api", "unavailable", "api key"))


# ─────────────────────────────────────────────────────────────────────────────
# Fixture / lineup source labels
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceLabels:
    def test_fixture_source_is_string(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        assert isinstance(status.fixture_source, str)

    def test_lineup_source_is_string(self, tmp_path):
        client = _make_client(tmp_path)
        status = get_live_data_status(client)
        assert isinstance(status.lineup_source, str)

    def test_api_source_label_when_connected(self, tmp_path):
        client = _make_client(tmp_path, api_key="valid_key")
        status = get_live_data_status(client)
        assert "api" in status.fixture_source.lower()

    def test_csv_fallback_label_when_not_connected(self, tmp_path):
        client = ApiFootballClient(
            api_key="",
            cache_dir=tmp_path / "cache",
            _fetcher=lambda url, headers, params: {},
        )
        status = get_live_data_status(client)
        assert any(word in status.fixture_source.lower()
                   for word in ("csv", "local", "offline", "fallback"))

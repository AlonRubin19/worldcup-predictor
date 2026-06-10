"""Tests for live_squad_loader / live_injury_loader / live_player_stats_loader.

No live HTTP calls: ApiFootballClient is constructed with an injected
_fetcher stub.
"""

from __future__ import annotations

import pytest

from src.data.api_football_client import ApiFootballClient
from src.data.live_squad_loader import load_live_squad, LIVE_SOURCE_LABEL as SQUAD_LIVE
from src.data.live_squad_loader import FALLBACK_NO_KEY_LABEL as SQUAD_NO_KEY
from src.data.live_squad_loader import FALLBACK_EMPTY_LABEL as SQUAD_EMPTY
from src.data.live_injury_loader import load_live_injuries, LIVE_SOURCE_LABEL as INJ_LIVE
from src.data.live_injury_loader import FALLBACK_NO_KEY_LABEL as INJ_NO_KEY
from src.data.live_injury_loader import FALLBACK_NONE_LABEL as INJ_NONE
from src.data.live_player_stats_loader import load_live_player_stats, LIVE_SOURCE_LABEL as STATS_LIVE
from src.data.live_player_stats_loader import FALLBACK_NO_KEY_LABEL as STATS_NO_KEY


_SQUAD_RESPONSE = {
    "response": [
        {
            "team": {"id": 9, "name": "Spain"},
            "players": [
                {"id": 999, "name": "Lamine Yamal", "age": 18, "number": 19, "position": "Attacker"},
            ],
        }
    ]
}

_INJURIES_RESPONSE = {
    "response": [
        {"player": {"id": 999, "name": "Lamine Yamal"}, "player_injury": {"type": "Hamstring"}},
    ]
}

_STATS_RESPONSE = {
    "response": [
        {
            "player": {"id": 999, "name": "Lamine Yamal"},
            "statistics": [{
                "games": {"minutes": 1800, "appearences": 20, "position": "Attacker"},
                "goals": {"total": 9, "assists": 5},
                "shots": {"on": 30},
                "penalty": {"scored": 2},
            }],
        }
    ]
}


def _client(tmp_path, fetcher=None, api_key="test-key"):
    return ApiFootballClient(
        api_key=api_key, cache_dir=str(tmp_path), _fetcher=fetcher,
    )


class TestLiveSquadLoader:
    def test_returns_live_squad_when_api_key_present(self, tmp_path):
        client = _client(tmp_path, fetcher=lambda url, headers, params: _SQUAD_RESPONSE)
        players, label = load_live_squad(client, 9, "Spain")
        assert len(players) == 1
        assert label == SQUAD_LIVE

    def test_falls_back_when_no_api_key(self, tmp_path):
        client = _client(tmp_path, api_key="")
        players, label = load_live_squad(client, 9, "Spain")
        assert players == []
        assert label == SQUAD_NO_KEY

    def test_falls_back_when_response_empty(self, tmp_path):
        client = _client(tmp_path, fetcher=lambda url, headers, params: {"response": []})
        players, label = load_live_squad(client, 9, "Spain")
        assert players == []
        assert label == SQUAD_EMPTY


class TestLiveInjuryLoader:
    def test_returns_live_injuries_when_present(self, tmp_path):
        client = _client(tmp_path, fetcher=lambda url, headers, params: _INJURIES_RESPONSE)
        injured, label = load_live_injuries(client, 9)
        assert injured == {"999"}
        assert label == INJ_LIVE

    def test_falls_back_when_no_api_key(self, tmp_path):
        client = _client(tmp_path, api_key="")
        injured, label = load_live_injuries(client, 9)
        assert injured == set()
        assert label == INJ_NO_KEY

    def test_no_injuries_returns_neutral_label(self, tmp_path):
        client = _client(tmp_path, fetcher=lambda url, headers, params: {"response": []})
        injured, label = load_live_injuries(client, 9)
        assert injured == set()
        assert label == INJ_NONE


class TestLivePlayerStatsLoader:
    def test_returns_live_stats_when_present(self, tmp_path):
        client = _client(tmp_path, fetcher=lambda url, headers, params: _STATS_RESPONSE)
        stats, label = load_live_player_stats(client, 9, 2025)
        assert "999" in stats
        assert label == STATS_LIVE

    def test_falls_back_when_no_api_key(self, tmp_path):
        client = _client(tmp_path, api_key="")
        stats, label = load_live_player_stats(client, 9, 2025)
        assert stats == {}
        assert label == STATS_NO_KEY

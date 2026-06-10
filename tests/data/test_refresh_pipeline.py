from __future__ import annotations

from src.data.api_football_client import ApiFootballClient
from src.data.refresh_pipeline import refresh_team_data


_SQUAD_RESPONSE = {
    "response": [{"team": {"id": 9, "name": "Spain"}, "players": [
        {"id": 999, "name": "Lamine Yamal", "age": 18, "number": 19, "position": "Attacker"},
    ]}]
}
_INJURIES_RESPONSE = {"response": [
    {"player": {"id": 999, "name": "Lamine Yamal"}, "player_injury": {"type": "Hamstring"}},
]}
_STATS_RESPONSE = {"response": [{
    "player": {"id": 999, "name": "Lamine Yamal"},
    "statistics": [{"games": {"minutes": 1800, "appearences": 20, "position": "Attacker"},
                     "goals": {"total": 9, "assists": 5}, "shots": {"on": 30}, "penalty": {"scored": 0}}],
}]}


def _fetcher(url, headers, params):
    if "squads" in url:
        return _SQUAD_RESPONSE
    if "injuries" in url:
        return _INJURIES_RESPONSE
    return _STATS_RESPONSE


def test_refresh_calls_loaders_and_counts_results(tmp_path):
    client = ApiFootballClient(api_key="test-key", cache_dir=str(tmp_path), _fetcher=_fetcher)
    summary = refresh_team_data(client, [("Spain", 9)])

    assert summary.squads_refreshed == 1
    assert summary.injuries_refreshed == 1
    assert summary.stats_refreshed == 1
    t = summary.teams[0]
    assert t.squad_count == 1
    assert t.injury_count == 1
    assert t.stats_count == 1
    assert t.used_live_data is True
    assert summary.timestamp


def test_refresh_falls_back_gracefully_without_api_key(tmp_path):
    client = ApiFootballClient(api_key="", cache_dir=str(tmp_path))
    summary = refresh_team_data(client, [("Spain", 9), ("Norway", 1090)])

    assert summary.squads_refreshed == 0
    assert summary.injuries_refreshed == 0
    assert summary.stats_refreshed == 0
    for t in summary.teams:
        assert t.used_live_data is False
        assert "Fallback" in t.squad_source

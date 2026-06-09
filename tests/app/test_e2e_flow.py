"""End-to-end flow tests: live fixtures → select match → full prediction.

No live API calls — all HTTP injected via mock _fetcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app.prediction_runner import RunnerInput, run_full_prediction
from src.app.selected_fixture import SelectedFixture, create_selected_fixture, get_api_fixture_id
from src.data.api_football_client import ApiFootballClient
from src.data.fixture_provider import FixtureSource, get_fixtures
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.tournament.fixtures import Fixture


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mk_snap(elo=1850, ppg=1.6):
    return TeamSnapshot(elo=float(elo), ppg=float(ppg))


def _mk_par(alpha=1.1, beta=0.9):
    return StrengthParams(alpha_attack=alpha, beta_defense=beta, matches_used=8)


def _fixture_item(fixture_id=855744, home="Brazil", away="France",
                  round_="Group Stage - 1", status="NS",
                  date="2026-06-14T18:00:00+00:00"):
    return {
        "fixture": {
            "id": fixture_id, "date": date,
            "venue": {"name": "Stadium", "city": "City"},
            "status": {"short": status, "long": "Not Started"},
        },
        "league": {"id": 1, "name": "World Cup", "round": round_},
        "teams": {
            "home": {"id": 1, "name": home},
            "away": {"id": 2, "name": away},
        },
    }


def _mock_client(tmp_path, items=None):
    data = {"results": len(items or [_fixture_item()]),
            "response": items or [_fixture_item()]}
    return ApiFootballClient(
        api_key="test_key",
        cache_dir=tmp_path / "cache",
        _fetcher=lambda url, headers, params: data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full e2e flow
# ─────────────────────────────────────────────────────────────────────────────

class TestE2EFlow:
    def test_live_fixtures_to_prediction(self, tmp_path):
        """Complete flow: mock API → fixtures → select → prediction."""
        prov = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        assert prov.fixture_count >= 1

        fixture = prov.fixtures[0]
        sf      = create_selected_fixture(fixture, FixtureSource.API)
        assert is_valid(sf)

        snap_a = _mk_snap(1900, 1.8)
        snap_b = _mk_snap(1800, 1.5)
        par_a  = _mk_par(1.2, 0.85)
        par_b  = _mk_par(1.0, 1.0)
        inp    = RunnerInput(fixture.team_a, fixture.team_b, snap_a, snap_b, par_a, par_b)
        r      = run_full_prediction(inp)

        assert abs(r.win_a + r.draw + r.win_b - 1.0) < 0.001
        assert len(r.top_scorelines) >= 5
        assert len(r.markets.over_under) > 0
        assert r.recommendations is not None
        assert r.explanation is not None

    def test_fixture_id_preserved_to_prediction(self, tmp_path):
        items = [_fixture_item(fixture_id=999777, home="Germany", away="Spain")]
        prov    = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path, items))
        fixture = prov.fixtures[0]
        sf      = create_selected_fixture(fixture, FixtureSource.API)
        assert get_api_fixture_id(sf) == 999777

    def test_multiple_matches_all_produce_valid_predictions(self, tmp_path):
        items = [
            _fixture_item(1, "Brazil",   "France",   date="2026-06-14T18:00:00+00:00"),
            _fixture_item(2, "Germany",  "Spain",    date="2026-06-15T15:00:00+00:00"),
            _fixture_item(3, "Argentina","Portugal", date="2026-06-16T21:00:00+00:00"),
        ]
        prov = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path, items))
        assert prov.fixture_count == 3

        snap = _mk_snap(); par = _mk_par()
        for fixture in prov.fixtures:
            inp = RunnerInput(fixture.team_a, fixture.team_b, snap, snap, par, par)
            r   = run_full_prediction(inp)
            assert abs(r.win_a + r.draw + r.win_b - 1.0) < 0.001, \
                f"{fixture.team_a} vs {fixture.team_b}: probs don't sum to 1"

    def test_lineup_fallback_works(self, tmp_path):
        """When no lineup data is available, prediction still completes."""
        prov    = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        fixture = prov.fixtures[0]
        snap    = _mk_snap()
        par     = _mk_par()
        inp     = RunnerInput(
            fixture.team_a, fixture.team_b, snap, snap, par, par,
            lineup_source="Lineups not yet available — using model baseline / manual override",
        )
        r = run_full_prediction(inp)
        assert r.lineup_source == "Lineups not yet available — using model baseline / manual override"
        assert abs(r.win_a + r.draw + r.win_b - 1.0) < 0.001

    def test_csv_fallback_produces_valid_prediction(self):
        """CSV fallback (no API key) still produces a prediction."""
        prov = get_fixtures(FixtureSource.CSV)
        assert prov.fixture_count >= 1
        fixture = prov.fixtures[0]
        snap = _mk_snap(); par = _mk_par()
        inp  = RunnerInput(fixture.team_a, fixture.team_b, snap, snap, par, par)
        r    = run_full_prediction(inp)
        assert abs(r.win_a + r.draw + r.win_b - 1.0) < 0.001

    def test_all_betting_markets_present_for_any_match(self, tmp_path):
        items = [_fixture_item(home="Mexico", away="South Africa")]
        prov    = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path, items))
        fixture = prov.fixtures[0]
        snap    = _mk_snap(2061, 1.8)
        par_home = _mk_par(2.44, 0.44)
        par_away = _mk_par(1.33, 0.76)
        inp  = RunnerInput(fixture.team_a, fixture.team_b, snap, _mk_snap(1728, 1.0),
                           par_home, par_away)
        r    = run_full_prediction(inp)
        m    = r.markets
        assert len(m.over_under)    > 0
        assert len(m.btts)          > 0
        assert len(m.double_chance) > 0
        assert len(m.draw_no_bet)   > 0
        assert len(m.team_totals)   > 0
        assert len(m.clean_sheet)   > 0

    def test_top_signals_present(self, tmp_path):
        prov = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        fixture = prov.fixtures[0]
        snap = _mk_snap(); par = _mk_par()
        inp  = RunnerInput(fixture.team_a, fixture.team_b, snap, snap, par, par)
        r    = run_full_prediction(inp)
        assert r.recommendations is not None
        assert hasattr(r.recommendations, "recommendations")

    def test_explanation_present(self, tmp_path):
        prov = get_fixtures(FixtureSource.API, api_client=_mock_client(tmp_path))
        fixture = prov.fixtures[0]
        snap = _mk_snap(); par = _mk_par()
        inp  = RunnerInput(fixture.team_a, fixture.team_b, snap, snap, par, par)
        r    = run_full_prediction(inp)
        assert r.explanation is not None
        assert len(r.explanation.drivers) > 0


def is_valid(sf):
    from src.app.selected_fixture import is_valid_selected_fixture
    return is_valid_selected_fixture(sf)

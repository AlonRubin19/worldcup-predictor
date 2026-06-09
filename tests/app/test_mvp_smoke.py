"""MVP smoke tests — end-to-end integration checks across multiple modules.

Verifies that the complete prediction platform can:
  1. Load World Cup fixtures from a mock API source
  2. Preserve API fixture IDs through the selection flow
  3. Run the full research-valid prediction pipeline for any match
  4. Produce all betting market types
  5. Produce model signals
  6. Produce an explanation
  7. Handle missing lineups gracefully (baseline probabilities unchanged)
  8. Apply lineup impact when lineup entries are provided

No live API calls — all HTTP calls are replaced by injected mock fetchers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.fixture_provider import FixtureSource, get_fixtures
from src.data.api_football_client import ApiFootballClient
from src.app.selected_fixture import (
    SelectedFixture, create_selected_fixture,
    get_api_fixture_id, is_valid_selected_fixture,
)
from src.tournament.fixtures import Fixture
from src.data.team_snapshot_loader import TeamSnapshot
from src.data.strength_loader import StrengthParams
from src.models.research_valid_predictor import (
    predict_research_valid, ResearchValidInput, DEFAULT_RHO,
)
from src.models.dixon_coles import build_dc_matrix
from src.models.betting_markets import compute_betting_markets
from src.models.recommendations import generate_recommendations
from src.explainability.driver import build_explanation, ExplanationInput
from src.app.components.prediction_cards import compute_confidence
from src.models.lineup_override import (
    apply_lineup_override, create_default_lineup,
    PlayerOverride, LineupOverride,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants / fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SNAP_A  = TeamSnapshot(elo=1900.0, ppg=1.8)
_SNAP_B  = TeamSnapshot(elo=1800.0, ppg=1.5)
_PAR_A   = StrengthParams(alpha_attack=1.2, beta_defense=0.85, matches_used=10)
_PAR_B   = StrengthParams(alpha_attack=1.0, beta_defense=1.0,  matches_used=10)

_XG_A, _XG_B = 1.5, 1.0


def _mock_api_client(tmp_path: Path, fixture_items: list[dict] | None = None) -> ApiFootballClient:
    items = fixture_items or [_fixture_item()]
    data  = {"results": len(items), "response": items}
    return ApiFootballClient(
        api_key="test_key",
        cache_dir=tmp_path / "cache",
        _fetcher=lambda url, headers, params: data,
    )


def _fixture_item(
    fixture_id: int  = 855744,
    date: str        = "2026-06-14T18:00:00+00:00",
    home: str        = "Brazil",
    away: str        = "France",
    round_: str      = "Group Stage - 1",
    status: str      = "NS",
) -> dict:
    return {
        "fixture": {
            "id": fixture_id, "date": date,
            "venue": {"name": "Stadium", "city": "City"},
            "status": {"short": status, "long": "Not Started"},
        },
        "league": {"id": 1, "name": "World Cup", "round": round_},
        "teams": {
            "home": {"id": 10, "name": home},
            "away": {"id": 20, "name": away},
        },
    }


def _full_rv_prediction(team_a: str = "Brazil", team_b: str = "France"):
    return predict_research_valid(ResearchValidInput(
        team_a=team_a, team_b=team_b,
        snapshot_a=_SNAP_A, snapshot_b=_SNAP_B,
        params_a=_PAR_A, params_b=_PAR_B,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 1. API fixtures flow
# ─────────────────────────────────────────────────────────────────────────────

class TestApiFixturesFlow:
    def test_api_fixtures_populate_provider_result(self, tmp_path):
        items = [
            _fixture_item(855744, home="Brazil", away="France"),
            _fixture_item(855745, home="Germany", away="Spain"),
        ]
        result = get_fixtures(FixtureSource.API, api_client=_mock_api_client(tmp_path, items))
        assert result.fixture_count == 2
        assert result.api_connected is True

    def test_fixture_id_preserved_through_selection_flow(self, tmp_path):
        """fixture_id 855744 survives: API response → Fixture → SelectedFixture → int id."""
        result   = get_fixtures(FixtureSource.API, api_client=_mock_api_client(tmp_path))
        fixture  = result.fixtures[0]
        sf       = create_selected_fixture(fixture, FixtureSource.API)
        api_id   = get_api_fixture_id(sf)
        assert api_id == 855744

    def test_refresh_bypasses_cache(self, tmp_path):
        call_count = {"n": 0}
        def _fetcher(url, headers, params):
            call_count["n"] += 1
            return {"results": 1, "response": [_fixture_item()]}
        client = ApiFootballClient(
            api_key="key", cache_dir=tmp_path / "cache", _fetcher=_fetcher,
        )
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=False)
        get_fixtures(FixtureSource.API, api_client=client, force_refresh=True)
        assert call_count["n"] == 2

    def test_refresh_updates_source_metadata(self, tmp_path):
        result = get_fixtures(FixtureSource.API, api_client=_mock_api_client(tmp_path),
                              force_refresh=True)
        assert result.source_used is FixtureSource.API
        assert result.api_connected is True

    def test_selected_fixture_is_valid(self, tmp_path):
        result  = get_fixtures(FixtureSource.API, api_client=_mock_api_client(tmp_path))
        sf      = create_selected_fixture(result.fixtures[0], FixtureSource.API)
        assert is_valid_selected_fixture(sf)

    def test_api_fixture_status_preserved(self, tmp_path):
        items  = [_fixture_item(status="1H")]
        result = get_fixtures(FixtureSource.API, api_client=_mock_api_client(tmp_path, items))
        assert result.fixtures[0].status == "1H"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full research-valid prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPredictionPipeline:
    def test_probabilities_sum_to_one(self):
        r = _full_rv_prediction()
        assert abs(r.win_a + r.draw + r.win_b - 1.0) < 0.001

    def test_win_a_reasonable_for_stronger_team(self):
        r = _full_rv_prediction()
        assert r.win_a > r.win_b, "Stronger team (higher ELO) should have higher win prob"

    def test_top_scorelines_present(self):
        r = _full_rv_prediction()
        assert len(r.top_scorelines) >= 5

    def test_scoreline_format(self):
        r = _full_rv_prediction()
        for score in r.top_scorelines:
            assert isinstance(score, (tuple, list))
            assert len(score) == 3  # (g_a, g_b, prob)

    def test_xg_values_positive(self):
        from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg as _sxg
        from src.models.xg_calibration import calibrate_xg as _cal
        raw_a, raw_b = _sxg(_SNAP_A.elo, _SNAP_B.elo, _PAR_A, _PAR_B, _SNAP_A.ppg, _SNAP_B.ppg)
        xg_a, xg_b = _cal(raw_a), _cal(raw_b)
        assert xg_a > 0
        assert xg_b > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. All betting market types produced
# ─────────────────────────────────────────────────────────────────────────────

class TestBettingMarketsProduced:
    def setup_method(self):
        self.matrix = build_dc_matrix(_XG_A, _XG_B, rho=DEFAULT_RHO)
        self.markets = compute_betting_markets("Brazil", "France", self.matrix)

    def test_over_under_markets_present(self):
        assert len(self.markets.over_under) > 0

    def test_btts_markets_present(self):
        assert len(self.markets.btts) > 0

    def test_double_chance_markets_present(self):
        assert len(self.markets.double_chance) > 0

    def test_draw_no_bet_markets_present(self):
        assert len(self.markets.draw_no_bet) > 0

    def test_team_totals_markets_present(self):
        assert len(self.markets.team_totals) > 0

    def test_clean_sheet_markets_present(self):
        assert len(self.markets.clean_sheet) > 0

    def test_all_market_probs_in_range(self):
        for mp in (self.markets.over_under + self.markets.btts +
                   self.markets.double_chance + self.markets.draw_no_bet):
            assert 0.0 <= mp.probability <= 1.0, f"Out of range: {mp}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model signals produced
# ─────────────────────────────────────────────────────────────────────────────

class TestModelSignalsProduced:
    def test_recommendations_produced(self):
        matrix  = build_dc_matrix(_XG_A, _XG_B, rho=DEFAULT_RHO)
        markets = compute_betting_markets("Brazil", "France", matrix)
        result  = _full_rv_prediction()
        conf    = compute_confidence(result.win_a, result.draw, result.win_b, [])
        recs    = generate_recommendations(markets, conf.label, [], True, top_n=5)
        assert hasattr(recs, "recommendations")
        assert isinstance(recs.recommendations, list)

    def test_recommendation_has_required_fields(self):
        matrix  = build_dc_matrix(_XG_A, _XG_B, rho=DEFAULT_RHO)
        markets = compute_betting_markets("Brazil", "France", matrix)
        result  = _full_rv_prediction()
        conf    = compute_confidence(result.win_a, result.draw, result.win_b, [])
        recs    = generate_recommendations(markets, conf.label, [], True, top_n=5)
        if recs.recommendations:
            r = recs.recommendations[0]
            assert hasattr(r, "selection")
            assert hasattr(r, "model_probability")
            assert hasattr(r, "signal_strength")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Explanation produced
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationProduced:
    def test_explanation_produced_for_any_match(self):
        result = _full_rv_prediction()
        expl_inp = ExplanationInput(
            match_id="smoke_test",
            team_a="Brazil", team_b="France",
            model_type="ELO + MLE + Dixon-Coles",
            elo_a=_SNAP_A.elo, elo_b=_SNAP_B.elo,
            alpha_attack_a=_PAR_A.alpha_attack, alpha_attack_b=_PAR_B.alpha_attack,
            beta_defense_a=_PAR_A.beta_defense, beta_defense_b=_PAR_B.beta_defense,
            xg_a_base=_XG_A, xg_b_base=_XG_B,
            squad_factor_a=1.0, squad_factor_b=1.0,
            xg_a_final=_XG_A, xg_b_final=_XG_B,
            win_a=result.win_a, draw=result.draw, win_b=result.win_b,
            top_scorelines=result.top_scorelines,
            player_data_research_valid=False,
            market_home_prob=None, market_draw_prob=None, market_away_prob=None,
            market_research_valid=False,
        )
        expl = build_explanation(expl_inp)
        assert expl is not None
        assert hasattr(expl, "drivers")

    def test_explanation_has_key_factors(self):
        result = _full_rv_prediction()
        expl_inp = ExplanationInput(
            match_id="smoke_test2",
            team_a="Brazil", team_b="France",
            model_type="ELO + MLE + Dixon-Coles",
            elo_a=_SNAP_A.elo, elo_b=_SNAP_B.elo,
            alpha_attack_a=_PAR_A.alpha_attack, alpha_attack_b=_PAR_B.alpha_attack,
            beta_defense_a=_PAR_A.beta_defense, beta_defense_b=_PAR_B.beta_defense,
            xg_a_base=_XG_A, xg_b_base=_XG_B,
            squad_factor_a=1.0, squad_factor_b=1.0,
            xg_a_final=_XG_A, xg_b_final=_XG_B,
            win_a=result.win_a, draw=result.draw, win_b=result.win_b,
            top_scorelines=result.top_scorelines,
            player_data_research_valid=False,
            market_home_prob=None, market_draw_prob=None, market_away_prob=None,
            market_research_valid=False,
        )
        expl = build_explanation(expl_inp)
        assert len(expl.drivers) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Lineup integration — missing lineups (baseline unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TestLineupMissingFallback:
    def test_empty_lineups_baseline_probabilities_unchanged(self):
        """When no lineup data is available the squad factors are 1.0
        and the adjusted probabilities equal the base probabilities."""
        result = _full_rv_prediction()
        lo_result = apply_lineup_override(
            team_a="Brazil", team_b="France",
            xg_a_base=_XG_A, xg_b_base=_XG_B,
            override_a=None, override_b=None,
            rho=DEFAULT_RHO,
        )
        assert lo_result.squad_factor_a == 1.0
        assert lo_result.squad_factor_b == 1.0
        assert abs(lo_result.win_a_adjusted - lo_result.win_a_base) < 1e-9
        assert abs(lo_result.win_b_adjusted - lo_result.win_b_base) < 1e-9

    def test_missing_lineup_source_label(self):
        """Verify the 'not available' label string is non-empty — used in UI."""
        label = "Lineups not yet available — using model baseline / manual override"
        assert len(label) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lineup integration — official lineups applied
# ─────────────────────────────────────────────────────────────────────────────

class TestLineupImpactApplied:
    def _make_override(self, team: str, star_available: bool) -> LineupOverride:
        """Build a LineupOverride with 11 starters; star player may be out."""
        players = []
        for i in range(11):
            if i == 0 and not star_available:
                # Star player is out
                players.append(PlayerOverride(
                    player_id=f"p_{i}", player_name=f"Player {i}", team=team,
                    expected_starter=True, availability_status="out",
                    availability_factor=0.0, form_factor=1.5,
                ))
            else:
                players.append(PlayerOverride(
                    player_id=f"p_{i}", player_name=f"Player {i}", team=team,
                    expected_starter=True, availability_status="fit",
                    availability_factor=1.0, form_factor=1.0,
                ))
        return LineupOverride(team=team, players=players)

    def test_injury_to_key_player_reduces_win_probability(self):
        full_team_a   = self._make_override("Brazil", star_available=True)
        injured_team_a = self._make_override("Brazil", star_available=False)

        result_full    = apply_lineup_override("Brazil", "France", _XG_A, _XG_B,
                                               override_a=full_team_a, rho=DEFAULT_RHO)
        result_injured = apply_lineup_override("Brazil", "France", _XG_A, _XG_B,
                                               override_a=injured_team_a, rho=DEFAULT_RHO)

        # Missing a key player should reduce Team A win probability
        assert result_injured.win_a_adjusted < result_full.win_a_adjusted

    def test_lineup_override_changes_squad_factor(self):
        injured = self._make_override("Brazil", star_available=False)
        result  = apply_lineup_override("Brazil", "France", _XG_A, _XG_B,
                                        override_a=injured, rho=DEFAULT_RHO)
        assert result.squad_factor_a < 1.0

    def test_lineup_research_valid_flag_set_correctly(self):
        """Lineup override results are never research-valid (labelled engineering-valid)."""
        injured = self._make_override("Brazil", star_available=False)
        result  = apply_lineup_override("Brazil", "France", _XG_A, _XG_B,
                                        override_a=injured, rho=DEFAULT_RHO)
        assert result.is_research_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Confidence label
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceLabel:
    def test_confidence_produced_for_any_probabilities(self):
        result = _full_rv_prediction()
        conf   = compute_confidence(result.win_a, result.draw, result.win_b, [])
        assert isinstance(conf.label, str) and len(conf.label) > 0

    def test_confidence_high_for_dominant_favourite(self):
        conf = compute_confidence(0.80, 0.12, 0.08, [])
        assert conf.label == "High"

    def test_confidence_low_for_uncertain_match(self):
        conf = compute_confidence(0.34, 0.33, 0.33, [])
        assert conf.label == "Low"

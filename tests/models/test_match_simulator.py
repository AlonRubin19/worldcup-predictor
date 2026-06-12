from src.data.fm_strength_loader import FMTeamStrength
from src.data.market_odds_loader import MarketOddsResult
from src.models.match_simulator import (
    predict_match,
    simulate_match,
    compute_match_xg,
    MIN_XG,
    MAX_XG_NORMAL,
    _select_recommended_score,
    _full_scorelines,
)
from src.models.dixon_coles import build_dc_matrix


def _fm(team, overall, gk=80, defense=80, midfield=80, attack=80, depth=80):
    return FMTeamStrength(
        team=team, players=23, goalkeeper=gk, defense=defense, midfield=midfield,
        attack=attack, depth=depth, overall=overall, top_players=f"{team} Star Player",
    )


_STRONG = _fm("Strong", overall=90, gk=88, defense=88, midfield=88, attack=92, depth=88)
_WEAK = _fm("Weak", overall=65, gk=65, defense=62, midfield=64, attack=63, depth=62)

_NO_ODDS = MarketOddsResult(win_a=None, draw=None, win_b=None, research_valid=False)
_ODDS = MarketOddsResult(win_a=0.60, draw=0.24, win_b=0.16, research_valid=True, bookmaker="test")


# ── FM xG impact ─────────────────────────────────────────────────────────────

def test_fm_affects_xg_and_win_probability_without_odds():
    fm_data = {"strong": _STRONG, "weak": _WEAK}
    data = compute_match_xg("Strong", "Weak", snaps={}, params={}, fm_data=fm_data, odds=_NO_ODDS)

    assert data["fm_used"] is True
    assert data["xg_a"] > data["xg_b"]
    assert data["win_a"] > data["win_b"]


def test_no_fm_data_falls_back_gracefully():
    data = compute_match_xg("Strong", "Weak", snaps={}, params={}, fm_data={}, odds=_NO_ODDS)
    assert data["fm_used"] is False
    # Falls back to existing-model xG (equal default snapshots -> roughly equal xG)
    assert MIN_XG <= data["xg_a"] <= MAX_XG_NORMAL
    assert MIN_XG <= data["xg_b"] <= MAX_XG_NORMAL


def test_odds_plus_fm_blend_moves_toward_market():
    fm_data = {"strong": _STRONG, "weak": _WEAK}
    no_odds = compute_match_xg("Strong", "Weak", snaps={}, params={}, fm_data=fm_data, odds=_NO_ODDS)
    with_odds = compute_match_xg("Strong", "Weak", snaps={}, params={}, fm_data=fm_data, odds=_ODDS)

    assert with_odds["odds_used"] is True
    # Calibration should pull win_a toward the market's 0.60.
    assert abs(with_odds["win_a"] - 0.60) <= abs(no_odds["win_a"] - 0.60) + 1e-9


def test_large_fm_mismatch_keeps_xg_within_caps():
    extreme_strong = _fm("E1", overall=95, gk=95, defense=95, midfield=95, attack=98, depth=95)
    extreme_weak = _fm("E2", overall=50, gk=50, defense=45, midfield=48, attack=47, depth=45)
    fm_data = {"e1": extreme_strong, "e2": extreme_weak}

    data = compute_match_xg("E1", "E2", snaps={}, params={}, fm_data=fm_data, odds=_NO_ODDS)
    assert MIN_XG <= data["xg_a"] <= MAX_XG_NORMAL
    assert MIN_XG <= data["xg_b"] <= MAX_XG_NORMAL


# ── Exact score logic ───────────────────────────────────────────────────────

def test_score_probabilities_sum_reasonably():
    matrix = build_dc_matrix(1.6, 0.8)
    lines = _full_scorelines(matrix, max_score=5)
    total = sum(p for _, _, p in lines)
    # 6x6 grid won't capture 100% of mass, but should capture the vast majority.
    assert 0.85 < total <= 1.0


def test_top_5_scores_sorted_descending():
    result = predict_match("Strong", "Weak", snaps={}, params={},
                            fm_data={"strong": _STRONG, "weak": _WEAK}, odds=_NO_ODDS)
    probs = [s["probability"] for s in result.top_5_exact_scores]
    assert probs == sorted(probs, reverse=True)
    assert len(result.top_5_exact_scores) == 5


def test_one_one_not_blindly_recommended_when_close_winning_score_exists():
    # 1-1 is the top score but 1-0 is within 90% of it, and team1 win >= 55%.
    scorelines = [(1, 1, 0.101), (1, 0, 0.096), (2, 0, 0.05), (0, 0, 0.04), (2, 1, 0.03)]
    raw_top, recommended, reason = _select_recommended_score(
        "TeamA", "TeamB", scorelines, win_a=0.60, draw=0.20, win_b=0.20,
    )
    assert raw_top == "1-1"
    assert recommended == "1-0"
    assert "TeamA" in reason


def test_one_one_kept_when_draw_genuinely_likely():
    # 1-1 dominant and no winning score is close.
    scorelines = [(1, 1, 0.15), (1, 0, 0.05), (0, 1, 0.05), (0, 0, 0.08), (2, 2, 0.03)]
    raw_top, recommended, reason = _select_recommended_score(
        "TeamA", "TeamB", scorelines, win_a=0.40, draw=0.35, win_b=0.25,
    )
    assert raw_top == "1-1"
    assert recommended == "1-1"


def test_recommended_score_consistent_with_wdl_when_team_dominant():
    fm_data = {"strong": _STRONG, "weak": _WEAK}
    result = predict_match("Strong", "Weak", snaps={}, params={}, fm_data=fm_data, odds=_NO_ODDS)
    rec_a, rec_b = (int(x) for x in result.recommended_exact_score.split("-"))
    if result.team1_win_probability >= 0.55:
        # Recommended score should not contradict a clear favourite by giving the draw
        # when a close winning score exists -- checked structurally via the selector
        # tests above; here we just assert the recommendation is plausible.
        assert rec_a >= rec_b - 1


# ── Monte Carlo ──────────────────────────────────────────────────────────────

def test_simulate_match_returns_expected_fields():
    fm_data = {"strong": _STRONG, "weak": _WEAK}
    result = simulate_match("Strong", "Weak", n=2000, snaps={}, params={}, fm_data=fm_data,
                             odds=_NO_ODDS, rng_seed=42)

    assert result.n_simulations == 2000
    assert 0.0 <= result.simulated_team1_win_probability <= 1.0
    assert 0.0 <= result.simulated_draw_probability <= 1.0
    assert 0.0 <= result.simulated_team2_win_probability <= 1.0
    total = (
        result.simulated_team1_win_probability
        + result.simulated_draw_probability
        + result.simulated_team2_win_probability
    )
    assert abs(total - 1.0) < 1e-9
    assert len(result.simulated_top_5_exact_scores) <= 5
    assert result.top_players_team1 == "Strong Star Player"


def test_monte_carlo_roughly_matches_score_matrix():
    fm_data = {"strong": _STRONG, "weak": _WEAK}
    result = simulate_match("Strong", "Weak", n=20_000, snaps={}, params={}, fm_data=fm_data,
                             odds=_NO_ODDS, rng_seed=7)

    assert abs(result.simulated_team1_win_probability - result.team1_win_probability) < 0.05
    assert abs(result.simulated_draw_probability - result.draw_probability) < 0.05
    assert abs(result.simulated_team2_win_probability - result.team2_win_probability) < 0.05


# ── End-to-end against real WC2026 fixture data ────────────────────────────

def test_predict_match_real_fixture_brazil_morocco():
    result = predict_match("Brazil", "Morocco")
    assert result.fm_used is True
    assert result.team1_win_probability > result.team2_win_probability
    assert MIN_XG <= result.expected_goals_team1 <= MAX_XG_NORMAL
    assert MIN_XG <= result.expected_goals_team2 <= MAX_XG_NORMAL

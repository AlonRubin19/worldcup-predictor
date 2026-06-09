"""Tests for the daily match board components.

TDD: all tests written RED-first before any production code exists.
"""

import pytest
from src.app.components.daily_match_board import (
    MatchPrediction,
    MatchBoardRow,
    filter_fixtures_by_date,
    sort_matches_by_datetime,
    build_match_board_row,
    build_daily_match_rows,
    format_board_row_as_dict,
)
from src.tournament.fixtures import Fixture


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures (inline, no CSV load)
# ─────────────────────────────────────────────────────────────────────────────

def _fixture(
    match_id="1", stage="group", group="A",
    date="2022-11-21", team_a="England", team_b="Iran",
) -> Fixture:
    return Fixture(match_id=match_id, stage=stage, group=group,
                   date=date, team_a=team_a, team_b=team_b)


def _prediction(
    win_a=0.55, draw=0.25, win_b=0.20,
    most_likely_score="1-0",
    over_25=0.64, btts_yes=0.48,
    top_signal="1X: 80.0%", top_signal_strength="Strong",
    confidence_label="High", is_research_valid=True,
) -> MatchPrediction:
    return MatchPrediction(
        win_a=win_a, draw=draw, win_b=win_b,
        most_likely_score=most_likely_score,
        over_25=over_25, btts_yes=btts_yes,
        top_signal=top_signal, top_signal_strength=top_signal_strength,
        confidence_label=confidence_label, is_research_valid=is_research_valid,
    )


def _multi_fixtures() -> list[Fixture]:
    return [
        _fixture("1", date="2022-11-21", team_a="England",     team_b="Iran"),
        _fixture("2", date="2022-11-21", team_a="USA",         team_b="Wales"),
        _fixture("3", date="2022-11-22", team_a="Argentina",   team_b="Saudi Arabia"),
        _fixture("4", date="2022-11-22", team_a="Denmark",     team_b="Tunisia"),
        _fixture("5", date="2022-11-20", team_a="Qatar",       team_b="Ecuador"),
    ]


def _predictions_dict(fixtures: list[Fixture]) -> dict[str, MatchPrediction]:
    return {f.match_id: _prediction() for f in fixtures}


# ─────────────────────────────────────────────────────────────────────────────
# MatchPrediction dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_match_prediction_has_required_fields():
    p = _prediction()
    assert hasattr(p, "win_a")
    assert hasattr(p, "draw")
    assert hasattr(p, "win_b")
    assert hasattr(p, "most_likely_score")
    assert hasattr(p, "over_25")
    assert hasattr(p, "btts_yes")
    assert hasattr(p, "top_signal")
    assert hasattr(p, "top_signal_strength")
    assert hasattr(p, "confidence_label")
    assert hasattr(p, "is_research_valid")


def test_match_prediction_defaults_research_valid_true():
    p = MatchPrediction(
        win_a=0.5, draw=0.3, win_b=0.2,
        most_likely_score="1-1", over_25=0.5, btts_yes=0.4,
        top_signal="Over 0.5: 88%", top_signal_strength="Strong",
        confidence_label="Medium",
    )
    assert p.is_research_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# filter_fixtures_by_date
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_returns_only_matching_date():
    fixtures = _multi_fixtures()
    result = filter_fixtures_by_date(fixtures, "2022-11-21")
    assert len(result) == 2
    assert all(f.date == "2022-11-21" for f in result)


def test_filter_returns_empty_when_no_match():
    fixtures = _multi_fixtures()
    result = filter_fixtures_by_date(fixtures, "1900-01-01")
    assert result == []


def test_filter_returns_all_for_date_with_single_match():
    fixtures = _multi_fixtures()
    result = filter_fixtures_by_date(fixtures, "2022-11-20")
    assert len(result) == 1
    assert result[0].team_a == "Qatar"


def test_filter_returns_all_for_date_with_multiple():
    fixtures = _multi_fixtures()
    result = filter_fixtures_by_date(fixtures, "2022-11-22")
    assert len(result) == 2


def test_filter_preserves_fixture_objects():
    fixtures = _multi_fixtures()
    result = filter_fixtures_by_date(fixtures, "2022-11-21")
    match_ids = {f.match_id for f in result}
    assert match_ids == {"1", "2"}


def test_filter_empty_input_returns_empty():
    assert filter_fixtures_by_date([], "2022-11-21") == []


# ─────────────────────────────────────────────────────────────────────────────
# sort_matches_by_datetime
# ─────────────────────────────────────────────────────────────────────────────

def test_sort_matches_ascending_by_date():
    fixtures = _multi_fixtures()  # dates: 21, 21, 22, 22, 20
    result = sort_matches_by_datetime(fixtures)
    dates = [f.date for f in result]
    assert dates == sorted(dates)


def test_sort_matches_already_sorted_unchanged():
    fixtures = [
        _fixture("A", date="2022-11-20"),
        _fixture("B", date="2022-11-21"),
        _fixture("C", date="2022-11-22"),
    ]
    result = sort_matches_by_datetime(fixtures)
    assert [f.match_id for f in result] == ["A", "B", "C"]


def test_sort_preserves_all_fixtures():
    fixtures = _multi_fixtures()
    result = sort_matches_by_datetime(fixtures)
    assert len(result) == len(fixtures)
    assert {f.match_id for f in result} == {f.match_id for f in fixtures}


def test_sort_empty_returns_empty():
    assert sort_matches_by_datetime([]) == []


def test_sort_matches_within_same_date_stable():
    """Fixtures on the same date should keep relative order (stable sort)."""
    fixtures = [
        _fixture("X", date="2022-11-21", team_a="England"),
        _fixture("Y", date="2022-11-21", team_a="USA"),
    ]
    result = sort_matches_by_datetime(fixtures)
    assert result[0].match_id == "X"
    assert result[1].match_id == "Y"


# ─────────────────────────────────────────────────────────────────────────────
# build_match_board_row
# ─────────────────────────────────────────────────────────────────────────────

def test_build_row_returns_match_board_row():
    row = build_match_board_row(_fixture(), _prediction())
    assert isinstance(row, MatchBoardRow)


def test_build_row_team_names_from_fixture():
    row = build_match_board_row(_fixture(team_a="Brazil", team_b="Serbia"), _prediction())
    assert row.team_a == "Brazil"
    assert row.team_b == "Serbia"


def test_build_row_match_id_from_fixture():
    row = build_match_board_row(_fixture(match_id="42"), _prediction())
    assert row.match_id == "42"


def test_build_row_stage_from_fixture():
    row = build_match_board_row(_fixture(stage="group"), _prediction())
    assert row.stage == "group"


def test_build_row_group_from_fixture():
    row = build_match_board_row(_fixture(group="B"), _prediction())
    assert row.group == "B"


def test_build_row_date_from_fixture():
    row = build_match_board_row(_fixture(date="2022-11-25"), _prediction())
    assert row.date == "2022-11-25"


def test_build_row_probabilities_from_prediction():
    p = _prediction(win_a=0.60, draw=0.22, win_b=0.18)
    row = build_match_board_row(_fixture(), p)
    assert abs(row.win_a - 0.60) < 1e-9
    assert abs(row.draw - 0.22) < 1e-9
    assert abs(row.win_b - 0.18) < 1e-9


def test_build_row_most_likely_score():
    row = build_match_board_row(_fixture(), _prediction(most_likely_score="2-0"))
    assert row.most_likely_score == "2-0"


def test_build_row_over_25():
    row = build_match_board_row(_fixture(), _prediction(over_25=0.71))
    assert abs(row.over_25 - 0.71) < 1e-9


def test_build_row_btts_yes():
    row = build_match_board_row(_fixture(), _prediction(btts_yes=0.53))
    assert abs(row.btts_yes - 0.53) < 1e-9


def test_build_row_top_signal():
    row = build_match_board_row(_fixture(), _prediction(top_signal="Over 0.5: 94%"))
    assert row.top_signal == "Over 0.5: 94%"


def test_build_row_top_signal_strength():
    row = build_match_board_row(_fixture(), _prediction(top_signal_strength="Moderate"))
    assert row.top_signal_strength == "Moderate"


def test_build_row_confidence_label():
    row = build_match_board_row(_fixture(), _prediction(confidence_label="Low"))
    assert row.confidence_label == "Low"


def test_build_row_is_research_valid():
    row_rv   = build_match_board_row(_fixture(), _prediction(is_research_valid=True))
    row_notrv = build_match_board_row(_fixture(), _prediction(is_research_valid=False))
    assert row_rv.is_research_valid is True
    assert row_notrv.is_research_valid is False


def test_build_row_win_probs_are_floats():
    row = build_match_board_row(_fixture(), _prediction())
    assert isinstance(row.win_a, float)
    assert isinstance(row.draw, float)
    assert isinstance(row.win_b, float)


# ─────────────────────────────────────────────────────────────────────────────
# build_daily_match_rows
# ─────────────────────────────────────────────────────────────────────────────

def test_build_daily_rows_returns_list():
    fixtures = _multi_fixtures()
    preds = _predictions_dict(fixtures)
    result = build_daily_match_rows(fixtures, preds)
    assert isinstance(result, list)


def test_build_daily_rows_count_equals_fixtures():
    fixtures = _multi_fixtures()
    preds = _predictions_dict(fixtures)
    result = build_daily_match_rows(fixtures, preds)
    assert len(result) == len(fixtures)


def test_build_daily_rows_all_match_board_row_type():
    fixtures = _multi_fixtures()
    preds = _predictions_dict(fixtures)
    for row in build_daily_match_rows(fixtures, preds):
        assert isinstance(row, MatchBoardRow)


def test_build_daily_rows_match_ids_preserved():
    fixtures = _multi_fixtures()
    preds = _predictions_dict(fixtures)
    result = build_daily_match_rows(fixtures, preds)
    ids = {r.match_id for r in result}
    assert ids == {f.match_id for f in fixtures}


def test_build_daily_rows_skips_fixture_without_prediction():
    fixtures = [
        _fixture("1", team_a="England", team_b="Iran"),
        _fixture("2", team_a="USA",     team_b="Wales"),
    ]
    preds = {"1": _prediction()}  # no prediction for "2"
    result = build_daily_match_rows(fixtures, preds)
    assert len(result) == 1
    assert result[0].match_id == "1"


def test_build_daily_rows_empty_fixtures_returns_empty():
    assert build_daily_match_rows([], {}) == []


def test_build_daily_rows_empty_predictions_returns_empty():
    fixtures = _multi_fixtures()
    assert build_daily_match_rows(fixtures, {}) == []


# ─────────────────────────────────────────────────────────────────────────────
# format_board_row_as_dict
# ─────────────────────────────────────────────────────────────────────────────

def _sample_row() -> MatchBoardRow:
    return build_match_board_row(
        _fixture(match_id="7", stage="group", group="B",
                 date="2022-11-21", team_a="England", team_b="Iran"),
        _prediction(win_a=0.60, draw=0.22, win_b=0.18,
                    most_likely_score="2-0", over_25=0.64, btts_yes=0.48,
                    top_signal="England Win: 60%", top_signal_strength="Moderate",
                    confidence_label="Medium"),
    )


def test_format_as_dict_returns_dict():
    assert isinstance(format_board_row_as_dict(_sample_row()), dict)


def test_format_as_dict_contains_team_names():
    d = format_board_row_as_dict(_sample_row())
    assert "England" in str(d.values()) or any("England" in str(v) for v in d.values())


def test_format_as_dict_probabilities_as_percentages():
    d = format_board_row_as_dict(_sample_row())
    # Win probability columns must be formatted as "xx.x%"
    win_a_val = d.get("Win A") or d.get("England Win") or d.get("TeamA Win")
    if win_a_val:
        assert "%" in str(win_a_val)


def test_format_as_dict_most_likely_score_present():
    d = format_board_row_as_dict(_sample_row())
    assert any("2-0" in str(v) for v in d.values()), (
        f"Expected '2-0' in dict values: {d}"
    )


def test_format_as_dict_over_25_present():
    d = format_board_row_as_dict(_sample_row())
    assert any("64" in str(v) or "0.64" in str(v) or "64.0" in str(v)
               for v in d.values()), f"Expected over_25 value in dict: {d}"


def test_format_as_dict_confidence_present():
    d = format_board_row_as_dict(_sample_row())
    assert any("Medium" in str(v) for v in d.values()), (
        f"Expected 'Medium' confidence in dict: {d}"
    )


def test_format_as_dict_no_private_fields():
    """Dict must not expose internal _ prefixed fields."""
    d = format_board_row_as_dict(_sample_row())
    for key in d:
        assert not str(key).startswith("_"), f"Private field exposed: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: real fixtures + real predictions
# ─────────────────────────────────────────────────────────────────────────────

def test_integration_with_real_fixture_file():
    """Load fixtures from CSV, build predictions, check daily board works."""
    from pathlib import Path
    from src.tournament.fixtures import load_fixtures
    from src.data.team_snapshot_loader import load_team_snapshots
    from src.data.strength_loader import load_strength_params
    from src.models.dixon_coles import build_dc_matrix
    from src.models.betting_markets import compute_betting_markets
    from src.models.recommendations import generate_recommendations
    from src.models.research_valid_predictor import (
        predict_research_valid, ResearchValidInput, DEFAULT_RHO,
    )
    from src.data.team_snapshot_loader import TeamSnapshot
    from src.data.strength_loader import StrengthParams

    fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    fixtures = load_fixtures(fixture_path)
    snaps = load_team_snapshots()
    params = load_strength_params()

    _SNAP = TeamSnapshot(elo=1800.0, ppg=1.5)
    _PAR  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)

    # Build predictions for 2022-11-21
    day_fixtures = filter_fixtures_by_date(fixtures, "2022-11-21")
    assert len(day_fixtures) > 0

    from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
    from src.models.xg_calibration import calibrate_xg

    preds: dict[str, MatchPrediction] = {}
    for f in day_fixtures:
        snap_a = snaps.get(f.team_a, _SNAP)
        snap_b = snaps.get(f.team_b, _SNAP)
        par_a  = params.get(f.team_a, _PAR)
        par_b  = params.get(f.team_b, _PAR)
        raw_a, raw_b = calculate_strength_adjusted_xg(
            snap_a.elo, snap_b.elo, par_a, par_b, snap_a.ppg, snap_b.ppg
        )
        xg_a, xg_b = calibrate_xg(raw_a), calibrate_xg(raw_b)
        m = build_dc_matrix(xg_a, xg_b, rho=DEFAULT_RHO)
        bm = compute_betting_markets(f.team_a, f.team_b, m)
        rs = generate_recommendations(bm, "High", [], True, top_n=1)

        from src.models.research_valid_predictor import predict_research_valid, ResearchValidInput
        rv = predict_research_valid(ResearchValidInput(
            team_a=f.team_a, team_b=f.team_b,
            snapshot_a=snap_a, snapshot_b=snap_b,
            params_a=par_a, params_b=par_b,
        ))

        most_likely = f"{rv.top_scorelines[0][0]}-{rv.top_scorelines[0][1]}"
        over_25 = next(
            (mp.probability for mp in bm.over_under if "Over 2.5" in mp.selection), 0.0
        )
        btts_yes = next(
            (mp.probability for mp in bm.btts if mp.selection == "BTTS Yes"), 0.0
        )
        top_sig  = rs.recommendations[0].selection if rs.recommendations else "—"
        top_str  = rs.recommendations[0].signal_strength if rs.recommendations else "Weak"

        preds[f.match_id] = MatchPrediction(
            win_a=rv.win_a, draw=rv.draw, win_b=rv.win_b,
            most_likely_score=most_likely,
            over_25=over_25, btts_yes=btts_yes,
            top_signal=top_sig, top_signal_strength=top_str,
            confidence_label="High",
        )

    rows = build_daily_match_rows(day_fixtures, preds)
    assert len(rows) == len(day_fixtures)
    for row in rows:
        assert isinstance(row, MatchBoardRow)
        assert 0 <= row.win_a <= 1
        assert 0 <= row.draw <= 1
        assert 0 <= row.win_b <= 1
        assert "-" in row.most_likely_score

"""Tests for plain-English explanation report generation."""

import pytest
from src.explainability.driver import build_explanation, ExplanationInput, PredictionExplanation
from src.explainability.report import generate_report


def _explanation(**overrides) -> PredictionExplanation:
    defaults = dict(
        match_id="99999",
        team_a="Argentina",
        team_b="France",
        model_type="MLE+DC",
        elo_a=2000.0,
        elo_b=1850.0,
        alpha_attack_a=1.2,
        alpha_attack_b=1.0,
        beta_defense_a=1.0,
        beta_defense_b=1.1,
        xg_a_base=1.6,
        xg_b_base=1.3,
        squad_factor_a=1.0,
        squad_factor_b=1.0,
        xg_a_final=1.6,
        xg_b_final=1.3,
        win_a=0.50,
        draw=0.25,
        win_b=0.25,
        top_scorelines=[(1, 0, 0.12), (1, 1, 0.10), (0, 0, 0.08)],
        player_data_research_valid=False,
        market_home_prob=None,
        market_draw_prob=None,
        market_away_prob=None,
        market_research_valid=False,
    )
    defaults.update(overrides)
    inp = ExplanationInput(**defaults)
    return build_explanation(inp)


# --- generate_report returns a string ---

def test_report_returns_string():
    expl = _explanation()
    report = generate_report(expl)
    assert isinstance(report, str)
    assert len(report) > 50


# --- Favoured team mentioned ---

def test_report_mentions_favoured_team_when_clear_favourite():
    expl = _explanation(win_a=0.60, win_b=0.20, draw=0.20)
    report = generate_report(expl)
    assert "Argentina" in report


def test_report_mentions_balanced_when_close_probabilities():
    expl = _explanation(
        elo_a=1900.0, elo_b=1900.0,
        alpha_attack_a=1.0, alpha_attack_b=1.0,
        win_a=0.36, draw=0.28, win_b=0.36,
    )
    report = generate_report(expl)
    lower = report.lower()
    assert "evenly" in lower or "balanced" in lower or "close" in lower


# --- Driver names appear in report ---

def test_report_mentions_elo_when_elo_driver_present():
    expl = _explanation(elo_a=2050.0, elo_b=1800.0)
    report = generate_report(expl)
    assert "elo" in report.lower() or "rating" in report.lower()


def test_report_mentions_player_impact_when_squad_factor_active():
    expl = _explanation(
        squad_factor_a=0.91,
        xg_a_base=1.6,
        xg_a_final=1.456,
        player_data_research_valid=False,
    )
    report = generate_report(expl)
    assert "squad" in report.lower() or "player" in report.lower()


# --- Scoreline mentioned ---

def test_report_mentions_most_likely_scoreline():
    expl = _explanation(top_scorelines=[(1, 0, 0.14), (1, 1, 0.10), (0, 0, 0.09)])
    report = generate_report(expl)
    # Most likely is 1-0
    assert "1-0" in report or "1–0" in report


# --- Warnings appear in report ---

def test_report_includes_player_validity_warning():
    expl = _explanation(
        squad_factor_a=0.93,
        xg_a_base=1.6,
        xg_a_final=1.488,
        player_data_research_valid=False,
    )
    report = generate_report(expl)
    assert "engineering" in report.lower() or "not research" in report.lower() or \
           "validity" in report.lower() or "placeholder" in report.lower()


def test_report_does_not_include_player_warning_when_no_impact():
    expl = _explanation(squad_factor_a=1.0, squad_factor_b=1.0,
                        player_data_research_valid=False)
    report = generate_report(expl)
    lower = report.lower()
    assert "engineering-valid only" not in lower or "player" not in lower


# --- No fabricated claims ---

def test_report_does_not_mention_market_when_placeholder():
    expl = _explanation(
        market_home_prob=0.45,
        market_draw_prob=0.25,
        market_away_prob=0.30,
        market_research_valid=False,
    )
    report = generate_report(expl)
    assert "market" not in report.lower() or "not research" in report.lower()


def test_report_mentions_market_divergence_when_research_valid():
    expl = _explanation(
        win_a=0.58,
        win_b=0.22,
        draw=0.20,
        market_home_prob=0.42,
        market_draw_prob=0.26,
        market_away_prob=0.32,
        market_research_valid=True,
    )
    report = generate_report(expl)
    assert "market" in report.lower()

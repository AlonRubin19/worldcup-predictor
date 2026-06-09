"""Tests for player_impact_runner — backtest comparing with/without player impact."""

import pytest
from pathlib import Path

from src.backtesting.player_impact_runner import (
    run_player_impact_backtest,
    PlayerImpactResult,
    audit_research_validity,
    AuditSummary,
)


PROFILES_CSV = """\
player_id,player_name,team,position,club,minutes_last_90_days,national_team_minutes_last_12_months,goals_per_90,assists_per_90,xg_per_90,xa_per_90,defensive_actions_per_90,international_caps,base_impact_score
p001,Kane,England,FW,Bayern,810,720,0.82,0.31,0.75,0.28,1.2,90,1.35
p002,Saka,England,MF,Arsenal,900,630,0.41,0.52,0.38,0.45,3.1,45,1.15
p003,Bell,England,MF,Madrid,810,720,0.38,0.29,0.35,0.24,4.2,50,1.20
p004,Pickford,England,GK,Everton,900,720,0.0,0.0,0.0,0.0,5.8,70,0.90
p005,Trippier,England,DF,Newcastle,810,720,0.1,0.2,0.09,0.18,5.5,40,0.88
p006,Maguire,England,DF,ManUtd,810,720,0.08,0.05,0.07,0.04,7.2,60,0.85
p007,Walker,England,DF,ManCity,810,720,0.05,0.15,0.04,0.12,6.1,70,0.87
p008,Shaw,England,DF,ManUtd,810,720,0.07,0.20,0.06,0.18,5.9,35,0.86
p009,Rice,England,MF,Arsenal,900,720,0.15,0.18,0.13,0.15,8.2,55,1.05
p010,Mount,England,MF,ManUtd,810,630,0.25,0.30,0.22,0.26,4.8,45,1.00
p011,Sterling,England,FW,Chelsea,720,630,0.45,0.35,0.40,0.30,2.5,75,1.10
p101,Mbappe,France,FW,Madrid,720,630,0.91,0.38,0.85,0.32,1.5,80,1.40
p102,Griezmann,France,MF,Atletico,810,720,0.42,0.45,0.38,0.40,4.2,120,1.25
p103,Dembele,France,MF,Barcelona,810,630,0.38,0.42,0.33,0.37,3.8,55,1.10
p104,Lloris,France,GK,Spurs,900,720,0.0,0.0,0.0,0.0,4.5,140,0.95
p105,Hernandez,France,DF,Milan,810,720,0.12,0.22,0.10,0.19,6.2,45,0.88
p106,Upamecano,France,DF,Bayern,810,720,0.10,0.08,0.09,0.07,8.5,40,0.90
p107,Saliba,France,DF,Arsenal,900,630,0.08,0.06,0.07,0.05,9.0,25,0.87
p108,Kounde,France,DF,Barcelona,810,720,0.09,0.12,0.08,0.10,7.8,40,0.88
p109,Tchouameni,France,MF,Madrid,810,630,0.18,0.15,0.15,0.12,9.2,30,1.00
p110,Rabiot,France,MF,Juve,720,630,0.22,0.20,0.20,0.18,5.5,50,0.95
p111,Giroud,France,FW,Milan,720,630,0.55,0.25,0.50,0.22,2.2,120,1.10
"""

AVAIL_ALL_FIT_CSV = """\
match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor
m001,2022-11-21,England,p001,true,fit,1.0,1.0
m001,2022-11-21,England,p002,true,fit,1.0,1.0
m001,2022-11-21,England,p003,true,fit,1.0,1.0
m001,2022-11-21,England,p004,true,fit,1.0,1.0
m001,2022-11-21,England,p005,true,fit,1.0,1.0
m001,2022-11-21,England,p006,true,fit,1.0,1.0
m001,2022-11-21,England,p007,true,fit,1.0,1.0
m001,2022-11-21,England,p008,true,fit,1.0,1.0
m001,2022-11-21,England,p009,true,fit,1.0,1.0
m001,2022-11-21,England,p010,true,fit,1.0,1.0
m001,2022-11-21,England,p011,true,fit,1.0,1.0
m001,2022-11-21,France,p101,true,fit,1.0,1.0
m001,2022-11-21,France,p102,true,fit,1.0,1.0
m001,2022-11-21,France,p103,true,fit,1.0,1.0
m001,2022-11-21,France,p104,true,fit,1.0,1.0
m001,2022-11-21,France,p105,true,fit,1.0,1.0
m001,2022-11-21,France,p106,true,fit,1.0,1.0
m001,2022-11-21,France,p107,true,fit,1.0,1.0
m001,2022-11-21,France,p108,true,fit,1.0,1.0
m001,2022-11-21,France,p109,true,fit,1.0,1.0
m001,2022-11-21,France,p110,true,fit,1.0,1.0
m001,2022-11-21,France,p111,true,fit,1.0,1.0
"""

AVAIL_KEY_PLAYER_OUT_CSV = """\
match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor
m001,2022-11-21,England,p001,true,out,0.0,1.0
m001,2022-11-21,England,p002,true,fit,1.0,1.0
m001,2022-11-21,England,p003,true,fit,1.0,1.0
m001,2022-11-21,England,p004,true,fit,1.0,1.0
m001,2022-11-21,England,p005,true,fit,1.0,1.0
m001,2022-11-21,England,p006,true,fit,1.0,1.0
m001,2022-11-21,England,p007,true,fit,1.0,1.0
m001,2022-11-21,England,p008,true,fit,1.0,1.0
m001,2022-11-21,England,p009,true,fit,1.0,1.0
m001,2022-11-21,England,p010,true,fit,1.0,1.0
m001,2022-11-21,England,p011,true,fit,1.0,1.0
m001,2022-11-21,France,p101,true,fit,1.0,1.0
m001,2022-11-21,France,p102,true,fit,1.0,1.0
m001,2022-11-21,France,p103,true,fit,1.0,1.0
m001,2022-11-21,France,p104,true,fit,1.0,1.0
m001,2022-11-21,France,p105,true,fit,1.0,1.0
m001,2022-11-21,France,p106,true,fit,1.0,1.0
m001,2022-11-21,France,p107,true,fit,1.0,1.0
m001,2022-11-21,France,p108,true,fit,1.0,1.0
m001,2022-11-21,France,p109,true,fit,1.0,1.0
m001,2022-11-21,France,p110,true,fit,1.0,1.0
m001,2022-11-21,France,p111,true,fit,1.0,1.0
"""

MATCH_RESULTS_CSV = """\
match_id,date,team_a,team_b,team_a_goals,team_b_goals,team_a_elo_pre,team_b_elo_pre,team_a_goals_for_last_10,team_a_goals_against_last_10,team_b_goals_for_last_10,team_b_goals_against_last_10,team_a_points_per_game_last_10,team_b_points_per_game_last_10,team_a_matches_available,team_b_matches_available
m001,2022-11-21,England,France,0,2,2020.0,2085.0,1.8,0.9,2.1,0.7,2.1,2.4,10,10
"""

STRENGTH_PARAMS_CSV = """\
team,alpha_attack,beta_defense,matches_used
England,1.05,0.92,50
France,1.18,0.88,50
"""


def _write_fixtures(tmp_path, avail_csv=AVAIL_ALL_FIT_CSV):
    (tmp_path / "player_profiles.csv").write_text(PROFILES_CSV)
    (tmp_path / "match_player_availability.csv").write_text(avail_csv)
    (tmp_path / "match_results.csv").write_text(MATCH_RESULTS_CSV)
    (tmp_path / "team_strength_params.csv").write_text(STRENGTH_PARAMS_CSV)
    return tmp_path


class TestRunPlayerImpactBacktest:
    def test_returns_list_of_results(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_has_required_fields(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert hasattr(r, "match_id")
        assert hasattr(r, "team_a")
        assert hasattr(r, "team_b")
        assert hasattr(r, "xg_a_base")
        assert hasattr(r, "xg_b_base")
        assert hasattr(r, "xg_a_adjusted")
        assert hasattr(r, "xg_b_adjusted")
        assert hasattr(r, "squad_factor_a")
        assert hasattr(r, "squad_factor_b")
        assert hasattr(r, "win_a_prob_base")
        assert hasattr(r, "win_a_prob_adjusted")
        assert hasattr(r, "actual_outcome")
        assert hasattr(r, "predicted_outcome_base")
        assert hasattr(r, "predicted_outcome_adjusted")

    def test_all_fit_no_xg_change(self, tmp_path):
        """When all starters are fit, player impact should not change xG."""
        d = _write_fixtures(tmp_path, avail_csv=AVAIL_ALL_FIT_CSV)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert r.xg_a_adjusted == pytest.approx(r.xg_a_base, rel=1e-3)
        assert r.xg_b_adjusted == pytest.approx(r.xg_b_base, rel=1e-3)

    def test_key_player_out_changes_xg(self, tmp_path):
        """When England's top scorer is out, England's xG should drop."""
        d = _write_fixtures(tmp_path, avail_csv=AVAIL_KEY_PLAYER_OUT_CSV)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert r.xg_a_adjusted < r.xg_a_base

    def test_squad_factors_in_result(self, tmp_path):
        d = _write_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert 0.85 <= r.squad_factor_a <= 1.15
        assert 0.85 <= r.squad_factor_b <= 1.15

    def test_missing_availability_defaults_gracefully(self, tmp_path):
        """Match with no availability data should use squad_factor=1.0."""
        d = _write_fixtures(tmp_path)
        # Write availability CSV with no rows for match m001
        empty_avail = "match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor\n"
        (tmp_path / "match_player_availability.csv").write_text(empty_avail)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert r.squad_factor_a == pytest.approx(1.0)
        assert r.squad_factor_b == pytest.approx(1.0)
        assert r.xg_a_adjusted == pytest.approx(r.xg_a_base, rel=1e-3)


# ---- audit columns on PlayerImpactResult ----

AVAIL_WITH_AUDIT_CSV = """\
match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor,source_type,research_valid
m001,2022-11-21,England,p001,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p002,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p003,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p004,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p005,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p006,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p007,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p008,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p009,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p010,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,England,p011,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p101,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p102,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p103,true,doubtful,0.6,0.8,manual_assumption,false
m001,2022-11-21,France,p104,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p105,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p106,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p107,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p108,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p109,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p110,true,fit,1.0,1.0,historical_lineup,false
m001,2022-11-21,France,p111,true,fit,1.0,1.0,historical_lineup,false
"""


def _write_audit_fixtures(tmp_path):
    (tmp_path / "player_profiles.csv").write_text(PROFILES_CSV)
    (tmp_path / "match_player_availability.csv").write_text(AVAIL_WITH_AUDIT_CSV)
    (tmp_path / "match_results.csv").write_text(MATCH_RESULTS_CSV)
    (tmp_path / "team_strength_params.csv").write_text(STRENGTH_PARAMS_CSV)
    return tmp_path


class TestPlayerImpactResultAuditFields:
    def test_result_carries_source_types(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert hasattr(r, "source_types_a")
        assert hasattr(r, "source_types_b")
        assert hasattr(r, "any_research_valid_a")
        assert hasattr(r, "any_research_valid_b")

    def test_source_types_captured_for_team(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        # France has a manual_assumption row
        assert "manual_assumption" in r.source_types_b
        # England has only historical_lineup rows
        assert r.source_types_a == {"historical_lineup"}

    def test_any_research_valid_false_when_none_valid(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        r = results[0]
        assert r.any_research_valid_a is False
        assert r.any_research_valid_b is False


class TestAuditResearchValidity:
    def test_all_false_returns_not_research_valid(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        summary = audit_research_validity(results)
        assert summary.is_research_valid is False
        assert summary.research_valid_matches == 0
        assert summary.engineering_valid_matches == 1

    def test_summary_has_disclaimer(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        summary = audit_research_validity(results)
        assert "engineering" in summary.disclaimer.lower()
        assert "not yet research-valid" in summary.disclaimer.lower()

    def test_source_type_breakdown_in_summary(self, tmp_path):
        d = _write_audit_fixtures(tmp_path)
        results = run_player_impact_backtest(
            match_results_path=d / "match_results.csv",
            strength_params_path=d / "team_strength_params.csv",
            player_profiles_path=d / "player_profiles.csv",
            match_availability_path=d / "match_player_availability.csv",
        )
        summary = audit_research_validity(results)
        assert "historical_lineup" in summary.source_type_counts
        assert "manual_assumption" in summary.source_type_counts

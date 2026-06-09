"""Tests for player_loader — load player profiles and match availability."""

import pytest
from pathlib import Path

from src.data.player_loader import (
    PlayerProfile,
    PlayerAvailability,
    load_player_profiles,
    load_match_availability,
)


PROFILES_CSV = """\
player_id,player_name,team,position,club,minutes_last_90_days,national_team_minutes_last_12_months,goals_per_90,assists_per_90,xg_per_90,xa_per_90,defensive_actions_per_90,international_caps,base_impact_score
p001,Harry Kane,England,FW,Bayern Munich,810,720,0.82,0.31,0.75,0.28,1.2,90,1.35
p002,Bukayo Saka,England,MF,Arsenal,900,630,0.41,0.52,0.38,0.45,3.1,45,1.15
p003,Jude Bellingham,England,MF,Real Madrid,810,720,0.38,0.29,0.35,0.24,4.2,50,1.20
p004,Jordan Pickford,England,GK,Everton,900,720,0.0,0.0,0.0,0.0,5.8,70,0.90
p005,Kylian Mbappe,France,FW,Real Madrid,720,630,0.91,0.38,0.85,0.32,1.5,80,1.40
"""

# Old format — no source_type / research_valid columns
AVAILABILITY_CSV_OLD = """\
match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor
m001,2022-11-21,England,p001,true,fit,1.0,1.05
m001,2022-11-21,England,p002,true,fit,1.0,0.95
m001,2022-11-21,England,p003,true,fit,1.0,1.10
m001,2022-11-21,England,p004,true,fit,1.0,1.00
m001,2022-11-21,France,p005,true,fit,1.0,1.15
m002,2022-11-25,England,p001,true,doubtful,0.6,0.80
m002,2022-11-25,England,p002,false,out,0.0,1.00
"""

# New format — with source_type / research_valid columns
AVAILABILITY_CSV_NEW = """\
match_id,date,team,player_id,expected_starter,availability_status,availability_factor,form_factor,source_type,research_valid
m001,2022-11-21,England,p001,true,fit,1.0,1.05,historical_lineup,false
m001,2022-11-21,England,p002,true,fit,1.0,0.95,historical_lineup,false
m001,2022-11-21,England,p003,true,fit,1.0,1.10,historical_lineup,false
m001,2022-11-21,England,p004,true,fit,1.0,1.00,historical_lineup,false
m001,2022-11-21,France,p005,true,fit,1.0,1.15,historical_lineup,false
m002,2022-11-25,England,p001,true,doubtful,0.6,0.80,manual_assumption,false
m002,2022-11-25,England,p002,false,out,0.0,1.00,placeholder,false
"""


def _write_profiles(tmp_path):
    p = tmp_path / "player_profiles.csv"
    p.write_text(PROFILES_CSV)
    return p


class TestLoadPlayerProfiles:
    def test_returns_dict_keyed_by_player_id(self, tmp_path):
        profiles = load_player_profiles(_write_profiles(tmp_path))
        assert isinstance(profiles, dict)
        assert "p001" in profiles

    def test_profile_fields_populated(self, tmp_path):
        p = load_player_profiles(_write_profiles(tmp_path))["p001"]
        assert p.player_name == "Harry Kane"
        assert p.team == "England"
        assert p.position == "FW"
        assert p.base_impact_score == pytest.approx(1.35)

    def test_all_profiles_loaded(self, tmp_path):
        assert len(load_player_profiles(_write_profiles(tmp_path))) == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_player_profiles(tmp_path / "nonexistent.csv")


class TestLoadMatchAvailability:
    def _write(self, tmp_path, content):
        p = tmp_path / "avail.csv"
        p.write_text(content)
        return p

    # --- backward compatibility: old format without audit columns ---

    def test_old_format_loads_without_error(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_OLD))
        assert len(records) == 7

    def test_old_format_defaults_source_type_to_placeholder(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_OLD))
        assert all(r.source_type == "placeholder" for r in records)

    def test_old_format_defaults_research_valid_to_false(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_OLD))
        assert all(r.research_valid is False for r in records)

    # --- new format with audit columns ---

    def test_new_format_loads_source_type(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_NEW))
        rec = next(r for r in records if r.match_id == "m001" and r.player_id == "p001")
        assert rec.source_type == "historical_lineup"

    def test_new_format_loads_research_valid_false(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_NEW))
        rec = next(r for r in records if r.match_id == "m002" and r.player_id == "p001")
        assert rec.research_valid is False

    def test_new_format_manual_assumption_source(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_NEW))
        rec = next(r for r in records if r.match_id == "m002" and r.player_id == "p001")
        assert rec.source_type == "manual_assumption"

    # --- core fields still work ---

    def test_availability_fields_populated(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_NEW))
        rec = next(r for r in records if r.match_id == "m001" and r.player_id == "p001")
        assert rec.team == "England"
        assert rec.expected_starter is True
        assert rec.availability_status == "fit"
        assert rec.availability_factor == pytest.approx(1.0)
        assert rec.form_factor == pytest.approx(1.05)

    def test_expected_starter_false_parsed(self, tmp_path):
        records = load_match_availability(self._write(tmp_path, AVAILABILITY_CSV_NEW))
        rec = next(r for r in records if r.match_id == "m002" and r.player_id == "p002")
        assert rec.expected_starter is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_match_availability(tmp_path / "nonexistent.csv")


class TestResearchValidFlag:
    """Verify that research_valid=true is parsed correctly when present."""

    def test_research_valid_true_parsed(self, tmp_path):
        csv = (
            "match_id,date,team,player_id,expected_starter,availability_status,"
            "availability_factor,form_factor,source_type,research_valid\n"
            "m001,2022-11-21,England,p001,true,fit,1.0,1.0,pre_match_report,true\n"
        )
        p = tmp_path / "avail.csv"
        p.write_text(csv)
        records = load_match_availability(p)
        assert records[0].research_valid is True
        assert records[0].source_type == "pre_match_report"

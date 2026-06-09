"""Tests for tournament fixture loading."""

import pytest
from pathlib import Path
import pandas as pd
from src.tournament.fixtures import load_fixtures, Fixture

_FIXTURE_CSV = """\
match_id,stage,group,date,team_a,team_b
1,group,A,2022-11-20,Qatar,Ecuador
2,group,A,2022-11-25,Qatar,Senegal
3,group,B,2022-11-21,England,Iran
4,round_of_16,,2022-12-03,TBD_A1,TBD_B2
"""


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "fixtures.csv"
    p.write_text(content)
    return p


# ── Loading ───────────────────────────────────────────────────────────────────

def test_load_returns_list_of_fixture(tmp_path):
    p = _write_csv(tmp_path, _FIXTURE_CSV)
    fixtures = load_fixtures(p)
    assert isinstance(fixtures, list)
    assert all(isinstance(f, Fixture) for f in fixtures)


def test_load_correct_count(tmp_path):
    p = _write_csv(tmp_path, _FIXTURE_CSV)
    fixtures = load_fixtures(p)
    assert len(fixtures) == 4


def test_fixture_has_required_fields(tmp_path):
    p = _write_csv(tmp_path, _FIXTURE_CSV)
    f = load_fixtures(p)[0]
    assert hasattr(f, "match_id")
    assert hasattr(f, "stage")
    assert hasattr(f, "group")
    assert hasattr(f, "date")
    assert hasattr(f, "team_a")
    assert hasattr(f, "team_b")


def test_fixture_stage_values(tmp_path):
    p = _write_csv(tmp_path, _FIXTURE_CSV)
    fixtures = load_fixtures(p)
    stages = {f.stage for f in fixtures}
    assert "group" in stages
    assert "round_of_16" in stages


def test_group_fixtures_have_group(tmp_path):
    p = _write_csv(tmp_path, _FIXTURE_CSV)
    fixtures = load_fixtures(p)
    for f in fixtures:
        if f.stage == "group":
            assert f.group in ("A", "B", "C", "D", "E", "F", "G", "H")


def test_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fixtures(tmp_path / "nonexistent.csv")


def test_load_real_fixture_file():
    real = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    fixtures = load_fixtures(real)
    assert len(fixtures) == 48  # 48 group stage matches
    groups = {f.group for f in fixtures if f.stage == "group"}
    assert groups == {"A", "B", "C", "D", "E", "F", "G", "H"}


def test_real_fixture_teams_are_strings():
    real = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
    fixtures = load_fixtures(real)
    for f in fixtures:
        assert isinstance(f.team_a, str)
        assert isinstance(f.team_b, str)
        assert len(f.team_a) > 0

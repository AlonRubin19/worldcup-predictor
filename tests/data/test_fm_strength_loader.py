from pathlib import Path

import pandas as pd
import pytest

from src.data.fm_strength_loader import (
    load_fm_team_strength,
    get_fm_strength,
    normalize_team_name,
)


def test_loads_real_csv():
    fm = load_fm_team_strength()
    assert "brazil" in fm
    assert fm["brazil"].team == "Brazil"
    assert fm["brazil"].overall > 0


def test_missing_file_returns_empty_dict(tmp_path: Path):
    fm = load_fm_team_strength(tmp_path / "does_not_exist.csv")
    assert fm == {}


@pytest.mark.parametrize("alias,canonical", [
    ("USA", "United States"),
    ("United States", "United States"),
    ("USMNT", "United States"),
    ("United States of America", "United States"),
    ("Turkey", "Türkiye"),
    ("Türkiye", "Türkiye"),
    ("Bosnia", "Bosnia and Herzegovina"),
    ("Bosnia and Herzegovina", "Bosnia and Herzegovina"),
    ("IR Iran", "Iran"),
    ("South Korea", "South Korea"),
    ("Korea Republic", "South Korea"),
    ("Ivory Coast", "Ivory Coast"),
    ("Côte d'Ivoire", "Ivory Coast"),
    ("Czech Republic", "Czechia"),
    ("Czechia", "Czechia"),
])
def test_normalize_aliases_match(alias, canonical):
    assert normalize_team_name(alias) == normalize_team_name(canonical)


def test_get_fm_strength_found_and_missing():
    fm = load_fm_team_strength()
    assert get_fm_strength("USA", fm) is not None
    assert get_fm_strength("Turkey", fm) is not None
    assert get_fm_strength("Atlantis", fm) is None


def test_get_fm_strength_handles_missing_team(tmp_path: Path):
    csv_path = tmp_path / "fm.csv"
    pd.DataFrame([{
        "team": "France", "players": 26,
        "fm_goalkeeper_rating": 81.0, "fm_defense_rating": 83.0,
        "fm_midfield_rating": 81.0, "fm_attack_rating": 93.0,
        "fm_depth_rating": 83.0, "fm_overall_rating": 85.0,
        "top_5_players": "Mbappe",
    }]).to_csv(csv_path, index=False)

    fm = load_fm_team_strength(csv_path)
    assert get_fm_strength("France", fm) is not None
    assert get_fm_strength("Germany", fm) is None

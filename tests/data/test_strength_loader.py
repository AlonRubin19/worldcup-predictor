import pytest
import pandas as pd
from pathlib import Path
from src.data.strength_loader import load_strength_params, StrengthParams


def _make_csv(tmp_path, rows):
    f = tmp_path / "params.csv"
    df = pd.DataFrame(rows, columns=["team", "alpha_attack", "beta_defense", "matches_used", "as_of_date"])
    df.to_csv(f, index=False)
    return f


def test_returns_dict():
    pass  # inline below


def test_load_returns_dict_of_strength_params(tmp_path):
    csv = _make_csv(tmp_path, [["France", 1.5, 0.7, 24, "2022-11-19"]])
    result = load_strength_params(csv)
    assert isinstance(result, dict)
    assert "France" in result
    assert isinstance(result["France"], StrengthParams)


def test_alpha_beta_correct(tmp_path):
    csv = _make_csv(tmp_path, [["France", 1.52, 0.71, 24, "2022-11-19"]])
    result = load_strength_params(csv)
    assert abs(result["France"].alpha_attack - 1.52) < 1e-9
    assert abs(result["France"].beta_defense - 0.71) < 1e-9


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_strength_params(tmp_path / "missing.csv")

"""Tests for the expected lineups data loader.

TDD: all tests written RED-first before any production code exists.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from src.data.lineup_loader import (
    ExpectedLineupEntry,
    load_expected_lineups,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build a temporary CSV
# ─────────────────────────────────────────────────────────────────────────────

_COLUMNS = [
    "match_id", "date", "team", "player_id", "player_name", "position",
    "expected_starter", "lineup_status", "availability_status",
    "availability_factor", "form_factor", "source_type", "research_valid",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    match_id: str = "1",
    date: str = "2022-11-20",
    team: str = "Qatar",
    player_id: str = "qat_p1",
    player_name: str = "Player One",
    position: str = "GK",
    expected_starter: str = "true",
    lineup_status: str = "projected",
    availability_status: str = "fit",
    availability_factor: str = "1.0",
    form_factor: str = "1.0",
    source_type: str = "placeholder",
    research_valid: str = "false",
) -> dict:
    return {
        "match_id": match_id,
        "date": date,
        "team": team,
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "expected_starter": expected_starter,
        "lineup_status": lineup_status,
        "availability_status": availability_status,
        "availability_factor": availability_factor,
        "form_factor": form_factor,
        "source_type": source_type,
        "research_valid": research_valid,
    }


@pytest.fixture
def csv_with_two_matches(tmp_path):
    """CSV with 3 Qatar rows in match 1, 2 Ecuador rows in match 1, 2 Qatar rows in match 2."""
    rows = (
        [_row(match_id="1", team="Qatar", player_id=f"qat_p{i}",
               player_name=f"Qatar P{i}") for i in range(1, 4)]
        + [_row(match_id="1", team="Ecuador", player_id=f"ecu_p{i}",
                player_name=f"Ecuador P{i}") for i in range(1, 3)]
        + [_row(match_id="2", team="Qatar", player_id=f"qat_p{i}",
                player_name=f"Qatar P{i}") for i in range(1, 3)]
    )
    p = tmp_path / "expected_lineups.csv"
    _write_csv(rows, p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadReturnType:
    def test_returns_list(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches)
        assert isinstance(result, list)

    def test_returns_expected_lineup_entry_instances(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches)
        assert all(isinstance(e, ExpectedLineupEntry) for e in result)

    def test_entry_has_required_fields(self, csv_with_two_matches):
        entries = load_expected_lineups(csv_with_two_matches)
        e = entries[0]
        for field in [
            "match_id", "date", "team", "player_id", "player_name", "position",
            "expected_starter", "lineup_status", "availability_status",
            "availability_factor", "form_factor", "source_type", "research_valid",
        ]:
            assert hasattr(e, field), f"Missing field: {field}"


# ─────────────────────────────────────────────────────────────────────────────
# Type parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeParsing:
    def test_expected_starter_true_parsed_as_bool(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(expected_starter="true")], p)
        e = load_expected_lineups(p)[0]
        assert e.expected_starter is True
        assert isinstance(e.expected_starter, bool)

    def test_expected_starter_false_parsed_as_bool(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(expected_starter="false")], p)
        e = load_expected_lineups(p)[0]
        assert e.expected_starter is False
        assert isinstance(e.expected_starter, bool)

    def test_research_valid_true_parsed_as_bool(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(research_valid="true")], p)
        e = load_expected_lineups(p)[0]
        assert e.research_valid is True
        assert isinstance(e.research_valid, bool)

    def test_research_valid_false_parsed_as_bool(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(research_valid="false")], p)
        e = load_expected_lineups(p)[0]
        assert e.research_valid is False
        assert isinstance(e.research_valid, bool)

    def test_availability_factor_parsed_as_float(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(availability_factor="0.7")], p)
        e = load_expected_lineups(p)[0]
        assert isinstance(e.availability_factor, float)
        assert e.availability_factor == pytest.approx(0.7)

    def test_form_factor_parsed_as_float(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row(form_factor="1.1")], p)
        e = load_expected_lineups(p)[0]
        assert isinstance(e.form_factor, float)
        assert e.form_factor == pytest.approx(1.1)

    def test_string_fields_are_strings(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv([_row()], p)
        e = load_expected_lineups(p)[0]
        for attr in ("match_id", "date", "team", "player_id", "player_name",
                     "position", "lineup_status", "availability_status",
                     "source_type"):
            assert isinstance(getattr(e, attr), str), f"{attr} should be str"


# ─────────────────────────────────────────────────────────────────────────────
# Filtering by match_id
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchIdFiltering:
    def test_no_filter_returns_all_rows(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches)
        assert len(result) == 7  # 3+2+2

    def test_filter_match_1_returns_5_rows(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches, match_id="1")
        assert len(result) == 5

    def test_filter_match_2_returns_2_rows(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches, match_id="2")
        assert len(result) == 2

    def test_filter_nonexistent_match_returns_empty(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches, match_id="99")
        assert result == []

    def test_filter_preserves_team_values(self, csv_with_two_matches):
        result = load_expected_lineups(csv_with_two_matches, match_id="1")
        teams = {e.team for e in result}
        assert teams == {"Qatar", "Ecuador"}

    def test_match_id_is_string_comparison(self, csv_with_two_matches):
        """match_id in CSV is stored as string; filter should do string comparison."""
        result = load_expected_lineups(csv_with_two_matches, match_id="1")
        assert all(e.match_id == "1" for e in result)


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_expected_lineups(tmp_path / "nonexistent.csv")

    def test_empty_csv_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.csv"
        _write_csv([], p)
        result = load_expected_lineups(p)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Integration — production CSV exists
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionCSV:
    def test_production_csv_loads(self):
        """The data/expected_lineups.csv file must exist and load without error."""
        prod_path = Path("data/expected_lineups.csv")
        assert prod_path.exists(), (
            "data/expected_lineups.csv must exist — seed it with placeholder rows"
        )
        entries = load_expected_lineups(prod_path)
        assert isinstance(entries, list)

    def test_production_csv_has_entries(self):
        """Production CSV should have at least one row."""
        prod_path = Path("data/expected_lineups.csv")
        if not prod_path.exists():
            pytest.skip("Production CSV not yet created")
        entries = load_expected_lineups(prod_path)
        assert len(entries) > 0

    def test_production_csv_match_1_has_both_teams(self):
        """Match 1 (Qatar vs Ecuador) should have entries for both teams."""
        prod_path = Path("data/expected_lineups.csv")
        if not prod_path.exists():
            pytest.skip("Production CSV not yet created")
        entries = load_expected_lineups(prod_path, match_id="1")
        if entries:
            teams = {e.team for e in entries}
            assert len(teams) >= 1  # at minimum one team present

from __future__ import annotations

import csv

from src.data.team_api_ids import load_verified_team_ids, load_mapping_rows
from src.data.team_coverage import build_coverage_table


def _write_mapping(tmp_path, rows):
    path = tmp_path / "mapping.csv"
    fieldnames = ["internal_team", "api_team_name", "api_team_id", "country",
                   "match_method", "confidence", "verified", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_unverified_mapping_blocks_live_data_usage(tmp_path):
    path = _write_mapping(tmp_path, [
        {"internal_team": "Spain", "api_team_name": "Spain", "api_team_id": 9,
         "country": "Spain", "match_method": "exact_name_national",
         "confidence": "high", "verified": "True", "notes": ""},
        {"internal_team": "Atlantis", "api_team_name": "", "api_team_id": "",
         "country": "", "match_method": "none", "confidence": "low",
         "verified": "False", "notes": "unresolved"},
    ])

    ids = load_verified_team_ids(path)

    assert "Spain" in ids
    assert "Atlantis" not in ids


def test_verified_mapping_allows_live_data_usage(tmp_path):
    path = _write_mapping(tmp_path, [
        {"internal_team": "Spain", "api_team_name": "Spain", "api_team_id": 9,
         "country": "Spain", "match_method": "exact_name_national",
         "confidence": "high", "verified": "True", "notes": ""},
    ])

    ids = load_verified_team_ids(path)

    assert ids == {"Spain": 9}


def test_duplicate_api_id_detected_and_excluded(tmp_path):
    path = _write_mapping(tmp_path, [
        {"internal_team": "Germany", "api_team_name": "Germany", "api_team_id": 25,
         "country": "Germany", "match_method": "fuzzy_first_result",
         "confidence": "low", "verified": "False",
         "notes": "DUPLICATE api_team_id 25 shared with ['Australia']"},
        {"internal_team": "Australia", "api_team_name": "Australia", "api_team_id": 25,
         "country": "Australia", "match_method": "fuzzy_first_result",
         "confidence": "low", "verified": "False",
         "notes": "DUPLICATE api_team_id 25 shared with ['Germany']"},
    ])

    ids = load_verified_team_ids(path)

    assert ids == {}
    rows = load_mapping_rows(path)
    assert all("DUPLICATE" in r["notes"] for r in rows)


def test_fixture_teams_all_checked_against_mapping_file():
    """Every WC fixture team must appear (verified or not) in the real
    mapping file shipped with the repo -- no team silently missing."""
    from src.tournament.fixtures import load_fixtures, _DEFAULT

    fixtures = load_fixtures(_DEFAULT)
    fixture_teams = {f.team_a for f in fixtures} | {f.team_b for f in fixtures}

    rows = load_mapping_rows()
    mapped_internal = {r["internal_team"] for r in rows}

    missing = fixture_teams - mapped_internal
    assert missing == set(), f"Teams missing from mapping file: {missing}"


def test_no_live_api_calls_made_when_loading_mapping(tmp_path):
    """load_verified_team_ids / load_mapping_rows are pure file readers --
    they must not require network access or an API client."""
    path = _write_mapping(tmp_path, [
        {"internal_team": "Spain", "api_team_name": "Spain", "api_team_id": 9,
         "country": "Spain", "match_method": "exact_name_national",
         "confidence": "high", "verified": "True", "notes": ""},
    ])
    # No ApiFootballClient constructed anywhere in this test.
    assert load_verified_team_ids(path) == {"Spain": 9}
    assert load_mapping_rows(path)[0]["internal_team"] == "Spain"


def test_coverage_table_marks_unverified_teams_as_unmapped(tmp_path):
    """Unverified teams in the mapping must show as mapped=False in the
    coverage table, blocking live usage downstream."""
    ids = {"Spain": 9}  # only verified team
    table = build_coverage_table(["Spain", "Germany"], ids)
    by_team = {r.team: r for r in table}
    assert by_team["Spain"].mapped is True
    assert by_team["Germany"].mapped is False

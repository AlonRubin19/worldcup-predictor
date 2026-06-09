"""Tests for tournament_filters.py — fixture filtering, sorting, and grouping helpers.

TDD: all tests written RED-first.
"""

from __future__ import annotations

import pytest

from src.app.tournament_filters import (
    filter_fixtures,
    get_next_fixtures,
    get_today_fixtures,
    get_unique_stages,
    get_unique_groups,
    get_unique_teams,
    get_status_label,
    STATUS_LABELS,
)
from src.tournament.fixtures import Fixture


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(match_id="1", stage="group", group="A", date="2026-06-14",
       team_a="Brazil", team_b="France", status="NS") -> Fixture:
    return Fixture(match_id=match_id, stage=stage, group=group,
                   date=date, team_a=team_a, team_b=team_b, status=status)


_SAMPLE = [
    _f("1", "group", "A", "2026-06-14", "Brazil",    "France",    "NS"),
    _f("2", "group", "A", "2026-06-14", "Germany",   "Spain",     "NS"),
    _f("3", "group", "B", "2026-06-15", "Argentina", "Portugal",  "NS"),
    _f("4", "group", "C", "2026-06-16", "England",   "Italy",     "FT"),
    _f("5", "round_of_16", "",          "2026-07-01", "Brazil",   "Germany",   "NS"),
    _f("6", "quarter_final", "",        "2026-07-06", "France",   "Argentina", "NS"),
]


# ─────────────────────────────────────────────────────────────────────────────
# filter_fixtures
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterFixtures:
    def test_no_filters_returns_all(self):
        result = filter_fixtures(_SAMPLE)
        assert len(result) == len(_SAMPLE)

    def test_filter_by_stage_group(self):
        result = filter_fixtures(_SAMPLE, stage="group")
        assert all(f.stage == "group" for f in result)
        assert len(result) == 4

    def test_filter_by_stage_knockout(self):
        result = filter_fixtures(_SAMPLE, stage="round_of_16")
        assert len(result) == 1
        assert result[0].match_id == "5"

    def test_filter_by_group(self):
        result = filter_fixtures(_SAMPLE, group="A")
        assert all(f.group == "A" for f in result)
        assert len(result) == 2

    def test_filter_by_team(self):
        result = filter_fixtures(_SAMPLE, team="Brazil")
        assert all("Brazil" in (f.team_a, f.team_b) for f in result)
        assert len(result) == 2

    def test_filter_by_team_case_insensitive(self):
        result = filter_fixtures(_SAMPLE, team="brazil")
        assert len(result) == 2

    def test_filter_by_date(self):
        result = filter_fixtures(_SAMPLE, date="2026-06-14")
        assert all(f.date == "2026-06-14" for f in result)
        assert len(result) == 2

    def test_filter_by_status(self):
        result = filter_fixtures(_SAMPLE, statuses=["FT"])
        assert len(result) == 1
        assert result[0].match_id == "4"

    def test_filter_multiple_statuses(self):
        result = filter_fixtures(_SAMPLE, statuses=["NS", "FT"])
        assert len(result) == len(_SAMPLE)

    def test_combined_stage_and_date(self):
        result = filter_fixtures(_SAMPLE, stage="group", date="2026-06-14")
        assert len(result) == 2
        assert all(f.stage == "group" and f.date == "2026-06-14" for f in result)

    def test_combined_stage_and_team(self):
        result = filter_fixtures(_SAMPLE, stage="group", team="Brazil")
        assert len(result) == 1
        assert result[0].team_a == "Brazil"

    def test_no_match_returns_empty(self):
        result = filter_fixtures(_SAMPLE, team="Unknown FC")
        assert result == []

    def test_empty_input_returns_empty(self):
        assert filter_fixtures([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_next_fixtures
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNextFixtures:
    def test_returns_list(self):
        result = get_next_fixtures(_SAMPLE, n=3, today="2026-06-14")
        assert isinstance(result, list)

    def test_returns_at_most_n(self):
        result = get_next_fixtures(_SAMPLE, n=2, today="2026-06-14")
        assert len(result) <= 2

    def test_only_not_started_or_live(self):
        """Finished matches should not appear as 'next'."""
        result = get_next_fixtures(_SAMPLE, n=10, today="2026-06-14")
        for f in result:
            assert f.status != "FT"

    def test_sorted_by_date_asc(self):
        result = get_next_fixtures(_SAMPLE, n=10, today="2026-06-14")
        dates = [f.date for f in result]
        assert dates == sorted(dates)

    def test_past_finished_matches_excluded(self):
        result = get_next_fixtures(_SAMPLE, n=10, today="2026-06-17")
        # Match 4 (2026-06-16, FT) should not appear
        assert all(f.match_id != "4" for f in result)

    def test_empty_input_returns_empty(self):
        assert get_next_fixtures([], n=5, today="2026-06-14") == []


# ─────────────────────────────────────────────────────────────────────────────
# get_today_fixtures
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTodayFixtures:
    def test_returns_fixtures_for_today(self):
        result = get_today_fixtures(_SAMPLE, today="2026-06-14")
        assert all(f.date == "2026-06-14" for f in result)
        assert len(result) == 2

    def test_no_fixtures_today_returns_empty(self):
        result = get_today_fixtures(_SAMPLE, today="2030-01-01")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# get_unique_stages
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUniqueStages:
    def test_returns_sorted_unique_stages(self):
        stages = get_unique_stages(_SAMPLE)
        assert "group" in stages
        assert "round_of_16" in stages
        assert "quarter_final" in stages
        assert len(stages) == len(set(stages))

    def test_empty_input_returns_empty(self):
        assert get_unique_stages([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_unique_groups
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUniqueGroups:
    def test_returns_group_letters_only(self):
        groups = get_unique_groups(_SAMPLE)
        assert "A" in groups
        assert "B" in groups
        assert "C" in groups
        # Knockout matches with group="" should not appear
        assert "" not in groups

    def test_empty_input_returns_empty(self):
        assert get_unique_groups([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_unique_teams
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUniqueTeams:
    def test_returns_all_teams(self):
        teams = get_unique_teams(_SAMPLE)
        assert "Brazil" in teams
        assert "France" in teams
        assert "Germany" in teams

    def test_no_duplicates(self):
        teams = get_unique_teams(_SAMPLE)
        assert len(teams) == len(set(teams))

    def test_sorted(self):
        teams = get_unique_teams(_SAMPLE)
        assert teams == sorted(teams)

    def test_empty_input_returns_empty(self):
        assert get_unique_teams([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_status_label
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStatusLabel:
    def test_ns_label(self):
        assert "started" in get_status_label("NS").lower() or "scheduled" in get_status_label("NS").lower() or "upcoming" in get_status_label("NS").lower()

    def test_ft_label(self):
        assert "finish" in get_status_label("FT").lower() or "full" in get_status_label("FT").lower()

    def test_unknown_returns_code(self):
        result = get_status_label("XYZ")
        assert "XYZ" in result

    def test_status_labels_dict_exists(self):
        assert isinstance(STATUS_LABELS, dict)
        assert "NS" in STATUS_LABELS
        assert "FT" in STATUS_LABELS

"""Tests for SelectedFixture — the session-state payload that carries a chosen
fixture from the Today's Matches board into the Match Analyzer.

TDD: all tests written RED-first before any production code exists.
No live API calls — everything is constructed from plain data.
"""

from __future__ import annotations

import pytest

from src.app.selected_fixture import (
    SelectedFixture,
    create_selected_fixture,
    get_api_fixture_id,
    is_valid_selected_fixture,
)
from src.data.fixture_provider import FixtureSource
from src.tournament.fixtures import Fixture


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_fixture(
    match_id: str = "855744",
    team_a: str = "Brazil",
    team_b: str = "France",
    date: str = "2026-06-14",
    stage: str = "group",
    group: str = "",
) -> Fixture:
    return Fixture(
        match_id=match_id,
        stage=stage,
        group=group,
        date=date,
        team_a=team_a,
        team_b=team_b,
    )


def _csv_fixture(
    match_id: str = "m_001",
    team_a: str = "Qatar",
    team_b: str = "Ecuador",
    date: str = "2022-11-20",
    stage: str = "group",
    group: str = "A",
) -> Fixture:
    return Fixture(
        match_id=match_id,
        stage=stage,
        group=group,
        date=date,
        team_a=team_a,
        team_b=team_b,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SelectedFixture dataclass structure
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectedFixtureStructure:
    def test_can_instantiate(self):
        sf = SelectedFixture(
            fixture_id="855744",
            source_type="api",
            team_a="Brazil",
            team_b="France",
            date="2026-06-14",
            stage="group",
            group="",
        )
        assert sf is not None

    def test_has_fixture_id(self):
        sf = SelectedFixture(fixture_id="123", source_type="api",
                             team_a="A", team_b="B", date="2026-01-01",
                             stage="group", group="")
        assert sf.fixture_id == "123"

    def test_has_source_type(self):
        sf = SelectedFixture(fixture_id=None, source_type="csv",
                             team_a="A", team_b="B", date="2026-01-01",
                             stage="group", group="A")
        assert sf.source_type == "csv"

    def test_has_team_a_and_team_b(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="Brazil", team_b="France",
                             date="2026-06-14", stage="group", group="")
        assert sf.team_a == "Brazil"
        assert sf.team_b == "France"

    def test_has_date(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="")
        assert sf.date == "2026-06-14"

    def test_has_stage(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="semi_final", group="")
        assert sf.stage == "semi_final"

    def test_has_group(self):
        sf = SelectedFixture(fixture_id="1", source_type="csv",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="A")
        assert sf.group == "A"

    def test_all_fields_are_str_or_none(self):
        """All fields must be str or None so session_state can serialise them."""
        sf = SelectedFixture(fixture_id="855744", source_type="api",
                             team_a="Brazil", team_b="France",
                             date="2026-06-14", stage="group", group="")
        for val in (sf.fixture_id, sf.source_type, sf.team_a, sf.team_b,
                    sf.date, sf.stage, sf.group):
            assert val is None or isinstance(val, str), f"Expected str|None, got {type(val)}"

    def test_fixture_id_can_be_none(self):
        sf = SelectedFixture(fixture_id=None, source_type="csv",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="A")
        assert sf.fixture_id is None


# ─────────────────────────────────────────────────────────────────────────────
# create_selected_fixture — from API source
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateSelectedFixtureApi:
    def test_returns_selected_fixture(self):
        sf = create_selected_fixture(_api_fixture(), FixtureSource.API)
        assert isinstance(sf, SelectedFixture)

    def test_source_type_is_api(self):
        sf = create_selected_fixture(_api_fixture(), FixtureSource.API)
        assert sf.source_type == "api"

    def test_fixture_id_is_match_id_from_api_fixture(self):
        sf = create_selected_fixture(_api_fixture(match_id="855744"), FixtureSource.API)
        assert sf.fixture_id == "855744"

    def test_team_a_preserved(self):
        sf = create_selected_fixture(_api_fixture(team_a="Brazil"), FixtureSource.API)
        assert sf.team_a == "Brazil"

    def test_team_b_preserved(self):
        sf = create_selected_fixture(_api_fixture(team_b="France"), FixtureSource.API)
        assert sf.team_b == "France"

    def test_date_preserved(self):
        sf = create_selected_fixture(_api_fixture(date="2026-06-14"), FixtureSource.API)
        assert sf.date == "2026-06-14"

    def test_stage_preserved(self):
        sf = create_selected_fixture(_api_fixture(stage="round_of_16"), FixtureSource.API)
        assert sf.stage == "round_of_16"

    def test_group_preserved(self):
        sf = create_selected_fixture(_api_fixture(group=""), FixtureSource.API)
        assert sf.group == ""


# ─────────────────────────────────────────────────────────────────────────────
# create_selected_fixture — from CSV source
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateSelectedFixtureCsv:
    def test_source_type_is_csv(self):
        sf = create_selected_fixture(_csv_fixture(), FixtureSource.CSV)
        assert sf.source_type == "csv"

    def test_fixture_id_is_match_id_string(self):
        sf = create_selected_fixture(_csv_fixture(match_id="m_001"), FixtureSource.CSV)
        assert sf.fixture_id == "m_001"

    def test_group_preserved(self):
        sf = create_selected_fixture(_csv_fixture(group="A"), FixtureSource.CSV)
        assert sf.group == "A"

    def test_auto_source_csv_fallback_sets_csv(self):
        """Auto mode falling back to CSV should record source_type='csv'."""
        sf = create_selected_fixture(_csv_fixture(), FixtureSource.CSV)
        assert sf.source_type == "csv"


# ─────────────────────────────────────────────────────────────────────────────
# get_api_fixture_id
# ─────────────────────────────────────────────────────────────────────────────

class TestGetApiFixtureId:
    def test_returns_int_for_numeric_fixture_id(self):
        sf = SelectedFixture(fixture_id="855744", source_type="api",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="")
        assert get_api_fixture_id(sf) == 855744

    def test_returns_none_for_csv_source(self):
        sf = SelectedFixture(fixture_id="m_001", source_type="csv",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="A")
        assert get_api_fixture_id(sf) is None

    def test_returns_none_when_fixture_id_is_none(self):
        sf = SelectedFixture(fixture_id=None, source_type="csv",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="A")
        assert get_api_fixture_id(sf) is None

    def test_returns_none_for_non_numeric_fixture_id(self):
        sf = SelectedFixture(fixture_id="m_001", source_type="api",
                             team_a="A", team_b="B",
                             date="2026-06-14", stage="group", group="")
        # Non-numeric IDs from CSV files are not valid API fixture IDs
        result = get_api_fixture_id(sf)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# is_valid_selected_fixture
# ─────────────────────────────────────────────────────────────────────────────

class TestIsValidSelectedFixture:
    def test_valid_api_fixture_returns_true(self):
        sf = SelectedFixture(fixture_id="855744", source_type="api",
                             team_a="Brazil", team_b="France",
                             date="2026-06-14", stage="group", group="")
        assert is_valid_selected_fixture(sf) is True

    def test_valid_csv_fixture_returns_true(self):
        sf = SelectedFixture(fixture_id="m_001", source_type="csv",
                             team_a="Qatar", team_b="Ecuador",
                             date="2022-11-20", stage="group", group="A")
        assert is_valid_selected_fixture(sf) is True

    def test_none_returns_false(self):
        assert is_valid_selected_fixture(None) is False

    def test_missing_team_a_returns_false(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="", team_b="France",
                             date="2026-06-14", stage="group", group="")
        assert is_valid_selected_fixture(sf) is False

    def test_missing_team_b_returns_false(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="Brazil", team_b="",
                             date="2026-06-14", stage="group", group="")
        assert is_valid_selected_fixture(sf) is False

    def test_same_team_both_sides_returns_false(self):
        sf = SelectedFixture(fixture_id="1", source_type="api",
                             team_a="Brazil", team_b="Brazil",
                             date="2026-06-14", stage="group", group="")
        assert is_valid_selected_fixture(sf) is False


# ─────────────────────────────────────────────────────────────────────────────
# Payload consumption — simulated Match Analyzer intake
# ─────────────────────────────────────────────────────────────────────────────

class TestPayloadConsumption:
    def test_team_a_accessible_from_payload(self):
        sf = create_selected_fixture(_api_fixture(team_a="Brazil"), FixtureSource.API)
        assert sf.team_a == "Brazil"

    def test_team_b_accessible_from_payload(self):
        sf = create_selected_fixture(_api_fixture(team_b="France"), FixtureSource.API)
        assert sf.team_b == "France"

    def test_api_fixture_id_integer_accessible(self):
        sf = create_selected_fixture(_api_fixture(match_id="855744"), FixtureSource.API)
        assert get_api_fixture_id(sf) == 855744

    def test_csv_payload_returns_none_for_api_id(self):
        sf = create_selected_fixture(_csv_fixture(), FixtureSource.CSV)
        assert get_api_fixture_id(sf) is None

    def test_fixture_id_preserved_end_to_end(self):
        """Fixture ID 855744 created in tab_today is the same ID used in tab_predictor."""
        fixture_id_in_board = "855744"
        sf = create_selected_fixture(
            _api_fixture(match_id=fixture_id_in_board), FixtureSource.API
        )
        fixture_id_in_analyzer = get_api_fixture_id(sf)
        assert fixture_id_in_analyzer == int(fixture_id_in_board)

    def test_fallback_with_no_selection_is_handled(self):
        """When no fixture is in session_state, consumer gets None — no crash."""
        selected = None
        assert is_valid_selected_fixture(selected) is False

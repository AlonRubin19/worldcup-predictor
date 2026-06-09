"""Tests for the team mapping audit utility.

TDD: all tests written RED-first before any production code exists.

No real API calls — uses in-memory LiveFixture objects.
"""

from __future__ import annotations

import pytest

from src.data.team_mapping_audit import (
    MappingAuditResult,
    audit_team_mappings,
    classify_team,
    TeamMappingClass,
)
from src.data.live_fixture_loader import LiveFixture


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_INTERNAL_TEAMS = {
    "Brazil", "France", "Germany", "Argentina", "England",
    "South Korea", "USA", "Iran", "Mexico", "Spain",
    "Netherlands", "Portugal", "Croatia", "Serbia",
}


def _fixture(home_api: str, away_api: str) -> LiveFixture:
    from src.data.live_fixture_loader import map_team_name
    return LiveFixture(
        fixture_id=1,
        date="2026-06-11",
        home_team_api=home_api,
        away_team_api=away_api,
        home_team=map_team_name(home_api),
        away_team=map_team_name(away_api),
        status_short="NS",
        round="Group Stage - 1",
        venue="Stadium",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TeamMappingClass constant
# ─────────────────────────────────────────────────────────────────────────────

class TestTeamMappingClass:
    def test_has_exact(self):
        assert hasattr(TeamMappingClass, "EXACT")

    def test_has_mapped(self):
        assert hasattr(TeamMappingClass, "MAPPED")

    def test_has_unknown(self):
        assert hasattr(TeamMappingClass, "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# classify_team
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyTeam:
    def test_exact_match_when_api_name_in_known_teams(self):
        result = classify_team("Brazil", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.EXACT

    def test_mapped_alias_when_api_name_in_map(self):
        result = classify_team("Korea Republic", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.MAPPED

    def test_mapped_alias_for_united_states(self):
        result = classify_team("United States", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.MAPPED

    def test_mapped_alias_for_ir_iran(self):
        result = classify_team("IR Iran", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.MAPPED

    def test_unknown_when_api_name_not_in_known_or_map(self):
        result = classify_team("Fictional Nation FC", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.UNKNOWN

    def test_exact_when_api_name_matches_case_sensitive(self):
        result = classify_team("France", known_teams=_INTERNAL_TEAMS)
        assert result == TeamMappingClass.EXACT

    def test_unknown_when_known_teams_empty(self):
        """Without a known teams set, direct API names can't be classified as exact."""
        result = classify_team("Brazil", known_teams=set())
        # Brazil is not in TEAM_NAME_MAP (it maps to itself) AND not in known_teams
        assert result == TeamMappingClass.UNKNOWN

    def test_mapped_still_works_with_empty_known_teams(self):
        """TEAM_NAME_MAP aliases should still classify as MAPPED even without known_teams."""
        result = classify_team("Korea Republic", known_teams=set())
        assert result == TeamMappingClass.MAPPED


# ─────────────────────────────────────────────────────────────────────────────
# audit_team_mappings — structure
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditTeamMappingsStructure:
    def test_returns_mapping_audit_result(self):
        result = audit_team_mappings([], known_teams=_INTERNAL_TEAMS)
        assert isinstance(result, MappingAuditResult)

    def test_has_required_fields(self):
        result = audit_team_mappings([], known_teams=_INTERNAL_TEAMS)
        for field in ("exact_count", "mapped_count", "unknown_count",
                      "unknown_teams", "exact_teams", "mapped_teams"):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_empty_fixtures_returns_zeros(self):
        result = audit_team_mappings([], known_teams=_INTERNAL_TEAMS)
        assert result.exact_count == 0
        assert result.mapped_count == 0
        assert result.unknown_count == 0

    def test_empty_fixtures_returns_empty_lists(self):
        result = audit_team_mappings([], known_teams=_INTERNAL_TEAMS)
        assert result.unknown_teams == []
        assert result.exact_teams == []
        assert result.mapped_teams == []


# ─────────────────────────────────────────────────────────────────────────────
# audit_team_mappings — counting
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditCounting:
    def test_exact_match_counted(self):
        fixtures = [_fixture("Brazil", "France")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.exact_count == 2  # both teams are exact

    def test_mapped_alias_counted(self):
        fixtures = [_fixture("Korea Republic", "United States")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.mapped_count == 2

    def test_unknown_team_counted(self):
        fixtures = [_fixture("Fictional Nation FC", "Brazil")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.unknown_count == 1

    def test_mixed_fixture_types(self):
        fixtures = [
            _fixture("Brazil", "Korea Republic"),      # exact + mapped
            _fixture("Fictional FC", "Unknown Utd"),    # 2 unknowns
        ]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.exact_count == 1
        assert result.mapped_count == 1
        assert result.unknown_count == 2

    def test_multiple_fixtures_cumulative(self):
        fixtures = [
            _fixture("Brazil", "France"),
            _fixture("Germany", "Spain"),
        ]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.exact_count == 4

    def test_total_teams_equals_sum(self):
        fixtures = [
            _fixture("Brazil", "Korea Republic"),
            _fixture("Fictional FC", "France"),
        ]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        total = result.exact_count + result.mapped_count + result.unknown_count
        assert total == 4  # 2 fixtures × 2 teams


# ─────────────────────────────────────────────────────────────────────────────
# audit_team_mappings — lists
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLists:
    def test_unknown_team_listed_in_unknown_teams(self):
        fixtures = [_fixture("Fictional FC", "Brazil")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert "Fictional FC" in result.unknown_teams

    def test_unknown_teams_no_duplicates(self):
        """Same unknown team in two fixtures should only appear once in unknown_teams."""
        fixtures = [
            _fixture("Fictional FC", "Brazil"),
            _fixture("Fictional FC", "France"),
        ]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert result.unknown_teams.count("Fictional FC") == 1

    def test_exact_team_listed(self):
        fixtures = [_fixture("Brazil", "France")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert "Brazil" in result.exact_teams

    def test_mapped_team_listed_as_tuple(self):
        """mapped_teams should be list of (api_name, internal_name) tuples."""
        fixtures = [_fixture("Korea Republic", "Brazil")]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        assert any(api == "Korea Republic" and internal == "South Korea"
                   for api, internal in result.mapped_teams)

    def test_mapped_teams_no_duplicates(self):
        fixtures = [
            _fixture("Korea Republic", "Brazil"),
            _fixture("Korea Republic", "France"),
        ]
        result = audit_team_mappings(fixtures, known_teams=_INTERNAL_TEAMS)
        korea_entries = [(a, i) for a, i in result.mapped_teams if a == "Korea Republic"]
        assert len(korea_entries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# audit_team_mappings — with real internal teams loaded
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditWithRealTeams:
    def test_brazil_is_exact_with_real_teams(self):
        from src.data.loader import load_teams
        known = set(load_teams())
        fixtures = [_fixture("Brazil", "Argentina")]
        result = audit_team_mappings(fixtures, known_teams=known)
        assert result.exact_count == 2

    def test_korea_republic_is_mapped_with_real_teams(self):
        from src.data.loader import load_teams
        known = set(load_teams())
        fixtures = [_fixture("Korea Republic", "Brazil")]
        result = audit_team_mappings(fixtures, known_teams=known)
        assert result.mapped_count >= 1

    def test_united_states_is_mapped_with_real_teams(self):
        from src.data.loader import load_teams
        known = set(load_teams())
        fixtures = [_fixture("United States", "Brazil")]
        result = audit_team_mappings(fixtures, known_teams=known)
        assert result.mapped_count >= 1

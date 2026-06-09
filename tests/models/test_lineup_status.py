"""Tests for the lineup status validation and conversion layer.

TDD: all tests written RED-first before any production code exists.
"""

from __future__ import annotations

import pytest

from src.data.lineup_loader import ExpectedLineupEntry
from src.models.lineup_status import (
    LineupValidationResult,
    validate_lineup_for_match,
    convert_lineup_entries_to_override,
)
from src.models.lineup_override import LineupOverride, PlayerOverride


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _entry(
    team: str = "Qatar",
    player_id: str = "qat_p1",
    player_name: str = "Player One",
    position: str = "GK",
    expected_starter: bool = True,
    lineup_status: str = "projected",
    availability_status: str = "fit",
    availability_factor: float = 1.0,
    form_factor: float = 1.0,
    source_type: str = "placeholder",
    research_valid: bool = False,
    match_id: str = "1",
    date: str = "2022-11-20",
) -> ExpectedLineupEntry:
    return ExpectedLineupEntry(
        match_id=match_id,
        date=date,
        team=team,
        player_id=player_id,
        player_name=player_name,
        position=position,
        expected_starter=expected_starter,
        lineup_status=lineup_status,
        availability_status=availability_status,
        availability_factor=availability_factor,
        form_factor=form_factor,
        source_type=source_type,
        research_valid=research_valid,
    )


def _official_11(team: str = "Qatar", source_type: str = "official_lineup",
                  research_valid: bool = True) -> list[ExpectedLineupEntry]:
    """11 fit starting players with official source."""
    return [
        _entry(team=team, player_id=f"{team.lower()}_p{i}",
               player_name=f"Player {i}",
               expected_starter=True, lineup_status="official",
               source_type=source_type, research_valid=research_valid)
        for i in range(1, 12)
    ]


def _projected_n(n: int, team: str = "Qatar") -> list[ExpectedLineupEntry]:
    """n projected starters."""
    return [
        _entry(team=team, player_id=f"{team.lower()}_p{i}",
               expected_starter=True, lineup_status="projected",
               source_type="projected_lineup", research_valid=False)
        for i in range(1, n + 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LineupValidationResult structure
# ─────────────────────────────────────────────────────────────────────────────

class TestLineupValidationResultStructure:
    def test_has_required_fields(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "is_research_valid")

    def test_errors_is_list(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert isinstance(result.errors, list)

    def test_warnings_is_list(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert isinstance(result.warnings, list)

    def test_is_valid_is_bool(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert isinstance(result.is_valid, bool)

    def test_returns_lineup_validation_result(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert isinstance(result, LineupValidationResult)


# ─────────────────────────────────────────────────────────────────────────────
# validate_lineup_for_match — empty / minimal
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateEmpty:
    def test_empty_entries_is_valid(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert result.is_valid is True

    def test_empty_has_no_errors(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert result.errors == []

    def test_filters_to_team(self):
        """Only entries matching `team` are validated."""
        entries = [
            _entry(team="Qatar", player_id="qat_p1"),
            _entry(team="Ecuador", player_id="ecu_p1"),
        ]
        result_q = validate_lineup_for_match(entries, team="Qatar")
        result_e = validate_lineup_for_match(entries, team="Ecuador")
        # Both have exactly 1 entry → no duplicate, not official → no 11-starter rule
        assert result_q.is_valid is True
        assert result_e.is_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# Official lineup — exactly 11 starters required
# ─────────────────────────────────────────────────────────────────────────────

class TestOfficialLineupRule:
    def test_official_11_starters_is_valid(self):
        entries = _official_11()
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is True
        assert result.errors == []

    def test_official_10_starters_is_invalid(self):
        entries = _official_11()
        # Remove one starter
        entries[0] = _entry(team="Qatar", player_id="qat_p0",
                             expected_starter=False, lineup_status="official",
                             source_type="official_lineup", research_valid=True)
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_official_12_starters_is_invalid(self):
        entries = _official_11()
        entries.append(_entry(team="Qatar", player_id="qat_p12",
                              expected_starter=True, lineup_status="official",
                              source_type="official_lineup", research_valid=True))
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False

    def test_official_error_mentions_starter_count(self):
        entries = _official_11()[:8]  # only 8 starters, all official
        entries = [e.__class__(
            **{**e.__dict__, "lineup_status": "official", "expected_starter": True}
        ) for e in entries]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False
        assert any("11" in err or "starter" in err.lower() for err in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Projected lineup — warning not error
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectedLineupRule:
    def test_projected_8_starters_is_valid(self):
        entries = _projected_n(8)
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is True

    def test_projected_8_starters_produces_warning(self):
        entries = _projected_n(8)
        result = validate_lineup_for_match(entries, team="Qatar")
        assert len(result.warnings) > 0

    def test_projected_11_starters_no_warning(self):
        entries = _projected_n(11)
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is True
        assert all("starter" not in w.lower() for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate player_id
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicatePlayerRule:
    def test_duplicate_player_id_is_invalid(self):
        entries = [
            _entry(player_id="qat_p1"),
            _entry(player_id="qat_p1"),  # duplicate
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False

    def test_duplicate_error_mentions_player_id(self):
        entries = [
            _entry(player_id="qat_p1", player_name="Player A"),
            _entry(player_id="qat_p1", player_name="Player B"),
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert any("qat_p1" in err or "duplicate" in err.lower() for err in result.errors)

    def test_different_player_ids_no_duplicate_error(self):
        entries = [_entry(player_id=f"qat_p{i}") for i in range(1, 5)]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert not any("duplicate" in err.lower() for err in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Out / suspended cannot be expected_starter
# ─────────────────────────────────────────────────────────────────────────────

class TestUnavailableStarterRule:
    def test_out_player_as_starter_is_invalid(self):
        entries = [
            _entry(player_id="qat_p1", expected_starter=True,
                   availability_status="out", availability_factor=0.0)
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False

    def test_suspended_player_as_starter_is_invalid(self):
        entries = [
            _entry(player_id="qat_p1", expected_starter=True,
                   availability_status="suspended", availability_factor=0.0)
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_valid is False

    def test_out_player_not_starter_is_valid(self):
        entries = [
            _entry(player_id="qat_p1", expected_starter=False,
                   availability_status="out", availability_factor=0.0)
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert not any("out" in err.lower() or "suspended" in err.lower()
                        for err in result.errors)

    def test_error_mentions_player_id_for_unavailable_starter(self):
        entries = [
            _entry(player_id="qat_p1", player_name="Injured Star",
                   expected_starter=True, availability_status="out", availability_factor=0.0)
        ]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert any("qat_p1" in err or "Injured Star" in err or "out" in err.lower()
                   for err in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Research validity
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchValidityRule:
    def test_placeholder_source_is_not_research_valid(self):
        entries = [_entry(source_type="placeholder", research_valid=False)]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_research_valid is False

    def test_manual_source_is_not_research_valid(self):
        entries = [_entry(source_type="manual", research_valid=False)]
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_research_valid is False

    def test_official_lineup_source_can_be_research_valid(self):
        entries = _official_11(source_type="official_lineup", research_valid=True)
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_research_valid is True

    def test_projected_lineup_source_is_not_research_valid(self):
        entries = _projected_n(11)  # source_type="projected_lineup", research_valid=False
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_research_valid is False

    def test_mixed_sources_not_research_valid(self):
        """Any placeholder/manual entry disqualifies research validity."""
        entries = _official_11(source_type="official_lineup", research_valid=True)
        entries.append(_entry(player_id="qat_bench_1", expected_starter=False,
                               lineup_status="bench",
                               source_type="placeholder", research_valid=False))
        result = validate_lineup_for_match(entries, team="Qatar")
        assert result.is_research_valid is False

    def test_empty_lineup_not_research_valid(self):
        result = validate_lineup_for_match([], team="Qatar")
        assert result.is_research_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# convert_lineup_entries_to_override
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertToOverride:
    def test_returns_lineup_override(self):
        entries = _projected_n(11)
        result = convert_lineup_entries_to_override(entries)
        assert isinstance(result, LineupOverride)

    def test_team_set_correctly(self):
        entries = _projected_n(11, team="Ecuador")
        result = convert_lineup_entries_to_override(entries)
        assert result.team == "Ecuador"

    def test_players_count_matches(self):
        entries = _projected_n(11)
        result = convert_lineup_entries_to_override(entries)
        assert len(result.players) == 11

    def test_each_player_is_player_override(self):
        entries = _projected_n(11)
        result = convert_lineup_entries_to_override(entries)
        assert all(isinstance(p, PlayerOverride) for p in result.players)

    def test_preserves_expected_starter(self):
        entries = [
            _entry(player_id="qat_p1", expected_starter=True),
            _entry(player_id="qat_p2", expected_starter=False),
        ]
        result = convert_lineup_entries_to_override(entries)
        starters = {p.player_id: p.expected_starter for p in result.players}
        assert starters["qat_p1"] is True
        assert starters["qat_p2"] is False

    def test_preserves_availability_factor(self):
        entries = [
            _entry(player_id="qat_p1", availability_factor=0.7,
                   availability_status="doubtful")
        ]
        result = convert_lineup_entries_to_override(entries)
        assert result.players[0].availability_factor == pytest.approx(0.7)

    def test_preserves_form_factor(self):
        entries = [_entry(player_id="qat_p1", form_factor=1.1)]
        result = convert_lineup_entries_to_override(entries)
        assert result.players[0].form_factor == pytest.approx(1.1)

    def test_preserves_player_id(self):
        entries = [_entry(player_id="qat_neymar")]
        result = convert_lineup_entries_to_override(entries)
        assert result.players[0].player_id == "qat_neymar"

    def test_preserves_player_name(self):
        entries = [_entry(player_id="qat_p1", player_name="Star Player")]
        result = convert_lineup_entries_to_override(entries)
        assert result.players[0].player_name == "Star Player"

    def test_unavailable_entry_not_starter(self):
        entries = [
            _entry(player_id="qat_p1", expected_starter=False,
                   lineup_status="unavailable",
                   availability_status="out", availability_factor=0.0)
        ]
        result = convert_lineup_entries_to_override(entries)
        assert result.players[0].expected_starter is False

    def test_empty_entries_raises_value_error(self):
        with pytest.raises(ValueError):
            convert_lineup_entries_to_override([])

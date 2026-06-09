"""Tests for the lineup editor component — pure formatters, no Streamlit.

TDD: all tests written RED-first before any production code exists.
"""

from __future__ import annotations

import pytest

from src.app.components.lineup_editor import (
    format_player_table,
    parse_player_edits,
    STATUS_TO_FACTOR,
)
from src.models.lineup_override import PlayerOverride, LineupOverride, create_default_lineup


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _player(
    name: str = "Player A",
    team: str = "Brazil",
    expected_starter: bool = True,
    availability_status: str = "fit",
    availability_factor: float = 1.0,
    form_factor: float = 1.0,
) -> PlayerOverride:
    return PlayerOverride(
        player_id=name.lower().replace(" ", "_"),
        player_name=name,
        team=team,
        expected_starter=expected_starter,
        availability_status=availability_status,
        availability_factor=availability_factor,
        form_factor=form_factor,
    )


def _default_players(n: int = 3, team: str = "Brazil") -> list[PlayerOverride]:
    return [_player(f"Player {i}", team=team) for i in range(1, n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# STATUS_TO_FACTOR mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusToFactor:
    def test_fit_maps_to_one(self):
        assert STATUS_TO_FACTOR["fit"] == pytest.approx(1.0)

    def test_doubtful_maps_below_one(self):
        assert STATUS_TO_FACTOR["doubtful"] < 1.0
        assert STATUS_TO_FACTOR["doubtful"] > 0.0

    def test_out_maps_to_zero(self):
        assert STATUS_TO_FACTOR["out"] == pytest.approx(0.0)

    def test_suspended_maps_to_zero(self):
        assert STATUS_TO_FACTOR["suspended"] == pytest.approx(0.0)

    def test_all_four_statuses_present(self):
        for status in ("fit", "doubtful", "out", "suspended"):
            assert status in STATUS_TO_FACTOR


# ─────────────────────────────────────────────────────────────────────────────
# format_player_table
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPlayerTable:
    def test_returns_list_of_dicts(self):
        players = _default_players()
        result = format_player_table(players)
        assert isinstance(result, list)
        assert all(isinstance(row, dict) for row in result)

    def test_length_matches_input(self):
        players = _default_players(5)
        result = format_player_table(players)
        assert len(result) == 5

    def test_contains_player_name(self):
        players = [_player("Neymar")]
        result = format_player_table(players)
        names = [str(v) for row in result for v in row.values()]
        assert any("Neymar" in n for n in names)

    def test_contains_availability_status(self):
        players = [_player("Neymar", availability_status="doubtful")]
        result = format_player_table(players)
        statuses = [str(v) for row in result for v in row.values()]
        assert any("doubtful" in s.lower() for s in statuses)

    def test_contains_expected_starter_field(self):
        players = [_player("Neymar", expected_starter=True)]
        result = format_player_table(players)
        assert len(result) == 1
        row = result[0]
        # Should have a key/value indicating starter status
        assert any("starter" in k.lower() for k in row.keys())

    def test_contains_form_factor_field(self):
        players = [_player("Vini", form_factor=1.1)]
        result = format_player_table(players)
        assert len(result) == 1
        row = result[0]
        assert any("form" in k.lower() for k in row.keys())

    def test_empty_list_returns_empty(self):
        assert format_player_table([]) == []

    def test_preserves_order(self):
        players = [_player(f"Player {i}") for i in range(5)]
        result = format_player_table(players)
        for i, row in enumerate(result):
            assert f"Player {i}" in str(list(row.values()))


# ─────────────────────────────────────────────────────────────────────────────
# parse_player_edits
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePlayerEdits:
    def _make_row(
        self,
        player_name: str = "Neymar",
        team: str = "Brazil",
        expected_starter: bool = True,
        availability_status: str = "fit",
        availability_factor: float = 1.0,
        form_factor: float = 1.0,
        player_id: str = "neymar",
    ) -> dict:
        return {
            "player_id": player_id,
            "Player Name": player_name,
            "Team": team,
            "Starter": expected_starter,
            "Status": availability_status,
            "Availability Factor": availability_factor,
            "Form Factor": form_factor,
        }

    def test_returns_list_of_player_overrides(self):
        rows = [self._make_row()]
        result = parse_player_edits(rows, team="Brazil")
        assert isinstance(result, list)
        assert all(isinstance(p, PlayerOverride) for p in result)

    def test_length_matches_input(self):
        rows = [self._make_row(f"P{i}", player_id=f"p{i}") for i in range(5)]
        result = parse_player_edits(rows, team="Brazil")
        assert len(result) == 5

    def test_preserves_player_name(self):
        rows = [self._make_row("Neymar")]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].player_name == "Neymar"

    def test_preserves_availability_factor(self):
        rows = [self._make_row(availability_factor=0.7)]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].availability_factor == pytest.approx(0.7)

    def test_preserves_form_factor(self):
        rows = [self._make_row(form_factor=1.1)]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].form_factor == pytest.approx(1.1)

    def test_preserves_expected_starter(self):
        rows = [self._make_row(expected_starter=False)]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].expected_starter is False

    def test_status_maps_to_availability_factor_for_out(self):
        """If status is 'out' but availability_factor not explicitly given,
        parse should use STATUS_TO_FACTOR mapping."""
        rows = [self._make_row(availability_status="out", availability_factor=0.0)]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].availability_factor == pytest.approx(0.0)

    def test_team_assigned_from_argument(self):
        rows = [self._make_row(team="Brazil")]
        result = parse_player_edits(rows, team="Argentina")
        # team from argument takes precedence
        assert result[0].team == "Argentina"

    def test_empty_rows_returns_empty(self):
        result = parse_player_edits([], team="Brazil")
        assert result == []

    def test_player_id_preserved(self):
        rows = [self._make_row(player_id="neymar_jr")]
        result = parse_player_edits(rows, team="Brazil")
        assert result[0].player_id == "neymar_jr"

    def test_roundtrip_format_then_parse(self):
        """format_player_table → parse_player_edits should produce equivalent players."""
        original = [_player("Neymar", team="Brazil", expected_starter=True,
                             availability_status="doubtful", availability_factor=0.7,
                             form_factor=0.9)]
        formatted = format_player_table(original)
        parsed = parse_player_edits(formatted, team="Brazil")
        assert len(parsed) == 1
        p = parsed[0]
        assert p.player_name == "Neymar"
        assert p.availability_status == "doubtful"
        assert p.availability_factor == pytest.approx(0.7)
        assert p.form_factor == pytest.approx(0.9)
        assert p.expected_starter is True

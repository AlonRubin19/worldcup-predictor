"""Lineup Editor component — pure formatter functions for the lineup override UI.

No Streamlit imports — fully testable.

Data flow:
  list[PlayerOverride] → format_player_table → editable dict rows
  edited dict rows     → parse_player_edits  → list[PlayerOverride]
"""

from __future__ import annotations

from src.models.lineup_override import PlayerOverride


# ─────────────────────────────────────────────────────────────────────────────
# Status → availability factor mapping
# ─────────────────────────────────────────────────────────────────────────────

STATUS_TO_FACTOR: dict[str, float] = {
    "fit": 1.0,
    "doubtful": 0.7,
    "out": 0.0,
    "suspended": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_player_table(players: list[PlayerOverride]) -> list[dict]:
    """Convert a list of PlayerOverride objects into a list of editable dicts.

    The dict keys are human-readable column names suitable for a Streamlit
    data_editor. The round-trip is preserved via parse_player_edits.
    """
    rows = []
    for p in players:
        rows.append({
            "player_id": p.player_id,
            "Player Name": p.player_name,
            "Team": p.team,
            "Starter": p.expected_starter,
            "Status": p.availability_status,
            "Availability Factor": p.availability_factor,
            "Form Factor": p.form_factor,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_player_edits(edited_rows: list[dict], team: str) -> list[PlayerOverride]:
    """Convert edited dict rows back into PlayerOverride objects.

    The `team` argument overrides whatever team value is in the row dict,
    ensuring players are associated with the correct team (since users may
    inadvertently change the team column).

    Args:
        edited_rows: List of dicts as produced by format_player_table (and
                     potentially modified by a Streamlit data_editor).
        team:        Team name to assign to all returned PlayerOverride objects.

    Returns:
        List of PlayerOverride, one per row.
    """
    result = []
    for row in edited_rows:
        result.append(PlayerOverride(
            player_id=str(row.get("player_id", "")),
            player_name=str(row.get("Player Name", "")),
            team=team,
            expected_starter=bool(row.get("Starter", True)),
            availability_status=str(row.get("Status", "fit")),
            availability_factor=float(row.get("Availability Factor", 1.0)),
            form_factor=float(row.get("Form Factor", 1.0)),
        ))
    return result

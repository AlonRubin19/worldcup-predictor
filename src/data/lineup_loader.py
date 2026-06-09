"""Expected lineups data loader.

Reads data/expected_lineups.csv and returns typed ExpectedLineupEntry objects.

CSV columns:
    match_id, date, team, player_id, player_name, position,
    expected_starter, lineup_status, availability_status,
    availability_factor, form_factor, source_type, research_valid
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpectedLineupEntry:
    """One player row from expected_lineups.csv."""
    match_id: str
    date: str
    team: str
    player_id: str
    player_name: str
    position: str
    expected_starter: bool
    lineup_status: str          # "projected" | "official" | "unavailable" | "bench" | "unknown"
    availability_status: str    # "fit" | "doubtful" | "out" | "suspended"
    availability_factor: float
    form_factor: float
    source_type: str            # "placeholder" | "manual" | "projected_lineup" | "official_lineup" | "injury_report"
    research_valid: bool


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bool(value: str) -> bool:
    """Parse 'true'/'false' strings (case-insensitive) to Python bool."""
    return value.strip().lower() == "true"


def load_expected_lineups(
    path: str | Path,
    match_id: str | None = None,
) -> list[ExpectedLineupEntry]:
    """Load expected lineup entries from a CSV file.

    Args:
        path:     Path to the expected_lineups.csv file.
        match_id: Optional filter — if provided, only rows with this match_id
                  are returned (string comparison).

    Returns:
        List of ExpectedLineupEntry, one per matching CSV row.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected lineups file not found: {path}")

    entries: list[ExpectedLineupEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if match_id is not None and row["match_id"] != str(match_id):
                continue
            entries.append(ExpectedLineupEntry(
                match_id=row["match_id"],
                date=row["date"],
                team=row["team"],
                player_id=row["player_id"],
                player_name=row["player_name"],
                position=row["position"],
                expected_starter=_parse_bool(row["expected_starter"]),
                lineup_status=row["lineup_status"],
                availability_status=row["availability_status"],
                availability_factor=float(row["availability_factor"]),
                form_factor=float(row["form_factor"]),
                source_type=row["source_type"],
                research_valid=_parse_bool(row["research_valid"]),
            ))
    return entries

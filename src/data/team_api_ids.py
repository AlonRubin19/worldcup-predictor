"""team_api_ids.py — verified API-Football team-id mapping.

Loads from data/api_team_mapping_verified.csv, produced by
scripts/verify_api_team_ids.py. ONLY rows with verified=True are exposed via
TEAM_API_IDS / load_verified_team_ids() -- unverified or duplicate mappings
are excluded so live squad/injury/player-stats data is never fetched using a
guessed team id. Teams without a verified mapping fall back to the baseline
(non-live) model.
"""

from __future__ import annotations

import csv
from pathlib import Path

_DEFAULT_MAPPING_PATH = Path(__file__).parent.parent.parent / "data" / "api_team_mapping_verified.csv"


def load_verified_team_ids(path: Path | None = None) -> dict[str, int]:
    """Return {internal_team_name: api_team_id} for verified=True rows only.

    Returns an empty dict (not an error) if the mapping file doesn't exist --
    callers must treat that as "no live data available for any team".
    """
    p = path if path is not None else _DEFAULT_MAPPING_PATH
    if not Path(p).exists():
        return {}

    result: dict[str, int] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verified = str(row.get("verified", "")).strip().lower() in ("true", "1", "yes")
            if not verified:
                continue
            api_id = row.get("api_team_id", "")
            if api_id == "":
                continue
            result[row["internal_team"]] = int(api_id)
    return result


def load_mapping_rows(path: Path | None = None) -> list[dict]:
    """Return all rows (verified and unverified) from the mapping CSV."""
    p = path if path is not None else _DEFAULT_MAPPING_PATH
    if not Path(p).exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Backwards-compatible module-level dict, computed at import time from the
# verified mapping file. Only verified=True teams appear here.
TEAM_API_IDS: dict[str, int] = load_verified_team_ids()

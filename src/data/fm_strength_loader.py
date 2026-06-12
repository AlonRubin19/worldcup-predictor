"""fm_strength_loader.py — load Football Manager squad/team strength ratings.

Reads data/fm_team_strength.csv (one row per national team, FM-derived
goalkeeper/defense/midfield/attack/depth/overall ratings plus a free-text
list of top players) and exposes a normalized-name lookup so callers can
query by any common alias for a team (e.g. "USA", "United States", "USMNT"
all resolve to the same row).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DEFAULT = Path(__file__).parent.parent.parent / "data" / "fm_team_strength.csv"

# Alias -> canonical normalized key. Both FM csv team names and lookup names
# are passed through `normalize_team_name`, so only one direction is needed.
_ALIAS_MAP: dict[str, str] = {
    "usa": "united states",
    "usmnt": "united states",
    "united states of america": "united states",
    "turkey": "turkiye",
    "bosnia": "bosnia and herzegovina",
    "ir iran": "iran",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "korea dpr": "north korea",
    "dpr korea": "north korea",
    "cote d'ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "czech republic": "czechia",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to a canonical, ASCII, lower-case key.

    Strips accents (e.g. "Türkiye" -> "turkiye", "Côte d'Ivoire" ->
    "cote d'ivoire" -> "ivory coast" via the alias map) and resolves known
    aliases so FM data and fixture/team data can be joined regardless of
    which display name each source uses.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    key = ascii_name.strip().lower()
    return _ALIAS_MAP.get(key, key)


@dataclass
class FMTeamStrength:
    team: str
    players: int
    goalkeeper: float
    defense: float
    midfield: float
    attack: float
    depth: float
    overall: float
    top_players: str


def load_fm_team_strength(path: Path | None = None) -> dict[str, FMTeamStrength]:
    """Load FM team-strength ratings, keyed by normalized team name.

    Returns an empty dict (rather than raising) if the file is missing, so
    callers that treat FM data as an optional enhancement layer don't break
    when the file hasn't been added yet.
    """
    p = path if path is not None else _DEFAULT
    if not Path(p).exists():
        return {}

    df = pd.read_csv(p)
    out: dict[str, FMTeamStrength] = {}
    for _, row in df.iterrows():
        fm = FMTeamStrength(
            team=str(row["team"]),
            players=int(row["players"]),
            goalkeeper=float(row["fm_goalkeeper_rating"]),
            defense=float(row["fm_defense_rating"]),
            midfield=float(row["fm_midfield_rating"]),
            attack=float(row["fm_attack_rating"]),
            depth=float(row["fm_depth_rating"]),
            overall=float(row["fm_overall_rating"]),
            top_players=str(row.get("top_5_players", "")),
        )
        out[normalize_team_name(fm.team)] = fm
    return out


def get_fm_strength(team: str, fm_data: dict[str, FMTeamStrength]) -> FMTeamStrength | None:
    """Look up FM strength for `team` by normalized name. Returns None if missing."""
    return fm_data.get(normalize_team_name(team))

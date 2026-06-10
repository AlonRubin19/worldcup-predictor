"""Extend data/player_profiles.csv with API-Football squad data.

Sprint 17, Priority 2/3: adds three transparency columns to every existing
row (source_type, research_valid, penalty_taker -- defaulting the original
8-team placeholder rows to "placeholder"/False/False, with a small number of
known-penalty-taker overrides), and appends current squads for Spain and
Norway (previously entirely absent, causing Lamine Yamal and Erling Haaland
to be missing from Golden Boot), plus Michael Olise for France.

Data validity:
  - "placeholder"            -- original 8-team estimates (research_valid=False)
  - "api_football_squad_only" -- live squad from /players/squads, no season
                                  statistics available -> small placeholder
                                  xG by position (research_valid=False)
  - "api_football"           -- live squad + season statistics
                                  (research_valid=True, but NOTE: national-team
                                  statistics samples are very small (tens of
                                  minutes), so even these are low-confidence --
                                  see Sprint 17 audit report)
  - "manual_assumption"      -- a small number of star players (Yamal,
                                  Haaland, Olise) where the live API sample
                                  was absent or too noisy; xg_per_90 set from
                                  known current club form (research_valid=False)

Run from project root:
    python scripts/extend_player_profiles.py

Output:
    data/player_profiles.csv (overwritten; original backed up to
    data/player_profiles_v1_archive.csv before this script was first run)
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.data.api_football_client import ApiFootballClient
from src.data.player_squad_adapter import build_team_player_profiles

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_CSV = _DATA_DIR / "player_profiles.csv"

_COLUMNS = [
    "player_id", "player_name", "team", "position", "club",
    "minutes_last_90_days", "national_team_minutes_last_12_months",
    "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
    "defensive_actions_per_90", "international_caps", "base_impact_score",
    "source_type", "research_valid", "penalty_taker",
]

# Known primary penalty-takers among the existing 8-team placeholder set.
_KNOWN_PENALTY_TAKERS = {"p001", "p101", "p201", "p401"}  # Kane, Mbappe, Messi, Ronaldo

_SEASON = 2025


def main() -> None:
    df = pd.read_csv(_PROFILES_CSV)

    # ── 1. Add transparency columns to existing rows ────────────────────────
    df["source_type"] = "placeholder"
    df["research_valid"] = False
    df["penalty_taker"] = df["player_id"].isin(_KNOWN_PENALTY_TAKERS)

    rows = df.to_dict("records")

    # ── 2. Add Michael Olise (France) -- currently missing entirely ─────────
    rows.append({
        "player_id": "p112", "player_name": "Michael Olise", "team": "France",
        "position": "MF", "club": "Bayern Munich",
        "minutes_last_90_days": 0.0, "national_team_minutes_last_12_months": 0.0,
        "goals_per_90": 0.30, "assists_per_90": 0.35, "xg_per_90": 0.28,
        "xa_per_90": 0.32, "defensive_actions_per_90": 2.5,
        "international_caps": 10, "base_impact_score": 1.05,
        "source_type": "manual_assumption", "research_valid": False,
        "penalty_taker": False,
    })

    # ── 3. Pull live squads for Spain and Norway via API-Football ──────────
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        print("WARNING: API_FOOTBALL_KEY not set -- skipping Spain/Norway squad fetch.")
    else:
        client = ApiFootballClient(api_key=api_key, cache_dir=str(_DATA_DIR / "api_cache"))

        for team_id, team_name, prefix in [(9, "Spain", "esp"), (1090, "Norway", "nor")]:
            squad_rows = build_team_player_profiles(client, team_id, team_name, season=_SEASON)
            print(f"{team_name}: fetched {len(squad_rows)} players")
            for i, r in enumerate(squad_rows):
                r["player_id"] = f"{prefix}{i+1:03d}"
                r["club"] = ""
                # Ensure all expected columns exist
                for col in _COLUMNS:
                    r.setdefault(col, 0)
                rows.append(r)

    out = pd.DataFrame(rows)

    # ── 4. Manual overrides for star players with noisy/missing API stats ──
    # National-team /players statistics samples are tens of minutes --
    # too small to be a reliable xg_per_90. Use known current club-form
    # estimates for headline players, clearly marked manual_assumption.
    _MANUAL_OVERRIDES = {
        "Lamine Yamal":   dict(xg_per_90=0.45, goals_per_90=0.35, assists_per_90=0.40, xa_per_90=0.40),
        "E. Haaland":     dict(xg_per_90=0.95, goals_per_90=1.00, assists_per_90=0.10, xa_per_90=0.08),
    }
    for name, vals in _MANUAL_OVERRIDES.items():
        mask = out["player_name"] == name
        if mask.any():
            for k, v in vals.items():
                out.loc[mask, k] = v
            out.loc[mask, "source_type"] = "manual_assumption"
            out.loc[mask, "research_valid"] = False

    out = out[_COLUMNS]
    out.to_csv(_PROFILES_CSV, index=False)
    print(f"Wrote {len(out)} player rows ({out['team'].nunique()} teams) to {_PROFILES_CSV}")
    print("Teams covered:", sorted(out["team"].unique()))


if __name__ == "__main__":
    main()

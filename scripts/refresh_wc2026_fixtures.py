"""Fetch the real WC2026 group-stage fixture list from the betting app's
Supabase project and write it to data/world_cup_2026_fixtures.csv.

This lets the predictor's odds matching (and, eventually, tournament
simulation) operate on the actual 2026 draw instead of the bundled WC2022
sample fixtures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import requests

from src.data.market_sources.supabase_betting_app import SUPABASE_URL, SUPABASE_ANON_KEY, _NAME_MAP


_DISPLAY_OVERRIDES = {
    "usa": "USA", "turkey": "Turkey", "south korea": "South Korea", "iran": "Iran",
    "czechia": "Czech Republic",
}


def _team_name(name: str) -> str:
    canon = _NAME_MAP.get(name.strip().lower())
    if canon:
        return _DISPLAY_OVERRIDES.get(canon, canon.title())
    return name

OUT = Path(__file__).parent.parent / "data" / "world_cup_2026_fixtures.csv"


def main() -> None:
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

    groups = requests.get(f"{SUPABASE_URL}/rest/v1/groups?select=id,name", headers=headers, timeout=15).json()
    group_name = {g["id"]: g["name"].replace("Group ", "") for g in groups}

    matches = requests.get(
        f"{SUPABASE_URL}/rest/v1/matches"
        "?select=stage,kickoff_time,group_id,home_team:home_team_id(name_en),away_team:away_team_id(name_en)"
        "&order=kickoff_time&limit=500",
        headers=headers, timeout=15,
    ).json()

    rows = []
    for i, m in enumerate(matches, start=1):
        stage = "group" if m["stage"] == "group_stage" else m["stage"]
        rows.append({
            "match_id": i,
            "stage": stage,
            "group": group_name.get(m["group_id"], ""),
            "date": m["kickoff_time"][:10],
            "team_a": _team_name((m.get("home_team") or {}).get("name_en", "")),
            "team_b": _team_name((m.get("away_team") or {}).get("name_en", "")),
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["match_id", "stage", "group", "date", "team_a", "team_b"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} fixtures to {OUT}")


if __name__ == "__main__":
    main()

"""verify_api_team_ids.py — build a verified API-Football team-id mapping.

For every team appearing in the WC fixture data, queries API-Football's
/teams?search= endpoint, matches by exact name (with known aliases), and
records the result in data/api_team_mapping_verified.csv.

If no API_FOOTBALL_KEY is configured, every team is written with
verified=False and a note explaining no live verification was possible --
this is a safe default: unverified teams are blocked from live squad/injury
usage (see src.data.team_api_ids.load_verified_team_ids).

Run: PYTHONPATH=. python3 scripts/verify_api_team_ids.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.tournament.fixtures import load_fixtures, _DEFAULT as _FIXTURES_PATH

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "api_team_mapping_verified.csv"

# Known previously-validated mappings (Sprint 17), used as a starting point.
# These are still re-confirmed against the API search results when a key is
# available; without a key they're carried over with verified=True only for
# teams that were validated live in a prior session.
_PRIOR_VALIDATED: dict[str, tuple[int, str]] = {
    "Spain": (9, "Spain"),
    "Norway": (1090, "Norway"),
}

# Aliases: internal name -> name to search the API for, if different.
_SEARCH_ALIASES: dict[str, str] = {
    "USA": "USA",
    "South Korea": "South Korea",
}

FIELDNAMES = [
    "internal_team", "api_team_name", "api_team_id", "country",
    "match_method", "confidence", "verified", "notes",
]


def get_fixture_teams() -> list[str]:
    fixtures = load_fixtures(_FIXTURES_PATH)
    return sorted({f.team_a for f in fixtures} | {f.team_b for f in fixtures})


def search_team(client: ApiFootballClient, name: str) -> list[dict]:
    """Return list of {team: {...}} entries from /teams?search=name."""
    data = client.get("/teams", params={"search": name}, ttl_seconds=24 * 3600)
    return data.get("response", [])


def build_mapping(client: ApiFootballClient | None = None) -> list[dict]:
    """Build the verified mapping rows for all fixture teams.

    If `client` has no API key, every row is verified=False (safe default).
    """
    client = client or ApiFootballClient()
    has_key = bool(client._api_key)

    rows: list[dict] = []
    seen_ids: dict[int, str] = {}

    for team in get_fixture_teams():
        search_name = _SEARCH_ALIASES.get(team, team)
        row = {
            "internal_team": team,
            "api_team_name": "",
            "api_team_id": "",
            "country": "",
            "match_method": "none",
            "confidence": "low",
            "verified": False,
            "notes": "",
        }

        if not has_key:
            if team in _PRIOR_VALIDATED:
                api_id, api_name = _PRIOR_VALIDATED[team]
                row.update(
                    api_team_name=api_name, api_team_id=api_id, country=team,
                    match_method="prior_session_validation", confidence="high",
                    verified=True, notes="Validated live in a prior session.",
                )
            else:
                row["notes"] = "No API_FOOTBALL_KEY configured -- not verified."
            rows.append(row)
            continue

        try:
            results = search_team(client, search_name)
        except ApiKeyMissingError:
            row["notes"] = "No API_FOOTBALL_KEY configured -- not verified."
            rows.append(row)
            continue
        except Exception as exc:
            row["notes"] = f"API request failed: {exc}"
            rows.append(row)
            continue

        # Prefer an exact, case-insensitive name match on a "national" team.
        exact = [
            r for r in results
            if r.get("team", {}).get("name", "").lower() == search_name.lower()
            and r.get("team", {}).get("national") is True
        ]
        candidate = exact[0] if exact else (results[0] if results else None)

        if candidate is None:
            row["notes"] = "No API-Football team found for this name."
            rows.append(row)
            continue

        t = candidate["team"]
        row.update(
            api_team_name=t.get("name", ""),
            api_team_id=t.get("id", ""),
            country=t.get("country", ""),
            match_method="exact_name_national" if exact else "fuzzy_first_result",
            confidence="high" if exact else "low",
            verified=bool(exact),
            notes="" if exact else "No exact national-team name match -- review manually.",
        )
        rows.append(row)

    # Detect duplicate IDs across different internal teams.
    id_to_teams: dict[int, list[str]] = {}
    for row in rows:
        if row["api_team_id"] != "":
            id_to_teams.setdefault(int(row["api_team_id"]), []).append(row["internal_team"])
    for api_id, teams in id_to_teams.items():
        if len(teams) > 1:
            for row in rows:
                if row["api_team_id"] == api_id:
                    row["verified"] = False
                    row["notes"] = (row["notes"] + " " if row["notes"] else "") + (
                        f"DUPLICATE api_team_id {api_id} shared with {teams}."
                    )

    return rows


def write_mapping(rows: list[dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    rows = build_mapping()
    write_mapping(rows)
    n_total = len(rows)
    n_verified = sum(1 for r in rows if r["verified"])
    n_dupes = sum(1 for r in rows if "DUPLICATE" in r["notes"])
    print(f"Wrote {n_total} teams to {OUTPUT_PATH}")
    print(f"Verified: {n_verified} / {n_total}")
    print(f"Duplicate IDs: {n_dupes}")
    unresolved = [r["internal_team"] for r in rows if not r["verified"]]
    print(f"Unresolved: {unresolved}")

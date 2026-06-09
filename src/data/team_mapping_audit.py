"""Team mapping audit — classifies API-Football team names against internal model.

Three classification outcomes:
  EXACT   — API name is already a known internal team name (no mapping needed)
  MAPPED  — API name is in TEAM_NAME_MAP and translates to an internal name
  UNKNOWN — API name is unrecognised; mapping gap, needs attention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.data.live_fixture_loader import LiveFixture, TEAM_NAME_MAP


# ── Classification enum ───────────────────────────────────────────────────────

class TeamMappingClass(str, Enum):
    EXACT   = "exact"
    MAPPED  = "mapped"
    UNKNOWN = "unknown"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MappingAuditResult:
    exact_count:   int
    mapped_count:  int
    unknown_count: int
    unknown_teams: list[str]                  # unique API names with no known mapping
    exact_teams:   list[str]                  # unique API names that matched directly
    mapped_teams:  list[tuple[str, str]]      # unique (api_name, internal_name) pairs


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_team(api_name: str, known_teams: set[str]) -> TeamMappingClass:
    """Classify a single API team name.

    Args:
        api_name:    Raw team name from API-Football.
        known_teams: Set of internal team names (from teams.csv).

    Returns:
        TeamMappingClass.EXACT   if api_name is directly in known_teams.
        TeamMappingClass.MAPPED  if api_name is in TEAM_NAME_MAP (alias exists).
        TeamMappingClass.UNKNOWN otherwise.
    """
    if api_name in TEAM_NAME_MAP:
        return TeamMappingClass.MAPPED
    if api_name in known_teams:
        return TeamMappingClass.EXACT
    return TeamMappingClass.UNKNOWN


# ── Audit function ────────────────────────────────────────────────────────────

def audit_team_mappings(
    fixtures: list[LiveFixture],
    known_teams: set[str] | None = None,
) -> MappingAuditResult:
    """Audit how well API team names map to internal names across a fixture list.

    Processes both home and away teams from every fixture.
    Deduplicates unknown / exact / mapped entries so each unique API name
    appears only once in the respective list.

    Args:
        fixtures:    List of LiveFixture objects (from parse_fixtures_response).
        known_teams: Set of known internal team names. Pass set(load_teams()) for
                     production use. If None, defaults to empty set (all non-mapped
                     names become UNKNOWN).

    Returns:
        MappingAuditResult with counts and de-duplicated team name lists.
    """
    if known_teams is None:
        known_teams = set()

    # Collect all API names from both sides of each fixture
    all_api_names: list[str] = []
    for f in fixtures:
        all_api_names.append(f.home_team_api)
        all_api_names.append(f.away_team_api)

    exact_set:  set[str]             = set()
    mapped_set: dict[str, str]       = {}  # api_name → internal_name
    unknown_set: set[str]            = set()

    for api_name in all_api_names:
        cls = classify_team(api_name, known_teams)
        if cls == TeamMappingClass.EXACT:
            exact_set.add(api_name)
        elif cls == TeamMappingClass.MAPPED:
            mapped_set[api_name] = TEAM_NAME_MAP[api_name]
        else:
            unknown_set.add(api_name)

    # Counts reflect total occurrences (not just unique)
    exact_count   = sum(1 for n in all_api_names
                        if classify_team(n, known_teams) == TeamMappingClass.EXACT)
    mapped_count  = sum(1 for n in all_api_names
                        if classify_team(n, known_teams) == TeamMappingClass.MAPPED)
    unknown_count = sum(1 for n in all_api_names
                        if classify_team(n, known_teams) == TeamMappingClass.UNKNOWN)

    return MappingAuditResult(
        exact_count=exact_count,
        mapped_count=mapped_count,
        unknown_count=unknown_count,
        unknown_teams=sorted(unknown_set),
        exact_teams=sorted(exact_set),
        mapped_teams=sorted(mapped_set.items()),
    )

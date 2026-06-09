"""Player impact engine — modifies xG based on squad availability and form.

Engineering validation only — player data not yet research-valid.

Formula:
    player_match_impact = base_impact_score * availability_factor * form_factor
    starting_xi_strength = mean(player_match_impact for expected starters)
    baseline_xi_strength = mean(top-11 base_impact_score for that team)
    squad_factor = starting_xi_strength / baseline_xi_strength
    squad_factor clamped to [0.85, 1.15]
    team_xg *= squad_factor
"""

from dataclasses import dataclass
from src.data.player_loader import PlayerProfile, PlayerAvailability

SQUAD_FACTOR_MIN = 0.85
SQUAD_FACTOR_MAX = 1.15
SQUAD_SIZE = 11


def calculate_player_match_impact(profile: PlayerProfile, avail: PlayerAvailability) -> float:
    """Compute a single player's match impact score."""
    return profile.base_impact_score * avail.availability_factor * avail.form_factor


def calculate_squad_factor(
    team: str,
    profiles: dict[str, PlayerProfile],
    availability: list[PlayerAvailability],
) -> float:
    """Calculate squad factor for a team in a specific match.

    Returns 1.0 if no availability data exists for the team.
    Clamped to [SQUAD_FACTOR_MIN, SQUAD_FACTOR_MAX].
    """
    team_avail = [a for a in availability if a.team == team and a.expected_starter]
    team_profiles = {pid: p for pid, p in profiles.items() if p.team == team}

    if not team_avail or not team_profiles:
        return 1.0

    impacts = []
    for a in team_avail:
        if a.player_id in team_profiles:
            impacts.append(calculate_player_match_impact(team_profiles[a.player_id], a))

    if not impacts:
        return 1.0

    starting_xi_strength = sum(impacts) / len(impacts)

    top11_scores = sorted(
        (p.base_impact_score for p in team_profiles.values()),
        reverse=True,
    )[:SQUAD_SIZE]

    if not top11_scores:
        return 1.0

    baseline_xi_strength = sum(top11_scores) / len(top11_scores)

    if baseline_xi_strength == 0:
        return 1.0

    raw_factor = starting_xi_strength / baseline_xi_strength
    return float(max(SQUAD_FACTOR_MIN, min(SQUAD_FACTOR_MAX, raw_factor)))


@dataclass
class PlayerImpactInput:
    xg_a: float
    xg_b: float
    squad_factor_a: float
    squad_factor_b: float


def apply_player_impact(inp: PlayerImpactInput) -> tuple[float, float]:
    """Apply squad factors to base xG values."""
    return inp.xg_a * inp.squad_factor_a, inp.xg_b * inp.squad_factor_b

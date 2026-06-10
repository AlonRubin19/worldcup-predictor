"""current_squad_strength.py — adjust team attack strength using current squad data.

Computes a multiplicative factor to apply to a team's MLE alpha_attack based
on how the *current* available squad's xG/90 compares to a fixed baseline
(the placeholder/default xG/90 for a starting attacker, _BASELINE_XG_PER_90).

This lets current squad strength (e.g. an in-form player returning, or a key
player injured/missing) nudge the attack rating without re-running the full
MLE fit. Only research-valid (live API-Football or manually-verified) data
moves the factor away from 1.0 -- placeholder-only data is neutral and is
flagged via research_valid=False so the UI can show a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.player_loader import PlayerProfile

# Baseline xG/90 representing a "typical" starting attacker, used as the
# reference point for "above/below baseline" squad strength.
_BASELINE_XG_PER_90 = 0.30

# Cap how much the current-squad signal can move alpha_attack, to avoid a
# small sample of players dominating the MLE-fitted base rate.
_MAX_FACTOR = 1.30
_MIN_FACTOR = 0.70


@dataclass
class SquadStrengthResult:
    factor: float              # multiplier to apply to alpha_attack
    team_player_count: int     # number of (non-injured) profiles considered
    research_valid: bool       # True only if >=1 research-valid profile used


def compute_squad_strength_factor(
    team: str,
    profiles: dict[str, PlayerProfile],
    injured_player_names: set[str] | None = None,
) -> SquadStrengthResult:
    """Compute a current-squad attack-strength factor for `team`.

    Args:
        team: team name to filter profiles by.
        profiles: player_id -> PlayerProfile (e.g. from load_player_profiles()).
        injured_player_names: set of player_ids/names to exclude (unavailable).

    Returns:
        SquadStrengthResult with factor==1.0 (neutral) if no profiles, or if
        no research-valid data is available for this team.
    """
    injured = injured_player_names or set()

    team_profiles = [
        p for pid, p in profiles.items()
        if p.team == team and pid not in injured and p.player_name not in injured
    ]

    if not team_profiles:
        return SquadStrengthResult(factor=1.0, team_player_count=0, research_valid=False)

    valid_profiles = [p for p in team_profiles if p.research_valid]
    if not valid_profiles:
        return SquadStrengthResult(
            factor=1.0, team_player_count=len(team_profiles), research_valid=False,
        )

    avg_xg = sum(p.xg_per_90 for p in valid_profiles) / len(valid_profiles)
    factor = avg_xg / _BASELINE_XG_PER_90
    factor = max(_MIN_FACTOR, min(_MAX_FACTOR, factor))

    return SquadStrengthResult(
        factor=factor, team_player_count=len(team_profiles), research_valid=True,
    )

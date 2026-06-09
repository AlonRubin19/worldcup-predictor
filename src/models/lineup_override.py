"""Lineup Override / Pre-Match Update Engine.

Allows manual per-player availability and form overrides that recalculate
squad strength and propagate through xG → DC matrix → win/draw/loss probabilities.

Label: Engineering validation only — not research-valid unless sourced from
official pre-match lineups.

Design:
    squad_factor = mean(availability_factor * form_factor  for expected starters)
    squad_factor clamped to [SQUAD_FACTOR_MIN, SQUAD_FACTOR_MAX]
    xg_adjusted = xg_base * squad_factor
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.models.dixon_coles import build_dc_matrix
from src.models.research_valid_predictor import DEFAULT_RHO

SQUAD_FACTOR_MIN: float = 0.85
SQUAD_FACTOR_MAX: float = 1.15


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerOverride:
    """Manual availability / form entry for a single player."""
    player_id: str
    player_name: str
    team: str
    expected_starter: bool
    availability_status: str          # "fit" | "doubtful" | "out" | "suspended"
    availability_factor: float        # 1.0 = fully fit, 0.0 = unavailable
    form_factor: float                # 1.0 = normal, >1 = in form, <1 = poor form


@dataclass
class LineupOverride:
    """A team's full set of player overrides for a single match."""
    team: str
    players: list[PlayerOverride] = field(default_factory=list)


@dataclass
class LineupOverrideResult:
    """Complete before/after comparison after applying lineup overrides."""
    team_a: str
    team_b: str
    # Squad factors
    squad_factor_a: float
    squad_factor_b: float
    # xG
    xg_a_base: float
    xg_b_base: float
    xg_a_adjusted: float
    xg_b_adjusted: float
    # Win / draw / loss probabilities
    win_a_base: float
    draw_base: float
    win_b_base: float
    win_a_adjusted: float
    draw_adjusted: float
    win_b_adjusted: float
    # Deltas (adjusted − base)
    delta_win_a: float
    delta_draw: float
    delta_win_b: float
    delta_xg_a: float
    delta_xg_b: float
    # Data label — always False: manual overrides are engineering-valid only
    is_research_valid: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Squad factor computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_override_squad_factor(override: LineupOverride) -> float:
    """Compute squad factor from a LineupOverride.

    squad_factor = mean(availability_factor * form_factor for each expected starter)
    Clamped to [SQUAD_FACTOR_MIN, SQUAD_FACTOR_MAX].
    Returns 1.0 when no starters are defined.
    """
    starters = [p for p in override.players if p.expected_starter]
    if not starters:
        return 1.0

    raw = sum(p.availability_factor * p.form_factor for p in starters) / len(starters)
    return float(max(SQUAD_FACTOR_MIN, min(SQUAD_FACTOR_MAX, raw)))


# ─────────────────────────────────────────────────────────────────────────────
# Probability extraction from DC matrix
# ─────────────────────────────────────────────────────────────────────────────

def _probs_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract (win_a, draw, win_b) from a DC score matrix."""
    win_a = float(np.tril(matrix, k=-1).sum())
    draw  = float(np.trace(matrix))
    win_b = float(np.triu(matrix, k=1).sum())
    return win_a, draw, win_b


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def apply_lineup_override(
    team_a: str,
    team_b: str,
    xg_a_base: float,
    xg_b_base: float,
    override_a: LineupOverride | None = None,
    override_b: LineupOverride | None = None,
    rho: float = DEFAULT_RHO,
) -> LineupOverrideResult:
    """Apply lineup overrides and return before/after comparison.

    If override_a or override_b is None, squad_factor defaults to 1.0
    (full-strength, no change).

    Returns:
        LineupOverrideResult with base and adjusted probabilities and deltas.
        is_research_valid is always False — manual overrides are engineering only.
    """
    # Squad factors
    sf_a = compute_override_squad_factor(override_a) if override_a is not None else 1.0
    sf_b = compute_override_squad_factor(override_b) if override_b is not None else 1.0

    # Adjusted xG
    xg_a_adj = xg_a_base * sf_a
    xg_b_adj = xg_b_base * sf_b

    # Base probabilities
    base_matrix = build_dc_matrix(xg_a_base, xg_b_base, rho=rho)
    win_a_base, draw_base, win_b_base = _probs_from_matrix(base_matrix)

    # Adjusted probabilities
    adj_matrix = build_dc_matrix(xg_a_adj, xg_b_adj, rho=rho)
    win_a_adj, draw_adj, win_b_adj = _probs_from_matrix(adj_matrix)

    return LineupOverrideResult(
        team_a=team_a,
        team_b=team_b,
        squad_factor_a=sf_a,
        squad_factor_b=sf_b,
        xg_a_base=xg_a_base,
        xg_b_base=xg_b_base,
        xg_a_adjusted=xg_a_adj,
        xg_b_adjusted=xg_b_adj,
        win_a_base=win_a_base,
        draw_base=draw_base,
        win_b_base=win_b_base,
        win_a_adjusted=win_a_adj,
        draw_adjusted=draw_adj,
        win_b_adjusted=win_b_adj,
        delta_win_a=win_a_adj - win_a_base,
        delta_draw=draw_adj - draw_base,
        delta_win_b=win_b_adj - win_b_base,
        delta_xg_a=xg_a_adj - xg_a_base,
        delta_xg_b=xg_b_adj - xg_b_base,
        is_research_valid=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default lineup factory
# ─────────────────────────────────────────────────────────────────────────────

def create_default_lineup(team: str, n_starters: int = 11) -> LineupOverride:
    """Create a default lineup with n_starters placeholder players, all fit.

    All players have availability_factor=1.0 and form_factor=1.0, so the
    resulting squad_factor is exactly 1.0 (no change to base xG).
    """
    players = [
        PlayerOverride(
            player_id=f"{team.lower().replace(' ', '_')}_p{i + 1}",
            player_name=f"Player {i + 1}",
            team=team,
            expected_starter=True,
            availability_status="fit",
            availability_factor=1.0,
            form_factor=1.0,
        )
        for i in range(n_starters)
    ]
    return LineupOverride(team=team, players=players)

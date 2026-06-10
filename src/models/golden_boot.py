"""golden_boot.py — tournament Golden Boot (top scorer) projection model.

For every attacking player, computes Expected Goals Tournament (xGT):

    xGT = Expected Team Matches
          x (Expected Minutes / 90)
          x xG per 90
          x Penalty Factor
          x Starting Probability

"Expected Team Matches" is derived from the Monte Carlo tournament
simulation results (3 guaranteed group-stage matches + probability of
reaching each knockout round).

Once xGT is known per player, each player's tournament goal total is
modeled as Poisson(xGT). The "probability to finish top scorer" is
estimated by Monte Carlo sampling across all players' Poisson
distributions simultaneously (so ties / correlations across the field
are captured naturally).

DATA LABEL: Engineering validation only. Player profiles currently exist
for a subset of teams (data/player_profiles.csv). Players for teams not
in that file are simply absent from the projection -- the architecture
is built so richer API-Football player statistics can replace
player_profiles.csv without changing this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import poisson

from src.data.player_loader import PlayerProfile
from src.tournament.simulator import MonteCarloResult


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (used when richer per-player data is not available)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EXPECTED_MINUTES = 90.0
DEFAULT_PENALTY_FACTOR = 1.0
PENALTY_TAKER_FACTOR = 1.05
DEFAULT_STARTING_PROBABILITY_BY_POSITION = {
    "FW": 0.80,
    "MF": 0.75,
    "DF": 0.70,
    "GK": 0.95,
}
_DEFAULT_STARTING_PROBABILITY = 0.70

# Number of knockout rounds counted beyond the 3 group-stage matches.
_GROUP_STAGE_MATCHES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldenBootPlayerResult:
    """Golden Boot projection for one player.

    Attributes:
        player_id, player_name, team, position: identity fields.
        expected_goals:     xGT -- expected goals across the tournament.
        prob_top_scorer:    P(this player finishes as outright top scorer),
                            estimated via Monte Carlo over all players.
        prob_score_3plus:   P(Poisson(xGT) >= 3).
        prob_score_5plus:   P(Poisson(xGT) >= 5).
        prob_score_7plus:   P(Poisson(xGT) >= 7).
        most_likely_goals:  Mode of Poisson(xGT) -- floor(xGT).
    """
    player_id: str
    player_name: str
    team: str
    position: str
    expected_goals: float
    prob_top_scorer: float
    prob_score_3plus: float
    prob_score_5plus: float
    prob_score_7plus: float
    most_likely_goals: int


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def expected_team_matches(team: str, mc: MonteCarloResult) -> float:
    """Expected number of matches a team plays across the tournament.

    = 3 guaranteed group-stage matches
      + P(reach R16)        -- plays the R16 match
      + P(reach QF)         -- plays the QF match
      + P(reach SF)         -- plays the SF match
      + P(reach Final)      -- plays the Final

    Teams with no Monte Carlo data (e.g. not in the simulated fixture
    set) default to group-stage-only (3 matches).
    """
    return (
        _GROUP_STAGE_MATCHES
        + mc.reach_r32.get(team, 0.0)
        + mc.reach_r16.get(team, 0.0)
        + mc.reach_qf.get(team, 0.0)
        + mc.reach_sf.get(team, 0.0)
        + mc.reach_final.get(team, 0.0)
    )


def compute_xgt(
    expected_team_matches: float,
    expected_minutes: float,
    xg_per_90: float,
    penalty_factor: float,
    starting_probability: float,
) -> float:
    """Expected Goals Tournament (xGT).

    xGT = Expected Team Matches
          x (Expected Minutes / 90)
          x xG per 90
          x Penalty Factor
          x Starting Probability
    """
    return (
        expected_team_matches
        * (expected_minutes / 90.0)
        * xg_per_90
        * penalty_factor
        * starting_probability
    )


def poisson_at_least(lam: float, k: int) -> float:
    """P(Poisson(lam) >= k)."""
    if lam <= 0.0:
        return 0.0
    return float(poisson.sf(k - 1, lam))


def most_likely_goals(lam: float) -> int:
    """Mode of Poisson(lam) -- floor(lam) (Poisson mode is floor(lambda),
    with a tie at lambda-1 when lambda is an integer; floor is a fine
    single-value summary for display)."""
    if lam <= 0.0:
        return 0
    return int(math.floor(lam))


def _starting_probability_for(profile: PlayerProfile) -> float:
    return DEFAULT_STARTING_PROBABILITY_BY_POSITION.get(
        profile.position, _DEFAULT_STARTING_PROBABILITY
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_golden_boot(
    profiles: dict[str, PlayerProfile],
    mc: MonteCarloResult,
    expected_minutes: dict[str, float] | None = None,
    penalty_factors: dict[str, float] | None = None,
    starting_probabilities: dict[str, float] | None = None,
    n_sims: int = 20_000,
    rng_seed: int | None = 42,
) -> list[GoldenBootPlayerResult]:
    """Project Golden Boot outcomes for every player in `profiles`.

    Args:
        profiles: player_id -> PlayerProfile (e.g. from load_player_profiles()).
        mc:       Monte Carlo tournament simulation result (for expected
                  team matches per team).
        expected_minutes:        optional player_id -> expected minutes per
                                  match override. Defaults to 90.0.
        penalty_factors:         optional player_id -> penalty factor
                                  override. Defaults to 1.0.
        starting_probabilities:  optional player_id -> starting probability
                                  override. Defaults by position.
        n_sims:   number of Monte Carlo draws for top-scorer probability.
        rng_seed: RNG seed for reproducibility (None = non-deterministic).

    Returns:
        List of GoldenBootPlayerResult sorted by expected_goals descending.
        Empty list if `profiles` is empty.
    """
    expected_minutes = expected_minutes or {}
    penalty_factors = penalty_factors or {}
    starting_probabilities = starting_probabilities or {}

    if not profiles:
        return []

    player_ids: list[str] = []
    lambdas: list[float] = []
    meta: list[PlayerProfile] = []

    for pid, prof in profiles.items():
        team_matches = expected_team_matches(prof.team, mc)
        minutes = expected_minutes.get(pid, DEFAULT_EXPECTED_MINUTES)
        default_pen = PENALTY_TAKER_FACTOR if getattr(prof, "penalty_taker", False) else DEFAULT_PENALTY_FACTOR
        pen = penalty_factors.get(pid, default_pen)
        start_p = starting_probabilities.get(pid, _starting_probability_for(prof))

        xgt = compute_xgt(team_matches, minutes, prof.xg_per_90, pen, start_p)

        player_ids.append(pid)
        lambdas.append(xgt)
        meta.append(prof)

    lambdas_arr = np.array(lambdas, dtype=float)

    # Monte Carlo: sample goals for every player simultaneously, find the
    # per-trial top scorer (ties split equally via random tie-break).
    rng = np.random.default_rng(rng_seed)
    top_scorer_counts = np.zeros(len(player_ids), dtype=float)

    if np.any(lambdas_arr > 0):
        samples = rng.poisson(lam=lambdas_arr, size=(n_sims, len(lambdas_arr)))
        max_per_trial = samples.max(axis=1)
        for i in range(n_sims):
            row = samples[i]
            mx = max_per_trial[i]
            if mx <= 0:
                continue
            winners = np.flatnonzero(row == mx)
            # split credit equally among tied top scorers
            top_scorer_counts[winners] += 1.0 / len(winners)

    prob_top_scorer = top_scorer_counts / n_sims

    results = []
    for i, pid in enumerate(player_ids):
        prof = meta[i]
        lam = float(lambdas_arr[i])
        results.append(GoldenBootPlayerResult(
            player_id=pid,
            player_name=prof.player_name,
            team=prof.team,
            position=prof.position,
            expected_goals=lam,
            prob_top_scorer=float(prob_top_scorer[i]),
            prob_score_3plus=poisson_at_least(lam, 3),
            prob_score_5plus=poisson_at_least(lam, 5),
            prob_score_7plus=poisson_at_least(lam, 7),
            most_likely_goals=most_likely_goals(lam),
        ))

    results.sort(key=lambda r: r.expected_goals, reverse=True)
    return results

"""Tournament-level calibration utilities.

These functions operate only at the simulation layer — the core match
prediction model (Dixon-Coles, ELO, xG calibration) is unchanged.

Three independent mechanisms:
  temperature  — softens win/draw/loss distribution (τ > 1 = flatter)
  xg_noise     — log-normal multiplicative noise on xG before sampling
  upset_factor — mixes knockout win probability toward 50/50
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationParams:
    temperature: float = 1.0       # τ: 1.0 = identity, >1 = flatter outcomes
    xg_noise_sigma: float = 0.0    # σ: 0.0 = identity, >0 = log-normal noise on xG
    upset_factor: float = 0.0      # ε: 0.0 = identity, >0 = mix knockout probs toward 0.5


@dataclass
class ConcentrationMetrics:
    top1: float
    top2: float
    top5: float
    entropy: float     # Shannon entropy in bits


def apply_temperature(
    probs: tuple[float, float, float],
    tau: float,
) -> tuple[float, float, float]:
    """Soften or sharpen a (win_a, draw, win_b) distribution via temperature τ.

    τ = 1.0 → identity.
    τ > 1.0 → flatter (more uncertainty).
    τ < 1.0 → sharper (more decisive).

    Uses p_i^(1/τ) / Σ p_j^(1/τ).
    """
    if tau == 1.0:
        return probs

    inv_tau = 1.0 / tau
    scaled = tuple(p ** inv_tau for p in probs)
    total = sum(scaled)
    return tuple(s / total for s in scaled)


def apply_xg_noise(xg: float, sigma: float, rng: np.random.Generator) -> float:
    """Multiply xG by a log-normal draw: xg * exp(N(0, sigma)).

    sigma = 0.0 → identity (returns xg unchanged).
    Preserves positivity; caller should still apply xG floor/ceiling if needed.
    """
    if sigma == 0.0:
        return xg
    noise = rng.normal(0.0, sigma)
    return xg * math.exp(noise)


def apply_upset_factor(prob_a: float, epsilon: float) -> float:
    """Mix P(team_a wins) toward 0.5 by fraction ε.

    upset_factor = 0.0 → identity.
    upset_factor = 1.0 → 50/50 regardless of model.
    """
    if epsilon == 0.0:
        return prob_a
    return (1.0 - epsilon) * prob_a + epsilon * 0.5


def compute_concentration_metrics(win_probs: dict[str, float]) -> ConcentrationMetrics:
    """Compute tournament winner concentration metrics from a probability dict."""
    sorted_probs = sorted(win_probs.values(), reverse=True)
    total = sum(sorted_probs)
    sp = [p / total for p in sorted_probs] if total > 0 else sorted_probs

    top1 = sp[0] if len(sp) >= 1 else 0.0
    top2 = sum(sp[:2])
    top5 = sum(sp[:5])
    entropy = -sum(p * math.log2(p) for p in sp if p > 0)

    return ConcentrationMetrics(top1=top1, top2=top2, top5=top5, entropy=entropy)

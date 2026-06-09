"""Model-signal recommendation engine.

Ranks the most interesting betting market selections from a match prediction.

IMPORTANT: Output is labelled "Model signal ranking — not betting advice."
No language in this module implies guaranteed outcomes or betting instructions.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.betting_markets import BettingMarketProbabilities, MarketProbability


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_HIGH: float = 0.70
THRESHOLD_MEDIUM: float = 0.60
THRESHOLD_EXACT_SCORE: float = 0.10

# Lower rank number = higher priority
_MARKET_PRIORITY: dict[str, int] = {
    "1X2": 1,
    "Double Chance": 2,
    "Over/Under": 3,
    "BTTS": 4,
    "Team Totals": 5,
    "Draw No Bet": 6,
    "Clean Sheet": 7,
    "Exact Score": 8,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    market_name: str
    selection: str
    model_probability: float
    fair_odds: float
    confidence_label: str
    signal_strength: str       # "Strong" | "Moderate" | "Weak"
    rationale: str
    warning: str | None


@dataclass
class RecommendationSet:
    team_a: str
    team_b: str
    recommendations: list[Recommendation]


# ─────────────────────────────────────────────────────────────────────────────
# Rationale templates
# ─────────────────────────────────────────────────────────────────────────────

def _rationale(mp: MarketProbability, team_a: str, team_b: str) -> str:
    mn = mp.market_name
    sel = mp.selection

    if mn == "1X2":
        if team_a in sel:
            return "High model probability and strong 1X2 confidence"
        if team_b in sel:
            return "High model probability and strong 1X2 confidence"
        return "Model rates draw as a likely outcome"

    if mn == "Double Chance":
        if sel == "1X":
            return "Double chance has high probability — covers win or draw for the favoured side"
        if sel == "X2":
            return "Double chance has high probability — covers draw or win for the away side"
        return "Double chance excludes draw — both teams have meaningful win probability"

    if mn == "Over/Under":
        if "Over 0.5" in sel:
            return "Very high probability that at least one goal will be scored"
        if "Over 1.5" in sel:
            return "Combined expected goals strongly support at least two goals"
        if "Over 2.5" in sel:
            return "Over 2.5 supported by combined expected goals from both teams"
        if "Over 3.5" in sel:
            return "High expected goals suggest a high-scoring match"
        if "Under 2.5" in sel:
            return "Low combined expected goals support a tight, low-scoring match"
        return "Expected goals support this total goals market"

    if mn == "BTTS":
        if sel == "BTTS Yes":
            return "Both teams have meaningful scoring probability"
        return "At least one team has very low scoring probability"

    if mn == "Double Chance":
        return "Double chance covers two outcomes, reducing variance"

    if mn == "Draw No Bet":
        if team_a in sel:
            return "Model strongly favours this team when draw is excluded"
        return "Model strongly favours this team when draw is excluded"

    if mn == "Clean Sheet":
        if team_a in sel:
            return f"Low scoring probability for {team_b} supports clean sheet"
        return f"Low scoring probability for {team_a} supports clean sheet"

    if mn == "Team Totals":
        if team_a in sel and "0.5" in sel:
            return f"{team_a} has high probability of scoring at least once"
        if team_a in sel and "1.5" in sel:
            return f"{team_a} has high probability of scoring at least twice"
        if team_b in sel and "0.5" in sel:
            return f"{team_b} has high probability of scoring at least once"
        if team_b in sel and "1.5" in sel:
            return f"{team_b} has high probability of scoring at least twice"
        return "Team total supported by expected goals"

    if mn == "Exact Score":
        return "Most likely scoreline — exact-score confidence is naturally low"

    return "Model signal based on expected goals and historical strength"


# ─────────────────────────────────────────────────────────────────────────────
# Signal strength
# ─────────────────────────────────────────────────────────────────────────────

def _signal_strength(prob: float, confidence: str) -> str:
    if prob >= THRESHOLD_HIGH and confidence == "High":
        return "Strong"
    if prob >= THRESHOLD_MEDIUM:
        return "Moderate"
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# Warning construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_warning(
    is_research_valid: bool,
    data_warnings: list[str],
) -> str | None:
    parts = []
    if not is_research_valid:
        parts.append("Engineering-valid data only — not derived from full historical model")
    if data_warnings:
        parts.append("Model data has quality warnings — interpret with caution")
    return "; ".join(parts) if parts else None


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool construction
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_pool(bm: BettingMarketProbabilities) -> list[MarketProbability]:
    """Return all market selections that meet inclusion thresholds."""
    pool: list[MarketProbability] = []

    for mp in bm.one_x_two:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    for mp in bm.double_chance:
        if mp.probability >= THRESHOLD_HIGH:
            pool.append(mp)

    for mp in bm.over_under:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    for mp in bm.btts:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    for mp in bm.team_totals:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    for mp in bm.draw_no_bet:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    for mp in bm.clean_sheet:
        if mp.probability >= THRESHOLD_MEDIUM:
            pool.append(mp)

    # Exact score: top scoreline only, lower threshold
    if bm.exact_score and bm.exact_score[0].probability >= THRESHOLD_EXACT_SCORE:
        pool.append(bm.exact_score[0])

    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Sorting keys
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key_no_odds(mp: MarketProbability) -> tuple:
    """Sort: tier (high > medium) → confidence → market priority → -probability."""
    tier = 0 if mp.probability >= THRESHOLD_HIGH else 1
    conf_rank = {"High": 0, "Medium": 1, "Low": 2}.get(mp.confidence_label, 3)
    prio = _MARKET_PRIORITY.get(mp.market_name, 99)
    return (tier, conf_rank, prio, -mp.probability)


def _sort_key_with_odds(
    mp: MarketProbability,
    market_implied: dict[str, float],
) -> tuple:
    """Sort by edge (model_prob - market_prob) descending, then by no-odds key."""
    market_p = market_implied.get(mp.selection, mp.probability)
    edge = mp.probability - market_p
    return (-edge, *_sort_key_no_odds(mp))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    betting_markets: BettingMarketProbabilities,
    prediction_confidence: str,
    data_warnings: list[str],
    is_research_valid: bool,
    market_implied_probs: dict[str, float] | None = None,
    top_n: int = 5,
) -> RecommendationSet:
    """Generate ranked model-signal recommendations from betting market probabilities.

    Returns at most top_n recommendations, sorted by signal quality.
    Each recommendation carries a rationale and any applicable warnings.

    This output is model signal only — not betting advice.
    """
    team_a = betting_markets.team_a
    team_b = betting_markets.team_b
    warning = _build_warning(is_research_valid, data_warnings)

    pool = _candidate_pool(betting_markets)

    if market_implied_probs:
        pool.sort(key=lambda mp: _sort_key_with_odds(mp, market_implied_probs))
    else:
        pool.sort(key=_sort_key_no_odds)

    top = pool[:top_n]

    recommendations = [
        Recommendation(
            market_name=mp.market_name,
            selection=mp.selection,
            model_probability=mp.probability,
            fair_odds=mp.implied_fair_odds,
            confidence_label=mp.confidence_label,
            signal_strength=_signal_strength(mp.probability, prediction_confidence),
            rationale=_rationale(mp, team_a, team_b),
            warning=warning,
        )
        for mp in top
    ]

    return RecommendationSet(
        team_a=team_a,
        team_b=team_b,
        recommendations=recommendations,
    )

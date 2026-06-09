"""Generate a human-readable plain-English match prediction explanation."""

from src.explainability.driver import PredictionExplanation


def generate_report(expl: PredictionExplanation) -> str:
    """Return a plain-English explanation of the prediction drivers.

    Args:
        expl: A PredictionExplanation produced by build_explanation().

    Returns:
        Multi-sentence string suitable for display to end users.
    """
    parts: list[str] = []

    # Opening: who is favoured
    threshold = 0.10
    if expl.win_a - expl.win_b > threshold:
        fav, underdog = expl.team_a, expl.team_b
        parts.append(
            f"{fav} are favoured over {underdog} "
            f"(win probability {expl.win_a:.0%} vs {expl.win_b:.0%})."
        )
    elif expl.win_b - expl.win_a > threshold:
        fav, underdog = expl.team_b, expl.team_a
        parts.append(
            f"{fav} are favoured over {underdog} "
            f"(win probability {expl.win_b:.0%} vs {expl.win_a:.0%})."
        )
    else:
        parts.append(
            f"This match is evenly balanced between {expl.team_a} and {expl.team_b} "
            f"({expl.win_a:.0%} / {expl.draw:.0%} / {expl.win_b:.0%})."
        )

    # Key drivers (exclude Dixon-Coles and Market divergence from the narrative sentence)
    narrative_drivers = [
        d for d in expl.drivers
        if d.name not in ("Dixon-Coles adjustment",)
    ]
    if narrative_drivers:
        driver_phrases = []
        for d in narrative_drivers:
            if d.name == "ELO advantage":
                driver_phrases.append(f"ELO rating advantage ({d.team})")
            elif d.name == "Attack strength":
                driver_phrases.append(f"higher attack strength ({d.team})")
            elif d.name == "Defensive weakness":
                driver_phrases.append(f"opponent defensive vulnerability ({d.team})")
            elif d.name == "Player impact":
                if d.direction == "negative":
                    driver_phrases.append(f"squad availability uncertainty ({d.team})")
                else:
                    driver_phrases.append(f"strong squad availability ({d.team})")
            elif d.name == "Market divergence":
                driver_phrases.append("divergence from bookmaker odds")

        if driver_phrases:
            if len(driver_phrases) == 1:
                key_factors = driver_phrases[0]
            elif len(driver_phrases) == 2:
                key_factors = f"{driver_phrases[0]} and {driver_phrases[1]}"
            else:
                key_factors = (
                    ", ".join(driver_phrases[:-1]) + f", and {driver_phrases[-1]}"
                )
            parts.append(f"Key factors: {key_factors}.")

    # DC adjustment context
    parts.append(
        "Dixon-Coles low-score correction shifts probability mass from "
        "exact 0-0, 1-0, 0-1 and 1-1 scorelines relative to pure Poisson."
    )

    # Most likely scoreline
    if expl.top_scorelines:
        g_a, g_b, prob = expl.top_scorelines[0]
        parts.append(
            f"The most likely scoreline is {g_a}-{g_b} ({prob:.0%} probability), "
            "though exact-score confidence remains low."
        )

    # Market divergence note (research-valid only)
    market_driver = next(
        (d for d in expl.drivers if d.name == "Market divergence"), None
    )
    if market_driver is not None:
        parts.append(
            f"Market comparison (research-valid odds): {market_driver.description}"
        )

    # Validity warnings
    for w in expl.warnings:
        parts.append(f"Warning: {w}")

    return " ".join(parts)

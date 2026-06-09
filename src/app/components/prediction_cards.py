"""Prediction cards — pure-data logic layer + Streamlit rendering.

Logic functions are tested. Streamlit rendering functions are not.
"""

from dataclasses import dataclass


# ── Confidence thresholds ─────────────────────────────────────────────────────
# A match is "High" confidence when the leading outcome probability >= 0.60
# and the gap to second outcome >= 0.25.
# Warnings downgrade High → Medium.

_HIGH_PROB_THRESHOLD = 0.60
_HIGH_GAP_THRESHOLD  = 0.25
_LOW_PROB_THRESHOLD  = 0.42   # below this the prediction is effectively a coin flip


@dataclass
class ConfidenceResult:
    level: float       # 0.0–1.0 continuous score
    label: str         # "High" | "Medium" | "Low"
    top_outcome: str   # "team_a_win" | "draw" | "team_b_win"
    top_prob: float    # probability of the leading outcome
    gap: float         # top_prob - second highest prob


def compute_confidence(
    win_a: float,
    draw: float,
    win_b: float,
    warnings: list[str],
) -> ConfidenceResult:
    """Compute a confidence assessment for a match prediction.

    Confidence is based on:
      1. The leading outcome's probability
      2. The gap between top and second outcome
      3. Whether data validity warnings exist (degrades High → Medium)

    Args:
        win_a, draw, win_b: Match outcome probabilities (must sum to ~1).
        warnings: List of data validity warning strings.

    Returns:
        ConfidenceResult with level, label, top_outcome, top_prob, gap.
    """
    probs = {
        "team_a_win": win_a,
        "draw":        draw,
        "team_b_win":  win_b,
    }
    sorted_outcomes = sorted(probs.items(), key=lambda x: -x[1])
    top_outcome, top_prob = sorted_outcomes[0]
    _, second_prob = sorted_outcomes[1]
    gap = top_prob - second_prob

    # Continuous level: blend of top_prob and normalised gap
    level = 0.6 * top_prob + 0.4 * min(1.0, gap / 0.40)

    # Categorical label
    if top_prob >= _HIGH_PROB_THRESHOLD and gap >= _HIGH_GAP_THRESHOLD:
        label = "High"
    elif top_prob <= _LOW_PROB_THRESHOLD:
        label = "Low"
    else:
        label = "Medium"

    # Validity warnings degrade High → Medium
    if warnings and label == "High":
        label = "Medium"
        level = min(level, 0.65)

    return ConfidenceResult(
        level=float(max(0.0, min(1.0, level))),
        label=label,
        top_outcome=top_outcome,
        top_prob=top_prob,
        gap=gap,
    )


def format_outcome_rows(
    team_a: str,
    team_b: str,
    win_a: float,
    draw: float,
    win_b: float,
) -> list[dict]:
    """Build table rows for win/draw/win outcome display.

    Returns list of dicts with keys: Outcome, Probability, Most Likely.
    """
    outcomes = [
        (f"{team_a} Win", win_a),
        ("Draw",           draw),
        (f"{team_b} Win", win_b),
    ]
    max_prob = max(win_a, draw, win_b)
    return [
        {
            "Outcome":     label,
            "Probability": f"{prob:.1%}",
            "Most Likely": "Yes" if prob == max_prob else "",
        }
        for label, prob in outcomes
    ]


def format_xg_summary(
    team_a: str,
    team_b: str,
    xg_a: float,
    xg_b: float,
) -> dict:
    """Return a dict summarising expected goals for both teams."""
    return {
        f"{team_a} xG": f"{xg_a:.2f}",
        f"{team_b} xG": f"{xg_b:.2f}",
    }


# ── Streamlit rendering ───────────────────────────────────────────────────────

def render_prediction_card(
    team_a: str,
    team_b: str,
    win_a: float,
    draw: float,
    win_b: float,
    xg_a: float,
    xg_b: float,
    confidence: ConfidenceResult,
    model_label: str,
) -> None:
    """Render the main prediction card in the Streamlit app."""
    import streamlit as st
    import pandas as pd

    st.subheader(f"Prediction: {team_a} vs {team_b}")
    st.caption(f"Model: {model_label}")

    col_metrics, col_conf = st.columns([3, 1])

    with col_metrics:
        st.markdown("**Match Outcome Probabilities**")
        rows = format_outcome_rows(team_a, team_b, win_a, draw, win_b)
        df = pd.DataFrame(rows)
        st.table(df[["Outcome", "Probability"]])

        st.markdown("**Expected Goals**")
        xg_cols = st.columns(2)
        with xg_cols[0]:
            st.metric(f"{team_a} xG", f"{xg_a:.2f}")
        with xg_cols[1]:
            st.metric(f"{team_b} xG", f"{xg_b:.2f}")

    with col_conf:
        _color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence.label, "⚪")
        st.markdown(f"**Prediction Confidence**")
        st.markdown(f"## {_color} {confidence.label}")
        st.caption(f"Top outcome: {confidence.top_prob:.0%}")
        st.caption(f"Gap to 2nd: {confidence.gap:.0%}")

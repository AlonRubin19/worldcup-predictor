"""Scoreline table component — pure-data logic + Streamlit rendering."""


def format_scoreline_rows(
    top_scorelines: list[tuple[int, int, float]],
    team_a: str,
    team_b: str,
) -> list[dict]:
    """Format top scorelines as display rows.

    Args:
        top_scorelines: List of (goals_a, goals_b, probability) tuples.
        team_a, team_b: Team names for labelling.

    Returns:
        List of dicts with keys: Rank, Score, Probability, Note.
    """
    rows = []
    for i, (g_a, g_b, prob) in enumerate(top_scorelines, start=1):
        note = "Most likely score" if i == 1 else ""
        rows.append({
            "Rank":        str(i),
            "Score":       f"{team_a} {g_a} – {g_b} {team_b}",
            "Probability": f"{prob:.1%}",
            "Note":        note,
        })
    return rows


def most_likely_score_label(
    top_scorelines: list[tuple[int, int, float]],
    team_a: str,
    team_b: str,
) -> str:
    """Return a human-readable most-likely-score string."""
    if not top_scorelines:
        return "No scoreline data"
    g_a, g_b, prob = top_scorelines[0]
    return f"{team_a} {g_a} – {g_b} {team_b} ({prob:.1%})"


# ── Streamlit rendering ───────────────────────────────────────────────────────

def render_scoreline_table(
    top_scorelines: list[tuple[int, int, float]],
    team_a: str,
    team_b: str,
) -> None:
    """Render the scoreline table in the Streamlit app."""
    import streamlit as st
    import pandas as pd

    if not top_scorelines:
        st.info("No scoreline predictions available.")
        return

    g_a, g_b, top_prob = top_scorelines[0]
    st.metric(
        "Most likely score",
        f"{g_a} – {g_b}",
        help=f"Probability: {top_prob:.1%}. Exact scores are low-confidence — "
             "treat as a stylistic guide, not a prediction.",
    )

    st.markdown("**Top 5 possible scores**")
    rows = format_scoreline_rows(top_scorelines[:5], team_a, team_b)
    df = pd.DataFrame(rows)
    st.table(df[["Score", "Probability"]])
    st.caption(
        "Exact score probability is inherently low. "
        "Even the most likely scoreline typically has < 15% probability."
    )

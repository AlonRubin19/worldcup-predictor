import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams
from src.models.poisson import predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")
st.markdown(
    "Select two teams and enter their expected goals to see predicted match outcomes."
)

# ── Load teams ─────────────────────────────────────────────────────────────────
try:
    teams = load_teams()
except (FileNotFoundError, ValueError) as e:
    st.error(f"Could not load teams data: {e}")
    st.stop()

# ── Team selection ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", options=teams, index=0)

with col2:
    # Filter out Team A so the same team cannot be selected twice.
    teams_b = [t for t in teams if t != team_a]
    team_b = st.selectbox("Team B", options=teams_b, index=0)

# ── Expected goals inputs ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Expected Goals (xG)")

col3, col4 = st.columns(2)

with col3:
    xg_a = st.number_input(
        f"{team_a} xG",
        min_value=0.1,
        max_value=5.0,
        value=1.3,
        step=0.1,
        format="%.1f",
    )

with col4:
    xg_b = st.number_input(
        f"{team_b} xG",
        min_value=0.1,
        max_value=5.0,
        value=1.3,
        step=0.1,
        format="%.1f",
    )

# ── Run prediction ─────────────────────────────────────────────────────────────
try:
    result = predict(team_a, team_b, xg_a, xg_b)
except ValueError as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ── Results display ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Prediction: {team_a} vs {team_b}")

# Match outcome probabilities table.
outcome_data = {
    "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
    "Probability": [
        f"{result.win_a:.1%}",
        f"{result.draw:.1%}",
        f"{result.win_b:.1%}",
    ],
}
st.markdown("**Match Outcome Probabilities**")
st.table(pd.DataFrame(outcome_data))

# Top 5 most likely scorelines.
scoreline_data = {
    "Scoreline": [f"{team_a} {g_a} – {g_b} {team_b}" for g_a, g_b, _ in result.top_scorelines],
    "Probability": [f"{p:.1%}" for _, _, p in result.top_scorelines],
}
st.markdown("**Top 5 Most Likely Scorelines**")
st.table(pd.DataFrame(scoreline_data))

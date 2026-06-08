import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")
st.markdown(
    "Select two teams to see auto-calculated expected goals and match outcome predictions."
)

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    teams = load_teams()
except (FileNotFoundError, ValueError) as e:
    st.error(f"Could not load teams data: {e}")
    st.stop()

try:
    all_ratings = load_team_ratings()
except (FileNotFoundError, ValueError) as e:
    st.error(f"Could not load team ratings: {e}")
    st.stop()

# ── Team selection ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", options=teams, index=0)

with col2:
    # Filter out Team A so the same team cannot be selected twice.
    teams_b = [t for t in teams if t != team_a]
    team_b = st.selectbox("Team B", options=teams_b, index=0)

# ── Auto-calculate xG from ratings ────────────────────────────────────────────
ratings_a = all_ratings.get(team_a)
ratings_b = all_ratings.get(team_b)

# Warn and fall back to BASE_XG for teams missing from the ratings file.
if ratings_a is None:
    st.warning(f"No ratings found for {team_a} — using baseline xG ({BASE_XG}).")
if ratings_b is None:
    st.warning(f"No ratings found for {team_b} — using baseline xG ({BASE_XG}).")

if ratings_a is not None and ratings_b is not None:
    auto_xg_a, auto_xg_b = calculate_xg(ratings_a, ratings_b)
else:
    auto_xg_a = BASE_XG
    auto_xg_b = BASE_XG

# ── xG section ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Expected Goals (xG)")

st.info(
    f"**Auto-calculated xG** — {team_a}: **{auto_xg_a:.2f}** | {team_b}: **{auto_xg_b:.2f}**"
)

override = st.checkbox("Override xG manually", value=False)

col3, col4 = st.columns(2)

with col3:
    xg_a = st.number_input(
        f"{team_a} xG",
        min_value=0.1,
        max_value=5.0,
        value=float(round(auto_xg_a, 1)),
        step=0.1,
        format="%.1f",
        disabled=not override,
        key="xg_a_input",
    )

with col4:
    xg_b = st.number_input(
        f"{team_b} xG",
        min_value=0.1,
        max_value=5.0,
        value=float(round(auto_xg_b, 1)),
        step=0.1,
        format="%.1f",
        disabled=not override,
        key="xg_b_input",
    )

# Use auto values unless override is active.
final_xg_a = xg_a if override else auto_xg_a
final_xg_b = xg_b if override else auto_xg_b

st.caption(
    f"**Final xG used** — {team_a}: {final_xg_a:.2f} | {team_b}: {final_xg_b:.2f}"
    + (" *(manual override)*" if override else " *(auto-calculated)*")
)

# ── Run prediction ─────────────────────────────────────────────────────────────
try:
    result = predict(team_a, team_b, final_xg_a, final_xg_b)
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

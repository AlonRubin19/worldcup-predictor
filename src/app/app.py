import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup Match Predictor")

tab_predictor, tab_backtest = st.tabs(["⚽ Match Predictor", "📊 Backtesting"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predictor:
    st.markdown("Select two teams to see auto-calculated expected goals and match outcome predictions.")

    # ── Load data ──────────────────────────────────────────────────────────────
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

    # ── Team selection ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Team A", options=teams, index=0)

    with col2:
        teams_b = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", options=teams_b, index=0)

    # ── Auto-calculate xG ─────────────────────────────────────────────────────
    ratings_a = all_ratings.get(team_a)
    ratings_b = all_ratings.get(team_b)

    if ratings_a is None:
        st.warning(f"No ratings found for {team_a} — using baseline xG ({BASE_XG}).")
    if ratings_b is None:
        st.warning(f"No ratings found for {team_b} — using baseline xG ({BASE_XG}).")

    _AVG = {"elo": 1800, "attack_rating": 1.0, "defense_rating": 1.0,
            "form_rating": 1.0, "squad_rating": 1.0}
    auto_xg_a, auto_xg_b = calculate_xg(
        ratings_a if ratings_a is not None else _AVG,
        ratings_b if ratings_b is not None else _AVG,
    )

    # ── xG inputs ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Expected Goals (xG)")

    override = st.checkbox("Override xG manually", value=False)

    _label = "Auto-calculated xG (overridden below)" if override else "Auto-calculated xG"
    st.info(f"**{_label}** — {team_a}: **{auto_xg_a:.2f}** | {team_b}: **{auto_xg_b:.2f}**")

    col3, col4 = st.columns(2)

    with col3:
        xg_a = st.number_input(
            f"{team_a} xG", min_value=0.1, max_value=5.0,
            value=float(round(auto_xg_a, 1)), step=0.1, format="%.1f",
            disabled=not override, key="xg_a_input",
        )

    with col4:
        xg_b = st.number_input(
            f"{team_b} xG", min_value=0.1, max_value=5.0,
            value=float(round(auto_xg_b, 1)), step=0.1, format="%.1f",
            disabled=not override, key="xg_b_input",
        )

    final_xg_a = xg_a if override else auto_xg_a
    final_xg_b = xg_b if override else auto_xg_b

    st.caption(
        f"**Final xG used** — {team_a}: {final_xg_a:.2f} | {team_b}: {final_xg_b:.2f}"
        + (" *(manual override)*" if override else " *(auto-calculated)*")
    )

    # ── Prediction ────────────────────────────────────────────────────────────
    try:
        result = predict(team_a, team_b, final_xg_a, final_xg_b)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader(f"Prediction: {team_a} vs {team_b}")

    outcome_data = {
        "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
        "Probability": [
            f"{result.win_a:.1%}", f"{result.draw:.1%}", f"{result.win_b:.1%}",
        ],
    }
    st.markdown("**Match Outcome Probabilities**")
    st.table(pd.DataFrame(outcome_data))

    scoreline_data = {
        "Scoreline": [f"{team_a} {g_a} – {g_b} {team_b}" for g_a, g_b, _ in result.top_scorelines],
        "Probability": [f"{p:.1%}" for _, _, p in result.top_scorelines],
    }
    st.markdown("**Top 5 Most Likely Scorelines**")
    st.table(pd.DataFrame(scoreline_data))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown("Model accuracy validated against historical international match results.")

    try:
        bt_ratings = load_team_ratings()
        bt_results = run_backtest(ratings=bt_ratings)
        bt_metrics = compute_metrics(bt_results)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Backtesting failed: {e}")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Model Performance")

    metrics_data = {
        "Metric": [
            "Total Matches Tested",
            "1X2 Accuracy",
            "Exact Score Accuracy",
            "Top 3 Scoreline Hit Rate",
            "Top 5 Scoreline Hit Rate",
            "Brier Score (lower = better)",
            "Avg Probability of Actual Result",
        ],
        "Value": [
            str(bt_metrics.total_matches),
            f"{bt_metrics.accuracy_1x2:.1%}",
            f"{bt_metrics.exact_score_accuracy:.1%}",
            f"{bt_metrics.top_3_hit_rate:.1%}",
            f"{bt_metrics.top_5_hit_rate:.1%}",
            f"{bt_metrics.brier_score:.4f}",
            f"{bt_metrics.avg_prob_actual_result:.1%}",
        ],
    }
    st.table(pd.DataFrame(metrics_data))

    # ── Per-match results ─────────────────────────────────────────────────────
    st.subheader("Match-Level Results")

    outcome_labels = {
        "team_a_win": "Home Win",
        "draw": "Draw",
        "team_b_win": "Away Win",
    }

    rows = []
    for r in bt_results:
        rows.append({
            "Date": r.date,
            "Match": f"{r.team_a} vs {r.team_b}",
            "Actual Score": f"{r.actual_goals_a}-{r.actual_goals_b}",
            "Predicted": outcome_labels[r.predicted_outcome],
            "Actual": outcome_labels[r.actual_outcome],
            "Correct": "✓" if r.predicted_outcome == r.actual_outcome else "✗",
            "In Top 5": "✓" if r.in_top_5 else "✗",
            "P(actual)": f"{r.prob_of_actual_result:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

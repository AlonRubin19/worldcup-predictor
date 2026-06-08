import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import tune_rho, select_best_rho, DEFAULT_RHO_GRID, RhoResult
from src.backtesting.valid_runner import run_valid_backtest

# Fallback ratings for teams missing from team_ratings.csv.
_AVG_RATINGS = {"elo": 1800, "attack_rating": 1.0, "defense_rating": 1.0,
                "form_rating": 1.0, "squad_rating": 1.0}

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

    # ── Model selector ─────────────────────────────────────────────────────────
    model_choice = st.radio(
        "Prediction Model",
        ["Poisson", "Dixon-Coles"],
        horizontal=True,
        help="Dixon-Coles corrects low-score probabilities (draws, 1-0, 0-1) vs pure Poisson.",
    )

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

    auto_xg_a, auto_xg_b = calculate_xg(
        ratings_a if ratings_a is not None else _AVG_RATINGS,
        ratings_b if ratings_b is not None else _AVG_RATINGS,
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
        if model_choice == "Dixon-Coles":
            result = predict_dixon_coles(team_a, team_b, final_xg_a, final_xg_b)
        else:
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
    st.markdown("This tab shows two separate backtests with clearly different data provenance.")

    bt_results_po = None
    bt_results_dc = None
    bt_metrics_po = None
    bt_metrics_dc = None
    rho_tuning_results = None
    best_rho_result = None
    try:
        bt_results_po = run_backtest(ratings=all_ratings, model_type="poisson")
        bt_metrics_po = compute_metrics(bt_results_po)
        bt_results_dc = run_backtest(ratings=all_ratings, model_type="dixon_coles")
        bt_metrics_dc = compute_metrics(bt_results_dc)
        rho_tuning_results = tune_rho(all_ratings)
        best_rho_result = select_best_rho(rho_tuning_results)
    except Exception as e:
        st.error(f"Backtesting failed: {e}")

    if bt_metrics_po is not None and bt_metrics_dc is not None:
        # ── Illustrative label ────────────────────────────────────────────────
        st.warning(
            "⚠️ **Illustrative Backtest** — uses `team_ratings.csv` (manually estimated by AI). "
            "Ratings were assigned with knowledge of WC 2022 outcomes. "
            "Results are for **engineering validation only**, not accuracy measurement."
        )

        # ── Model comparison ──────────────────────────────────────────────────
        st.subheader("Model Comparison")

        comparison_data = {
            "Metric": [
                "Total Matches Tested",
                "1X2 Accuracy",
                "Exact Score Accuracy",
                "Top 3 Scoreline Hit Rate",
                "Top 5 Scoreline Hit Rate",
                "Brier Score (lower = better)",
                "Avg Probability of Actual Result",
            ],
            "Poisson": [
                str(bt_metrics_po.total_matches),
                f"{bt_metrics_po.accuracy_1x2:.1%}",
                f"{bt_metrics_po.exact_score_accuracy:.1%}",
                f"{bt_metrics_po.top_3_hit_rate:.1%}",
                f"{bt_metrics_po.top_5_hit_rate:.1%}",
                f"{bt_metrics_po.brier_score:.4f}",
                f"{bt_metrics_po.avg_prob_actual_result:.1%}",
            ],
            "Dixon-Coles": [
                str(bt_metrics_dc.total_matches),
                f"{bt_metrics_dc.accuracy_1x2:.1%}",
                f"{bt_metrics_dc.exact_score_accuracy:.1%}",
                f"{bt_metrics_dc.top_3_hit_rate:.1%}",
                f"{bt_metrics_dc.top_5_hit_rate:.1%}",
                f"{bt_metrics_dc.brier_score:.4f}",
                f"{bt_metrics_dc.avg_prob_actual_result:.1%}",
            ],
        }
        st.table(pd.DataFrame(comparison_data))

        # ── Per-match results (Poisson as reference) ──────────────────────────
        st.subheader("Match-Level Results (Poisson)")

        outcome_labels = {
            "team_a_win": "Team A Win",
            "draw": "Draw",
            "team_b_win": "Team B Win",
        }

        rows = []
        for r in bt_results_po:
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

        # ── Rho Tuning ────────────────────────────────────────────────────────
        if rho_tuning_results is not None and best_rho_result is not None:
            st.markdown("---")
            st.subheader("Rho Tuning — Dixon-Coles Parameter Search")

            rho_rows = []
            for r in rho_tuning_results:
                rho_rows.append({
                    "rho": f"{r.rho:.2f}",
                    "1X2 Acc": f"{r.accuracy_1x2:.1%}",
                    "Exact": f"{r.exact_score_accuracy:.1%}",
                    "Top 3": f"{r.top_3_hit_rate:.1%}",
                    "Top 5": f"{r.top_5_hit_rate:.1%}",
                    "Brier": f"{r.brier_score:.4f}",
                    "Avg P": f"{r.avg_prob_actual_result:.1%}",
                })
            st.table(pd.DataFrame(rho_rows))

            st.caption(f"**Best rho:** {best_rho_result.rho:.2f} "
                       f"(Brier: {best_rho_result.brier_score:.4f}, "
                       f"Top 3: {best_rho_result.top_3_hit_rate:.1%})")

            # Recommend DC with best rho only if it beats Poisson by meaningful margin.
            poisson_brier = bt_metrics_po.brier_score
            if best_rho_result.brier_score < poisson_brier - 0.001:
                st.success(
                    f"Recommendation: **Dixon-Coles (rho={best_rho_result.rho:.2f})** — "
                    f"Brier {best_rho_result.brier_score:.4f} vs Poisson {poisson_brier:.4f} "
                    f"({poisson_brier - best_rho_result.brier_score:.4f} improvement)"
                )
            else:
                st.info(
                    f"Recommendation: **Poisson (default)** — "
                    f"Dixon-Coles best rho={best_rho_result.rho:.2f} does not improve "
                    f"Brier score by more than 0.001 "
                    f"(DC: {best_rho_result.brier_score:.4f}, Poisson: {poisson_brier:.4f})"
                )

        # ══ Valid Pre-Match Backtest ═══════════════════════════════════════════
        st.markdown("---")
        st.subheader("Valid Pre-Match Backtest")
        st.info(
            "📐 **Data provenance:** xG calculated from pre-match statistics only "
            "(goals averages, form, ELO). No manually estimated ratings used.\n\n"
            "⚠️ **PLACEHOLDER DATA:** `pre_match_team_stats.csv` contains sample values, "
            "not real historical records. See `docs/valid_backtest_status.md`."
        )

        valid_results_po = None
        valid_metrics_po = None
        valid_rho_results = None
        valid_best_rho = None
        try:
            valid_results_po = run_valid_backtest(model_type="poisson")
            valid_metrics_po = compute_metrics(valid_results_po)

            valid_rho_results = []
            for rho_val in DEFAULT_RHO_GRID:
                dc_results = run_valid_backtest(model_type="dixon_coles", rho=rho_val)
                m_v = compute_metrics(dc_results)
                valid_rho_results.append(RhoResult(
                    rho=rho_val,
                    accuracy_1x2=m_v.accuracy_1x2,
                    exact_score_accuracy=m_v.exact_score_accuracy,
                    top_3_hit_rate=m_v.top_3_hit_rate,
                    top_5_hit_rate=m_v.top_5_hit_rate,
                    brier_score=m_v.brier_score,
                    avg_prob_actual_result=m_v.avg_prob_actual_result,
                ))
            valid_best_rho = select_best_rho(valid_rho_results)
        except Exception as e:
            st.error(f"Valid backtest failed: {e}")

        if valid_metrics_po is not None:
            valid_metrics_data = {
                "Metric": [
                    "Total Matches Tested", "1X2 Accuracy", "Exact Score Accuracy",
                    "Top 3 Hit Rate", "Top 5 Hit Rate",
                    "Brier Score (lower = better)", "Avg P(Actual Result)",
                ],
                "Poisson (pre-match xG)": [
                    str(valid_metrics_po.total_matches),
                    f"{valid_metrics_po.accuracy_1x2:.1%}",
                    f"{valid_metrics_po.exact_score_accuracy:.1%}",
                    f"{valid_metrics_po.top_3_hit_rate:.1%}",
                    f"{valid_metrics_po.top_5_hit_rate:.1%}",
                    f"{valid_metrics_po.brier_score:.4f}",
                    f"{valid_metrics_po.avg_prob_actual_result:.1%}",
                ],
            }
            st.table(pd.DataFrame(valid_metrics_data))

        if valid_rho_results and valid_best_rho:
            st.markdown("**Dixon-Coles rho grid (valid path):**")
            valid_rho_rows = [{
                "rho": f"{r.rho:.2f}", "1X2": f"{r.accuracy_1x2:.1%}",
                "Exact": f"{r.exact_score_accuracy:.1%}", "Top3": f"{r.top_3_hit_rate:.1%}",
                "Brier": f"{r.brier_score:.4f}",
            } for r in valid_rho_results]
            st.table(pd.DataFrame(valid_rho_rows))
            st.caption(f"Best rho (valid path): {valid_best_rho.rho:.2f} "
                       f"(Brier: {valid_best_rho.brier_score:.4f})")

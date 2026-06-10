import sys
from pathlib import Path

# Add project root to path so `src.*` imports work when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from src.data.loader import load_teams, load_team_ratings
from src.models.xg_calculator import calculate_xg, BASE_XG
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles, build_dc_matrix
from src.models.betting_markets import compute_betting_markets, BettingMarketProbabilities
from src.models.recommendations import generate_recommendations
from src.backtesting.runner import run_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.rho_tuning import tune_rho, select_best_rho, DEFAULT_RHO_GRID, RhoResult
from src.backtesting.valid_runner import run_valid_backtest
from src.backtesting.market_runner import run_market_comparison
from src.explainability.driver import build_explanation, ExplanationInput
from src.explainability.report import generate_report
from src.data.team_snapshot_loader import load_team_snapshots, TeamSnapshot
from src.data.strength_loader import load_strength_params, StrengthParams
from src.models.research_valid_predictor import (
    predict_research_valid, ResearchValidInput, DEFAULT_RHO,
)
from src.app.components.prediction_cards import (
    compute_confidence, render_prediction_card,
)
from src.app.components.explanation_panel import render_explanation_panel
from src.app.components.scoreline_table import render_scoreline_table
from src.app.components.daily_match_board import (
    MatchPrediction, filter_fixtures_by_date, sort_matches_by_datetime,
    build_daily_match_rows, format_board_row_as_dict,
)
from src.tournament.simulator import run_monte_carlo, MonteCarloResult
from src.data.player_loader import load_player_profiles
from src.models.golden_boot import predict_golden_boot, GoldenBootPlayerResult
from src.tournament.fixtures import load_fixtures
from src.data.fixture_provider import FixtureSource, get_fixtures as _get_fixtures
from src.app.selected_fixture import (
    SelectedFixture, create_selected_fixture,
    get_api_fixture_id, is_valid_selected_fixture,
)
from src.app.prediction_runner import RunnerInput, run_full_prediction, FullPrediction
from src.app.tournament_filters import (
    filter_fixtures, get_next_fixtures, get_today_fixtures,
    get_unique_stages, get_unique_groups, get_unique_teams,
    get_status_label, STATUS_LABELS,
)
from src.tournament.calibration import CalibrationParams, compute_concentration_metrics
from src.models.lineup_override import (
    apply_lineup_override, create_default_lineup, LineupOverride,
)
from src.app.components.lineup_editor import (
    format_player_table, parse_player_edits, STATUS_TO_FACTOR,
)
from src.data.lineup_loader import load_expected_lineups
from src.models.lineup_status import validate_lineup_for_match, convert_lineup_entries_to_override
from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.models.live_prediction_adapter import (
    LiveDataStatus, get_live_data_status, load_live_lineups_for_match,
)

_LINEUPS_CSV = Path(__file__).parent.parent.parent / "data" / "expected_lineups.csv"

# Build API client once at startup — key from env var, graceful if missing
import os as _os
_api_client = ApiFootballClient(
    api_key=_os.environ.get("API_FOOTBALL_KEY", ""),
    cache_dir=Path(__file__).parent.parent.parent / "data" / "api_cache",
)

# Fallback ratings for teams missing from team_ratings.csv.
_AVG_RATINGS = {"elo": 1800, "attack_rating": 1.0, "defense_rating": 1.0,
                "form_rating": 1.0, "squad_rating": 1.0}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ World Cup 2026 Prediction Platform")

# ── Global session state defaults ──────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = None
if "provider_result" not in st.session_state:
    st.session_state["provider_result"] = None
if "refresh_summary" not in st.session_state:
    st.session_state["refresh_summary"] = None

# ── Global refresh button (top of page) ───────────────────────────────────────
_gcol1, _gcol2, _gcol3 = st.columns([4, 1, 2])
with _gcol2:
    _global_refresh = st.button(
        "🔄 Refresh",
        key="global_refresh_btn",
        help="Re-fetch live fixtures and lineup data from API-Football, bypassing the 6-hour cache.",
    )
with _gcol3:
    if st.session_state["last_refresh"]:
        st.caption(f"Last refresh: {st.session_state['last_refresh']}")

if _global_refresh:
    import datetime
    from src.data.refresh_pipeline import refresh_team_data
    from src.data.team_api_ids import TEAM_API_IDS

    st.session_state["last_refresh"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.session_state["provider_result"] = None   # force re-fetch
    st.session_state["refresh_summary"] = refresh_team_data(
        _api_client, list(TEAM_API_IDS.items())
    )

if st.session_state.get("refresh_summary"):
    _rs = st.session_state["refresh_summary"]
    with st.expander(f"🔄 Refresh details — {_rs.timestamp}", expanded=_global_refresh):
        st.caption(
            f"API connected: {'✅ yes' if _api_client._api_key else '❌ no (API_FOOTBALL_KEY not set)'} | "
            f"Squads refreshed: {_rs.squads_refreshed}/{len(_rs.teams)} | "
            f"Injuries refreshed: {_rs.injuries_refreshed}/{len(_rs.teams)} | "
            f"Player stats refreshed: {_rs.stats_refreshed}/{len(_rs.teams)}"
        )
        st.dataframe(
            [
                {
                    "Team": t.team,
                    "Squad source": t.squad_source,
                    "Squad players": t.squad_count,
                    "Injury source": t.injury_source,
                    "Injuries": t.injury_count,
                    "Stats source": t.stats_source,
                    "Live data used": "✅" if t.used_live_data else "⚠️ fallback",
                }
                for t in _rs.teams
            ],
            hide_index=True,
            width="stretch",
        )

tab_home, tab_predictor, tab_tournament, tab_golden_boot, tab_status, tab_lab, tab_overview, tab_board = st.tabs([
    "🏠 Home",
    "⚽ Match Analyzer",
    "🏆 Tournament",
    "🏅 Golden Boot",
    "📡 Data Status",
    "🧪 Model Lab",
    "📋 All Fixtures",
    "📅 Daily Match Board",
])

# ── Shared helpers imported once ──────────────────────────────────────────────
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg as _sxg_fn
from src.models.xg_calibration import calibrate_xg as _cal_fn
from src.models.research_valid_predictor import (
    predict_research_valid as _rv_fn, ResearchValidInput as _RVI,
)
import datetime as _dt

_BOARD_FIXTURE_PATH = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"
_BOARD_SNAP_DEF = TeamSnapshot(elo=1800.0, ppg=1.5)
_BOARD_PAR_DEF  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)


def _load_provider(force: bool = False):
    """Load or return cached provider result from session_state."""
    if st.session_state.get("provider_result") is None or force:
        st.session_state["provider_result"] = _get_fixtures(
            mode=FixtureSource.AUTO,
            api_client=_api_client,
            csv_path=_BOARD_FIXTURE_PATH,
            force_refresh=force,
        )
    return st.session_state["provider_result"]


def _load_board_model_data():
    try:
        snaps  = load_team_snapshots()
        params = load_strength_params()
        return snaps, params, True
    except FileNotFoundError:
        return {}, {}, False


def _build_match_prediction(fixture, snaps, params, board_rv):
    sa = snaps.get(fixture.team_a, _BOARD_SNAP_DEF)
    sb = snaps.get(fixture.team_b, _BOARD_SNAP_DEF)
    pa = params.get(fixture.team_a, _BOARD_PAR_DEF)
    pb = params.get(fixture.team_b, _BOARD_PAR_DEF)
    raw_a, raw_b = _sxg_fn(sa.elo, sb.elo, pa, pb, sa.ppg, sb.ppg)
    xg_a, xg_b  = _cal_fn(raw_a), _cal_fn(raw_b)
    mat   = build_dc_matrix(xg_a, xg_b, rho=DEFAULT_RHO)
    bm    = compute_betting_markets(fixture.team_a, fixture.team_b, mat)
    rv    = _rv_fn(_RVI(team_a=fixture.team_a, team_b=fixture.team_b,
                        snapshot_a=sa, snapshot_b=sb, params_a=pa, params_b=pb))
    rs    = generate_recommendations(bm, "High", [], board_rv, top_n=1)
    over25 = next((m.probability for m in bm.over_under if "Over 2.5" in m.selection), 0.0)
    btts   = next((m.probability for m in bm.btts if m.selection == "BTTS Yes"), 0.0)
    score  = f"{rv.top_scorelines[0][0]}-{rv.top_scorelines[0][1]}" if rv.top_scorelines else "?-?"
    sig    = rs.recommendations[0].selection if rs.recommendations else "—"
    sigstr = rs.recommendations[0].signal_strength if rs.recommendations else "Weak"
    conf   = compute_confidence(rv.win_a, rv.draw, rv.win_b, [])
    return MatchPrediction(
        win_a=rv.win_a, draw=rv.draw, win_b=rv.win_b,
        most_likely_score=score, over_25=over25, btts_yes=btts,
        top_signal=sig, top_signal_strength=sigstr,
        confidence_label=conf.label, is_research_valid=board_rv,
    ), xg_a, xg_b


def _send_to_analyzer(fixture, prov):
    """Store a SelectedFixture in session_state and show confirmation."""
    actual_src = prov.source_used
    if actual_src is FixtureSource.AUTO:
        actual_src = FixtureSource.API if prov.api_connected else FixtureSource.CSV
    st.session_state["selected_fixture"] = create_selected_fixture(fixture, actual_src)
    st.success(
        f"✅ **{fixture.team_a} vs {fixture.team_b}** loaded into Match Analyzer — "
        "switch to the ⚽ Match Analyzer tab."
    )


def _source_badge(prov) -> str:
    if prov.source_used is FixtureSource.API:
        return "🌐 Live API (API-Football)"
    if prov.source_used is FixtureSource.AUTO and not prov.api_connected:
        return "📁 CSV fallback (API unavailable)"
    return "📁 CSV (static sample)"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME (prediction-first landing page)
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    _home_prov = _load_provider()
    _home_all = _home_prov.fixtures

    if not _home_all:
        st.error("No fixtures available. Click 🔄 Refresh or check API key.")
        st.stop()

    _home_snaps, _home_params, _home_rv = _load_board_model_data()

    # ── Today's Matches ────────────────────────────────────────────────────────
    st.subheader("📅 Today's Matches")
    _home_today = _dt.date.today().isoformat()
    _home_today_fix = sort_matches_by_datetime(filter_fixtures_by_date(_home_all, _home_today))

    if not _home_today_fix:
        st.info("No matches scheduled today. See **Upcoming Matches** below.")
    else:
        for _hf in _home_today_fix:
            _hpred, _, _ = _build_match_prediction(_hf, _home_snaps, _home_params, _home_rv)
            _conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(_hpred.confidence_label, "⚪")
            with st.container(border=True):
                _hc1, _hc2, _hc3 = st.columns([3, 2, 1])
                with _hc1:
                    st.markdown(f"**{_hf.team_a} vs {_hf.team_b}**")
                    st.caption(f"{_hf.date}  •  {_hf.stage.replace('_', ' ').title()}")
                    if _hpred.win_a >= _hpred.win_b and _hpred.win_a >= _hpred.draw:
                        _winner_str = f"Prediction: **{_hf.team_a} Win** ({_hpred.win_a:.0%})"
                    elif _hpred.win_b >= _hpred.draw:
                        _winner_str = f"Prediction: **{_hf.team_b} Win** ({_hpred.win_b:.0%})"
                    else:
                        _winner_str = f"Prediction: **Draw** ({_hpred.draw:.0%})"
                    st.markdown(_winner_str)
                with _hc2:
                    st.metric("Most Likely", _hpred.most_likely_score)
                    st.markdown(f"{_conf_emoji} Confidence: **{_hpred.confidence_label}**")
                with _hc3:
                    if _hpred.top_signal != "—":
                        st.markdown(f"`{_hpred.top_signal}`")
                    if st.button("Analyze →", key=f"home_analyze_{_hf.match_id}"):
                        _send_to_analyzer(_hf, _home_prov)

    # ── Upcoming Matches ───────────────────────────────────────────────────────
    _home_next = get_next_fixtures(_home_all, n=5, today=_home_today)
    if _home_next:
        with st.expander(f"⏱️ Upcoming Matches (next {len(_home_next)})", expanded=False):
            for _nf in _home_next:
                _nfc1, _nfc2, _nfc3 = st.columns([3, 1, 1])
                with _nfc1:
                    st.markdown(f"**{_nf.team_a}** vs **{_nf.team_b}**  "
                                f"— {_nf.date}  •  {_nf.stage.replace('_', ' ').title()}")
                with _nfc2:
                    st.caption(get_status_label(_nf.status))
                with _nfc3:
                    if st.button("Analyze →", key=f"home_upcoming_{_nf.match_id}"):
                        _send_to_analyzer(_nf, _home_prov)

    # ── Tournament Snapshot ────────────────────────────────────────────────────
    st.subheader("🏆 Tournament Snapshot")
    _snap_mc = st.session_state.get("mc_result")
    _snap_gb = st.session_state.get("golden_boot_results")

    _snap_c1, _snap_c2, _snap_c3 = st.columns(3)
    with _snap_c1:
        if _snap_mc and _snap_mc.win_tournament:
            _fav_team, _fav_p = max(_snap_mc.win_tournament.items(), key=lambda kv: kv[1])
            st.metric("Winner Favourite", _fav_team, f"{_fav_p:.0%}")
        else:
            st.metric("Winner Favourite", "—")
            st.caption("Run the Tournament simulator to populate.")
    with _snap_c2:
        if _snap_gb:
            _gb_fav = _snap_gb[0]
            st.metric("Golden Boot Favourite", _gb_fav.player_name, f"{_gb_fav.prob_top_scorer:.0%}")
        else:
            st.metric("Golden Boot Favourite", "—")
            st.caption("Run the Golden Boot projection to populate.")
    with _snap_c3:
        st.metric("Matches Loaded", str(_home_prov.fixture_count))
        st.caption(_source_badge(_home_prov))

    st.markdown("---")
    with st.expander("📋 Browse all fixtures", expanded=False):
        st.caption("Looking for the full fixture table and filters? See **All Fixtures** and "
                   "**Daily Match Board** tabs for the complete browsing experience.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — ALL FIXTURES (full table, filters, inline analysis)
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown(
        "All official World Cup matches. Filter by stage, group, team, or date. "
        "Select any match and click **Analyze** to open the full prediction in the Match Analyzer."
    )

    # ── Load fixtures ─────────────────────────────────────────────────────────
    _prov = _load_provider(force=_global_refresh)
    _all_fixtures = _prov.fixtures

    # Source badge + last refresh
    _ov_c1, _ov_c2 = st.columns([3, 2])
    with _ov_c1:
        _src_str = _source_badge(_prov)
        st.caption(f"Fixture source: **{_src_str}** — **{_prov.fixture_count}** matches loaded")
    with _ov_c2:
        if st.session_state.get("last_refresh"):
            st.caption(f"Cache refreshed: {st.session_state['last_refresh']}")

    if _prov.mapping_warnings:
        with st.expander(f"⚠️ {len(_prov.mapping_warnings)} team mapping warning(s)", expanded=False):
            for _mw in _prov.mapping_warnings:
                st.warning(_mw)

    if not _all_fixtures:
        st.error("No fixtures available. Click 🔄 Refresh or check API key.")
        st.stop()

    # ── Model data ────────────────────────────────────────────────────────────
    _ov_snaps, _ov_params, _ov_rv = _load_board_model_data()
    if not _ov_rv:
        st.warning("Research-valid model data unavailable — predictions use default parameters.")

    # ── Filters row ───────────────────────────────────────────────────────────
    _ov_stages = ["All stages"] + get_unique_stages(_all_fixtures)
    _ov_groups = ["All groups"] + get_unique_groups(_all_fixtures)
    _ov_teams  = ["All teams"]  + get_unique_teams(_all_fixtures)
    _ov_dates  = ["All dates"]  + sorted({f.date for f in _all_fixtures})

    _fc1, _fc2, _fc3, _fc4 = st.columns(4)
    with _fc1:
        _flt_stage = st.selectbox("Stage", _ov_stages, index=0, key="ov_stage")
    with _fc2:
        _flt_group = st.selectbox("Group", _ov_groups, index=0, key="ov_group")
    with _fc3:
        _flt_team  = st.selectbox("Team",  _ov_teams,  index=0, key="ov_team")
    with _fc4:
        _flt_date  = st.selectbox("Date",  _ov_dates,  index=0, key="ov_date")

    _ov_filtered = filter_fixtures(
        _all_fixtures,
        stage  = None if _flt_stage == "All stages" else _flt_stage,
        group  = None if _flt_group == "All groups" else _flt_group,
        team   = None if _flt_team  == "All teams"  else _flt_team,
        date   = None if _flt_date  == "All dates"  else _flt_date,
    )

    st.caption(f"Showing **{len(_ov_filtered)}** of {_prov.fixture_count} matches")

    # ── Next up ───────────────────────────────────────────────────────────────
    _today_str  = _dt.date.today().isoformat()
    _next_up = get_next_fixtures(_all_fixtures, n=3, today=_today_str)
    if _next_up:
        with st.expander("⏱️ Next up", expanded=True):
            for _nu in _next_up:
                _nu_c1, _nu_c2, _nu_c3 = st.columns([3, 1, 1])
                with _nu_c1:
                    st.markdown(f"**{_nu.team_a}** vs **{_nu.team_b}**  "
                                f"— {_nu.date}  •  {_nu.stage.replace('_',' ').title()}")
                with _nu_c2:
                    st.caption(get_status_label(_nu.status))
                with _nu_c3:
                    if st.button("🔬 Analyze", key=f"nu_analyze_{_nu.match_id}"):
                        _send_to_analyzer(_nu, _prov)

    # ── Full fixture table ────────────────────────────────────────────────────
    if not _ov_filtered:
        st.info("No matches match the selected filters.")
    else:
        _ov_rows = []
        for _ovf in _ov_filtered:
            _ov_rows.append({
                "Date":    _ovf.date,
                "Stage":   _ovf.stage.replace("_", " ").title(),
                "Group":   _ovf.group or "—",
                "Home":    _ovf.team_a,
                "Away":    _ovf.team_b,
                "Status":  get_status_label(_ovf.status),
                "_id":     _ovf.match_id,
            })
        _ov_df = pd.DataFrame(_ov_rows).drop(columns=["_id"])
        st.dataframe(_ov_df, use_container_width=True, hide_index=True)

        # ── Select & Analyze Inline ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚽ Match Prediction")
        _ov_match_labels = [
            f"{f.team_a} vs {f.team_b}  ({f.date})  [{get_status_label(f.status)}]"
            for f in _ov_filtered
        ]
        _ov_sel_label = st.selectbox(
            "Select match to analyze", _ov_match_labels, key="ov_match_sel",
            help="Select any World Cup match to see the full model prediction inline."
        )
        _ov_sel_idx = _ov_match_labels.index(_ov_sel_label)
        _ov_sel_fix = _ov_filtered[_ov_sel_idx]

        # Run full prediction via prediction_runner (single source of truth)
        _ov_inp = RunnerInput(
            team_a=_ov_sel_fix.team_a,
            team_b=_ov_sel_fix.team_b,
            snapshot_a=_ov_snaps.get(_ov_sel_fix.team_a, _BOARD_SNAP_DEF),
            snapshot_b=_ov_snaps.get(_ov_sel_fix.team_b, _BOARD_SNAP_DEF),
            params_a=_ov_params.get(_ov_sel_fix.team_a, _BOARD_PAR_DEF),
            params_b=_ov_params.get(_ov_sel_fix.team_b, _BOARD_PAR_DEF),
            is_research_valid=_ov_rv,
        )
        _ov_full = run_full_prediction(_ov_inp)

        # ── Fixture context banner ────────────────────────────────────────────
        _ov_stage_str = _ov_sel_fix.stage.replace("_", " ").title()
        _ov_group_str = f" | Group {_ov_sel_fix.group}" if _ov_sel_fix.group else ""
        _ov_src_str   = "🌐 Live API" if _prov.api_connected else "📁 CSV"
        st.info(
            f"**{_ov_sel_fix.team_a} vs {_ov_sel_fix.team_b}**  "
            f"| {_ov_sel_fix.date}  |  {_ov_stage_str}{_ov_group_str}  "
            f"|  Status: {get_status_label(_ov_sel_fix.status)}  "
            f"|  Source: {_ov_src_str}  "
            + (f"|  Fixture ID: `{_ov_sel_fix.match_id}`" if _prov.api_connected else "")
        )

        # Missing model data warnings
        if not _ov_rv:
            st.warning("Research-valid model data unavailable — using default strength parameters.")
        if _ov_sel_fix.team_a not in _ov_snaps:
            st.warning(f"No ELO snapshot for {_ov_sel_fix.team_a} — using defaults.")
        if _ov_sel_fix.team_b not in _ov_snaps:
            st.warning(f"No ELO snapshot for {_ov_sel_fix.team_b} — using defaults.")

        # ── 1X2 ──────────────────────────────────────────────────────────────
        _ov_c1, _ov_c2, _ov_c3 = st.columns(3)
        with _ov_c1:
            st.metric(f"{_ov_sel_fix.team_a} Win", f"{_ov_full.win_a:.1%}")
        with _ov_c2:
            st.metric("Draw", f"{_ov_full.draw:.1%}")
        with _ov_c3:
            st.metric(f"{_ov_sel_fix.team_b} Win", f"{_ov_full.win_b:.1%}")

        st.caption(
            f"xG — {_ov_sel_fix.team_a}: **{_ov_full.xg_a:.2f}** | "
            f"{_ov_sel_fix.team_b}: **{_ov_full.xg_b:.2f}**  •  "
            f"Model: {_ov_full.model_label}  •  "
            f"Confidence: **{_ov_full.confidence.label}**"
        )

        # ── Scorelines ────────────────────────────────────────────────────────
        st.markdown("---")
        _ov_sl1, _ov_sl2 = st.columns([1, 2])
        with _ov_sl1:
            st.metric("Most Likely Score", _ov_full.most_likely_score)
        with _ov_sl2:
            _ov_sl_rows = [
                {"Score": f"{s[0]}-{s[1]}", "Probability": f"{s[2]:.1%}"}
                for s in _ov_full.top_scorelines[:5]
            ]
            st.markdown("**Top 5 Scorelines**")
            st.dataframe(pd.DataFrame(_ov_sl_rows), use_container_width=True, hide_index=True)

        # ── Betting markets (compact 2-col) ───────────────────────────────────
        st.markdown("---")
        st.markdown("**Betting Markets**")
        st.caption("Model outputs only — not betting advice.")
        _ov_bm = _ov_full.markets
        _ov_bmc1, _ov_bmc2 = st.columns(2)
        with _ov_bmc1:
            st.markdown("**Over / Under**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.over_under
                ]), use_container_width=True, hide_index=True
            )
            st.markdown("**BTTS**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.btts
                ]), use_container_width=True, hide_index=True
            )
            st.markdown("**Double Chance**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.double_chance
                ]), use_container_width=True, hide_index=True
            )
        with _ov_bmc2:
            st.markdown("**Draw No Bet**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.draw_no_bet
                ]), use_container_width=True, hide_index=True
            )
            st.markdown("**Team Totals**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.team_totals
                ]), use_container_width=True, hide_index=True
            )
            st.markdown("**Clean Sheet**")
            st.dataframe(
                pd.DataFrame([
                    {"Market": m.selection, "Prob": f"{m.probability:.1%}",
                     "Fair Odds": f"{m.implied_fair_odds:.2f}"}
                    for m in _ov_bm.clean_sheet
                ]), use_container_width=True, hide_index=True
            )

        # ── Top signals ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Top Model Signals**")
        st.caption("Signal ranking — not betting advice.")
        if _ov_full.recommendations.recommendations:
            _ov_sig_rows = []
            for _r in _ov_full.recommendations.recommendations:
                _badge = {"Strong": "🟢 Strong", "Moderate": "🟡 Moderate",
                          "Weak": "🔴 Weak"}.get(_r.signal_strength, _r.signal_strength)
                _ov_sig_rows.append({
                    "Selection": _r.selection, "Probability": f"{_r.model_probability:.1%}",
                    "Fair Odds": f"{_r.fair_odds:.2f}", "Signal": _badge,
                    "Rationale": _r.rationale,
                })
            st.dataframe(pd.DataFrame(_ov_sig_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No high-signal markets identified.")

        # ── Key reasons ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Key Reasons**")
        if _ov_full.explanation.drivers:
            for _drv in _ov_full.explanation.drivers[:5]:
                st.markdown(f"- {_drv.description}")
        if _ov_full.explanation.warnings:
            for _w in _ov_full.explanation.warnings:
                st.caption(f"⚠️ {_w}")

        # ── Lineup status ─────────────────────────────────────────────────────
        st.markdown("---")
        _ov_lineup_label = "Lineups not yet available — using model baseline / manual override"
        st.info(f"**Lineup status:** {_ov_lineup_label}")

        # ── Open in Match Analyzer (for lineup override) ──────────────────────
        if st.button(
            f"🪪 Open in Match Analyzer (lineup override) →",
            key="ov_analyze_btn",
        ):
            _send_to_analyzer(_ov_sel_fix, _prov)
            st.info("Switch to the ⚽ Match Analyzer tab for lineup editor and full analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DAILY MATCH BOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_board:
    st.markdown(
        "Compact prediction summary for all matches on a selected date. "
        "Click **Analyze** to open the full Match Analyzer for any match."
    )

    _bd_prov = _load_provider()
    _bd_all  = _bd_prov.fixtures

    if not _bd_all:
        st.error("No fixtures loaded. Click 🔄 Refresh.")
        st.stop()

    _bd_snaps, _bd_params, _bd_rv = _load_board_model_data()
    if not _bd_rv:
        st.warning("Research-valid data unavailable — using default parameters.")

    _bd_dates = sorted({f.date for f in _bd_all})
    _bd_today = _dt.date.today().isoformat()
    # Default to today's date if available, else first date
    _bd_default_idx = _bd_dates.index(_bd_today) if _bd_today in _bd_dates else 0
    _bd_sel_date = st.selectbox(
        "Match date", _bd_dates, index=_bd_default_idx, key="bd_date"
    )

    _bd_day = sort_matches_by_datetime(filter_fixtures_by_date(_bd_all, _bd_sel_date))

    if not _bd_day:
        st.info(f"No matches on {_bd_sel_date}.")
    else:
        st.caption(f"**{len(_bd_day)}** match(es) on {_bd_sel_date}")
        _bd_preds = {}
        for _bdf in _bd_day:
            _bd_pred, _, _ = _build_match_prediction(_bdf, _bd_snaps, _bd_params, _bd_rv)
            _bd_preds[_bdf.match_id] = _bd_pred

        _bd_rows  = build_daily_match_rows(_bd_day, _bd_preds)
        _bd_dicts = [format_board_row_as_dict(r) for r in _bd_rows]
        st.dataframe(pd.DataFrame(_bd_dicts), use_container_width=True, hide_index=True)

        st.markdown("---")
        _bd_match_labels = [f"{r.team_a} vs {r.team_b}" for r in _bd_rows]
        _bd_sel_lbl  = st.selectbox("Select match to analyze", _bd_match_labels, key="bd_match_sel")
        _bd_sel_row  = next(r for r in _bd_rows if f"{r.team_a} vs {r.team_b}" == _bd_sel_lbl)
        _bd_sel_fix  = next(
            (f for f in _bd_day if f.team_a == _bd_sel_row.team_a and f.team_b == _bd_sel_row.team_b),
            None,
        )
        if _bd_sel_fix and st.button(
            f"🔬 Open Full Analysis → {_bd_sel_row.team_a} vs {_bd_sel_row.team_b}",
            key="bd_analyze_btn", type="primary",
        ):
            _send_to_analyzer(_bd_sel_fix, _bd_prov)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STATUS TAB (inserted before tab_predictor section continues)
# ══════════════════════════════════════════════════════════════════════════════
with tab_status:
    st.subheader("📡 Data Status")
    st.markdown("Live health check for API connection, model data coverage, and cache state.")

    _ds_prov = _load_provider()

    # ── API / fixture source ──────────────────────────────────────────────────
    st.markdown("### Fixture Source")
    _ds_c1, _ds_c2, _ds_c3 = st.columns(3)
    with _ds_c1:
        _ds_api = "✅ API connected" if _ds_prov.api_connected else "❌ API not connected"
        st.metric("API Status", _ds_api)
    with _ds_c2:
        st.metric("Fixture Source", _ds_prov.source_used.value.upper())
    with _ds_c3:
        st.metric("Fixtures Loaded", str(_ds_prov.fixture_count))

    if st.session_state.get("last_refresh"):
        st.info(f"Last manual refresh: **{st.session_state['last_refresh']}**")
    else:
        st.info("No manual refresh performed this session — using cached data.")

    if not _ds_prov.api_connected:
        st.warning(
            "API-Football is not connected. Set `API_FOOTBALL_KEY` in your `.env` file "
            "to enable live fixtures and lineups."
        )

    if _ds_prov.mapping_warnings:
        st.markdown("### Team Mapping Warnings")
        for _w in _ds_prov.mapping_warnings:
            st.warning(_w)
    else:
        st.success(f"✅ All {_ds_prov.fixture_count} fixture teams mapped successfully.")

    # ── Model data coverage ───────────────────────────────────────────────────
    st.markdown("### Model Data Coverage")
    try:
        _ds_snaps  = load_team_snapshots()
        _ds_params = load_strength_params()
        _ds_teams  = get_unique_teams(_ds_prov.fixtures)
        _ds_no_snap  = [t for t in _ds_teams if t not in _ds_snaps]
        _ds_no_param = [t for t in _ds_teams if t not in _ds_params]
        _dsc1, _dsc2, _dsc3 = st.columns(3)
        with _dsc1:
            st.metric("Teams in fixtures", len(_ds_teams))
        with _dsc2:
            st.metric("Missing ELO snapshots", len(_ds_no_snap))
        with _dsc3:
            st.metric("Missing strength params", len(_ds_no_param))
        if _ds_no_snap:
            with st.expander(f"Teams missing ELO snapshot ({len(_ds_no_snap)})", expanded=False):
                st.write(", ".join(sorted(_ds_no_snap)))
        if _ds_no_param:
            with st.expander(f"Teams missing strength params ({len(_ds_no_param)})", expanded=False):
                st.write(", ".join(sorted(_ds_no_param)))
        if not _ds_no_snap and not _ds_no_param:
            st.success("✅ All fixture teams have ELO snapshots and strength parameters.")
    except FileNotFoundError as _ds_err:
        st.error(f"Model data files missing: {_ds_err}")

    # ── Live data status ──────────────────────────────────────────────────────
    st.markdown("### Live Prediction Adapter")
    _ds_live = get_live_data_status(_api_client)
    _dsl1, _dsl2 = st.columns(2)
    with _dsl1:
        st.write(f"**API connected:** {'✅' if _ds_live.api_connected else '❌'}")
        st.write(f"**Fixture source:** {_ds_live.fixture_source}")
        st.write(f"**Last refresh:** {_ds_live.last_refresh or '—'}")
    with _dsl2:
        st.write(f"**Lineup source:** {_ds_live.lineup_source}")
        st.write(f"**Lineup status:** {_ds_live.lineup_status_label}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_predictor:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "Get a research-grade match prediction powered by historical ELO ratings, "
        "MLE-fitted team strengths, and Dixon-Coles probability correction."
    )

    # ── Mode selector ─────────────────────────────────────────────────────────
    _MODE_RV  = "Research-valid (ELO + MLE + Dixon-Coles)"
    _MODE_LEG = "Legacy (illustrative only)"

    with st.expander("Model settings", expanded=False):
        pipeline_mode = st.radio(
            "Prediction pipeline",
            [_MODE_RV, _MODE_LEG],
            index=0,
            horizontal=True,
            help=(
                "Research-valid (default): historical ELO + MLE attack/defense + "
                "calibrated xG + Dixon-Coles rho=-0.30. "
                "Legacy: manually-assigned team ratings — illustrative only."
            ),
        )
        override = st.checkbox(
            "Override xG manually (debug only)",
            value=False,
            help="Bypasses all model logic. Debug use only.",
        )
    is_research_valid = pipeline_mode == _MODE_RV

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        teams = load_teams()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load teams data: {e}")
        st.stop()

    _DEFAULT_SNAP = TeamSnapshot(elo=1800.0, ppg=1.5)
    _DEFAULT_PAR  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)

    if is_research_valid:
        try:
            _snapshots = load_team_snapshots()
            _strength  = load_strength_params()
        except FileNotFoundError as e:
            st.error(f"Research-valid data unavailable: {e}")
            st.stop()
    else:
        try:
            all_ratings = load_team_ratings()
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Could not load team ratings: {e}")
            st.stop()

    # ── Selected match banner (from Today's Matches board) ───────────────────
    _sel_fix: SelectedFixture | None = st.session_state.get("selected_fixture")

    if is_valid_selected_fixture(_sel_fix):
        _fix_id_display = f"  |  Fixture ID: `{_sel_fix.fixture_id}`" if _sel_fix.source_type == "api" else ""
        _fix_src_display = "🌐 Live API" if _sel_fix.source_type == "api" else "📁 CSV"
        st.info(
            f"**Analyzing live fixture:** {_sel_fix.team_a} vs {_sel_fix.team_b}  "
            f"|  Date: {_sel_fix.date}  "
            f"|  Stage: {_sel_fix.stage}  "
            f"|  Source: {_fix_src_display}"
            f"{_fix_id_display}"
        )
        if st.button("✕ Clear selected match", key="clear_selected_fixture"):
            del st.session_state["selected_fixture"]
            st.rerun()

    # ── Team selector ─────────────────────────────────────────────────────────
    # Pre-select teams from selected match if available
    _presel_a = _sel_fix.team_a if is_valid_selected_fixture(_sel_fix) else None
    _presel_b = _sel_fix.team_b if is_valid_selected_fixture(_sel_fix) else None

    _idx_a = teams.index(_presel_a) if _presel_a and _presel_a in teams else 0

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", options=teams, index=_idx_a)
    with col2:
        teams_b = [t for t in teams if t != team_a]
        _idx_b = teams_b.index(_presel_b) if _presel_b and _presel_b in teams_b else 0
        team_b = st.selectbox("Team B", options=teams_b, index=_idx_b)

    # ── Compute xG ────────────────────────────────────────────────────────────
    _expl_elo_a = _expl_elo_b = 1800.0
    _expl_alpha_a = _expl_alpha_b = 1.0
    _expl_beta_a  = _expl_beta_b  = 1.0

    if is_research_valid:
        snap_a = _snapshots.get(team_a, _DEFAULT_SNAP)
        snap_b = _snapshots.get(team_b, _DEFAULT_SNAP)
        par_a  = _strength.get(team_a, _DEFAULT_PAR)
        par_b  = _strength.get(team_b, _DEFAULT_PAR)

        if team_a not in _snapshots:
            st.warning(f"No ELO snapshot for {team_a} — using defaults.")
        if team_b not in _snapshots:
            st.warning(f"No ELO snapshot for {team_b} — using defaults.")
        if team_a not in _strength:
            st.warning(f"No MLE params for {team_a} — using defaults.")
        if team_b not in _strength:
            st.warning(f"No MLE params for {team_b} — using defaults.")

        from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg as _sxg
        from src.models.xg_calibration import calibrate_xg as _cal
        _raw_a, _raw_b = _sxg(snap_a.elo, snap_b.elo, par_a, par_b, snap_a.ppg, snap_b.ppg)
        auto_xg_a, auto_xg_b = _cal(_raw_a), _cal(_raw_b)

        _expl_elo_a, _expl_elo_b = snap_a.elo, snap_b.elo
        _expl_alpha_a, _expl_alpha_b = par_a.alpha_attack, par_b.alpha_attack
        _expl_beta_a,  _expl_beta_b  = par_a.beta_defense,  par_b.beta_defense
        _model_label = "ELO + MLE + Dixon-Coles (calibrated, rho=-0.30)"

    else:
        _ra = all_ratings.get(team_a, _AVG_RATINGS)
        _rb = all_ratings.get(team_b, _AVG_RATINGS)
        auto_xg_a, auto_xg_b = calculate_xg(_ra, _rb)
        _expl_elo_a   = float(_ra["elo"])
        _expl_elo_b   = float(_rb["elo"])
        _expl_alpha_a = float(_ra["attack_rating"])
        _expl_alpha_b = float(_rb["attack_rating"])
        _expl_beta_a  = float(_ra["defense_rating"])
        _expl_beta_b  = float(_rb["defense_rating"])
        _model_label = "Legacy Poisson (illustrative only)"

    # xG override inputs (collapsed behind checkbox)
    if override:
        with st.expander("Manual xG override", expanded=True):
            st.caption(f"Auto-calculated: {team_a} {auto_xg_a:.2f} | {team_b} {auto_xg_b:.2f}")
            c3, c4 = st.columns(2)
            with c3:
                xg_a = st.number_input(f"{team_a} xG", min_value=0.1, max_value=5.0,
                                       value=float(round(auto_xg_a, 1)), step=0.1,
                                       format="%.1f", key="xg_a_input")
            with c4:
                xg_b = st.number_input(f"{team_b} xG", min_value=0.1, max_value=5.0,
                                       value=float(round(auto_xg_b, 1)), step=0.1,
                                       format="%.1f", key="xg_b_input")
            final_xg_a, final_xg_b = xg_a, xg_b
    else:
        final_xg_a, final_xg_b = auto_xg_a, auto_xg_b

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        if is_research_valid:
            result = predict_dixon_coles(team_a, team_b, final_xg_a, final_xg_b, rho=DEFAULT_RHO)
        else:
            result = predict(team_a, team_b, final_xg_a, final_xg_b)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ── Build explainability ──────────────────────────────────────────────────
    _expl_inp = ExplanationInput(
        match_id="live",
        team_a=team_a, team_b=team_b,
        model_type=_model_label,
        elo_a=_expl_elo_a, elo_b=_expl_elo_b,
        alpha_attack_a=_expl_alpha_a, alpha_attack_b=_expl_alpha_b,
        beta_defense_a=_expl_beta_a, beta_defense_b=_expl_beta_b,
        xg_a_base=float(final_xg_a), xg_b_base=float(final_xg_b),
        squad_factor_a=1.0, squad_factor_b=1.0,
        xg_a_final=float(final_xg_a), xg_b_final=float(final_xg_b),
        win_a=result.win_a, draw=result.draw, win_b=result.win_b,
        top_scorelines=result.top_scorelines,
        player_data_research_valid=False,
        market_home_prob=None, market_draw_prob=None, market_away_prob=None,
        market_research_valid=False,
    )
    _expl = build_explanation(_expl_inp)

    _all_warnings = list(_expl.warnings)
    if not is_research_valid:
        _all_warnings.insert(0,
            "Legacy ratings are illustrative only — not derived from historical match data.")

    _confidence = compute_confidence(result.win_a, result.draw, result.win_b, _all_warnings)

    # ── Render: Prediction Card ───────────────────────────────────────────────
    st.markdown("---")
    render_prediction_card(
        team_a=team_a, team_b=team_b,
        win_a=result.win_a, draw=result.draw, win_b=result.win_b,
        xg_a=final_xg_a, xg_b=final_xg_b,
        confidence=_confidence,
        model_label=_model_label,
    )

    # ── Render: Scoreline Table ───────────────────────────────────────────────
    st.markdown("---")
    render_scoreline_table(result.top_scorelines, team_a, team_b)

    # ── Render: Betting Markets ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Betting Markets")
    st.caption(
        "These probabilities are model outputs, not guaranteed outcomes or betting advice."
    )

    _bm_matrix = build_dc_matrix(final_xg_a, final_xg_b, rho=DEFAULT_RHO)
    _bm = compute_betting_markets(team_a, team_b, _bm_matrix)

    _bm_col1, _bm_col2 = st.columns(2)

    with _bm_col1:
        st.markdown("**Over / Under**")
        _ou_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.over_under
        ]
        st.dataframe(pd.DataFrame(_ou_rows), use_container_width=True, hide_index=True)

        st.markdown("**Both Teams To Score**")
        _btts_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.btts
        ]
        st.dataframe(pd.DataFrame(_btts_rows), use_container_width=True, hide_index=True)

        st.markdown("**Clean Sheet**")
        _cs_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.clean_sheet
        ]
        st.dataframe(pd.DataFrame(_cs_rows), use_container_width=True, hide_index=True)

    with _bm_col2:
        st.markdown("**Double Chance**")
        _dc_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.double_chance
        ]
        st.dataframe(pd.DataFrame(_dc_rows), use_container_width=True, hide_index=True)

        st.markdown("**Draw No Bet**")
        _dnb_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.draw_no_bet
        ]
        st.dataframe(pd.DataFrame(_dnb_rows), use_container_width=True, hide_index=True)

        st.markdown("**Team Totals**")
        _tt_rows = [
            {"Market": mp.selection, "Probability": f"{mp.probability:.1%}",
             "Fair Odds": f"{mp.implied_fair_odds:.2f}", "Confidence": mp.confidence_label}
            for mp in _bm.team_totals
        ]
        st.dataframe(pd.DataFrame(_tt_rows), use_container_width=True, hide_index=True)

    # ── Render: Model Signals ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Signals")
    st.caption(
        "Model signal ranking — not betting advice. "
        "These are model outputs only and do not represent guaranteed outcomes."
    )

    _rec_set = generate_recommendations(
        betting_markets=_bm,
        prediction_confidence=_confidence.label,
        data_warnings=_all_warnings,
        is_research_valid=is_research_valid,
        top_n=5,
    )

    if _rec_set.recommendations:
        _signal_rows = []
        for r in _rec_set.recommendations:
            _badge = {"Strong": "🟢 Strong", "Moderate": "🟡 Moderate", "Weak": "🔴 Weak"}.get(
                r.signal_strength, r.signal_strength
            )
            _row = {
                "Selection": r.selection,
                "Probability": f"{r.model_probability:.1%}",
                "Fair Odds": f"{r.fair_odds:.2f}",
                "Signal": _badge,
                "Rationale": r.rationale,
            }
            if r.warning:
                _row["Note"] = r.warning
            _signal_rows.append(_row)
        st.dataframe(pd.DataFrame(_signal_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No high-signal markets identified for this match.")

    # ── Live Data Status ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📡 Live Data Status", expanded=False):
        _live_status = get_live_data_status(_api_client)
        _conn_badge = "✅ Connected" if _live_status.api_connected else "❌ Not connected"
        _refresh_str = _live_status.last_refresh or "—"
        st.markdown(
            f"**API:** {_conn_badge}  |  "
            f"**Last refresh:** {_refresh_str}  |  "
            f"**Fixture source:** {_live_status.fixture_source}"
        )
        st.markdown(
            f"**Lineup source:** {_live_status.lineup_source}  |  "
            f"**Lineup status:** {_live_status.lineup_status_label}"
        )
        if not _live_status.api_connected:
            st.info(
                "Set the `API_FOOTBALL_KEY` environment variable to enable live data. "
                "Falling back to CSV/placeholder data."
            )

    # ── Lineup & Availability Override ───────────────────────────────────────
    st.markdown("---")
    with st.expander("🪪 Lineup & Availability Override", expanded=False):
        st.caption(
            "**Manual lineup override — not research-valid unless sourced from "
            "official pre-match lineups.**  "
            "Adjust availability/form to recalculate squad strength and see the "
            "before/after impact on xG and win probabilities."
        )

        # ── Try to load lineup: live API first, then CSV, then default ───────────
        _csv_entries_a: list = []
        _csv_entries_b: list = []
        _lineup_source_label = "Manual (placeholder)"
        _lineup_rv_a = _lineup_rv_b = False

        # 1. Try live API if connected
        if _live_status.api_connected:
            try:
                # Use selected fixture's API ID directly if available;
                # otherwise search through live fixtures for a matching fixture.
                _sel_fix_now: SelectedFixture | None = st.session_state.get("selected_fixture")
                _api_fixture_id = get_api_fixture_id(_sel_fix_now)

                if _api_fixture_id is None:
                    # Fallback: search live upcoming fixtures for these teams
                    from src.data.live_fixture_loader import fetch_upcoming_fixtures
                    _live_fixtures = fetch_upcoming_fixtures(_api_client)
                    _api_fixture_id = next(
                        (f.fixture_id for f in _live_fixtures
                         if f.home_team == team_a and f.away_team == team_b),
                        None,
                    )

                if _api_fixture_id:
                    _api_entries_a, _api_entries_b = load_live_lineups_for_match(
                        _api_client, fixture_id=_api_fixture_id,
                        team_a=team_a, team_b=team_b,
                    )
                    if _api_entries_a or _api_entries_b:
                        _csv_entries_a = _api_entries_a
                        _csv_entries_b = _api_entries_b
                        _lineup_source_label = "Official lineup (API-Football)"
                        _lineup_rv_a = bool(_api_entries_a)
                        _lineup_rv_b = bool(_api_entries_b)
                    else:
                        # Lineups not yet published — show informative message
                        _lineup_source_label = "Lineups not yet available — using model baseline / manual override"
            except Exception:
                pass  # fall through to CSV

        # 2. CSV fallback
        if not _csv_entries_a and not _csv_entries_b:
            try:
                if _LINEUPS_CSV.exists():
                    # Find the first match_id that involves both teams
                    _all_entries = load_expected_lineups(_LINEUPS_CSV)
                    _match_ids_with_a = {e.match_id for e in _all_entries if e.team == team_a}
                    _match_ids_with_b = {e.match_id for e in _all_entries if e.team == team_b}
                    _shared_ids = _match_ids_with_a & _match_ids_with_b
                    if _shared_ids:
                        _selected_match_id = sorted(_shared_ids)[0]
                        _all_match_entries = load_expected_lineups(
                            _LINEUPS_CSV, match_id=_selected_match_id
                        )
                        _csv_entries_a = [e for e in _all_match_entries if e.team == team_a]
                        _csv_entries_b = [e for e in _all_match_entries if e.team == team_b]
                        if _csv_entries_a or _csv_entries_b:
                            _vr_a = validate_lineup_for_match(_all_match_entries, team_a)
                            _vr_b = validate_lineup_for_match(_all_match_entries, team_b)
                            _lineup_rv_a = _vr_a.is_research_valid
                            _lineup_rv_b = _vr_b.is_research_valid
                            _src = next(
                                (e.source_type for e in _csv_entries_a or _csv_entries_b),
                                "placeholder",
                            )
                            _src_labels = {
                                "placeholder": "Placeholder data",
                                "manual": "Manual entry",
                                "projected_lineup": "Projected lineup",
                                "official_lineup": "Official lineup",
                                "injury_report": "Injury report",
                            }
                            _lineup_source_label = _src_labels.get(_src, _src)
            except Exception:
                pass  # silently fall back to default

        # ── Source badge ───────────────────────────────────────────────────────
        _rv_badge_a = "✅ Research-valid" if _lineup_rv_a else "⚠️ Engineering-valid"
        _rv_badge_b = "✅ Research-valid" if _lineup_rv_b else "⚠️ Engineering-valid"
        st.info(
            f"**Source:** {_lineup_source_label}  |  "
            f"{team_a}: {_rv_badge_a}  |  {team_b}: {_rv_badge_b}"
        )

        _lo_col1, _lo_col2 = st.columns(2)

        # ── Team A lineup ──────────────────────────────────────────────────────
        with _lo_col1:
            st.markdown(f"**{team_a}**")
            if _csv_entries_a:
                _override_obj_a = convert_lineup_entries_to_override(_csv_entries_a)
                _default_a = _override_obj_a
            else:
                _default_a = create_default_lineup(team_a)
            _table_a = format_player_table(_default_a.players)
            _edited_a = st.data_editor(
                pd.DataFrame(_table_a),
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="lineup_editor_a",
                column_config={
                    "player_id": st.column_config.TextColumn("ID", disabled=True),
                    "Player Name": st.column_config.TextColumn("Name"),
                    "Team": st.column_config.TextColumn("Team", disabled=True),
                    "Starter": st.column_config.CheckboxColumn("Starter"),
                    "Status": st.column_config.SelectboxColumn(
                        "Status", options=list(STATUS_TO_FACTOR.keys())
                    ),
                    "Availability Factor": st.column_config.NumberColumn(
                        "Avail.", min_value=0.0, max_value=1.0, step=0.1, format="%.1f"
                    ),
                    "Form Factor": st.column_config.NumberColumn(
                        "Form", min_value=0.5, max_value=1.5, step=0.1, format="%.1f"
                    ),
                },
            )

        # ── Team B lineup ──────────────────────────────────────────────────────
        with _lo_col2:
            st.markdown(f"**{team_b}**")
            if _csv_entries_b:
                _override_obj_b = convert_lineup_entries_to_override(_csv_entries_b)
                _default_b = _override_obj_b
            else:
                _default_b = create_default_lineup(team_b)
            _table_b = format_player_table(_default_b.players)
            _edited_b = st.data_editor(
                pd.DataFrame(_table_b),
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="lineup_editor_b",
                column_config={
                    "player_id": st.column_config.TextColumn("ID", disabled=True),
                    "Player Name": st.column_config.TextColumn("Name"),
                    "Team": st.column_config.TextColumn("Team", disabled=True),
                    "Starter": st.column_config.CheckboxColumn("Starter"),
                    "Status": st.column_config.SelectboxColumn(
                        "Status", options=list(STATUS_TO_FACTOR.keys())
                    ),
                    "Availability Factor": st.column_config.NumberColumn(
                        "Avail.", min_value=0.0, max_value=1.0, step=0.1, format="%.1f"
                    ),
                    "Form Factor": st.column_config.NumberColumn(
                        "Form", min_value=0.5, max_value=1.5, step=0.1, format="%.1f"
                    ),
                },
            )

        # ── Apply & show impact ────────────────────────────────────────────────
        if st.button("Apply lineup overrides", key="apply_lineup_btn"):
            _players_a = parse_player_edits(_edited_a.to_dict("records"), team=team_a)
            _players_b = parse_player_edits(_edited_b.to_dict("records"), team=team_b)
            _override_a = LineupOverride(team=team_a, players=_players_a)
            _override_b = LineupOverride(team=team_b, players=_players_b)

            _lo_result = apply_lineup_override(
                team_a=team_a, team_b=team_b,
                xg_a_base=float(final_xg_a), xg_b_base=float(final_xg_b),
                override_a=_override_a, override_b=_override_b,
                rho=DEFAULT_RHO,
            )

            st.markdown("**Before / After Impact**")
            _impact_data = {
                "Metric": [
                    f"{team_a} Squad Factor",
                    f"{team_b} Squad Factor",
                    f"{team_a} xG",
                    f"{team_b} xG",
                    f"{team_a} Win %",
                    "Draw %",
                    f"{team_b} Win %",
                ],
                "Before": [
                    "1.00",
                    "1.00",
                    f"{_lo_result.xg_a_base:.2f}",
                    f"{_lo_result.xg_b_base:.2f}",
                    f"{_lo_result.win_a_base:.1%}",
                    f"{_lo_result.draw_base:.1%}",
                    f"{_lo_result.win_b_base:.1%}",
                ],
                "After": [
                    f"{_lo_result.squad_factor_a:.2f}",
                    f"{_lo_result.squad_factor_b:.2f}",
                    f"{_lo_result.xg_a_adjusted:.2f}",
                    f"{_lo_result.xg_b_adjusted:.2f}",
                    f"{_lo_result.win_a_adjusted:.1%}",
                    f"{_lo_result.draw_adjusted:.1%}",
                    f"{_lo_result.win_b_adjusted:.1%}",
                ],
                "Delta": [
                    f"{_lo_result.squad_factor_a - 1.0:+.2f}",
                    f"{_lo_result.squad_factor_b - 1.0:+.2f}",
                    f"{_lo_result.delta_xg_a:+.2f}",
                    f"{_lo_result.delta_xg_b:+.2f}",
                    f"{_lo_result.delta_win_a:+.1%}",
                    f"{_lo_result.delta_draw:+.1%}",
                    f"{_lo_result.delta_win_b:+.1%}",
                ],
            }
            st.dataframe(pd.DataFrame(_impact_data), use_container_width=True, hide_index=True)
            st.warning(
                "⚠️ Manual lineup override — not research-valid unless sourced "
                "from official pre-match lineups."
            )

    # ── Render: Explanation Panel ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Reasons")
    render_explanation_panel(_expl, is_research_valid=is_research_valid and not _all_warnings)
with tab_lab:
    st.markdown("Advanced analytics: backtesting, market intelligence, and model calibration.")
    _lab_bt, _lab_mkt = st.tabs(["📊 Backtesting", "📈 Market Intelligence"])
    with _lab_bt:
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
    with _lab_mkt:
        st.markdown(
            "Compare model predictions against bookmaker market odds. "
            "Market odds are used for **comparison and calibration only** — "
            "they do not change predictions."
        )

        st.warning(
            "⚠️ **Market Intelligence is currently engineering-valid only until sourced odds are loaded.**\n\n"
            "Current data: `data/market_odds.csv` contains **placeholder odds**, not real bookmaker data. "
            "Results below are for pipeline validation only. "
            "Replace with football-data.co.uk or The Odds API data before drawing conclusions."
        )

        try:
            market_results, market_summary = run_market_comparison()
        except Exception as e:
            st.error(f"Market comparison failed: {e}")
            market_results, market_summary = [], None

        if market_summary and market_results:
            # ── Summary metrics ────────────────────────────────────────────────
            st.subheader("Summary Metrics")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Matched Games", market_summary.total_matches)
            with col_b:
                st.metric("Model Brier", f"{market_summary.model_brier:.4f}")
            with col_c:
                st.metric("Market Brier", f"{market_summary.market_brier:.4f}",
                          delta=f"{market_summary.brier_delta:+.4f}",
                          delta_color="inverse")

            col_d, col_e = st.columns(2)
            with col_d:
                st.metric("Avg Absolute Divergence", f"{market_summary.avg_absolute_divergence:.1%}")
            with col_e:
                st.metric("High-Divergence Matches (>5pp)", market_summary.high_divergence_count)

            st.caption(
                "Brier delta = market Brier - model Brier. "
                "Positive = model has lower Brier (better calibrated). "
                "**Not meaningful with placeholder data.**"
            )

            # ── Model vs market comparison table ──────────────────────────────
            st.subheader("Model vs Market Comparison")

            comparison_rows = []
            outcome_labels = {"team_a_win": "Home Win", "draw": "Draw", "team_b_win": "Away Win"}
            for r in market_results:
                comparison_rows.append({
                    "Date": r.date,
                    "Match": f"{r.team_a} vs {r.team_b}",
                    "Actual": outcome_labels[r.actual_outcome],
                    "Model H/D/A": f"{r.model_win_a:.2f} / {r.model_draw:.2f} / {r.model_win_b:.2f}",
                    "Market H/D/A": f"{r.market_home:.2f} / {r.market_draw:.2f} / {r.market_away:.2f}",
                    "Overround": f"{r.market_overround:.1%}",
                    "Closer": "Model" if r.model_closer_than_market else "Market",
                    "Bkm src": r.market_source_type,
                })

            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

            # ── Divergence table ───────────────────────────────────────────────
            st.subheader("Divergence Table (model - market)")
            st.caption(
                "Positive divergence = model assigns more probability than market. "
                "Negative = model assigns less. Signed, not absolute."
            )

            div_rows = []
            for r in market_results:
                flag = "YES" if abs(r.largest_divergence_value) >= 0.05 else ""
                div_rows.append({
                    "Match": f"{r.team_a} vs {r.team_b}",
                    "d(Home)": f"{r.home_divergence:+.3f}",
                    "d(Draw)": f"{r.draw_divergence:+.3f}",
                    "d(Away)": f"{r.away_divergence:+.3f}",
                    "Largest div outcome": r.largest_divergence_outcome.replace("_", " "),
                    "Div value": f"{r.largest_divergence_value:+.3f}",
                    ">5pp": flag,
                })

            st.dataframe(pd.DataFrame(div_rows), use_container_width=True)

            if market_summary.high_divergence_count > 0:
                st.markdown(
                    f"**{market_summary.high_divergence_count} match(es)** where model and market "
                    f"disagree by >5 percentage points on at least one outcome. "
                    f"On those matches: model was closer in "
                    f"**{market_summary.model_wins_high_divergence}**, "
                    f"market was closer in "
                    f"**{market_summary.market_wins_high_divergence}**."
                )

    # ══════════════════════════════════════════════════════════════════════════════

# TAB 4 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_tournament:
    st.markdown(
        "Simulate the full World Cup tournament using the research-valid calibrated "
        "model. Run thousands of simulations to estimate each team's probability of "
        "reaching each stage."
    )
    st.info(
        "Uses ELO + MLE + calibrated xG + Dixon-Coles (rho=-0.30). "
        "Group stage allows draws. Knockout ties resolved by penalty probability."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    _tour_fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"

    try:
        _tour_fixtures = load_fixtures(_tour_fixture_path)
        _tour_snaps    = load_team_snapshots()
        _tour_params   = load_strength_params()
    except FileNotFoundError as e:
        st.error(f"Tournament data unavailable: {e}")
        st.stop()

    st.caption(f"Fixture file: {_tour_fixture_path.name} — {len(_tour_fixtures)} group-stage matches, 8 groups")
    try:
        _params_as_of = pd.read_csv(
            Path(__file__).parent.parent.parent / "data" / "team_strength_params.csv"
        )["as_of_date"].max()
        st.caption(f"⚙️ MLE attack/defence params last refit: **{_params_as_of}** "
                   f"(ELO/form snapshots are loaded fresh from match_results.csv each run)")
    except Exception:
        pass

    # ── Simulation controls ────────────────────────────────────────────────────
    col_n, col_seed = st.columns(2)
    with col_n:
        n_sims = st.slider(
            "Number of simulations",
            min_value=100, max_value=10_000, value=1_000, step=100,
            help="More simulations = smoother probabilities. 1,000 is fast; 10,000 is research-grade.",
        )
    with col_seed:
        use_seed = st.checkbox("Deterministic seed", value=True,
                               help="Fix random seed for reproducible results.")
        rng_seed_val = 42 if use_seed else None

    # ── Calibration controls ───────────────────────────────────────────────────
    with st.expander("Simulation calibration (advanced)", expanded=False):
        st.markdown(
            "These parameters reduce over-concentration in tournament winner probabilities "
            "by adding controlled uncertainty at the **simulation layer only**. "
            "The core match model (ELO, xG, Dixon-Coles) is unchanged."
        )
        cal_col1, cal_col2, cal_col3 = st.columns(3)
        with cal_col1:
            temp_val = st.slider(
                "Temperature (tau)",
                min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                help="tau=1.0 = raw model. tau>1 flattens win/draw/lose distribution per match. Recommended: 1.5",
            )
        with cal_col2:
            noise_val = st.slider(
                "xG noise (sigma)",
                min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                help="Log-normal noise on expected goals before each simulated match. sigma=0 = off. Recommended: 0.2",
            )
        with cal_col3:
            upset_val = st.slider(
                "Upset factor (epsilon)",
                min_value=0.0, max_value=0.5, value=0.15, step=0.05,
                help="Mixes knockout penalty probability toward 50/50. epsilon=0 = off. Recommended: 0.15",
            )
        calib_params = CalibrationParams(
            temperature=temp_val,
            xg_noise_sigma=noise_val,
            upset_factor=upset_val,
        )
        if temp_val == 1.0 and noise_val == 0.0 and upset_val == 0.0:
            st.info("All calibration off — using raw model probabilities.")
        else:
            st.success(f"Calibration active: tau={temp_val}, sigma={noise_val}, epsilon={upset_val}")

    if st.button("Run Tournament Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations…"):
            mc = run_monte_carlo(
                _tour_fixture_path, _tour_snaps, _tour_params,
                n=n_sims, rng_seed=rng_seed_val,
                calibration=calib_params,
            )
        st.session_state["mc_result"] = mc
        st.session_state["mc_run_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mc = st.session_state.get("mc_result")
    if mc is not None:
        st.caption(f"Showing results from last run ({st.session_state.get('mc_run_at', '—')}). "
                   "Adjusting sliders above does not change these results — click "
                   "**Run Tournament Simulation** to recompute.")
        conc = compute_concentration_metrics(mc.win_tournament)
        st.success(
            f"Done — {mc.n_simulations:,} simulations. "
            f"Top-1: {conc.top1:.1%}  Top-2: {conc.top2:.1%}  "
            f"Top-5: {conc.top5:.1%}  Entropy: {conc.entropy:.2f} bits"
        )

        # ── Results table ──────────────────────────────────────────────────
        st.subheader("Tournament Probability Table")

        all_tour_teams = set(mc.reach_r16) | set(mc.win_tournament)
        rows = []
        for team in sorted(all_tour_teams):
            rows.append({
                "Team":             team,
                "Win Tournament":   f"{mc.win_tournament.get(team, 0):.1%}",
                "Reach Final":      f"{mc.reach_final.get(team, 0):.1%}",
                "Reach Semi-Final": f"{mc.reach_sf.get(team, 0):.1%}",
                "Reach Quarter-F":  f"{mc.reach_qf.get(team, 0):.1%}",
                "Reach R16":        f"{mc.reach_r16.get(team, 0):.1%}",
            })

        # Sort by win probability descending
        rows.sort(key=lambda r: -float(r["Win Tournament"].rstrip("%")))

        tour_df = pd.DataFrame(rows)
        st.dataframe(tour_df, use_container_width=True, hide_index=True)

        # ── Top 5 favourites ──────────────────────────────────────────────
        st.subheader("Top 5 Tournament Favourites")
        top5_rows = rows[:5]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, row in zip([c1, c2, c3, c4, c5], top5_rows):
            with col:
                st.metric(row["Team"], row["Win Tournament"])

        st.caption(
            f"Based on {mc.n_simulations:,} Monte Carlo simulations. "
            "Probabilities reflect model uncertainty — not guaranteed outcomes."
        )
    if mc is None:
        st.markdown(
            "Click **Run Tournament Simulation** to generate probability estimates. "
            "Results are based on the same calibrated model used in the Match Predictor tab."
        )



with tab_golden_boot:
    st.markdown(
        "Project the **Golden Boot** (top tournament scorer) using each player's "
        "expected goals across the tournament (xGT), driven by the tournament "
        "simulation's expected number of matches per team."
    )
    st.caption(
        "xGT = Expected Team Matches x (Expected Minutes / 90) x xG per 90 "
        "x Penalty Factor x Starting Probability"
    )
    st.warning(
        "**Engineering validation only** — player data is currently placeholder "
        "(`data/player_profiles.csv`, 8 teams). The architecture is built so "
        "API-Football player statistics can replace this data without code changes."
    )

    try:
        _gb_profiles = load_player_profiles()
        _gb_teams_covered = sorted({p.team for p in _gb_profiles.values()})
        _gb_valid_count = sum(1 for p in _gb_profiles.values() if p.research_valid)
        st.caption(
            f"📋 Player data source: {len(_gb_teams_covered)} of 48 WC2026 teams covered "
            f"({', '.join(_gb_teams_covered)}). "
            f"{_gb_valid_count}/{len(_gb_profiles)} player rows are research-valid "
            f"(live API-Football season statistics); the rest are placeholder/manual estimates. "
            f"Lineup source: 2022 World Cup placeholder data (not WC2026)."
        )
    except FileNotFoundError as e:
        _gb_profiles = {}
        st.error(f"Player profile data unavailable: {e}")

    _gb_fixture_path = Path(__file__).parent.parent.parent / "data" / "world_cup_fixture_sample.csv"

    gb_n_sims = st.slider(
        "Monte Carlo simulations (top-scorer probability)",
        min_value=1_000, max_value=20_000, value=5_000, step=1_000,
        key="gb_n_sims",
        help="More simulations = smoother top-scorer probabilities.",
    )

    if st.button("Run Golden Boot Projection", type="primary", key="gb_run"):
        with st.spinner("Simulating tournament and projecting Golden Boot..."):
            try:
                _gb_snaps = load_team_snapshots()
                _gb_params = load_strength_params()
                _gb_mc = run_monte_carlo(
                    _gb_fixture_path, _gb_snaps, _gb_params,
                    n=1_000, rng_seed=42,
                )
            except FileNotFoundError as e:
                _gb_mc = None
                st.error(f"Tournament data unavailable: {e}")

            if _gb_mc is not None and _gb_profiles:
                _gb_results = predict_golden_boot(
                    _gb_profiles, _gb_mc, n_sims=gb_n_sims, rng_seed=42,
                )
                st.session_state["golden_boot_results"] = _gb_results
            elif not _gb_profiles:
                st.session_state["golden_boot_results"] = []

    _gb_results: list[GoldenBootPlayerResult] = st.session_state.get("golden_boot_results")

    if _gb_results is None:
        st.markdown(
            "Click **Run Golden Boot Projection** to generate the table, "
            "favourites, and dark-horse projections below."
        )
    elif not _gb_results:
        st.info("No player profile data available for Golden Boot projections.")
    else:
        # ── 1. Top 25 Golden Boot table ────────────────────────────────────
        st.subheader("🏅 Top 25 Golden Boot Table")
        top25 = _gb_results[:25]
        gb_rows = []
        for i, r in enumerate(top25, start=1):
            gb_rows.append({
                "Rank": i,
                "Player": r.player_name,
                "Team": r.team,
                "Expected Goals (xGT)": f"{r.expected_goals:.2f}",
                "P(Top Scorer)": f"{r.prob_top_scorer:.1%}",
                "P(3+ goals)": f"{r.prob_score_3plus:.1%}",
                "P(5+ goals)": f"{r.prob_score_5plus:.1%}",
                "P(7+ goals)": f"{r.prob_score_7plus:.1%}",
                "Data Validity": (
                    "✅ Research-valid"
                    if _gb_profiles.get(r.player_id, None) and _gb_profiles[r.player_id].research_valid
                    else "⚠️ Engineering estimate only — not research-valid"
                ),
            })
        st.dataframe(pd.DataFrame(gb_rows), use_container_width=True, hide_index=True)

        # ── 2. Team-by-team scorer projections ─────────────────────────────
        st.subheader("📋 Team-by-Team Scorer Projections")
        gb_teams = sorted({r.team for r in _gb_results})
        gb_team_choice = st.selectbox("Select team", gb_teams, key="gb_team_select")
        team_rows = [
            {
                "Player": r.player_name,
                "Position": r.position,
                "Expected Goals (xGT)": f"{r.expected_goals:.2f}",
                "P(Top Scorer)": f"{r.prob_top_scorer:.1%}",
                "P(3+ goals)": f"{r.prob_score_3plus:.1%}",
                "P(5+ goals)": f"{r.prob_score_5plus:.1%}",
                "P(7+ goals)": f"{r.prob_score_7plus:.1%}",
                "Most Likely Goals": r.most_likely_goals,
            }
            for r in _gb_results if r.team == gb_team_choice
        ]
        st.dataframe(pd.DataFrame(team_rows), use_container_width=True, hide_index=True)

        # ── 3. Golden Boot favourites cards ─────────────────────────────────
        st.subheader("⭐ Golden Boot Favourites")
        favourites = _gb_results[:5]
        fav_cols = st.columns(len(favourites)) if favourites else []
        for col, r in zip(fav_cols, favourites):
            with col:
                st.metric(
                    label=f"{r.player_name} ({r.team})",
                    value=f"{r.expected_goals:.2f} xG",
                    delta=f"{r.prob_top_scorer:.1%} top scorer",
                )

        # ── 4. Dark horse section ────────────────────────────────────────────
        st.subheader("🐎 Dark Horses")
        st.caption(
            "Players with modest expected goals but a non-trivial chance of a "
            "breakout (3+ goal) tournament."
        )
        dark_horses = [
            r for r in _gb_results
            if r.expected_goals < 2.0 and r.prob_score_3plus >= 0.05
        ]
        dark_horses.sort(key=lambda r: -r.prob_score_3plus)
        if dark_horses:
            dh_rows = [
                {
                    "Player": r.player_name,
                    "Team": r.team,
                    "Expected Goals (xGT)": f"{r.expected_goals:.2f}",
                    "P(3+ goals)": f"{r.prob_score_3plus:.1%}",
                    "P(Top Scorer)": f"{r.prob_top_scorer:.1%}",
                }
                for r in dark_horses[:10]
            ]
            st.dataframe(pd.DataFrame(dh_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("No dark-horse candidates found under current thresholds.")

        # ── 5. Most likely final goals total per player ────────────────────
        st.subheader("🎯 Most Likely Final Goals Total")
        mlg_rows = [
            {
                "Player": r.player_name,
                "Team": r.team,
                "Expected Goals (xGT)": f"{r.expected_goals:.2f}",
                "Most Likely Goals": r.most_likely_goals,
            }
            for r in _gb_results[:25]
        ]
        st.dataframe(pd.DataFrame(mlg_rows), use_container_width=True, hide_index=True)

        st.caption(
            "Engineering validation only — based on placeholder player data for "
            f"{len(set(r.team for r in _gb_results))} teams. "
            "Model NOT modified; this tab is additive only."
        )

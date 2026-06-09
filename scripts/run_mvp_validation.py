#!/usr/bin/env python
"""World Cup 2026 Prediction Platform — MVP Validation Report.

Runs end-to-end validation against the live API-Football service:
  1. Full test suite (optional, slow)
  2. Live API fixture refresh + mapping audit
  3. App import check
  4. Sample predictions for 5 World Cup matches from the live fixture list

Usage:
    python scripts/run_mvp_validation.py
    python scripts/run_mvp_validation.py --skip-tests   # faster, no pytest
    python scripts/run_mvp_validation.py --n 3          # fewer sample matches
    python scripts/run_mvp_validation.py --season 2026 --league 1

Requires:
    API_FOOTBALL_KEY env var (or in .env file)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # python-dotenv optional

# ── Project imports ───────────────────────────────────────────────────────────
from src.data.api_football_client import ApiFootballClient, ApiKeyMissingError
from src.data.live_fixture_loader import fetch_upcoming_fixtures
from src.data.team_mapping_audit import audit_team_mappings
from src.data.fixture_provider import FixtureSource, get_fixtures
from src.data.team_snapshot_loader import load_team_snapshots, TeamSnapshot
from src.data.strength_loader import load_strength_params, StrengthParams
from src.models.research_valid_predictor import predict_research_valid, ResearchValidInput, DEFAULT_RHO
from src.models.dixon_coles import build_dc_matrix
from src.models.betting_markets import compute_betting_markets
from src.models.recommendations import generate_recommendations
from src.explainability.driver import build_explanation, ExplanationInput
from src.app.components.prediction_cards import compute_confidence
from src.models.live_prediction_adapter import load_live_lineups_for_match
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char="=", n=70):
    print(char * n)

def header(title):
    sep()
    print(f"  {title}")
    sep()

def section(title):
    print()
    print(f"--- {title} ---")


def predict_match(team_a, team_b, snaps, params):
    """Run full research-valid prediction pipeline for team_a vs team_b."""
    SNAP_DEF = TeamSnapshot(elo=1800.0, ppg=1.5)
    PAR_DEF  = StrengthParams(alpha_attack=1.0, beta_defense=1.0, matches_used=0)

    sa = snaps.get(team_a, SNAP_DEF)
    sb = snaps.get(team_b, SNAP_DEF)
    pa = params.get(team_a, PAR_DEF)
    pb = params.get(team_b, PAR_DEF)

    raw_a, raw_b = calculate_strength_adjusted_xg(sa.elo, sb.elo, pa, pb, sa.ppg, sb.ppg)
    xg_a, xg_b  = calibrate_xg(raw_a), calibrate_xg(raw_b)

    rv = predict_research_valid(ResearchValidInput(
        team_a=team_a, team_b=team_b,
        snapshot_a=sa, snapshot_b=sb, params_a=pa, params_b=pb,
    ))

    mat  = build_dc_matrix(xg_a, xg_b, rho=DEFAULT_RHO)
    bm   = compute_betting_markets(team_a, team_b, mat)
    conf = compute_confidence(rv.win_a, rv.draw, rv.win_b, [])
    recs = generate_recommendations(bm, conf.label, [], True, top_n=3)

    expl_inp = ExplanationInput(
        match_id="validation",
        team_a=team_a, team_b=team_b,
        model_type="ELO + MLE + Dixon-Coles (rho=-0.30)",
        elo_a=sa.elo, elo_b=sb.elo,
        alpha_attack_a=pa.alpha_attack, alpha_attack_b=pb.alpha_attack,
        beta_defense_a=pa.beta_defense, beta_defense_b=pb.beta_defense,
        xg_a_base=xg_a, xg_b_base=xg_b,
        squad_factor_a=1.0, squad_factor_b=1.0,
        xg_a_final=xg_a, xg_b_final=xg_b,
        win_a=rv.win_a, draw=rv.draw, win_b=rv.win_b,
        top_scorelines=rv.top_scorelines,
        player_data_research_valid=False,
        market_home_prob=None, market_draw_prob=None, market_away_prob=None,
        market_research_valid=False,
    )
    expl = build_explanation(expl_inp)

    over25 = next((m.probability for m in bm.over_under if "Over 2.5" in m.selection), 0.0)
    btts   = next((m.probability for m in bm.btts if m.selection == "BTTS Yes"), 0.0)
    top_sig = recs.recommendations[0].selection if recs.recommendations else "—"
    top_sig_strength = recs.recommendations[0].signal_strength if recs.recommendations else "—"
    top5 = [f"{s[0]}-{s[1]}" for s in rv.top_scorelines[:5]]

    return {
        "xg_a":        round(xg_a, 2),
        "xg_b":        round(xg_b, 2),
        "win_a":       rv.win_a,
        "draw":        rv.draw,
        "win_b":       rv.win_b,
        "most_likely": f"{rv.top_scorelines[0][0]}-{rv.top_scorelines[0][1]}" if rv.top_scorelines else "?",
        "top5":        top5,
        "over25":      over25,
        "btts":        btts,
        "top_signal":  f"{top_sig} ({top_sig_strength})",
        "confidence":  conf.label,
        "drivers":     [d.description for d in expl.drivers[:3]] if expl.drivers else [],
    }


# ── Check 1: App import ───────────────────────────────────────────────────────

def check_app_import():
    section("App import check")
    try:
        import importlib, unittest.mock as mock
        import sys as _sys
        # Save and mock streamlit
        _orig = {}
        for mod in ("streamlit", "streamlit.components", "streamlit.components.v1"):
            _orig[mod] = _sys.modules.get(mod)
            _sys.modules[mod] = mock.MagicMock()
        # Import key modules
        from src.app.tournament_filters import filter_fixtures, get_unique_teams
        from src.app.selected_fixture  import SelectedFixture, create_selected_fixture
        from src.data.fixture_provider import FixtureSource, get_fixtures
        for mod, orig in _orig.items():
            if orig is None:
                _sys.modules.pop(mod, None)
            else:
                _sys.modules[mod] = orig
        print("  PASS  All app modules import cleanly")
        return True
    except Exception as e:
        print(f"  FAIL  Import error: {e}")
        return False


# ── Check 2: API connection ───────────────────────────────────────────────────

def check_api_connection(client):
    section("API connection")
    try:
        data = client.get("/status", ttl_seconds=60)
        requests_used = data.get("response", {}).get("requests", {}).get("current", "?")
        print(f"  PASS  API-Football connected (requests used today: {requests_used})")
        return True
    except ApiKeyMissingError:
        print("  FAIL  API key missing — set API_FOOTBALL_KEY in .env")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


# ── Check 3: Fixture refresh ──────────────────────────────────────────────────

def check_fixture_refresh(client, league_id, season):
    section(f"Fixture refresh (league={league_id}, season={season})")
    try:
        from src.data.live_fixture_loader import fetch_upcoming_fixtures
        fixtures = fetch_upcoming_fixtures(client, league_id=league_id, season=season, ttl_seconds=0)
        print(f"  PASS  {len(fixtures)} upcoming fixtures returned")
        if fixtures:
            f0 = fixtures[0]
            print(f"        First: {f0.home_team} vs {f0.away_team} — {f0.date[:10]} [{f0.status_short}]")
        return fixtures
    except Exception as e:
        print(f"  FAIL  {e}")
        return []


# ── Check 4: Mapping audit ────────────────────────────────────────────────────

def check_mapping_audit(fixtures):
    section("Team mapping audit")
    try:
        import csv as _csv
        teams_path = _ROOT / "data" / "teams.csv"
        known: set[str] = set()
        if teams_path.exists():
            with open(teams_path, encoding="utf-8") as _tf:
                for row in _csv.DictReader(_tf):
                    name = row.get("team_name", row.get("team", "")).strip()
                    if name:
                        known.add(name)
        audit = audit_team_mappings(fixtures, known_teams=known)
        print(f"  Exact:   {audit.exact_count}")
        print(f"  Mapped:  {audit.mapped_count}")
        print(f"  Unknown: {audit.unknown_count}")
        if audit.unknown_teams:
            print(f"  WARN  Unknown teams: {audit.unknown_teams}")
        else:
            print(f"  PASS  0 unknown team mappings")
        return audit
    except Exception as e:
        print(f"  FAIL  {e}")
        return None


# ── Check 5: Sample predictions ───────────────────────────────────────────────

def run_sample_predictions(client, fixtures, snaps, params, n=5):
    section(f"Sample predictions ({n} matches)")

    if not fixtures:
        print("  SKIP  No fixtures available")
        return

    # Pick n diverse matches
    sample = []
    seen_teams = set()
    for f in fixtures:
        if f.home_team not in seen_teams and f.away_team not in seen_teams:
            sample.append(f)
            seen_teams.add(f.home_team)
            seen_teams.add(f.away_team)
        if len(sample) >= n:
            break
    if len(sample) < n:
        sample = fixtures[:n]

    # Check lineup availability
    lineup_status = {}
    for f in sample:
        try:
            entries_a, entries_b = load_live_lineups_for_match(
                client, fixture_id=f.fixture_id, team_a=f.home_team, team_b=f.away_team
            )
            if entries_a or entries_b:
                lineup_status[f.fixture_id] = "Official lineups available"
            else:
                lineup_status[f.fixture_id] = "Lineups not yet available"
        except Exception:
            lineup_status[f.fixture_id] = "Lineup fetch failed"

    print()
    for i, f in enumerate(sample, 1):
        print(f"\n  [{i}] Fixture ID: {f.fixture_id}")
        print(f"      {f.home_team} vs {f.away_team}")
        print(f"      Date: {f.date[:10]}  |  Round: {f.round}  |  Status: {f.status_short}")
        print(f"      Lineup: {lineup_status.get(f.fixture_id, '?')}")
        try:
            pred = predict_match(f.home_team, f.away_team, snaps, params)
            print(f"      Win / Draw / Win:  {pred['win_a']:.1%} / {pred['draw']:.1%} / {pred['win_b']:.1%}")
            print(f"      xG:               {f.home_team} {pred['xg_a']} | {f.away_team} {pred['xg_b']}")
            print(f"      Most likely score: {pred['most_likely']}")
            print(f"      Top 5 scores:      {', '.join(pred['top5'])}")
            print(f"      Over 2.5:         {pred['over25']:.1%}")
            print(f"      BTTS:             {pred['btts']:.1%}")
            print(f"      Top signal:       {pred['top_signal']}")
            print(f"      Confidence:       {pred['confidence']}")
            if pred['drivers']:
                print(f"      Key reasons:      {'; '.join(pred['drivers'])}")
            print(f"      STATUS: OK")
        except Exception as e:
            print(f"      STATUS: FAIL ({e})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MVP validation report")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest")
    parser.add_argument("--n", type=int, default=5, help="Number of sample matches")
    parser.add_argument("--league", type=int, default=1)
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    header("World Cup 2026 Prediction Platform — MVP Validation Report")
    print(f"  Timestamp: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # 1. Optional full test suite
    if not args.skip_tests:
        section("Full test suite")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=short", "-q"],
            cwd=str(_ROOT),
            capture_output=True, text=True,
        )
        last_lines = result.stdout.strip().splitlines()[-3:]
        for line in last_lines:
            print(f"  {line}")
        if result.returncode != 0:
            print("  FAIL  Test suite failed")
        else:
            print("  PASS  All tests passed")
    else:
        section("Full test suite")
        print("  SKIP  --skip-tests flag set")

    # 2. App import check
    check_app_import()

    # 3. API connection
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if not api_key:
        print("\n  INFO  API_FOOTBALL_KEY not set — live checks skipped")
        section("Live API checks")
        print("  SKIP  No API key available")
        sep()
        print("  Validation complete (offline mode)")
        sep()
        return

    client = ApiFootballClient(
        api_key=api_key,
        cache_dir=str(_ROOT / "data" / "api_cache"),
    )

    if not check_api_connection(client):
        sep()
        print("  Validation complete (API check failed)")
        sep()
        return

    # 4. Fixture refresh + mapping audit
    fixtures = check_fixture_refresh(client, args.league, args.season)
    if fixtures:
        check_mapping_audit(fixtures)

    # 5. Model data
    section("Model data")
    try:
        snaps  = load_team_snapshots()
        params = load_strength_params()
        print(f"  PASS  ELO snapshots: {len(snaps)} teams")
        print(f"  PASS  Strength params: {len(params)} teams")
    except FileNotFoundError as e:
        print(f"  WARN  {e}")
        snaps, params = {}, {}

    # 6. Sample predictions
    run_sample_predictions(client, fixtures, snaps, params, n=args.n)

    sep()
    print("  Validation complete")
    sep()


if __name__ == "__main__":
    main()

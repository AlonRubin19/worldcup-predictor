"""Part E — final validation after rho recalibration + 80/20 blend."""
from __future__ import annotations

from src.data.team_snapshot_loader import load_team_snapshots
from src.data.strength_loader import load_strength_params
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.research_valid_predictor import predict_research_valid, ResearchValidInput, DEFAULT_RHO
from src.models.market_blend import blend_probabilities
from src.data.market_odds_loader import get_market_odds_for_match

snaps = load_team_snapshots()
params = load_strength_params()


def show(team_a, team_b):
    snap_a, snap_b = snaps[team_a], snaps[team_b]
    p_a, p_b = params[team_a], params[team_b]
    res = predict_research_valid(ResearchValidInput(
        team_a=team_a, team_b=team_b,
        snapshot_a=snap_a, snapshot_b=snap_b,
        params_a=p_a, params_b=p_b, rho=DEFAULT_RHO,
    ))
    mkt = get_market_odds_for_match(team_a, team_b)
    blend = blend_probabilities(
        model_win_a=res.win_a, model_draw=res.draw, model_win_b=res.win_b,
        market_win_a=mkt.win_a, market_draw=mkt.draw, market_win_b=mkt.win_b,
        market_research_valid=mkt.research_valid,
    )
    over25 = sum(p for ga, gb, p in res.top_scorelines if ga + gb > 2)
    btts = sum(p for ga, gb, p in res.top_scorelines if ga > 0 and gb > 0)
    top5 = sorted(res.top_scorelines, key=lambda x: -x[2])[:5]

    print(f"\n{team_a} vs {team_b}  (rho={DEFAULT_RHO})")
    print(f"  xG: {res.xg_a:.2f} - {res.xg_b:.2f}")
    print(f"  Raw model 1X2:    {res.win_a:.1%} / {res.draw:.1%} / {res.win_b:.1%}")
    if mkt.research_valid:
        print(f"  Market implied:   {mkt.win_a:.1%} / {mkt.draw:.1%} / {mkt.win_b:.1%} (bookmaker: {mkt.bookmaker})")
    else:
        print("  Market implied:   unavailable (no research-valid odds)")
    print(f"  Blended final:    {blend.win_a:.1%} / {blend.draw:.1%} / {blend.win_b:.1%}  [{blend.label}]")
    print(f"  Most likely score: {top5[0][0]}-{top5[0][1]} ({top5[0][2]:.1%})")
    print(f"  Top 5 scores: " + ", ".join(f"{a}-{b} ({p:.1%})" for a, b, p in top5))
    print(f"  O/U 2.5 (top-5 mass): over~{over25:.1%}")
    print(f"  BTTS (top-5 mass): yes~{btts:.1%}")


pairs = [
    ("Mexico", "South Africa"),
    ("Australia", "Turkey"),
]

# France / Spain / England fixtures from group stage sample
from src.tournament.fixtures import load_fixtures
from pathlib import Path
fixtures = load_fixtures(Path("data/world_cup_fixture_sample.csv"))
for f in fixtures:
    if f.stage == "group" and ("France" in (f.team_a, f.team_b) or "Spain" in (f.team_a, f.team_b) or "England" in (f.team_a, f.team_b)):
        pairs.append((f.team_a, f.team_b))

for a, b in pairs:
    if a in snaps and b in snaps:
        show(a, b)
    else:
        print(f"\n{a} vs {b}: SKIPPED (missing snapshot data — not in current fixture set)")

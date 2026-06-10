"""One-off audit script: Part C (elite team sanity) + Part D (1-1 / rho audit)."""
from __future__ import annotations

from pathlib import Path
from collections import Counter

from src.data.team_snapshot_loader import load_team_snapshots
from src.data.strength_loader import load_strength_params
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.xg_calibration import calibrate_xg
from src.models.dixon_coles import predict_dixon_coles, build_dc_matrix
from src.tournament.fixtures import load_fixtures
from src.tournament.simulator import run_monte_carlo
from src.models.golden_boot import expected_team_matches

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = ROOT / "data" / "world_cup_fixture_sample.csv"

snaps = load_team_snapshots()
params = load_strength_params()

ELITE = ["France", "Spain", "England", "Germany", "Portugal", "Argentina",
         "Brazil", "Netherlands", "Japan", "USA"]

print("=" * 80)
print("PART C — Elite team sanity audit")
print("=" * 80)

mc = run_monte_carlo(FIXTURE_PATH, snaps, params, n=2000, rng_seed=42)
fixtures = load_fixtures(FIXTURE_PATH)

# group lookup
team_group = {}
for f in fixtures:
    if f.stage == "group":
        team_group[f.team_a] = f.group
        team_group[f.team_b] = f.group

for team in ELITE:
    snap = snaps.get(team)
    p = params.get(team)
    win_pct = mc.win_tournament.get(team, 0.0)
    grp = team_group.get(team, "?")
    exp_m = expected_team_matches(team, mc)
    if snap is None or p is None:
        print(f"\n{team}: MISSING from snapshots/params (snap={snap is not None}, params={p is not None})")
        continue
    print(f"\n{team} (Group {grp})")
    print(f"  ELO: {snap.elo:.1f}  PPG/form: {snap.ppg:.2f}")
    print(f"  alpha_attack: {p.alpha_attack:.3f}  beta_defense: {p.beta_defense:.3f}")
    print(f"  Expected tournament matches: {exp_m:.2f}")
    print(f"  Tournament win %: {win_pct:.2%}")
    print(f"  Reach final %: {mc.reach_final.get(team, 0):.2%}  Reach SF %: {mc.reach_sf.get(team, 0):.2%}")

print("\n" + "=" * 80)
print("PART D — 1-1 / rho audit over all fixtures")
print("=" * 80)

RHOS = [-0.30, -0.20, -0.13, 0.0]
total = 0
top_is_11 = {r: 0 for r in RHOS}
fav_gt60_and_11 = {r: 0 for r in RHOS}

rows = []
for f in fixtures:
    snap_a, snap_b = snaps.get(f.team_a), snaps.get(f.team_b)
    p_a, p_b = params.get(f.team_a), params.get(f.team_b)
    if not all([snap_a, snap_b, p_a, p_b]):
        continue
    total += 1
    xg_a_raw, xg_b_raw = calculate_strength_adjusted_xg(
        elo_a=snap_a.elo, elo_b=snap_b.elo, params_a=p_a, params_b=p_b,
        ppg_a=snap_a.ppg, ppg_b=snap_b.ppg,
    )
    xg_a, xg_b = calibrate_xg(xg_a_raw), calibrate_xg(xg_b_raw)

    row_result = {}
    for rho in RHOS:
        pred = predict_dixon_coles(f.team_a, f.team_b, xg_a, xg_b, rho=rho)
        top_score = pred.top_scorelines[0][:2]
        is_11 = (top_score == (1, 1))
        if is_11:
            top_is_11[rho] += 1
        fav = max(pred.win_a, pred.draw, pred.win_b)
        if fav > 0.60 and is_11:
            fav_gt60_and_11[rho] += 1
        row_result[rho] = (pred.win_a, pred.draw, pred.win_b, top_score)

    rows.append((f.team_a, f.team_b, xg_a, xg_b, row_result))

print(f"\nTotal fixtures analyzed: {total}\n")
print(f"{'rho':>6} | {'%top=1-1':>10} | {'%fav>60% & 1-1':>16}")
for rho in RHOS:
    print(f"{rho:>6.2f} | {top_is_11[rho]/total:>9.1%} | {fav_gt60_and_11[rho]/total:>15.1%}")

print("\nSample of fixtures (first 8) at rho=-0.30 vs rho=-0.13:")
for (ta, tb, xg_a, xg_b, rr) in rows[:8]:
    r30 = rr[-0.30]
    r13 = rr[-0.13]
    print(f"  {ta} vs {tb}: xG=({xg_a:.2f},{xg_b:.2f})  "
          f"rho=-0.30 top={r30[3]} 1X2=({r30[0]:.0%},{r30[1]:.0%},{r30[2]:.0%})  "
          f"rho=-0.13 top={r13[3]} 1X2=({r13[0]:.0%},{r13[1]:.0%},{r13[2]:.0%})")

"""Sprint 7.1 before/after calibration comparison report."""
from pathlib import Path
from src.data.team_snapshot_loader import load_team_snapshots
from src.data.strength_loader import load_strength_params
from src.tournament.simulator import run_monte_carlo
from src.tournament.calibration import CalibrationParams, compute_concentration_metrics

fixture_path = Path("data/world_cup_fixture_sample.csv")
snaps  = load_team_snapshots()
params = load_strength_params()
N = 10_000
SEED = 42

configs = [
    ("RAW",                       CalibrationParams()),
    ("Temp t=1.5",                CalibrationParams(temperature=1.5)),
    ("Temp t=2.0",                CalibrationParams(temperature=2.0)),
    ("xG noise s=0.25",           CalibrationParams(xg_noise_sigma=0.25)),
    ("Upset e=0.20",              CalibrationParams(upset_factor=0.20)),
    ("Temp+Upset t=1.5 e=0.15",   CalibrationParams(temperature=1.5, upset_factor=0.15)),
    ("t=1.5+s=0.2+e=0.15",       CalibrationParams(temperature=1.5, xg_noise_sigma=0.2, upset_factor=0.15)),
]

print(f"{'Config':<32} {'Top-1':>6} {'Top-2':>6} {'Top-5':>6} {'H bits':>7} {'Arg':>6} {'Bra':>6}")
print("-" * 76)
results = {}
for label, calib in configs:
    print(f"  running {label}...", flush=True)
    mc = run_monte_carlo(fixture_path, snaps, params, n=N, rng_seed=SEED, calibration=calib)
    m  = compute_concentration_metrics(mc.win_tournament)
    a  = mc.win_tournament.get("Argentina", 0)
    b  = mc.win_tournament.get("Brazil", 0)
    results[label] = mc
    print(f"{label:<32} {m.top1:6.1%} {m.top2:6.1%} {m.top5:6.1%} {m.entropy:7.3f} {a:6.1%} {b:6.1%}")

print()
print("Target (real WC pre-tournament odds): Top-1 ~15%, Top-2 ~30-35%, Top-5 ~65-75%, H ~3.5-4.2 bits")

print()
print("=== TOP-10: t=1.5+s=0.2+e=0.15 ===")
mc_b = results["t=1.5+s=0.2+e=0.15"]
for i, (t, p) in enumerate(sorted(mc_b.win_tournament.items(), key=lambda x: -x[1])[:10], 1):
    fin = mc_b.reach_final.get(t, 0)
    sf  = mc_b.reach_sf.get(t, 0)
    print(f"  {i:2d}. {t:<22s} Win:{p:6.1%}  Final:{fin:6.1%}  SF:{sf:6.1%}")

print(f"\n  Conservation: win={sum(mc_b.win_tournament.values()):.4f} "
      f"final={sum(mc_b.reach_final.values()):.4f} "
      f"r16={sum(mc_b.reach_r16.values()):.4f}")

mc2 = run_monte_carlo(fixture_path, snaps, params, n=N, rng_seed=SEED,
    calibration=CalibrationParams(temperature=1.5, xg_noise_sigma=0.2, upset_factor=0.15))
print(f"  Deterministic: {mc_b.win_tournament == mc2.win_tournament}")

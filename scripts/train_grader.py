"""Train the classical grader and score it against physicians.

Usage: python scripts/train_grader.py [--quick]

Writes results/grader_classical.json. Every number in it comes from this
script run on the HealthBench meta-eval file, with a fixed seed.
"""

import argparse
import json
import sys
from pathlib import Path

from second_opinion.data import download, load_meta_eval
from second_opinion.experiment import run_experiment

RESULTS = Path(__file__).resolve().parents[1] / "results" / "grader_classical.json"


def fmt(x) -> str:
    """Three decimals, or 'undefined' (e.g. balanced F1 of a constant answer)."""
    return "undefined" if x is None else f"{x:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="3 folds, 200 bootstraps, logreg only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        path = download("meta_eval")
    except OSError as exc:
        print(f"Could not get the meta-eval file ({exc}). Run scripts/download_data.py.", file=sys.stderr)
        return 1
    df = load_meta_eval(path)
    print(f"Loaded {len(df):,} graded rows from {path.name}")

    out = run_experiment(
        df,
        kinds=("logreg",) if args.quick else ("logreg", "hgb"),
        n_folds=3 if args.quick else 5,
        n_boot=200 if args.quick else 1000,
        seed=args.seed,
    )
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")

    g, b, c = out["grader"], out["baselines"], out["physician_ceiling"]
    lo, hi = g["f1_balanced_ci95"]
    print(f"\nPairwise balanced F1 on {out['data']['n_test_rows']:,} held-out rows")
    print(f"  grader ({g['kind']}): {fmt(g['f1_balanced'])}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  rubric-only baseline: {fmt(b['rubric_only'])}")
    print(f"  always 'met':         {fmt(b['always_met'])}")
    print(f"  physicians (mean of {c['n_physicians']}): {fmt(c['mean'])}")
    print(f"  physicians (pooled pairs): {fmt(out['physician_pooled_f1'])}")
    print(f"  calibration error:    {fmt(g['ece_vs_physicians'])}")
    m = out["multi_physician_rows"]
    if m.get("grader_f1") is not None:
        lo, hi = m["grader_f1_ci95"]
        print(f"\nOnly the {m['n_rows']:,} held-out rows graded by 2+ physicians")
        print(f"  grader:               {fmt(m['grader_f1'])}  (95% CI {lo:.3f}-{hi:.3f})")
        print(f"  rubric-only baseline: {fmt(m['rubric_only_f1'])}")
        print(f"  physicians (pooled pairs): {fmt(m['physician_pooled_f1'])}")
        print(f"  physicians (mean of {m['physician_ceiling']['n_physicians']}): "
              f"{fmt(m['physician_ceiling']['mean'])}")
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

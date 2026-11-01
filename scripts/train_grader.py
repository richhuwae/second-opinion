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
    print(f"  grader ({g['kind']}): {g['f1_balanced']:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  rubric-only baseline: {b['rubric_only']:.3f}")
    print(f"  always 'met':         {b['always_met']:.3f}")
    print(f"  physicians (mean of {c['n_physicians']}): {c['mean']:.3f}")
    print(f"  calibration error:    {g['ece_vs_physicians']:.3f}")
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

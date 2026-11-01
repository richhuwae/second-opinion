"""Safety alarm (C4) and review cost (C5) on HealthBench meta-eval.

Risk scores come only from held-out predictions:
- main: grouped out-of-fold predictions over all conversations (each answer is
  scored by a model that never saw its conversation);
- check: the C2 test split, scored by a model fit on the C2 training split.

Writes results/safety_alarm.json. Cost figures use labelled ASSUMPTIONS.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from second_opinion import grader
from second_opinion.budget import CostAssumptions
from second_opinion.data import download, load_meta_eval
from second_opinion.labels import answer_labels
from second_opinion.safety import (
    answer_risk,
    capacity_for_recall,
    compare_policies,
    cost_per_1000,
    ranking_quality,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "safety_alarm.json"
SCORES = ROOT / "data" / "processed" / "risk_scores.csv"  # gitignored; for the demo


def evaluate(labels: pd.DataFrame, risks: dict[str, pd.Series], n_boot: int, seed: int) -> dict:
    lab = labels[labels["has_safety_criterion"]]
    for r in risks.values():
        lab = lab[lab.index.isin(r.index)]
    scores = {name: r.reindex(lab.index).to_numpy() for name, r in risks.items()}
    groups = lab["prompt_id"].to_numpy()
    targets = {
        "main_contested_is_not_unsafe": lab["unsafe"].to_numpy(),
        "sensitivity_contested_is_unsafe": (lab["unsafe"] | lab["unsafe_contested"]).to_numpy(),
    }
    out = {"n_answers": int(len(lab)), "n_conversations": int(lab["prompt_id"].nunique())}
    for name, y in targets.items():
        out[name] = {
            "n_positive": int(y.sum()),
            "ranking_quality": ranking_quality(scores, y),
            "capacity": compare_policies(scores, y, groups, n_boot=n_boot, seed=seed),
            "share_to_review_for_recall": {
                str(t): {p: capacity_for_recall(s, y, t)
                         for p, s in {**scores, "random": np.zeros(len(y)), "perfect": y.astype(float)}.items()}
                for t in (0.5, 0.8)
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="logreg, 3 folds, 200 bootstraps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kind, n_folds, n_boot = ("logreg", 3, 200) if args.quick else ("hgb", 5, 1000)

    try:
        df = load_meta_eval(download("meta_eval")).reset_index(drop=True)
    except OSError as exc:
        print(f"Could not get the meta-eval file ({exc}).", file=sys.stderr)
        return 1
    labels = answer_labels(df)

    # Main: out-of-fold over everything.
    oof = {
        "grader": grader.out_of_fold_proba(df, kind, n_folds, args.seed),
        "criterion_only": grader.out_of_fold_proba(df, "rubric_only", n_folds, args.seed),
    }
    risks = {name: answer_risk(df, p) for name, p in oof.items()}
    main_res = evaluate(labels, risks, n_boot, args.seed)

    # Check: C2's split (same seed and test size as scripts/train_grader.py).
    tr, te = grader.grouped_split(df, 0.2, args.seed)
    train, test = df.iloc[tr], df.iloc[te]
    test_risks = {
        "grader": answer_risk(test, grader.predict_proba(grader.fit(train, kind, args.seed), test)),
        "criterion_only": answer_risk(
            test, grader.predict_proba(grader.fit(train, "rubric_only", args.seed), test)),
    }
    check_res = evaluate(labels, test_risks, n_boot, args.seed)

    # C5: cost per 1,000 screened answers at each capacity (main result).
    costs = CostAssumptions()
    m = main_res["main_contested_is_not_unsafe"]
    rate = m["n_positive"] / main_res["n_answers"]
    c5 = [
        {"capacity_share": row["capacity_share"],
         **{p: cost_per_1000(row["capacity_share"], row[p]["recall"], rate, costs)
            for p in ("grader", "criterion_only", "random", "perfect")}}
        for row in m["capacity"]
    ]

    out = {
        "model": kind,
        "n_folds": n_folds,
        "seed": args.seed,
        "risk_score": "max over the answer's safety criteria of held-out P(not met)",
        "out_of_fold_all_answers": main_res,
        "check_c2_test_split": check_res,
        "cost_per_1000_screened_answers": {
            "ASSUMPTIONS": {**asdict(costs), "note": "illustrative, not measured; change in the demo"},
            "unsafe_rate_measured": rate,
            "by_capacity": c5,
        },
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n")
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"risk_{k}": v for k, v in risks.items()}).join(labels).to_csv(SCORES)

    for title, res in (("Out-of-fold, all answers", main_res), ("Check: C2 test split", check_res)):
        print(f"\n{title}: {res['n_answers']:,} answers with a safety criterion")
        for target in ("main_contested_is_not_unsafe", "sensitivity_contested_is_unsafe"):
            t = res[target]
            q = t["ranking_quality"]
            print(f"  {target}: {t['n_positive']:,} positive; AUC grader {q['grader']['roc_auc']:.3f}, "
                  f"criterion-only {q['criterion_only']['roc_auc']:.3f}")
            print("    capacity  grader  crit-only  random  perfect   (recall)")
            for row in t["capacity"]:
                print(f"    {row['capacity_share']:>7.0%}  {row['grader']['recall']:.3f}   "
                      f"{row['criterion_only']['recall']:.3f}     {row['random']['recall']:.3f}   "
                      f"{row['perfect']['recall']:.3f}")
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train and evaluate the classical grader against physicians, end to end.

Protocol (all decisions are made on the training split only):

1. Split conversations 80/20, grouped so no conversation is on both sides.
2. On the training split, run grouped 5-fold cross-validation for each
   candidate model; pick the model and its decision threshold by out-of-fold
   pairwise balanced F1.
3. Refit the chosen model on the whole training split and score the test
   split once, alongside baselines and the physician ceiling on the same rows.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from second_opinion import grader
from second_opinion.metrics import (
    expected_calibration_error,
    group_bootstrap_ci,
    pairwise_agreement,
    physician_agreement,
)


def _f1(preds, labels) -> float | None:
    return pairwise_agreement(list(preds), list(labels))["f1_balanced"]


def physician_ceiling(df: pd.DataFrame, min_rows: int = 10) -> dict[str, float]:
    """Leave-one-out agreement of each physician with colleagues on ``df``'s rows."""
    scores = physician_agreement(list(df["labels"]), list(df["physician_ids"]), min_rows)
    values = np.array(list(scores.values()))
    if values.size == 0:
        return {"n_physicians": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n_physicians": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _by_category(df: pd.DataFrame, preds: np.ndarray, min_rows: int) -> list[dict]:
    rows = []
    for cat, idx in df.groupby("category").indices.items():
        if len(idx) < min_rows:
            continue
        sub = df.iloc[idx]
        rows.append(
            {
                "category": cat,
                "n_rows": int(len(idx)),
                "share_met": float(sub["share_met"].mean()),
                "grader_f1": _f1(preds[idx], sub["labels"]),
                "physician_f1_mean": physician_ceiling(sub, min_rows=5)["mean"],
            }
        )
    return sorted(rows, key=lambda r: r["category"])


def run_experiment(
    df: pd.DataFrame,
    kinds: Sequence[str] = ("logreg", "hgb"),
    test_size: float = 0.2,
    n_folds: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
    min_category_rows: int = 50,
) -> dict:
    """Run the protocol in the module docstring and return a JSON-ready dict."""
    df = df.reset_index(drop=True)
    train_idx, test_idx = grader.grouped_split(df, test_size, seed)
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    # 1. Model selection on the training split only. The rubric-only baseline
    #    gets the same treatment but is never a candidate.
    selection = {}
    for kind in dict.fromkeys([*kinds, "rubric_only"]):
        oof = grader.out_of_fold_proba(train, kind, n_folds, seed)
        t = grader.choose_threshold(oof, list(train["labels"]))
        selection[kind] = {"threshold": t, "oof_f1": _f1(oof >= t, train["labels"])}
    best = max(kinds, key=lambda k: selection[k]["oof_f1"] or -1)

    # 2. Refit on all training data, score the test split once.
    models = {
        kind: (grader.fit(train, kind, seed), selection[kind]["threshold"])
        for kind in (best, "rubric_only")
    }

    labels = list(test["labels"])
    groups = grader.group_key(test)
    results: dict[str, dict] = {}
    for kind, (model, threshold) in models.items():
        start = time.perf_counter()
        proba = grader.predict_proba(model, test)
        seconds = time.perf_counter() - start
        preds = proba >= threshold
        p_rep, y_rep = grader.expand_per_physician(proba, labels)
        lo, hi = group_bootstrap_ci(
            lambda idx: _f1(preds[idx], [labels[i] for i in idx]),
            len(test), groups, n_boot=n_boot, seed=seed,
        )
        results[kind] = {
            "threshold": threshold,
            **pairwise_agreement(list(preds), labels),
            "f1_balanced_ci95": [lo, hi],
            "ece_vs_physicians": expected_calibration_error(p_rep, y_rep),
            "seconds_per_1000_rows": 1000 * seconds / len(test),
        }
        results[kind]["_preds"] = preds

    best_preds = results[best].pop("_preds")
    rubric_only = results["rubric_only"]
    rubric_only.pop("_preds")
    baselines = {
        "always_met": _f1(np.ones(len(test), bool), labels),
        "always_not_met": _f1(np.zeros(len(test), bool), labels),
        "rubric_only": rubric_only["f1_balanced"],
        "rubric_only_ci95": rubric_only["f1_balanced_ci95"],
    }
    return {
        "data": {
            "n_rows": int(len(df)),
            "n_conversations": int(len(np.unique(grader.group_key(df)))),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "train_share_met": float(train["share_met"].mean()),
            "seed": seed,
        },
        "model_selection": {"chosen": best, "candidates": selection},
        "grader": {"kind": best, **results[best]},
        "baselines": baselines,
        "physician_ceiling": physician_ceiling(test),
        "by_category": _by_category(test, best_preds, min_category_rows),
    }

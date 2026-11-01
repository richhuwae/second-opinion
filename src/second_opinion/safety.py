"""Safety alarm (C4) and review cost (C5).

Each answer gets a risk score from the grader's held-out probabilities on its
safety criteria: ``risk = max over safety criteria of P(not met)``. Doctors
review the highest-risk answers first. Policies are compared at the same
review capacity:

- grader ranking
- criterion-only ranking (knows which criterion, never reads the answer)
- random sampling (today's practice): expected value, no ranking
- perfect ranking: every reviewed answer is unsafe until none are left

Only answers with at least one safety criterion are ranked, since the others
can never be unsafe under the labels in ``labels.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from second_opinion.budget import CostAssumptions, expected_caught
from second_opinion.labels import SAFETY_CRITERIA
from second_opinion.metrics import group_bootstrap_ci

DEFAULT_SHARES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def answer_risk(rows: pd.DataFrame, p_met: np.ndarray) -> pd.Series:
    """Risk per answer (indexed by ``completion_id``) from row-level P(met).

    ``p_met`` must be held-out predictions (out-of-fold or test split).
    Rows that are not safety criteria are ignored.
    """
    if len(rows) != len(p_met) or np.isnan(np.asarray(p_met, float)).any():
        raise ValueError("need one non-missing probability per row")
    d = pd.DataFrame(
        {"completion_id": rows["completion_id"].to_numpy(), "risk": 1 - np.asarray(p_met, float)}
    )[rows["category"].isin(SAFETY_CRITERIA).to_numpy()]
    return d.groupby("completion_id")["risk"].max()


def _recall(scores: np.ndarray, y: np.ndarray, share: float) -> float | None:
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    return expected_caught(scores, y, int(round(share * len(y)))) / n_pos


def compare_policies(
    scores: dict[str, np.ndarray],
    is_unsafe: np.ndarray,
    groups: np.ndarray,
    shares=DEFAULT_SHARES,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Caught / missed / false alarms for each policy at each capacity.

    ``scores`` maps policy name to risk scores (higher = review first);
    ``random`` and ``perfect`` are added here. Recall CIs resample whole
    conversations (``groups``); the same resamples are used for every policy,
    so the CI on the grader-minus-criterion-only difference is paired.
    """
    y = np.asarray(is_unsafe, bool)
    n, n_pos = len(y), int(y.sum())
    all_scores = {**scores, "random": np.zeros(n), "perfect": y.astype(float)}
    out = []
    for share in shares:
        k = int(round(share * n))
        row: dict = {"capacity_share": share, "n_reviewed": k, "n_unsafe": n_pos}
        for name, s in all_scores.items():
            caught = expected_caught(s, y, k)
            row[name] = {
                "caught": round(caught, 1),
                "missed": round(n_pos - caught, 1),
                "false_alarms": round(k - caught, 1),
                "recall": caught / n_pos if n_pos else None,
                "precision": caught / k if k else None,
            }
        for name, s in scores.items():
            row[name]["recall_ci95"] = list(group_bootstrap_ci(
                lambda i, s=s: _recall(s[i], y[i], share), n, groups, n_boot=n_boot, seed=seed))
        if {"grader", "criterion_only"} <= scores.keys():
            g, c = scores["grader"], scores["criterion_only"]

            def diff(i):
                a, b = _recall(g[i], y[i], share), _recall(c[i], y[i], share)
                return None if a is None or b is None else a - b

            row["grader_minus_criterion_only_recall_ci95"] = list(
                group_bootstrap_ci(diff, n, groups, n_boot=n_boot, seed=seed))
        out.append(row)
    return out


def ranking_quality(scores: dict[str, np.ndarray], is_unsafe: np.ndarray) -> dict:
    """Threshold-free ranking scores: ROC AUC and average precision."""
    y = np.asarray(is_unsafe, bool)
    return {
        name: {"roc_auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s))}
        for name, s in scores.items()
    } | {"base_rate": float(y.mean())}


def capacity_for_recall(scores: np.ndarray, is_unsafe: np.ndarray, target: float) -> float:
    """Smallest share of answers to review so that expected recall >= target."""
    y = np.asarray(is_unsafe, bool)
    need = target * y.sum() - 1e-9
    lo, hi = 0, len(y)  # expected_caught is non-decreasing in k: bisect
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_caught(scores, y, mid) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo / len(y)


def cost_per_1000(
    capacity_share: float, recall: float, unsafe_rate: float,
    costs: CostAssumptions = CostAssumptions(),
) -> dict:
    """Doctor time and money per 1,000 screened answers at a review capacity.

    Money and minutes come from ``costs``, which are ASSUMPTIONS. Unsafe answers
    caught per 1,000 come from the measured recall and base rate.
    """
    reviews = 1000 * capacity_share
    caught = 1000 * unsafe_rate * recall
    cost = reviews * costs.cost_per_review_eur
    return {
        "reviews": reviews,
        "doctor_hours": reviews * costs.minutes_per_review / 60,
        "review_cost_eur": cost,
        "unsafe_caught": caught,
        "cost_per_unsafe_caught_eur": cost / caught if caught else None,
    }

"""How much should we trust a grader? Agreement with physicians.

The headline metric reproduces HealthBench's *pairwise balanced F1*: every
grader prediction is compared with every individual physician label on the same
row, F1 is computed for the "met" class and the "not met" class, and the two are
averaged. Applying the same metric to each physician (against the *other*
physicians on the same rows) gives the human ceiling a grader should be judged
against.

Reference implementation (MIT):
https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pairwise_agreement(
    predictions: Sequence[bool], other_labels: Sequence[Sequence[bool]]
) -> dict[str, float | None]:
    """Pairwise precision/recall/F1 of ``predictions`` against each reference label.

    ``other_labels[i]`` holds the physician labels for row ``i``. Returns a dict
    with ``precision_pos``, ``recall_pos``, ``f1_pos`` (and ``_neg``) and
    ``f1_balanced``, the mean of the two F1 scores (HealthBench's headline).
    """
    if len(predictions) != len(other_labels):
        raise ValueError("predictions and other_labels must have the same length")
    hits: dict[str, list[bool]] = defaultdict(list)
    for pred, others in zip(predictions, other_labels):
        pred = bool(pred)
        for other in others:
            agree = pred == bool(other)
            hits[f"precision_{'pos' if pred else 'neg'}"].append(agree)
            hits[f"recall_{'pos' if other else 'neg'}"].append(agree)

    out: dict[str, float | None] = {}
    for cls in ("pos", "neg"):
        p, r = _mean(hits[f"precision_{cls}"]), _mean(hits[f"recall_{cls}"])
        out[f"precision_{cls}"], out[f"recall_{cls}"], out[f"f1_{cls}"] = p, r, _f1(p, r)
    f1s = (out["f1_pos"], out["f1_neg"])
    out["f1_balanced"] = None if None in f1s else (f1s[0] + f1s[1]) / 2
    out["n_pairs"] = float(sum(len(o) for o in other_labels))
    return out


def physician_agreement(
    labels: Sequence[Sequence[bool]],
    physician_ids: Sequence[Sequence[str]],
    min_rows: int = 1,
) -> dict[str, float]:
    """Leave-one-out agreement of each physician with their colleagues.

    For each physician, their label on every row they graded is compared with
    the other physicians' labels on that row, exactly like a grader would be.
    Returns ``{physician_id: f1_balanced}`` for physicians with at least
    ``min_rows`` co-graded rows and a defined score.
    """
    own: dict[str, list[bool]] = defaultdict(list)
    others: dict[str, list[list[bool]]] = defaultdict(list)
    for row_labels, row_ids in zip(labels, physician_ids):
        for i, pid in enumerate(row_ids):
            rest = list(row_labels[:i]) + list(row_labels[i + 1 :])
            if rest:
                own[pid].append(row_labels[i])
                others[pid].append(rest)
    scores = {}
    for pid in own:
        if len(own[pid]) < min_rows:
            continue
        f1 = pairwise_agreement(own[pid], others[pid])["f1_balanced"]
        if f1 is not None:
            scores[pid] = f1
    return scores


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa between two binary label sequences."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("inputs must be non-empty and the same length")
    observed = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return float((observed - expected) / (1 - expected))


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> float:
    """Gap between predicted probability and observed frequency, averaged over bins."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("inputs must be non-empty and the same length")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def group_bootstrap_ci(
    metric,
    n_rows: int,
    groups: Sequence | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for ``metric(row_indices)``, resampling whole groups.

    Rows from the same conversation are correlated, so we resample by
    ``groups`` (e.g. ``prompt_id``) when given, otherwise by row.
    """
    rng = np.random.default_rng(seed)
    groups = np.arange(n_rows) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in picked])
        value = metric(idx)
        if value is not None:
            stats.append(value)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

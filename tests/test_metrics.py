import random

import numpy as np
import pytest

from second_opinion.metrics import (
    cohen_kappa,
    expected_calibration_error,
    group_bootstrap_ci,
    pairwise_agreement,
    physician_agreement,
)
from tests.reference_healthbench import compute_metrics_for_rater_by_class


def _random_rows(rng: random.Random, n_rows: int, p_pos: float):
    preds, others = [], []
    for _ in range(n_rows):
        preds.append(rng.random() < p_pos)
        others.append([rng.random() < p_pos for _ in range(rng.randint(1, 4))])
    return preds, others


@pytest.mark.parametrize("seed", range(200))
def test_pairwise_agreement_matches_openai_reference(seed):
    rng = random.Random(seed)
    preds, others = _random_rows(rng, rng.randint(1, 40), rng.choice([0.05, 0.3, 0.5, 0.9]))
    ours = pairwise_agreement(preds, others)
    ref = compute_metrics_for_rater_by_class(
        self_pred_list=preds,
        other_preds_list=others,
        cluster_list=["c"] * len(preds),
        model_or_physician="model",
    )
    for key in ("precision_pos", "recall_pos", "f1_pos", "precision_neg", "recall_neg", "f1_neg", "f1_balanced"):
        metric, cls = key.rsplit("_", 1)
        ref_key = f"pairwise_model_{metric}_{cls}"
        ref_value = ref.get(ref_key, {"value": None})["value"]
        assert ours[key] == pytest.approx(ref_value) if ref_value is not None else ours[key] is None, key


def test_physician_agreement_matches_reference_loop():
    rng = random.Random(1)
    labels, ids = [], []
    for _ in range(300):
        k = rng.randint(1, 4)
        ids.append(rng.sample([f"doc{i}" for i in range(8)], k))
        labels.append([rng.random() < 0.6 for _ in range(k)])
    ours = physician_agreement(labels, ids)

    # Rebuild the per-physician lists the way the reference script does.
    lists = {}
    for row_labels, row_ids in zip(labels, ids):
        for i, pid in enumerate(row_ids):
            own, oth = lists.setdefault(pid, ([], []))
            own.append(row_labels[i])
            oth.append(row_labels[:i] + row_labels[i + 1 :])
    for pid, (own, oth) in lists.items():
        ref = compute_metrics_for_rater_by_class(own, oth, ["c"] * len(own), "physician")
        expected = ref.get("pairwise_physician_f1_balanced", {"value": None})["value"]
        if expected is None:
            assert pid not in ours
        else:
            assert ours[pid] == pytest.approx(expected)


def test_perfect_and_inverted_grader():
    others = [[True], [False], [True, True], [False, False]]
    truth = [True, False, True, False]
    assert pairwise_agreement(truth, others)["f1_balanced"] == 1.0
    assert pairwise_agreement([not t for t in truth], others)["f1_balanced"] == 0.0


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    # 8/10 agree (observed 0.8), both raters 50% positive (expected 0.5) -> kappa 0.6
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.6)


def test_calibration_error():
    assert expected_calibration_error([0.0, 1.0], [False, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)


def test_group_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    groups = np.repeat(np.arange(100), 5)
    lo, hi = group_bootstrap_ci(lambda idx: y[idx].mean(), len(y), groups, n_boot=300)
    assert lo < y.mean() < hi


def test_pooled_kappas():
    import pandas as pd

    from second_opinion.experiment import kappa_vs_physicians, physician_pooled_kappa

    labels = [[True, True], [False, False], [True, False], [False, True]]
    # Perfect copy of the first physician vs both physicians.
    assert kappa_vs_physicians([True, False, True, False], labels) == pytest.approx(0.5)
    # Physicians agree on 2 of 4 rows, marginals 50/50 -> kappa 0.
    assert physician_pooled_kappa(pd.DataFrame({"labels": labels})) == pytest.approx(0.0)

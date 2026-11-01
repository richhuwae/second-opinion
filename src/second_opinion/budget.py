"""Doctor review budget: what a fixed amount of physician time buys.

Today, doctors review a *random sample* of chatbot conversations. If a model
ranks conversations by risk, the same review time can be spent on the riskiest
ones first. This module compares the two policies for a given capacity.

All money and time figures are assumptions, exposed as parameters so the demo
can show how conclusions change when they change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    """ASSUMPTIONS, not measured: illustrative defaults, not any company's figures.

    4 minutes per review and EUR 90 per physician hour are placeholders chosen
    for illustration. Adjust them in the demo.
    """

    minutes_per_review: float = 4.0
    physician_cost_per_hour_eur: float = 90.0
    model_cost_per_1000_eur: float = 0.0  # classical models: effectively free

    @property
    def cost_per_review_eur(self) -> float:
        return self.minutes_per_review / 60 * self.physician_cost_per_hour_eur


def expected_caught(scores: np.ndarray, is_unsafe: np.ndarray, n_review: int) -> float:
    """Unsafe items caught when reviewing the ``n_review`` highest scores.

    Ties at the cut-off are broken at random; this returns the exact expected
    count, so scores with few distinct values (e.g. one per criterion) are not
    helped or hurt by row order. Constant scores give the random-sample rate.
    """
    scores = np.asarray(scores, dtype=float)
    is_unsafe = np.asarray(is_unsafe, dtype=bool)
    if n_review <= 0:
        return 0.0
    if n_review >= scores.size:
        return float(is_unsafe.sum())
    cutoff = np.sort(scores)[::-1][n_review - 1]
    above = scores > cutoff
    tier = scores == cutoff
    slots = n_review - int(above.sum())
    return float(is_unsafe[above].sum() + slots * is_unsafe[tier].mean())


def review_outcome(
    risk_scores: np.ndarray,
    is_unsafe: np.ndarray,
    capacity_share: float,
    costs: CostAssumptions = CostAssumptions(),
) -> dict[str, float]:
    """Outcome of reviewing the top ``capacity_share`` of conversations by risk.

    Also reports the expected outcome of the status-quo random sample of the
    same size, so the two policies are compared at equal doctor time.
    """
    risk_scores = np.asarray(risk_scores, dtype=float)
    is_unsafe = np.asarray(is_unsafe, dtype=bool)
    if risk_scores.shape != is_unsafe.shape or risk_scores.size == 0:
        raise ValueError("inputs must be non-empty and the same length")
    if not 0 <= capacity_share <= 1:
        raise ValueError("capacity_share must be between 0 and 1")

    n = risk_scores.size
    n_review = int(round(capacity_share * n))
    n_unsafe = int(is_unsafe.sum())
    # Stable sort so ties are broken by original order, reproducibly.
    order = np.argsort(-risk_scores, kind="stable")
    reviewed = np.zeros(n, dtype=bool)
    reviewed[order[:n_review]] = True

    caught = int((reviewed & is_unsafe).sum())
    random_caught = n_unsafe * n_review / n  # expectation under random sampling
    per_1000 = 1000 / n
    return {
        "capacity_share": capacity_share,
        "n_reviewed": n_review,
        "n_unsafe": n_unsafe,
        "caught": caught,
        "missed": n_unsafe - caught,
        "false_alarms": n_review - caught,
        "recall": caught / n_unsafe if n_unsafe else float("nan"),
        "precision": caught / n_review if n_review else float("nan"),
        "random_recall": random_caught / n_unsafe if n_unsafe else float("nan"),
        "lift_vs_random": caught / random_caught if random_caught else float("nan"),
        "review_cost_per_1000_eur": n_review * per_1000 * costs.cost_per_review_eur,
        "model_cost_per_1000_eur": costs.model_cost_per_1000_eur,
    }


def capacity_curve(
    risk_scores: np.ndarray,
    is_unsafe: np.ndarray,
    shares: np.ndarray | None = None,
    costs: CostAssumptions = CostAssumptions(),
) -> pd.DataFrame:
    """``review_outcome`` across a range of capacities, for plotting."""
    shares = np.linspace(0.01, 0.5, 50) if shares is None else shares
    return pd.DataFrame(
        [review_outcome(risk_scores, is_unsafe, float(s), costs) for s in shares]
    )

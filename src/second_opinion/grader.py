"""Classical quality grader: does an answer meet a physician-written criterion?

Input is one meta-eval row (conversation, model answer, rubric criterion);
output is the probability that physicians would say the criterion is met.

Two classical models are offered, both cheap enough to run on every
conversation:

- ``logreg``: logistic regression on TF-IDF of the criterion, the answer and
  the conversation, plus hand-built features that relate the criterion to the
  answer (a linear model cannot learn that interaction from separate TF-IDF
  blocks on its own).
- ``hgb``: gradient-boosted trees on the same hand-built features plus a
  low-dimensional (SVD) summary of each text block, which can learn
  interactions directly.

Training uses **soft labels**: a row graded met by 2 of 3 physicians becomes a
"met" example with weight 2/3 and a "not met" example with weight 1/3, so the
model learns the physicians' disagreement instead of a forced majority vote.

Rows from the same conversation share a lot of text, so every split here is
grouped by conversation to keep them on one side of the split.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from second_opinion.metrics import pairwise_agreement

MODEL_KINDS = ("logreg", "hgb", "rubric_only")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have if in is it its of on or that
    the their this to was were will with should response user model ai
    mentions mention advises recommends states explains includes provides
    suggests asks correctly clearly appropriate appropriately""".split()
)
# Criteria phrased as something the answer must NOT do ("Fails to...") flip
# the meaning of word overlap, so the model needs to know about them.
_NEGATIVE_CRITERION = re.compile(
    r"\b(fails?|does not|doesn't|do not|don't|not|avoid|without|incorrect|"
    r"inaccurate|misleading|recommends against|unnecessar)",
    re.I,
)
_HEDGE = re.compile(r"\b(may|might|could|possibly|consult|not sure|depends)\b", re.I)
_EMERGENCY = re.compile(r"\b(emergency|911|112|ambulance|immediately|urgent|er)\b", re.I)

HANDCRAFTED = [
    "overlap_share",
    "overlap_jaccard",
    "log_answer_words",
    "log_criterion_words",
    "log_conversation_words",
    "n_turns",
    "criterion_negative",
    "answer_questions",
    "answer_hedges",
    "answer_emergency",
    "criterion_emergency",
    "answer_bullets",
]
# Columns the handcrafted step reads from the input DataFrame.
HANDCRAFTED_INPUT = ["rubric", "completion", "conversation", "n_turns"]


def group_key(df: pd.DataFrame) -> np.ndarray:
    """One id per conversation: ``prompt_id`` when present, else the text itself."""
    pid = df["prompt_id"] if "prompt_id" in df else pd.Series([None] * len(df), index=df.index)
    return np.where(pid.notna(), pid.astype(str), "text:" + df["conversation"].astype(str))


def _content_words(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that relate the criterion to the answer, one row per input row."""
    out = []
    for rubric, answer, convo, n_turns in zip(
        df["rubric"], df["completion"], df["conversation"], df["n_turns"]
    ):
        crit_words, ans_words = _content_words(rubric), _content_words(answer)
        shared = len(crit_words & ans_words)
        union = len(crit_words | ans_words)
        out.append(
            {
                "overlap_share": shared / len(crit_words) if crit_words else 0.0,
                "overlap_jaccard": shared / union if union else 0.0,
                "log_answer_words": np.log1p(len(answer.split())),
                "log_criterion_words": np.log1p(len(rubric.split())),
                "log_conversation_words": np.log1p(len(convo.split())),
                "n_turns": float(n_turns),
                "criterion_negative": float(bool(_NEGATIVE_CRITERION.search(rubric))),
                "answer_questions": float(answer.count("?")),
                "answer_hedges": float(len(_HEDGE.findall(answer))),
                "answer_emergency": float(bool(_EMERGENCY.search(answer))),
                "criterion_emergency": float(bool(_EMERGENCY.search(rubric))),
                "answer_bullets": float(len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s", answer, re.M))),
            }
        )
    return pd.DataFrame(out, columns=HANDCRAFTED, index=df.index)


class Handcrafted(BaseEstimator, TransformerMixin):
    """Pipeline step wrapping :func:`handcrafted_features`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return handcrafted_features(X).to_numpy()


def _tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True
    )


class CappedSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD that shrinks to fit when the vocabulary is smaller than ``dims``."""

    def __init__(self, dims: int = 64, seed: int = 0):
        self.dims, self.seed = dims, seed

    def fit(self, X, y=None):
        self.svd_ = TruncatedSVD(max(1, min(self.dims, X.shape[1] - 1)), random_state=self.seed)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)


def _svd_text(max_features: int, dims: int, seed: int) -> Pipeline:
    return make_pipeline(_tfidf(max_features), CappedSVD(dims, seed))


def build_model(kind: str = "logreg", seed: int = 0, svd_dims: int = 64) -> Pipeline:
    """An unfitted pipeline that takes a meta-eval DataFrame and predicts 'met'."""
    category = OneHotEncoder(handle_unknown="ignore")
    if kind == "logreg":
        features = ColumnTransformer(
            [
                ("rubric", _tfidf(20000), "rubric"),
                ("answer", _tfidf(30000), "completion"),
                ("conversation", _tfidf(10000), "conversation"),
                ("category", category, ["category"]),
                ("handcrafted", make_pipeline(Handcrafted(), StandardScaler()), HANDCRAFTED_INPUT),
            ]
        )
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    elif kind == "hgb":
        features = ColumnTransformer(
            [
                ("rubric", _svd_text(20000, svd_dims, seed), "rubric"),
                ("answer", _svd_text(30000, svd_dims, seed), "completion"),
                ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
                ("handcrafted", Handcrafted(), HANDCRAFTED_INPUT),
            ]
        )
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=False, random_state=seed
        )
    elif kind == "rubric_only":
        # Baseline that never reads the answer: how much of the label is just
        # "this kind of criterion is usually met"? A real grader must beat it.
        features = ColumnTransformer(
            [("rubric", _tfidf(20000), "rubric"), ("category", category, ["category"])]
        )
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    return Pipeline([("features", features), ("clf", clf)])



def soft_label_training_set(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each row as a weighted 'met' and a weighted 'not met' example.

    Returns ``(rows, y, w)``: positions into ``df``, the label of each copy and
    its weight. Weights are the share of physicians on each side, so each row
    contributes a total weight of 1. Zero-weight copies are dropped.
    """
    share = df["share_met"].to_numpy(dtype=float)
    rows = np.concatenate([np.arange(len(df)), np.arange(len(df))])
    y = np.concatenate([np.ones(len(df), bool), np.zeros(len(df), bool)])
    w = np.concatenate([share, 1 - share])
    keep = w > 0
    return rows[keep], y[keep], w[keep]


def fit(df: pd.DataFrame, kind: str = "logreg", seed: int = 0, **build_kwargs) -> Pipeline:
    """Fit a grader with soft labels; returns a pipeline that predicts on DataFrames.

    Features are extracted once per original row and the matrix is then
    duplicated for the two weighted copies, which halves the text processing.
    """
    model = build_model(kind, seed, **build_kwargs)
    features, clf = model.named_steps["features"], model.named_steps["clf"]
    X = features.fit_transform(df.reset_index(drop=True))
    rows, y, w = soft_label_training_set(df)
    clf.fit(X[rows], y, sample_weight=w)
    return model


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df)[:, list(model.classes_).index(True)]


def grouped_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test row indices with every conversation on exactly one side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(df, groups=group_key(df)))
    return train, test


def out_of_fold_proba(
    df: pd.DataFrame, kind: str = "logreg", n_folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Out-of-fold probabilities from grouped cross-validation on ``df``."""
    groups = group_key(df)
    n_folds = min(n_folds, len(np.unique(groups)))
    oof = np.full(len(df), np.nan)
    for tr, va in GroupKFold(n_splits=n_folds).split(df, groups=groups):
        model = fit(df.iloc[tr], kind, seed)
        oof[va] = predict_proba(model, df.iloc[va])
    return oof


def choose_threshold(
    probabilities: np.ndarray,
    labels: Sequence[Sequence[bool]],
    grid: np.ndarray | None = None,
) -> float:
    """Threshold that maximises pairwise balanced F1 against physician labels.

    Must be chosen on training data only (e.g. out-of-fold predictions).
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = pairwise_agreement(list(probabilities >= t), labels)["f1_balanced"]
        if f1 is not None and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def expand_per_physician(
    probabilities: np.ndarray, labels: Sequence[Sequence[bool]]
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each probability once per physician label, for calibration checks."""
    p = np.repeat(np.asarray(probabilities, float), [len(x) for x in labels])
    y = np.fromiter((v for x in labels for v in x), dtype=bool)
    return p, y


def fit_with_threshold(
    df: pd.DataFrame, kind: str = "logreg", n_folds: int = 5, seed: int = 0
) -> tuple[Pipeline, float, np.ndarray]:
    """Fit on all of ``df``; pick the threshold from grouped out-of-fold predictions."""
    oof = out_of_fold_proba(df, kind, n_folds, seed)
    threshold = choose_threshold(oof, list(df["labels"]))
    return fit(df, kind, seed), threshold, oof


__all__ = [
    "MODEL_KINDS",
    "build_model",
    "choose_threshold",
    "expand_per_physician",
    "fit",
    "fit_with_threshold",
    "group_key",
    "grouped_split",
    "handcrafted_features",
    "out_of_fold_proba",
    "predict_proba",
    "soft_label_training_set",
]

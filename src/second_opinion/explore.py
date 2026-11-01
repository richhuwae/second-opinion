"""Facts about the meta-eval data needed before defining "good" and "unsafe".

Everything here is descriptive: counts and shares computed from the file, used
to ground the label definitions in docs/PLAN.md (task C1).
"""

from __future__ import annotations

import pandas as pd

from second_opinion.grader import _NEGATIVE_CRITERION, group_key

# Cheap French detector for optional extension 6: share of common French words.
_FRENCH = frozenset("le la les des est une pour avec pas que qui dans je vous mon".split())


def looks_french(text: str, min_share: float = 0.15) -> bool:
    words = text.lower().split()
    return bool(words) and sum(w in _FRENCH for w in words) / len(words) >= min_share


def summarize(df: pd.DataFrame) -> dict:
    """Headline counts, label agreement, and breakdowns by theme."""
    convo = group_key(df)
    per_convo = df.assign(_c=convo).groupby("_c")
    by_cat = (
        df.groupby("category")
        .agg(rows=("row_id", "size"), share_met=("share_met", "mean"), unanimous=("unanimous", "mean"))
        .sort_values("rows", ascending=False)
    )
    first_user = df["conversation"].str.split("\n\n").str[0]
    return {
        "n_rows": int(len(df)),
        "n_conversations": int(per_convo.ngroups),
        "n_answers": int(df["completion_id"].nunique()) if df["completion_id"].notna().any() else None,
        "n_physicians": int(len({p for ids in df["physician_ids"] for p in ids})),
        "physicians_per_row": df["n_physicians"].value_counts().sort_index().to_dict(),
        "rows_per_conversation_median": float(per_convo.size().median()),
        "share_met_overall": float(df["share_met"].mean()),
        "share_unanimous": float(df["unanimous"].mean()),
        "share_unanimous_multi_physician": float(df.loc[df["n_physicians"] > 1, "unanimous"].mean()),
        "share_majority_tie": float(df["majority_met"].isna().mean()),
        "share_negative_criteria": float(
            df["rubric"].map(lambda r: bool(_NEGATIVE_CRITERION.search(r))).mean()
        ),
        "n_turns_distribution": df["n_turns"].value_counts().sort_index().to_dict(),
        "share_french_conversations": float(first_user.map(looks_french).mean()),
        "by_category": by_cat.round(3).reset_index().to_dict(orient="records"),
    }

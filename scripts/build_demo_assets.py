"""Build the files the Streamlit demo loads (run once, offline; never at app startup).

Writes app/assets/:
- grader.joblib: the exact C2 model (gradient boosting fit on the C2 training
  split). Before saving, its test-split score is checked against
  results/grader_classical.json, so the demo runs the model the results came from.
- risk_scores.csv: per answer with a safety criterion: held-out risk from C4
  (out-of-fold grader and criterion-only), the C2 model's risk on its test
  split, and the labels. Ids and numbers only, no benchmark text.
- examples.jsonl: about 24 test-split answers (one unsafe, one good per safety
  criterion) with text, physician votes and the C2 model's held-out scores.
  Every row keeps the HealthBench canary string.
- criteria.json: the 12 safety criteria, their text and why each counts.

Needs data/processed/risk_scores.csv from scripts/safety_alarm.py.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from second_opinion import grader
from second_opinion.data import download, load_meta_eval, read_jsonl
from second_opinion.experiment import _f1
from second_opinion.labels import SAFETY_REASONS, answer_labels
from second_opinion.safety import answer_risk

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"
OOF = ROOT / "data" / "processed" / "risk_scores.csv"
C2 = ROOT / "results" / "grader_classical.json"
MAX_ANSWER_CHARS = 1800
MAX_CONVO_CHARS = 1200


def main() -> int:
    if not OOF.exists():
        print(f"Missing {OOF}; run scripts/safety_alarm.py first.", file=sys.stderr)
        return 1
    path = download("meta_eval")
    df = load_meta_eval(path).reset_index(drop=True)
    canary = {r.get("canary") for r in read_jsonl(path)}
    if len(canary) != 1:
        raise ValueError(f"expected one canary string, found {canary}")
    canary = canary.pop()
    c2 = json.loads(C2.read_text())
    if c2["grader"]["kind"] != "hgb":
        raise ValueError("C2 did not choose gradient boosting; the demo expects it")

    # 1. The exact C2 model, checked against the reported score.
    tr, te = grader.grouped_split(df, 0.2, c2["data"]["seed"])
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    model = grader.fit(train, "hgb", c2["data"]["seed"])
    p_test = grader.predict_proba(model, test)
    f1 = _f1(p_test >= c2["grader"]["threshold"], test["labels"])
    if abs(f1 - c2["grader"]["f1_balanced"]) > 1e-9:
        raise ValueError(f"rebuilt model scores {f1}, results say {c2['grader']['f1_balanced']}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ASSETS / "grader.joblib", compress=3)  # lossless compression
    size_mb = (ASSETS / "grader.joblib").stat().st_size / 1e6
    print(f"grader.joblib: {size_mb:.1f} MB; test F1 {f1:.4f} matches results")

    # 2. Risk scores (no text).
    labels = answer_labels(df)
    oof = pd.read_csv(OOF, index_col=0)
    risk = oof.loc[oof["has_safety_criterion"], ["risk_grader", "risk_criterion_only", "theme",
                                                 "unsafe", "unsafe_contested", "good"]]
    risk["risk_c2_model_test"] = answer_risk(test, p_test).reindex(risk.index)
    risk.index.name = "completion_id"
    risk.to_csv(ASSETS / "risk_scores.csv", float_format="%.6g")

    # 3. Examples from the test split: per safety criterion, one clear-cut
    #    unsafe and one good answer (physicians unanimous), shortest first.
    test = test.assign(p_met=p_test)
    test_risk = risk["risk_c2_model_test"].dropna()
    examples = []
    for cat, reason in SAFETY_REASONS.items():
        rows = test[(test["category"] == cat)
                    & (test["completion"].str.len() <= MAX_ANSWER_CHARS)
                    & (test["conversation"].str.len() <= MAX_CONVO_CHARS)]
        for want in ("unsafe", "good"):
            ok = rows["labels"].map(lambda x, w=want: (not any(x)) if w == "unsafe" else all(x))
            ids = [c for c in rows.loc[ok, "completion_id"] if labels.loc[c, want]]
            cand = rows[rows["completion_id"].isin(ids)]
            if cand.empty:
                continue
            pick = cand.assign(_n=cand["completion"].str.len()).sort_values(["_n", "completion_id"]).iloc[0]
            answer_rows = test[test["completion_id"] == pick["completion_id"]]
            examples.append({
                "completion_id": pick["completion_id"],
                "safety_criterion": cat,
                "physician_label": want,
                "conversation": pick["conversation"],
                "n_turns": int(pick["n_turns"]),
                "answer": pick["completion"],
                "risk_c2_model": float(test_risk[pick["completion_id"]]),
                "criteria": [
                    {"category": r["category"], "rubric": r["rubric"], "labels": list(r["labels"]),
                     "p_met": float(r["p_met"])}
                    for _, r in answer_rows.iterrows()
                ],
                "canary": canary,
            })
    with open(ASSETS / "examples.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"examples.jsonl: {len(examples)} answers")

    # 4. The 12 safety criteria.
    rubric = df.groupby("category")["rubric"].first()
    criteria = [{"category": c, "rubric": rubric[c], "reason": r, "canary": canary}
                for c, r in SAFETY_REASONS.items()]
    (ASSETS / "criteria.json").write_text(json.dumps(criteria, indent=1) + "\n")
    for p in sorted(ASSETS.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

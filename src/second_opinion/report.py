"""Numbers for the README, write-up and model card, read from results/*.json.

Docs mark every computed number like this::

    <!--n:alarm_recall_at_10_grader-->38%<!--/n-->

and every generated table like this::

    <!--table:theme_gaps-->
    | ... |
    <!--/table-->

``fill`` rewrites the text between the markers from the results files, so a
doc can never drift from the results without a test failing
(``tests/test_report.py``). Run ``python scripts/update_docs.py`` after
rerunning an experiment.

Functions take the results directory as an argument: the package never builds
paths from its own location.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

POLICIES = ("grader", "criterion_only", "random", "perfect")
POLICY_LABELS = {
    "grader": "Grader ranking",
    "criterion_only": "Criterion-only ranking",
    "random": "Random sample (today)",
    "perfect": "Perfect ranking (upper bound)",
}
NUMBER = re.compile(r"<!--n:([a-z0-9_]+)-->(.*?)<!--/n-->", re.S)
TABLE = re.compile(r"(<!--table:([a-z0-9_]+)-->\n)(.*?)(<!--/table-->)", re.S)


def load_results(results_dir: Path) -> dict:
    """All results files, keyed by file stem."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))}


def pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def count(x: float) -> str:
    return f"{int(round(x)):,}"


def ci(lo_hi, fmt=num) -> str:
    return f"{fmt(lo_hi[0])}–{fmt(lo_hi[1])}"


def _theme(category: str) -> str:
    return category.removeprefix("cluster:")


def capacity_row(alarm: dict, share: float, target: str = "main_contested_is_not_unsafe",
                 split: str = "out_of_fold_all_answers") -> dict:
    for row in alarm[split][target]["capacity"]:
        if abs(row["capacity_share"] - share) < 1e-9:
            return row
    raise KeyError(f"no capacity {share} in results")


def numbers(r: dict) -> dict[str, str]:
    """Every number the docs may quote, formatted as it appears in the text."""
    g = r["grader_classical"]
    a = r["safety_alarm"]
    lab = r["label_summary"]
    data = r["data_summary"]
    main = a["out_of_fold_all_answers"]["main_contested_is_not_unsafe"]
    sens = a["out_of_fold_all_answers"]["sensitivity_contested_is_unsafe"]
    check = a["check_c2_test_split"]["main_contested_is_not_unsafe"]
    cost = a["cost_per_1000_screened_answers"]
    assume = cost["ASSUMPTIONS"]
    gr = g["grader"]
    n: dict[str, str] = {}

    # Data and labels.
    n["data_rows"] = count(data["n_rows"])
    n["data_conversations"] = count(data["n_conversations"])
    n["data_answers"] = count(data["n_answers"])
    n["data_physicians"] = count(data["n_physicians"])
    n["data_share_two_physicians"] = pct(data["physicians_per_row"]["2"] / data["n_rows"])
    n["data_share_met"] = pct(data["share_met_overall"])
    n["data_share_tie"] = pct(data["share_majority_tie"])
    n["data_share_french"] = pct(data["share_french_conversations"], 1)
    n["n_safety_criteria"] = count(lab["n_safety_criteria"])
    n["answers_with_safety"] = count(lab["answers_with_safety_criterion"])
    n["answers_unsafe"] = count(lab["answers_unsafe"])
    n["answers_contested"] = count(lab["answers_unsafe_contested"])
    n["share_unsafe"] = pct(lab["share_unsafe_of_covered"])
    uncovered = [t for t in lab["by_theme"] if t["covered"] == 0]
    n["n_themes_uncovered"] = count(len(uncovered))
    n["answers_uncovered"] = count(sum(t["answers"] for t in uncovered))
    n["share_uncovered"] = pct(sum(t["answers"] for t in uncovered) / lab["n_answers"])
    n["n_themes"] = count(len(lab["by_theme"]))

    # C2/C3 grader vs physicians (C2 test split).
    n["train_rows"] = count(g["data"]["n_train_rows"])
    n["test_rows"] = count(g["data"]["n_test_rows"])
    n["test_physicians"] = count(g["physician_ceiling"]["n_physicians"])
    n["grader_f1"] = num(gr["f1_balanced"])
    n["grader_f1_ci"] = ci(gr["f1_balanced_ci95"])
    n["rubric_f1"] = num(g["baselines"]["rubric_only"])
    n["rubric_f1_ci"] = ci(g["baselines"]["rubric_only_ci95"])
    n["phys_f1_mean"] = num(g["physician_ceiling"]["mean"])
    n["phys_f1_pooled"] = num(g["physician_pooled_f1"])
    n["phys_f1_min"] = num(g["physician_ceiling"]["min"])
    n["phys_f1_max"] = num(g["physician_ceiling"]["max"])
    n["grader_kappa"] = num(gr["kappa_vs_physicians"], 2)
    n["grader_kappa_ci"] = ci(gr["kappa_ci95"], lambda x: num(x, 2))
    n["rubric_kappa"] = num(g["baselines"]["rubric_only_kappa"], 2)
    n["rubric_kappa_ci"] = ci(g["baselines"]["rubric_only_kappa_ci95"], lambda x: num(x, 2))
    n["phys_kappa"] = num(g["physician_pooled_kappa"], 2)
    n["grader_ece"] = num(gr["ece_vs_physicians"])
    n["grader_threshold"] = num(gr["threshold"], 2)
    n["grader_recall_neg"] = pct(gr["recall_neg"])
    n["grader_precision_neg"] = pct(gr["precision_neg"])
    n["grader_recall_pos"] = pct(gr["recall_pos"])
    n["grader_precision_pos"] = pct(gr["precision_pos"])
    n["grader_ms_per_row"] = num(gr["seconds_per_1000_rows"], 1)
    cands = g["model_selection"]["candidates"]
    n["cv_f1_hgb"] = num(cands["hgb"]["oof_f1"])
    n["cv_f1_logreg"] = num(cands["logreg"]["oof_f1"])
    cats = [c for c in g["by_category"] if c["grader_f1"] is not None]
    below = [c for c in cats if c["grader_f1"] < c["physician_f1_mean"]]
    n["n_criteria_compared"] = count(len(cats))
    n["n_criteria_below_phys"] = count(len(below))
    n["n_criteria_no_grader_f1"] = count(len(g["by_category"]) - len(cats))

    # C4 safety alarm.
    q = main["ranking_quality"]
    n["auc_grader"] = num(q["grader"]["roc_auc"], 2)
    n["auc_criterion"] = num(q["criterion_only"]["roc_auc"], 2)
    n["ap_grader"] = num(q["grader"]["average_precision"], 2)
    n["ap_criterion"] = num(q["criterion_only"]["average_precision"], 2)
    n["auc_grader_contested"] = num(sens["ranking_quality"]["grader"]["roc_auc"], 2)
    n["auc_criterion_contested"] = num(sens["ranking_quality"]["criterion_only"]["roc_auc"], 2)
    n["auc_grader_check"] = num(check["ranking_quality"]["grader"]["roc_auc"], 2)
    n["alarm_folds"] = count(a["n_folds"])
    for t in ("0.5", "0.8"):
        tag = str(round(float(t) * 100))
        for p in POLICIES:
            n[f"share_for_{tag}_{p}"] = pct(main["share_to_review_for_recall"][t][p])
            n[f"share_for_{tag}_{p}_contested"] = pct(sens["share_to_review_for_recall"][t][p])
            n[f"share_for_{tag}_{p}_check"] = pct(check["share_to_review_for_recall"][t][p])
    s50 = main["share_to_review_for_recall"]["0.5"]
    n["time_ratio_random_vs_grader_50"] = num(s50["random"] / s50["grader"], 1)
    row10 = capacity_row(a, 0.10)
    for p in POLICIES:
        n[f"recall_10_{p}"] = pct(row10[p]["recall"])
        n[f"precision_10_{p}"] = pct(row10[p]["precision"])
        n[f"caught_10_{p}"] = count(row10[p]["caught"])
    n["recall_10_grader_ci"] = ci(row10["grader"]["recall_ci95"], pct)
    n["recall_10_criterion_only_ci"] = ci(row10["criterion_only"]["recall_ci95"], pct)
    n["reviewed_10"] = count(row10["n_reviewed"])
    n["ratio_10_grader_vs_random"] = num(row10["grader"]["recall"] / row10["random"]["recall"], 1)
    c10 = capacity_row(a, 0.10, split="check_c2_test_split")
    n["recall_10_grader_check"] = pct(c10["grader"]["recall"])
    s10 = capacity_row(a, 0.10, "sensitivity_contested_is_unsafe")
    n["recall_10_grader_contested"] = pct(s10["grader"]["recall"])
    n["recall_10_perfect_contested"] = pct(s10["perfect"]["recall"])

    # C5 cost view: ASSUMPTIONS.
    n["assume_minutes"] = num(assume["minutes_per_review"], 0)
    n["assume_eur_per_hour"] = num(assume["physician_cost_per_hour_eur"], 0)
    c = next(x for x in cost["by_capacity"] if abs(x["capacity_share"] - 0.10) < 1e-9)
    n["cost10_reviews"] = count(c["grader"]["reviews"])
    n["cost10_hours"] = num(c["grader"]["doctor_hours"], 1)
    n["cost10_eur"] = count(c["grader"]["review_cost_eur"])
    for p in POLICIES:
        n[f"cost10_caught_{p}"] = num(c[p]["unsafe_caught"], 0)
        n[f"cost10_per_caught_{p}"] = num(c[p]["cost_per_unsafe_caught_eur"], 0)
    return n


def theme_gaps_table(r: dict, k: int = 5) -> str:
    """The k criteria where the grader trails physicians most (C2 test split)."""
    cats = [c for c in r["grader_classical"]["by_category"] if c["grader_f1"] is not None]
    cats.sort(key=lambda c: c["grader_f1"] - c["physician_f1_mean"])
    lines = ["| Criterion | Test rows | Grader F1 | Physician F1 (mean) | Gap |", "|---|---|---|---|---|"]
    for c in cats[:k]:
        lines.append(
            f"| `{_theme(c['category'])}` | {count(c['n_rows'])} | {num(c['grader_f1'], 2)} "
            f"| {num(c['physician_f1_mean'], 2)} | {num(c['grader_f1'] - c['physician_f1_mean'], 2)} |")
    return "\n".join(lines) + "\n"


def theme_coverage_table(r: dict) -> str:
    """Answers per theme, and how many the safety alarm can rank."""
    lines = ["| Theme | Answers | Rankable by the alarm | Unsafe | Contested |", "|---|---|---|---|---|"]
    for t in r["label_summary"]["by_theme"]:
        lines.append(f"| {t['theme']} | {count(t['answers'])} | {count(t['covered'])} "
                     f"| {count(t['unsafe'])} | {count(t['contested'])} |")
    return "\n".join(lines) + "\n"


def capacity_table(r: dict) -> str:
    """Share of unsafe answers caught at each review capacity (main result)."""
    main = r["safety_alarm"]["out_of_fold_all_answers"]["main_contested_is_not_unsafe"]
    head = "| Review capacity | " + " | ".join(POLICY_LABELS[p] for p in POLICIES) + " |"
    lines = [head, "|---" * (len(POLICIES) + 1) + "|"]
    for row in main["capacity"]:
        lines.append(f"| {pct(row['capacity_share'])} | "
                     + " | ".join(pct(row[p]["recall"]) for p in POLICIES) + " |")
    return "\n".join(lines) + "\n"


TABLES = {
    "theme_gaps": theme_gaps_table,
    "theme_coverage": theme_coverage_table,
    "capacity": capacity_table,
}


def fill(text: str, r: dict) -> str:
    """Rewrite every marked number and table from the results. Unknown keys raise."""
    vals = numbers(r)

    def sub_number(m):
        if m.group(1) not in vals:
            raise KeyError(f"unknown number key: {m.group(1)}")
        return f"<!--n:{m.group(1)}-->{vals[m.group(1)]}<!--/n-->"

    def sub_table(m):
        if m.group(2) not in TABLES:
            raise KeyError(f"unknown table key: {m.group(2)}")
        return m.group(1) + TABLES[m.group(2)](r) + m.group(4)

    return TABLE.sub(sub_table, NUMBER.sub(sub_number, text))


def unmarked_numbers(text: str, allowed: tuple[str, ...] = ()) -> list[str]:
    """Result-like numbers (decimals, percentages, thousands) outside markers.

    Code, links, markers and generated tables are skipped. ``allowed`` lists
    literal strings that are units or names rather than results (e.g. "1,000").
    """
    t = TABLE.sub("", text)
    t = NUMBER.sub("", t)
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"`[^`]*`", "", t)
    t = re.sub(r"\]\([^)]*\)", "]", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    for a in allowed:
        t = t.replace(a, "")
    return re.findall(r"\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?%", t)


def capacity_curve(r: dict) -> list[dict]:
    """Points for the headline chart: share of unsafe answers caught vs capacity.

    Every point is a value from results/safety_alarm.json (main result): the
    tested capacities, plus the capacities that reach 50% and 80% recall. The
    perfect policy also gets its corner, where capacity equals the unsafe rate.
    """
    main = r["safety_alarm"]["out_of_fold_all_answers"]["main_contested_is_not_unsafe"]
    rate = main["n_positive"] / r["safety_alarm"]["out_of_fold_all_answers"]["n_answers"]
    pts = [{"policy": p, "capacity_share": 0.0, "recall": 0.0, "source": "origin"} for p in POLICIES]
    for row in main["capacity"]:
        for p in POLICIES:
            pts.append({"policy": p, "capacity_share": row["capacity_share"],
                        "recall": row[p]["recall"], "source": "capacity"})
    for t, by_p in main["share_to_review_for_recall"].items():
        for p in POLICIES:
            pts.append({"policy": p, "capacity_share": by_p[p], "recall": float(t),
                        "source": f"share_for_recall_{t}"})
    pts.append({"policy": "perfect", "capacity_share": rate, "recall": 1.0, "source": "unsafe_rate"})
    pts.sort(key=lambda d: (POLICIES.index(d["policy"]), d["capacity_share"], d["recall"]))
    return pts

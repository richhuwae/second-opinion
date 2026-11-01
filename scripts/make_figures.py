"""Headline chart: share of unsafe answers caught vs review capacity, four policies.

Reads results/safety_alarm.json only (main result: out-of-fold risk scores,
contested answers counted as not unsafe). Writes
docs/figures/caught_vs_capacity.png and, next to it, the exact points plotted
(caught_vs_capacity.csv), which tests/test_report.py checks against the results.

Needs matplotlib: pip install -e ".[docs]"
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from second_opinion.report import POLICIES, POLICY_LABELS, capacity_curve, load_results  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "caught_vs_capacity.png"
CSV = OUT.with_suffix(".csv")

# Reference categorical palette (blue, orange) for the two learned rankings;
# the two reference policies are neutral greys, dashed.
STYLE = {
    "grader": dict(color="#2a78d6", lw=2.4, ls="-", marker="o"),
    "criterion_only": dict(color="#eb6834", lw=2.0, ls="-", marker="o"),
    "random": dict(color="#52514e", lw=1.6, ls="--", marker=None),
    "perfect": dict(color="#8a8984", lw=1.6, ls=":", marker=None),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> int:
    pts = capacity_curve(load_results(ROOT / "results"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "capacity_share", "recall", "source"])
        w.writeheader()
        w.writerows(pts)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("white")
    xmax = 0.5
    for p in POLICIES:
        xs = [d["capacity_share"] for d in pts if d["policy"] == p and d["capacity_share"] <= xmax + 1e-9]
        ys = [d["recall"] for d in pts if d["policy"] == p and d["capacity_share"] <= xmax + 1e-9]
        st = STYLE[p]
        ax.plot(xs, ys, color=st["color"], lw=st["lw"], ls=st["ls"], label=POLICY_LABELS[p], zorder=3,
                marker=st["marker"], markersize=4.5, markeredgecolor="white", markeredgewidth=1)
        ax.annotate(POLICY_LABELS[p].replace(" (upper bound)", "").replace(" (today)", ""),
                    (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)
    ax.axvline(0.10, color=MUTED, lw=0.8, zorder=1)
    ax.text(0.106, 0.03, "10% capacity", color=MUTED, fontsize=9, rotation=90, va="bottom")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Review capacity: share of answers a doctor reads", color=MUTED)
    ax.set_ylabel("Share of unsafe answers caught", color=MUTED)
    ax.set_title("Same doctor time, more unsafe answers caught", loc="left", color=INK,
                 fontsize=13, fontweight="bold")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.text(0.01, 0.005, "HealthBench meta-eval, held-out (out-of-fold) risk scores. "
             "Points: values in results/safety_alarm.json; lines join them.",
             fontsize=7.5, color=MUTED)
    fig.subplots_adjust(right=0.80, bottom=0.14)
    fig.savefig(OUT, facecolor="white")
    print(f"Wrote {OUT.relative_to(ROOT)} and {CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

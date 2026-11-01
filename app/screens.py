"""Screens of the Second Opinion demo (imported by streamlit_app.py).

Loads a pre-trained model and precomputed results from app/assets/ and
results/. Nothing is trained at startup. Build the assets with
scripts/build_demo_assets.py.
"""

from __future__ import annotations

import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from second_opinion.budget import CostAssumptions
from second_opinion.demo import (
    ASSETS,
    ROOT,
    alarm_threshold,
    budget_table,
    build_input,
    load_assets,
    load_results,
    recall_curve,
    short_name,
    simulated_day,
)
from second_opinion.grader import predict_proba

REPO = "https://github.com/richhuwae/second-opinion"


@st.cache_resource
def model():
    return joblib.load(ASSETS / "grader.joblib")


@st.cache_data
def assets():
    return load_assets()


@st.cache_data
def results():
    return load_results()


A, R = assets(), results()
RISK = A["risk"]
MET_THRESHOLD = R["grader_classical"]["grader"]["threshold"]
DEFAULTS = CostAssumptions()
POLICY_COLORS = {"Grader": "#1f4e9c", "Criterion-only": "#7fb2e5", "Random": "#8c8c8c", "Perfect": "#2e9e5b"}


def sidebar() -> float:
    st.sidebar.title("🩺 Second Opinion")
    st.sidebar.caption("A safety monitor for a health chatbot, checked against physicians.")
    share = st.sidebar.slider(
        "Doctors review this share of answers", 1, 50, 10, format="%d%%",
        key="capacity", help="Sets the review budget and the alarm threshold on every screen.",
    ) / 100
    st.sidebar.caption(
        "Data: HealthBench meta-eval (OpenAI, MIT licence). Answers are written by models for the "
        "benchmark, not real patient conversations."
    )
    st.sidebar.markdown(f"[Code that reproduces every number]({REPO})")
    return share


def verdict(labels: list[bool]) -> str:
    met = sum(labels)
    return "met" if 2 * met > len(labels) else "not met" if 2 * met < len(labels) else "tie"


def screen_try():
    share = st.session_state.get("capacity", 10) / 100
    thr = alarm_threshold(RISK["risk_c2_model_test"], share)
    st.header("Try a conversation")
    st.write(
        f"The alarm fires when risk is at least **{thr:.2f}**: the level at which doctors would "
        f"review {share:.0%} of answers (set in the sidebar). Risk = the model's highest "
        "probability that a safety criterion is **not** met."
    )
    tab_ex, tab_own = st.tabs(["Benchmark example", "Your own text"])

    with tab_ex:
        ex = A["examples"]
        i = st.selectbox(
            "Pick an example (held-out: the model never saw this conversation)",
            range(len(ex)), format_func=lambda j: f"{j + 1}. {short_name(ex[j]['safety_criterion'])}",
        )
        e = ex[i]
        with st.expander("Conversation", expanded=True):
            st.text(e["conversation"])
        st.markdown("**Bot answer**")
        st.info(e["answer"])
        risk = e["risk_c2_model"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk", f"{risk:.2f}")
        c2.metric("Alarm", "🚨 sent to a doctor" if risk >= thr else "no alarm")
        c3.metric("Physicians' label", e["physician_label"])
        st.dataframe(
            [{
                "Criterion": short_name(c["category"]),
                "Safety criterion": "yes" if c["category"] == e["safety_criterion"] else "no",
                "Physician votes": " ".join("✅" if x else "❌" for x in c["labels"]),
                "Physicians": verdict(c["labels"]),
                "Model P(met)": round(c["p_met"], 2),
                "Model says": "met" if c["p_met"] >= MET_THRESHOLD else "not met",
            } for c in e["criteria"]],
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Examples are chosen by rule (shortest answer with a unanimous physician verdict), "
                   f"not for looking good; some show the model getting it wrong. "
                   f"'Model says met' uses the C2 threshold {MET_THRESHOLD:.2f}.")

    with tab_own:
        crit = {c["category"]: c for c in A["criteria"]}
        cat = st.selectbox("Safety criterion", list(crit), format_func=short_name)
        st.caption(f"Why it counts: {crit[cat]['reason']}")
        with st.expander("Full criterion text given to physicians"):
            st.text(crit[cat]["rubric"])
        user = st.text_area("User's message", placeholder="e.g. I have chest pain when I climb stairs…")
        answer = st.text_area("Bot's answer", height=160)
        if st.button("Grade this answer", type="primary", disabled=not (user.strip() and answer.strip())):
            p = float(predict_proba(model(), build_input(cat, crit[cat]["rubric"], user, answer))[0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Model P(criterion met)", f"{p:.2f}")
            c2.metric("Risk", f"{1 - p:.2f}")
            c3.metric("Alarm", "🚨 sent to a doctor" if 1 - p >= thr else "no alarm")
        st.warning(
            "The model learned from benchmark answers written by other models. Text that looks "
            "different (other languages, very long or very short answers) can get unreliable scores. "
            "This is a demonstration, not medical advice."
        )


def screen_monitor():
    share = st.session_state.get("capacity", 10) / 100
    thr = alarm_threshold(RISK["risk_grader"], share)
    st.header("Monitor")
    st.warning("**Simulated day.** 600 benchmark answers given random arrival times. Not real traffic.")
    day = simulated_day(RISK, thr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answers", f"{len(day):,}")
    c2.metric("Flagged for review", f"{int(day['flagged'].sum()):,}")
    c3.metric("Unsafe per physicians", f"{int(day['unsafe'].sum()):,}")
    c4.metric("Flag agrees with physicians", f"{day['agrees'].mean():.0%}")
    by_hour = day.groupby("hour").agg(flagged=("flagged", "sum"), unsafe=("unsafe", "sum"),
                                      agreement=("agrees", "mean")).reset_index()
    left, right = st.columns(2)
    with left:
        fig = px.bar(by_hour.melt("hour", ["flagged", "unsafe"], var_name="series", value_name="answers"),
                     x="hour", y="answers", color="series", barmode="group", color_discrete_sequence=["#1f4e9c", "#e07b39"],
                     title="Flagged by the grader vs unsafe per physicians, by hour")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(by_hour, x="hour", y="agreement", markers=True, range_y=[0, 1],
                      title="Agreement with physicians, by hour")
        st.plotly_chart(fig, use_container_width=True)
    th = day.groupby("theme").agg(flagged=("flagged", "sum"), unsafe=("unsafe", "sum")).reset_index()
    th["theme"] = th["theme"].str.replace("_", " ")
    fig = px.bar(th.melt("theme", ["flagged", "unsafe"], var_name="series", value_name="answers"),
                 x="answers", y="theme", color="series", barmode="group", color_discrete_sequence=["#1f4e9c", "#e07b39"], orientation="h",
                 title="Flags by theme")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Scores are held-out (out-of-fold): no answer is scored by a model that saw its conversation. "
               "Hedging and complex-response answers are not monitored: they have no safety criterion.")


def screen_budget():
    share = st.session_state.get("capacity", 10) / 100
    st.header("Review budget")
    st.write(f"Doctors review **{share:.0%}** of answers (change it in the sidebar). "
             "Which answers they review decides how many unsafe ones they catch.")
    c1, c2, c3 = st.columns(3)
    contested = c1.toggle("Contested counts as unsafe", value=False,
                          help="Contested = physicians split evenly. Off = main result. On = sensitivity check.")
    minutes = c2.number_input("ASSUMPTION: minutes per review", 0.5, 60.0, DEFAULTS.minutes_per_review, 0.5)
    rate = c3.number_input("ASSUMPTION: physician cost per hour (EUR)", 10.0, 500.0,
                           DEFAULTS.physician_cost_per_hour_eur, 5.0)
    costs = CostAssumptions(minutes_per_review=minutes, physician_cost_per_hour_eur=rate)
    t = budget_table(RISK, share, contested, costs)
    g, rnd = t.iloc[0], t.iloc[2]
    m1, m2, m3 = st.columns(3)
    m1.metric("Unsafe caught per 1,000 answers: grader", f"{g['Unsafe caught']:.0f}",
              f"{g['Unsafe caught'] - rnd['Unsafe caught']:+.0f} vs random sample")
    m2.metric("Missed by the grader", f"{g['Unsafe missed']:.0f}")
    m3.metric("False alarms (grader)", f"{g['False alarms']:.0f}")
    st.markdown(f"Same doctor time for every policy: **{g['Doctor hours']:.1f} hours** and "
                f"**EUR {g['Review cost (EUR)']:,.0f}** per 1,000 answers (from the assumptions above).")
    st.dataframe(t.drop(columns=["Doctor hours", "Review cost (EUR)"]).style.format({
        "Unsafe caught": "{:.0f}", "Unsafe missed": "{:.0f}", "False alarms": "{:.0f}",
        "Share of unsafe caught": "{:.0%}", "EUR per unsafe caught": "{:,.1f}",
    }), hide_index=True, use_container_width=True)
    curve = recall_curve(RISK, contested)
    fig = px.line(curve, x="Capacity", y="Share of unsafe caught", color="Policy",
                  color_discrete_map=POLICY_COLORS, line_dash="Policy",
                  line_dash_map={"Grader": "solid", "Criterion-only": "solid", "Random": "dot", "Perfect": "dash"},
                  title="Share of unsafe answers caught, by review capacity")
    fig.add_vline(x=share, line_dash="dot")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Per 1,000 answers that have a safety criterion. Risk scores are held-out (out-of-fold). "
        "Minutes per review and cost per hour are assumptions, not measurements; the model's own "
        "running cost is treated as zero. Ties in scores are broken at random (expected value)."
    )


def screen_under_the_hood():
    g = R["grader_classical"]
    s = R["safety_alarm"]["out_of_fold_all_answers"]
    st.header("Under the hood")
    st.subheader("Does the grader agree with physicians as often as physicians agree with each other?")
    rows = [
        ("Grader (gradient boosting)", g["grader"]["f1_balanced"], g["grader"]["kappa_vs_physicians"]),
        ("Criterion-only baseline (never reads the answer)", g["baselines"]["rubric_only"],
         g["baselines"]["rubric_only_kappa"]),
        ("Physicians: mean of individuals", g["physician_ceiling"]["mean"], None),
        ("Physicians: pooled pairs", g["physician_pooled_f1"], g["physician_pooled_kappa"]),
    ]
    st.dataframe([{"": n, "Balanced F1": f"{f:.3f}", "Cohen's kappa": "–" if k is None else f"{k:.3f}"}
                  for n, f, k in rows], hide_index=True, use_container_width=True)
    lo, hi = g["grader"]["f1_balanced_ci95"]
    st.caption(f"{g['data']['n_test_rows']:,} held-out rows. Grader F1 95% CI {lo:.3f}–{hi:.3f}, resampled by "
               f"conversation. Calibration error {g['grader']['ece_vs_physicians']:.3f}.")
    st.subheader("By theme: grader vs physicians")
    cats = [c for c in g["by_category"] if c["grader_f1"] is not None and c["physician_f1_mean"] is not None]
    bc = pd.DataFrame([{"criterion": short_name(c["category"]), "Grader": c["grader_f1"],
                        "Physicians (mean)": c["physician_f1_mean"]} for c in cats])
    bc = bc.sort_values("Grader")
    fig = px.bar(bc.melt("criterion", var_name="series", value_name="balanced F1"), x="balanced F1",
                 y="criterion", color="series", barmode="group", color_discrete_sequence=["#1f4e9c", "#e07b39"], orientation="h", height=900)
    st.plotly_chart(fig, use_container_width=True)
    worse = sum(c["grader_f1"] < c["physician_f1_mean"] for c in cats)
    st.caption(f"The grader is below physicians on {worse} of {len(cats)} criteria. Its overall score is close "
               "to physicians mostly because it learns which criteria are usually met.")
    st.subheader("Safety alarm: ranking quality")
    st.dataframe([{
        "Unsafe means": label, "Unsafe answers": s[k]["n_positive"],
        "AUC grader": round(s[k]["ranking_quality"]["grader"]["roc_auc"], 3),
        "AUC criterion-only": round(s[k]["ranking_quality"]["criterion_only"]["roc_auc"], 3),
    } for label, k in (("majority 'not met' (main)", "main_contested_is_not_unsafe"),
                       ("also physicians split (sensitivity)", "sensitivity_contested_is_unsafe"))],
        hide_index=True, use_container_width=True)
    st.subheader("Limitations")
    text = (ROOT / "docs" / "labels.md").read_text()
    lim = re.split(r"^## Limitations\s*$", text, flags=re.M)
    st.markdown(lim[1] if len(lim) > 1 else "See docs/labels.md.")
    st.markdown(
        "- Agreement between physicians is itself moderate; physician agreement is the ceiling, not truth.\n"
        f"- Every number here is produced by code in the [repository]({REPO}): `results/*.json`."
    )


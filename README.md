# Second Opinion

**A cost-aware safety monitor for a health chatbot, measured against physicians.**

> 🚧 Work in progress. Results will appear here only once they have been computed
> from the data and can be reproduced from this repository.

## The problem

Health chatbots answer patients' questions and point them to the right care.
Doctors cannot read every conversation, so today they review a **random sample**.
That raises three questions:

1. **Can a model grade conversations the way doctors do?** And how close is it to
   how much doctors agree *with each other*?
2. **Can it flag the unsafe ones** so that limited doctor time goes to the riskiest
   conversations first, instead of a random sample?
3. **What does it cost** per 1,000 conversations, in money and in doctor time?

## The approach

```
 every conversation ──► quality grader ──► safety alarm ──► doctor review queue ──► feedback
                        (is it good?)       (is it risky?)    (ranked by risk,        becomes
                                                               sized by capacity)      training data
```

| Stage | Question | Status |
|---|---|---|
| Data | Physician-graded health conversations, cleaned and validated | 🟡 loader built and tested; data not yet downloaded |
| Quality grader | Does an answer meet a physician-written criterion? Classical models first | ⬜ |
| Trust check | Agreement with physicians vs. physician-to-physician agreement, calibration | 🟡 metrics built and verified against OpenAI's reference code |
| Safety alarm | Which conversations should a doctor see, given a fixed review capacity? | 🟡 review-budget simulation built and tested |
| Cost view | Cost and doctor time per 1,000 conversations | 🟡 built into the budget simulation |
| Interactive demo | Try a conversation, move the review-budget slider | ⬜ |

**Planned extensions** (only after the core is solid): routing between cheap and
expensive graders, an LLM judge, turning doctor feedback into training examples,
failure-mode clustering, and a French-language slice.

## Data

[HealthBench](https://openai.com/index/healthbench/) (OpenAI, 2025, MIT licence):
health conversations with grading criteria written by physicians. The *meta-eval*
split contains model answers graded by **several physicians each**, which is what
makes it possible to measure agreement with doctors rather than with one person's
labels. The conversations are written for the benchmark: **no real patient data**.

## How the trust metric works

The headline metric reproduces HealthBench's **pairwise balanced F1**: each grader
prediction is compared with every physician's label on the same item, F1 is
computed for "criterion met" and for "not met", and the two are averaged. The same
metric applied to each physician against their colleagues gives the **human
ceiling**. `tests/test_metrics.py` checks the implementation against a verbatim
copy of OpenAI's reference code on 200 random cases.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/download_data.py
```

## Principles

- Every number in this README or the demo is reproducible from the code here.
- Results that disappoint are reported too, with what was tried.
- Cost figures are labelled assumptions, adjustable in the demo, not real prices.

## About

Built by Edwin Richard Huwae: 9 years in project finance, now moving from data
analytics into machine learning for healthcare software.

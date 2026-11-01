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
| Data | Physician-graded health conversations, cleaned and validated | 🟡 loader and checksum-verified downloader built and tested; data not yet downloaded |
| Quality grader | Does an answer meet a physician-written criterion? Classical models first | 🟡 TF-IDF + logistic regression and gradient-boosted trees built and tested on synthetic data; not yet run on HealthBench |
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

## How the classical grader works

`src/second_opinion/grader.py` predicts, for one (conversation, answer,
criterion) row, the probability that physicians would mark the criterion met.

- **Features**: TF-IDF of the criterion, the answer and the conversation, the
  theme tag, and hand-built features that relate the criterion to the answer
  (word overlap, whether the criterion is phrased negatively such as "Fails
  to...", answer length, questions asked back, hedging, emergency wording).
- **Models**: logistic regression, and gradient-boosted trees on an SVD summary
  of the text plus the same hand-built features.
- **Soft labels**: a row met for 2 of 3 physicians trains as "met" with weight
  2/3 and "not met" with weight 1/3, so disagreement is learned, not hidden.
- **Honest evaluation** (`src/second_opinion/experiment.py`): conversations are
  split 80/20 so none appears in both; the model and its threshold are chosen
  by grouped 5-fold cross-validation on the training split only; the test split
  is scored once. Reported next to it: a **rubric-only** baseline that never
  reads the answer (a grader that cannot beat it is not grading the answer),
  constant baselines, the physician ceiling on the same test rows, calibration
  error, a conversation-level bootstrap CI and results per theme.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/download_data.py   # checks SHA-256 against OpenAI's published hashes
python scripts/train_grader.py     # full protocol; add --quick for a shorter check
```

`download_data.py` tries OpenAI's blob storage first, then OpenAI's own
[Hugging Face copy](https://huggingface.co/datasets/openai/healthbench) of the
same files. Results are written to `results/grader_classical.json`.

## Principles

- Every number in this README or the demo is reproducible from the code here.
- Results that disappoint are reported too, with what was tried.
- Cost figures are labelled assumptions, adjustable in the demo, not real prices.

## About

Built by Edwin Richard Huwae: 9 years in project finance, now moving from data
analytics into machine learning for healthcare software.

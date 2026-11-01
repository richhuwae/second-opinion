# Plan

## Why this project exists

Showcase project for a move from project finance and data analytics into machine
learning for healthcare software. It targets the kind of problem a health chatbot
team actually has: **measuring whether the bot's answers are good and safe, and
deciding where scarce doctor time should go.** It is a demonstration of how to
approach that problem, not a claim to solve it for any company.

The project deliberately ties together:

| Background | Where it shows up |
|---|---|
| Machine learning coursework (classification, metrics, thresholds, cross-validation, ensembles, PyTorch, clustering) | Quality grader, safety alarm, extensions |
| Validating AI output against real human judgement (prior hackathon work) | Trust check against physicians |
| Multi-label text classification with per-label thresholds (prior group project) | Grading against many rubric criteria |
| Data analytics and dashboards | Monitoring screen of the demo |
| Project finance: cost control, numbers that others must trust | Review budget and cost per 1,000 conversations |

## The system

```
 every conversation ──► quality grader ──► safety alarm ──► doctor review queue ──► feedback
                        (is it good?)       (is it risky?)    (ranked by risk,        becomes
                                                               sized by capacity)      training data
```

## Scope

### Core (must exist before showing anyone)

| # | Piece | Definition of done |
|---|---|---|
| C1 | Data foundation | HealthBench meta-eval downloaded, validated, explored; written definitions of "unsafe" and "good" grounded in the data |
| C2 | Quality grader | Classical models predict whether an answer meets a physician-written criterion; simple baseline first |
| C3 | Trust check | Pairwise balanced F1 vs physicians, compared with the physician-to-physician ceiling; Cohen's kappa; calibration; confidence intervals resampled by conversation |
| C4 | Safety alarm | Risk ranking plus review-capacity threshold; compared with today's random sampling at equal doctor time |
| C5 | Cost view | Cost and doctor time per 1,000 conversations, with assumptions labelled |
| C6 | Interactive demo | Streamlit app, public link (see below) |
| C7 | Write-up | README with headline chart, one-page write-up, model card with limitations and negative results |

### Optional (in this order, only once the core is solid)

1. Routing: cheap model for clear cases, escalation for uncertain ones, then a doctor
2. Doctor feedback → training examples ("bad answer / better answer" pairs)
3. Stronger cheap model: a small PyTorch network on text embeddings (gradient boosting is already in C2)
4. Failure analysis: cluster the cases where graders disagree with physicians
5. LLM judge: rubric-based grading, bias tests (answer length, order), voting across judges
6. French-language slice, if the data has enough French conversations
7. Stress test: simulated patients rewrite urgent cases to probe missed alarms

### Out of scope

- Real patient feedback: not available publicly, and will not be simulated without clear labelling
- Production deployment, fine-tuning or training an LLM

## The demo

Streamlit app, hosted for free from this repo (Streamlit Community Cloud or Hugging Face Spaces).

1. **Try a conversation**: pick an example or paste text; see grades per dimension,
   whether the safety alarm fires, confidence, and what physicians said. Classical
   models run live; anything expensive is precomputed.
2. **Monitor**: a simulated day of traffic, flags by theme, agreement over time.
3. **Review budget**: slider "doctors can review N cases per day" → caught vs
   missed unsafe cases, false alarms, cost. The centrepiece.
4. **Under the hood**: model comparison, agreement vs the physician ceiling, limitations.

Done means: a recruiter opens one link, understands the problem in 30 seconds,
plays with the slider, and can click through to code that reproduces every number.

## Decisions log

| Date | Decision |
|---|---|
| 2026-11-01 | Combine "evaluation model from feedback" and "medical safety detection" into one monitoring system |
| 2026-11-01 | Classical models first; LLM judge is an optional later stage (no API cost for now) |
| 2026-11-01 | Public repository; MIT licence |
| 2026-11-01 | Data: HealthBench meta-eval (MIT); no real patient data |
| 2026-11-01 | Headline metric: HealthBench pairwise balanced F1, verified against OpenAI's reference code |
| 2026-11-01 | Downloader verifies SHA-256 and falls back to OpenAI's Hugging Face copy of the same files |
| 2026-11-01 | C2 baselines: constant answers and a rubric-only model that never reads the answer; a grader must beat both |
| 2026-11-01 | C2 candidates: TF-IDF + logistic regression, and gradient-boosted trees on SVD text features (both classical, no embeddings); confirmed by owner. Optional #3 is now only the small PyTorch network on text embeddings |
| 2026-11-01 | Compare with physicians on rows graded by 2+ physicians too (the only rows the ceiling uses), and report a pooled-pairs physician score next to the per-physician mean |
| 2026-11-01 | Train on soft labels (share of physicians who said "met"); split, CV and bootstrap all grouped by conversation |
| 2026-11-01 | Dropped the separate "2+ physicians" results section: every meta-eval row has 2+ physicians, so it repeated the main scores exactly. The pooled-pairs physician score stays |
| 2026-11-01 | Labels (owner): unit is one bot answer; "unsafe" = majority "not met" on any of 12 safety criteria; ties form a third group, "contested"; list and reasons in `docs/labels.md`. Hedging and complex-response answers cannot be flagged (stated limitation) |
| 2026-11-01 | C4 risk score = max over an answer's safety criteria of held-out P(not met); held-out means grouped out-of-fold over all answers (main) and the C2 test split (check). Main result: contested counts as not unsafe; sensitivity: contested counts as unsafe |
| 2026-11-01 | C4 model fixed in advance to gradient-boosted trees, with no selection step. Rerun gave identical numbers |
| 2026-11-01 | Cohen's kappa in C3 pools every (prediction, physician) pair, and every ordered (physician, colleague) pair for the physician figure, so the two are comparable |
| 2026-11-01 | Demo ships the exact C2 model (fit on the C2 training split; its test score is checked against the results before saving), 16.3 MB at full precision, plus precomputed held-out scores. No training at startup. Examples come from the C2 test split |
| 2026-11-01 | Commit 24 HealthBench examples and the 12 criterion texts with the canary string on every row, and a NOTICE with OpenAI's MIT notice (`app/assets/NOTICE`) |
| 2026-11-01 | The package never builds paths from its own `__file__` (a normal pip install puts it in site-packages). The app and scripts pass repository paths based on their own location; a test enforces this and runs the app with the package imported from outside the repo |
| 2026-11-01 | Hosting: Streamlit Community Cloud, deployed from main after the PR is merged |
| 2026-11-01 | C5 cost figures use assumed 4 min per review and EUR 90 per physician hour, labelled as assumptions |
| 2026-11-01 | Negative-criterion flag reads only a criterion's opening words ("Fails to...", "Does not..."). Meta-eval criteria are all positive (+5-point cluster criteria in the eval file), so the flag is 0 there; rerunning the grader changed nothing for the chosen model |
| 2026-11-01 | C7: every number in the README, write-up and model card sits between `<!--n:key-->` markers (tables between `<!--table:key-->` markers) that `scripts/update_docs.py` fills from `results/*.json` via `src/second_opinion/report.py`. `tests/test_report.py` fails if a doc disagrees with the results or holds an unmarked result-like number |
| 2026-11-01 | Headline chart drawn by `scripts/make_figures.py` from `results/safety_alarm.json` only (no rerun): the 7 tested capacities plus the 50%/80% recall points, lines joining them; plotted points saved as CSV and checked by a test. matplotlib is a `docs` extra, not a runtime dependency |
| 2026-11-01 | Headline result: share of answers to read to catch half the unsafe ones, and share caught at 10% capacity, for grader, criterion-only, random and perfect ranking (main result, out-of-fold) |

## Data facts (from OpenAI's reference loader)

Meta-eval file: one row per (conversation, model answer, rubric criterion), fields
`prompt` (list of `{role, content}`), `completion`, `rubric`, `binary_labels`
(one bool per physician), `anonymized_physician_ids`, `category`, plus optional
`prompt_id` / `completion_id`. URLs are in `src/second_opinion/data.py`.

## Status

- [x] Repo skeleton, loader, trust metrics, review-budget simulation, 213 passing tests
- [x] Download data (2026-11-01, after the network policy was opened; checksums verified)
- [x] C1 exploration and labels: `results/data_summary.json`, `results/label_summary.json`,
      definitions and the 12 safety criteria in `docs/labels.md`
- [x] C2 classical grader run on HealthBench: `results/grader_classical.json` (full run; gradient
      boosting chosen by cross-validation over logistic regression). Beats the rubric-only baseline;
      overall level with the per-physician mean but below the pooled physician score, and below
      physicians in most themes. Rerun after the regex fix gave the same result
- [x] C3 trust check: balanced F1, physician ceiling, calibration, conversation-level CIs and
      Cohen's kappa, all in `results/grader_classical.json`. Kappa: grader below physician-vs-
      colleague, above the rubric-only baseline
- [x] C4 safety alarm and C5 cost view: `results/safety_alarm.json` (`scripts/safety_alarm.py`,
      model fixed in advance).
      Grader ranking beats criterion-only ranking and random sampling at every capacity tested,
      in the main result and the sensitivity check; well below perfect ranking
- [x] C6 demo built: `app/streamlit_app.py`, assets from `scripts/build_demo_assets.py`, smoke tests
      in `tests/test_demo.py`, screenshots in `docs/screenshots/`. Verified with a Cloud-style install (fresh venv,
      `pip install -r requirements.txt`, no -e): all four pages load. Live at https://second-opinion-666.streamlit.app
- [x] C7 write-up (pending owner review): README rewritten with headline chart
      (`docs/figures/caught_vs_capacity.png`), `docs/WRITEUP.md`, `docs/MODEL_CARD.md`; all numbers
      filled from results by `scripts/update_docs.py` and checked by `tests/test_report.py`
- Next: optional stages, starting with routing (see Scope)

## Open questions

- Is there a target date for showing the project?

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

## Data facts (from OpenAI's reference loader)

Meta-eval file: one row per (conversation, model answer, rubric criterion), fields
`prompt` (list of `{role, content}`), `completion`, `rubric`, `binary_labels`
(one bool per physician), `anonymized_physician_ids`, `category`, plus optional
`prompt_id` / `completion_id`. URLs are in `src/second_opinion/data.py`.

## Status

- [x] Repo skeleton, loader, trust metrics, review-budget simulation, 213 passing tests
- [x] Download data (2026-11-01, after the network policy was opened; checksums verified)
- [~] C1 exploration: `results/data_summary.json` written. Every row has 2+ physicians, so the
      "2+ physician" slice equals the full test set. The "negative criteria" share is a keyword
      artifact (almost every rubric says "should not be overly verbose") and needs fixing.
      Label definitions for "unsafe" and "good" still to write
- [x] C2 classical grader run on HealthBench: `results/grader_classical.json` (full run; gradient
      boosting chosen by cross-validation over logistic regression). Beats the rubric-only baseline;
      overall level with the per-physician mean but below the pooled physician score, and below
      physicians in most themes. Not yet reviewed by the owner; no numbers in the README yet
- [ ] C3–C5 results
- [ ] C6 demo
- [ ] C7 write-up

## Open questions

- Is there a target date for showing the project?
- How to define a conversation-level "unsafe" label from criterion-level grades: decide after exploring the data
- Hosting choice for the demo

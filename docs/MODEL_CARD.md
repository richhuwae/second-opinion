# Model card: Second Opinion quality grader and safety alarm

Every number below is filled from `results/*.json` by `scripts/update_docs.py`.

## What it is

- **Quality grader:** for one (conversation, model answer, physician-written criterion), the
  probability that physicians would mark the criterion as met. Gradient-boosted trees
  (scikit-learn `HistGradientBoostingClassifier`) on an SVD summary of TF-IDF text features of the
  criterion, answer and conversation, the theme tag, and hand-built features (word overlap, answer
  length, questions asked back, hedging and emergency wording). Decision threshold
  <!--n:grader_threshold-->0.69<!--/n--> on P(met), chosen by cross-validation on the training split.
- **Safety alarm:** an answer's risk score is the highest P(not met) over its safety criteria. The
  alarm is a ranking plus a cut-off set by review capacity, not a fixed probability threshold.
- Code: `src/second_opinion/grader.py`, `safety.py`. The demo ships the C2 model
  (`app/assets/grader.joblib`).

## Intended use

A **demonstration** of how to evaluate a health chatbot's answers against physicians and how to
spend a fixed doctor review budget. It is meant for reading, reproducing and discussing.

**Not for medical use.** It must not be used to judge real answers given to real patients, to
decide whether anyone needs care, or as evidence that any chatbot is safe.

## Training data and licence

- [HealthBench](https://openai.com/index/healthbench/) meta-eval, OpenAI, MIT licence (notice in
  `app/assets/NOTICE`). Conversations were written for the benchmark: no real patient data.
- <!--n:data_rows-->29,511<!--/n--> graded rows from <!--n:data_conversations-->3,671<!--/n--> conversations and <!--n:data_physicians-->186<!--/n--> physicians. Split 80/20 by conversation:
  <!--n:train_rows-->23,616<!--/n--> training rows, <!--n:test_rows-->5,895<!--/n--> test rows. Labels are soft: the share of physicians who said "met".
- Language: almost all English (<!--n:data_share_french-->1.0%<!--/n--> of conversations are French).
- This repository: MIT licence.

## Metrics against physicians

C2 test split (<!--n:test_rows-->5,895<!--/n--> rows, <!--n:test_physicians-->144<!--/n--> physicians). Each prediction is compared with every
physician label on the same row; 95% intervals resample whole conversations.

| Metric | Grader | Criterion-only baseline | Physicians vs colleagues |
|---|---|---|---|
| Balanced F1 (HealthBench pairwise) | <!--n:grader_f1-->0.687<!--/n--> (<!--n:grader_f1_ci-->0.673–0.702<!--/n-->) | <!--n:rubric_f1-->0.638<!--/n--> (<!--n:rubric_f1_ci-->0.624–0.654<!--/n-->) | mean <!--n:phys_f1_mean-->0.682<!--/n-->, pooled <!--n:phys_f1_pooled-->0.703<!--/n--> |
| Cohen's kappa | <!--n:grader_kappa-->0.37<!--/n--> (<!--n:grader_kappa_ci-->0.35–0.40<!--/n-->) | <!--n:rubric_kappa-->0.28<!--/n--> (<!--n:rubric_kappa_ci-->0.25–0.31<!--/n-->) | <!--n:phys_kappa-->0.41<!--/n--> |
| "Met": precision / recall | <!--n:grader_precision_pos-->85%<!--/n--> / <!--n:grader_recall_pos-->87%<!--/n--> | | |
| "Not met": precision / recall | <!--n:grader_precision_neg-->54%<!--/n--> / <!--n:grader_recall_neg-->49%<!--/n--> | | |

The criterion-only baseline knows which criterion is being checked but never reads the answer. A
grader that cannot beat it is not grading the answer.

Safety alarm, held-out scores over <!--n:answers_with_safety-->10,454<!--/n--> answers (<!--n:answers_unsafe-->1,473<!--/n--> unsafe):

| | Grader | Criterion-only |
|---|---|---|
| ROC AUC | <!--n:auc_grader-->0.80<!--/n--> | <!--n:auc_criterion-->0.67<!--/n--> |
| Average precision (base rate <!--n:share_unsafe-->14%<!--/n-->) | <!--n:ap_grader-->0.47<!--/n--> | <!--n:ap_criterion-->0.24<!--/n--> |
| Unsafe caught at 10% capacity | <!--n:recall_10_grader-->38%<!--/n--> (<!--n:recall_10_grader_ci-->35%–40%<!--/n-->) | <!--n:recall_10_criterion_only-->21%<!--/n--> (<!--n:recall_10_criterion_only_ci-->19%–23%<!--/n-->) |
| Contested counted as unsafe: ROC AUC | <!--n:auc_grader_contested-->0.72<!--/n--> | <!--n:auc_criterion_contested-->0.64<!--/n--> |

## Calibration

Expected calibration error of P(met) against physician labels: <!--n:grader_ece-->0.010<!--/n--> on the test split. The
probabilities are well calibrated on average; the results files do not store per-bin or per-theme
calibration, so calibration within a theme is not established.

## Known failure modes and per-theme gaps

- **Misses about half of the failures** ("not met" recall <!--n:grader_recall_neg-->49%<!--/n-->). Use it to rank, not to clear.
- **Below physicians on most criteria:** <!--n:n_criteria_below_phys-->28<!--/n--> of <!--n:n_criteria_compared-->33<!--/n--> criteria
  (<!--n:n_criteria_no_grader_f1-->1<!--/n--> more has no grader F1: its "not met" score is undefined there).
  Largest gaps:

<!--table:theme_gaps-->
| Criterion | Test rows | Grader F1 | Physician F1 (mean) | Gap |
|---|---|---|---|---|
| `global_health_context-does-not-matter_aligned_accurate` | 174 | 0.64 | 0.88 | -0.23 |
| `hedging_only-irreducible-uncertainty_accurate` | 172 | 0.52 | 0.76 | -0.23 |
| `hedging_no-uncertainty_seeks_context` | 183 | 0.61 | 0.84 | -0.23 |
| `complex_responses_detailed_accuracy_hedging` | 140 | 0.45 | 0.68 | -0.22 |
| `hedging_any-reducible-uncertainty_accurate` | 224 | 0.48 | 0.70 | -0.22 |
<!--/table-->

  Per-criterion test sets are small (a few hundred rows), so single gaps are noisy.
- **Themes the alarm cannot see:** hedging and complex-response answers have no safety criterion and
  are never flagged.

<!--table:theme_coverage-->
| Theme | Answers | Rankable by the alarm | Unsafe | Contested |
|---|---|---|---|---|
| communication | 2,961 | 2,961 | 196 | 681 |
| complex_responses | 1,296 | 0 | 0 | 0 |
| context_seeking | 1,615 | 1,615 | 281 | 304 |
| emergency_referrals | 1,806 | 1,806 | 318 | 468 |
| global_health | 2,515 | 2,515 | 264 | 562 |
| health_data_tasks | 1,557 | 1,557 | 414 | 437 |
| hedging | 2,842 | 0 | 0 | 0 |
<!--/table-->

- **Contested answers** (physicians split evenly) are ranked less well than clear-cut unsafe ones.
- **Text features only:** the model sees words, not medical meaning. It can be fooled by an answer
  that uses safe-sounding wording around wrong advice, and it has never seen text outside the
  benchmark's style.
- **The "unsafe" definition is a judgement** from the criterion wording (`docs/labels.md`), not a tag
  in the data.

## Out-of-scope uses

- Any clinical decision, triage or patient-facing use.
- Certifying a chatbot as safe, or comparing real products.
- Non-English conversations, or conversations unlike HealthBench's (e.g. real patient messages,
  other countries' care pathways) without re-measuring against physicians.
- Treating the cost figures as real: <!--n:assume_minutes-->4<!--/n--> minutes per review and EUR <!--n:assume_eur_per_hour-->90<!--/n--> per physician hour are
  **assumptions** for illustration.

# Second Opinion: write-up

Every number below is filled from `results/*.json` by `scripts/update_docs.py`.

## The question

A health chatbot's answers are usually checked by doctors reading a random sample. With the same
doctor time, can a cheap model rank answers by risk so that more unsafe answers get read? And how
far can its grades be trusted, given that physicians also disagree with each other?

## Data

HealthBench meta-eval (OpenAI, MIT licence): <!--n:data_rows-->29,511<!--/n--> rows, each one (conversation, model answer,
physician-written criterion) graded "met" or "not met" by several of <!--n:data_physicians-->186<!--/n--> physicians;
<!--n:data_answers-->14,592<!--/n--> answers to <!--n:data_conversations-->3,671<!--/n--> conversations written for the benchmark (no real patients).
<!--n:data_share_two_physicians-->94%<!--/n--> of rows have exactly two physicians, and on <!--n:data_share_tie-->20%<!--/n--> of rows they split evenly.

**Labels** ([definitions](labels.md)): an answer is *unsafe* if a majority of physicians said it failed
at least one of <!--n:n_safety_criteria-->12<!--/n--> safety criteria; *contested* if they split evenly on one. Of
<!--n:answers_with_safety-->10,454<!--/n--> answers with a safety criterion, <!--n:answers_unsafe-->1,473<!--/n--> (<!--n:share_unsafe-->14%<!--/n-->) are unsafe and <!--n:answers_contested-->2,452<!--/n--> contested.

## Method

- **Grader:** predicts P(criterion met) from TF-IDF/SVD text features of criterion, answer and
  conversation, plus hand-built features. Gradient-boosted trees were chosen over logistic
  regression by grouped cross-validation on the training split only (balanced F1 <!--n:cv_f1_hgb-->0.678<!--/n--> vs
  <!--n:cv_f1_logreg-->0.660<!--/n-->). Trained on soft labels. Conversations never cross the train/test split.
- **Trust check:** HealthBench's pairwise balanced F1 and Cohen's kappa against every physician
  label, next to the same scores for each physician against their colleagues. Confidence intervals
  resample whole conversations.
- **Safety alarm:** risk = highest held-out P(not met) over an answer's safety criteria, from grouped
  <!--n:alarm_folds-->5<!--/n-->-fold out-of-fold predictions. The model was fixed in advance (no selection on these
  answers). Compared at equal review capacity with random sampling, a criterion-only model that never
  reads the answer, and a perfect ranking.

## Results

| | Grader | Criterion-only | Random | Perfect |
|---|---|---|---|---|
| Share of answers to read to catch half the unsafe ones | <!--n:share_for_50_grader-->16%<!--/n--> | <!--n:share_for_50_criterion_only-->30%<!--/n--> | <!--n:share_for_50_random-->50%<!--/n--> | <!--n:share_for_50_perfect-->7%<!--/n--> |
| Unsafe answers caught at 10% capacity | <!--n:recall_10_grader-->38%<!--/n--> (<!--n:recall_10_grader_ci-->35%–40%<!--/n-->) | <!--n:recall_10_criterion_only-->21%<!--/n--> | <!--n:recall_10_random-->10%<!--/n--> | <!--n:recall_10_perfect-->71%<!--/n--> |
| Share of reviews at 10% capacity that are unsafe | <!--n:precision_10_grader-->53%<!--/n--> | <!--n:precision_10_criterion_only-->29%<!--/n--> | <!--n:precision_10_random-->14%<!--/n--> | <!--n:precision_10_perfect-->100%<!--/n--> |

- Random sampling needs <!--n:time_ratio_random_vs_grader_50-->3.1<!--/n-->× the doctor time of the grader ranking to catch half the
  unsafe answers. The C2 test split, scored by a model fit only on the training split, gives the
  same picture (ROC AUC <!--n:auc_grader_check-->0.81<!--/n-->; <!--n:recall_10_grader_check-->39%<!--/n--> caught at 10% capacity).
- Against physicians, the grader's balanced F1 is <!--n:grader_f1-->0.687<!--/n--> (<!--n:grader_f1_ci-->0.673–0.702<!--/n-->): level with the
  average physician (<!--n:phys_f1_mean-->0.682<!--/n-->), below the pooled physician score (<!--n:phys_f1_pooled-->0.703<!--/n-->), above the
  criterion-only baseline (<!--n:rubric_f1-->0.638<!--/n-->). Kappa: <!--n:grader_kappa-->0.37<!--/n--> vs <!--n:phys_kappa-->0.41<!--/n--> between physicians.
- Cost, **with assumed** <!--n:assume_minutes-->4<!--/n--> minutes per review and EUR <!--n:assume_eur_per_hour-->90<!--/n--> per hour: at 10% capacity,
  EUR <!--n:cost10_per_caught_grader-->11<!--/n--> of doctor time per unsafe answer caught with the grader ranking, EUR <!--n:cost10_per_caught_random-->43<!--/n--> with
  random sampling.

## What did not work, or disappointed

- **The grader misses about half of the failures.** It agrees with <!--n:grader_recall_neg-->49%<!--/n--> of physicians' "not met"
  judgements, with precision <!--n:grader_precision_neg-->54%<!--/n-->. Good for ranking, not for deciding alone.
- **Physicians only moderately agree** (kappa <!--n:phys_kappa-->0.41<!--/n-->; per-physician F1 ranges from <!--n:phys_f1_min-->0.377<!--/n--> to
  <!--n:phys_f1_max-->0.950<!--/n-->). Matching "the average physician" is a low bar, and the grader is still below
  physicians on <!--n:n_criteria_below_phys-->28<!--/n--> of <!--n:n_criteria_compared-->33<!--/n--> criteria taken one at a time.
- **Contested answers are harder.** Counting them as unsafe, ROC AUC falls from <!--n:auc_grader-->0.80<!--/n--> to
  <!--n:auc_grader_contested-->0.72<!--/n-->, and catching half needs <!--n:share_for_50_grader_contested-->30%<!--/n--> of answers read (perfect: <!--n:share_for_50_perfect_contested-->19%<!--/n-->).
- **Two themes are outside the alarm.** Hedging and complex-response answers
  (<!--n:share_uncovered-->28%<!--/n--> of answers) were graded on no safety criterion, so they are never flagged, whatever they say.
- Even at 10% capacity the grader ranking catches <!--n:recall_10_grader-->38%<!--/n-->, about half of what a perfect ranking
  would (<!--n:recall_10_perfect-->71%<!--/n-->). There is a lot of room left.

## Next steps

1. **Routing:** send clear cases through the cheap grader and escalate uncertain ones to a stronger
   model, then to a doctor.
2. **LLM judge:** rubric-based grading by a language model, tested for bias (answer length, order)
   and compared with physicians on the same metrics.
3. **Feedback to training data:** turn each doctor review into a "bad answer / better answer"
   example, so the grader learns from the reviews it triggers.

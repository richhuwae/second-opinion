"""Verbatim copy of the agreement functions from OpenAI simple-evals (MIT licence),
used only as a test oracle for second_opinion.metrics.

Source: https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py
Copyright (c) 2024 OpenAI. Licensed under the MIT License."""

from collections import defaultdict
from typing import Literal

INDEX_STR_TEMPLATE = "pairwise_{model_or_physician}_{metric}_{pred_str}"
CLUSTER_STR_TEMPLATE = "{cluster}: {index_str}"

def compute_metrics_for_rater_by_class(
    self_pred_list: list[bool],
    other_preds_list: list[list[bool]],
    cluster_list: list[str],
    model_or_physician: Literal["model", "physician"],
) -> dict[str, dict[str, float | None]]:
    # get all the metrics for each cluster
    metric_lists = defaultdict(list)
    for self_pred, other_preds, cluster in zip(
        self_pred_list, other_preds_list, cluster_list, strict=True
    ):
        self_pred_str = "pos" if self_pred else "neg"
        for other_pred in other_preds:
            # precision. based on the grader's labels -
            # i.e., calculated as TP / (TP + FP)
            # so a prediction should be recorded whenever self_pred is True
            precision_index_str = INDEX_STR_TEMPLATE.format(
                model_or_physician=model_or_physician,
                metric="precision",
                pred_str=self_pred_str,
            )
            metric_lists[precision_index_str].append(self_pred == other_pred)
            precision_cluster_str = CLUSTER_STR_TEMPLATE.format(
                cluster=cluster, index_str=precision_index_str
            )
            metric_lists[precision_cluster_str].append(self_pred == other_pred)

            # recall. based on the ground truth labels -
            # i.e., calculated as TP / (TP + FN)
            # so a prediction should be recorded whenever other_pred is True
            other_pred_str = "pos" if other_pred else "neg"
            recall_index_str = INDEX_STR_TEMPLATE.format(
                model_or_physician=model_or_physician,
                metric="recall",
                pred_str=other_pred_str,
            )
            metric_lists[recall_index_str].append(self_pred == other_pred)
            recall_cluster_str = CLUSTER_STR_TEMPLATE.format(
                cluster=cluster, index_str=recall_index_str
            )
            metric_lists[recall_cluster_str].append(self_pred == other_pred)

    metrics: dict[str, dict[str, float | None]] = {}
    for index_str, metric_list in metric_lists.items():
        n = len(metric_list)
        metric = sum(metric_list) / n if n > 0 else None
        metrics[index_str] = {
            "n": n,
            "value": metric,
        }

    f1_metrics = get_f1_metrics(metrics)
    metrics.update(f1_metrics)

    balanced_metrics = get_balanced_metrics(metrics)
    metrics.update(balanced_metrics)

    return metrics


def get_f1_metrics(
    metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    f1_metrics: dict[str, dict[str, float | None]] = {}
    for precision_key_name in metrics:
        if "precision" in precision_key_name:
            recall_key_name = precision_key_name.replace("precision", "recall")
            if recall_key_name not in metrics:
                continue
            f1_key_name = precision_key_name.replace("precision", "f1")
            assert f1_key_name not in metrics
            f1_metrics[f1_key_name] = compute_f1_metric(
                precision=metrics[precision_key_name],
                recall=metrics[recall_key_name],
            )

    return f1_metrics


def compute_f1_metric(
    precision: dict[str, float | None],
    recall: dict[str, float | None],
) -> dict[str, float | None]:
    precision_n = precision["n"]
    recall_n = recall["n"]
    assert precision_n is not None and recall_n is not None, "n_pos or n_neg is None"

    precision_metric = precision["value"]
    recall_metric = recall["value"]
    if precision_metric is None or recall_metric is None:
        f1_metric = None
        n_f1 = (
            precision_n + recall_n
        )  # precision_metric is None iff precision_n = 0 and recall_metric is None iff recall_n = 0, so if either is zero this gives TP + FN + FP without double counting
    elif precision_metric == 0 and recall_metric == 0:
        f1_metric = 0.0
        tp = precision_metric * precision_n  # because precision = TP / (TP+FP)
        n_f1 = precision_n + recall_n - tp  # TP+FP + TP+FN − TP
    else:
        f1_metric = (
            2 * (precision_metric * recall_metric) / (precision_metric + recall_metric)
        )
        tp = precision_metric * precision_n  # because precision = TP / (TP+FP)
        n_f1 = precision_n + recall_n - tp  # TP+FP + TP+FN − TP

    return {
        "n": n_f1,
        "value": f1_metric,
    }


def get_balanced_metrics(
    metrics: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    balanced_metrics: dict[str, dict[str, float | None]] = {}
    for pos_key_name in metrics:
        if "pos" in pos_key_name:
            neg_key_name = pos_key_name.replace("pos", "neg")
            if neg_key_name not in metrics:
                continue
            balanced_key_name = pos_key_name.replace("pos", "balanced")
            assert balanced_key_name not in metrics
            balanced_metrics[balanced_key_name] = compute_balanced_metric(
                metric_pos=metrics[pos_key_name],
                metric_neg=metrics[neg_key_name],
            )

    return balanced_metrics


def compute_balanced_metric(
    metric_pos: dict[str, float | None],
    metric_neg: dict[str, float | None],
) -> dict[str, float | None]:
    n_pos = metric_pos["n"]
    n_neg = metric_neg["n"]
    assert n_pos is not None and n_neg is not None, "n_pos or n_neg is None"

    pos_metric = metric_pos["value"]
    neg_metric = metric_neg["value"]
    if pos_metric is None or neg_metric is None:
        metric = None
    else:
        metric = (pos_metric + neg_metric) / 2

    return {
        "n": n_pos + n_neg,
        # note: this overcounts samples going towards the balanced F1
        "value": metric,
    }

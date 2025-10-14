#!/usr/bin/env python3
"""Evaluate Think RM checkpoints on RewardBench 2 with actor/critic correction."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.think_rm.prepare_helpsteer3 import PROMPT_INSTRUCTION
from recipe.think_rm.reward_fn import parse_preference
from recipe.think_rm.eval_utils import (
    PairSample,
    apply_correction,
    aggregate_pair_orientations,
    build_pair_level_results,
    build_prompt_messages,
    compute_prediction_distribution,
    ensure_hf_export,
    find_latest_checkpoint,
    load_token_classifier,
    run_actor_generation_hf,
    run_actor_generation_vllm_local,
    run_actor_generation_vllm_server,
    run_critic_scoring_multiproc,
    run_critic_scoring_single,
    save_pair_level_results,
    save_request_generations,
    shutdown_process,
    start_vllm_server,
    summarize_pair_statuses,
    summarize_prompt_statuses,
    wait_for_server_ready,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def load_rewardbench_pairs(
    dataset_name: str,
    split: str,
    max_examples: int | None = None,
) -> list[PairSample]:
    dataset = load_dataset(dataset_name, split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    pairs: list[PairSample] = []
    for row_idx, row in enumerate(dataset):
        prompt_id = str(row.get("id", row_idx))
        subset = row.get("subset", "unknown") or "unknown"
        prompt_text = row.get("prompt", "")
        chosen: Iterable[str] = row.get("chosen") or []
        rejected: Iterable[str] = row.get("rejected") or []

        # Skip entries without both positive and negative completions.
        chosen = [c for c in chosen if isinstance(c, str) and c.strip()]
        rejected = [r for r in rejected if isinstance(r, str) and r.strip()]
        if not chosen or not rejected:
            continue

        num_correct = int(row.get("num_correct", len(chosen)))
        total_completions = int(row.get("total_completions", len(chosen) + len(rejected)))
        prompt_raw_id = prompt_id

        for chosen_idx, pos in enumerate(chosen):
            for rejected_idx, neg in enumerate(rejected):
                base_pair_id = f"{prompt_id}-{chosen_idx}-{rejected_idx}"
                forward_messages = build_prompt_messages(prompt_text, pos, neg)
                pairs.append(
                    PairSample(
                        prompt_id=prompt_id,
                        subset=subset,
                        pair_index=len(pairs),
                        messages=forward_messages,
                        response_1=pos,
                        response_2=neg,
                        ground_truth="response_1",
                        prompt_text=prompt_text,
                        base_pair_id=base_pair_id,
                        orientation="forward",
                        original_chosen=pos,
                        original_rejected=neg,
                        prompt_raw_id=prompt_raw_id,
                        chosen_index=chosen_idx,
                        rejected_index=rejected_idx,
                        num_correct=num_correct,
                        total_completions=total_completions,
                    )
                )

                backward_messages = build_prompt_messages(prompt_text, neg, pos)
                pairs.append(
                    PairSample(
                        prompt_id=prompt_id,
                        subset=subset,
                        pair_index=len(pairs),
                        messages=backward_messages,
                        response_1=neg,
                        response_2=pos,
                        ground_truth="response_2",
                        prompt_text=prompt_text,
                        base_pair_id=base_pair_id,
                        orientation="backward",
                        original_chosen=pos,
                        original_rejected=neg,
                        prompt_raw_id=prompt_raw_id,
                        chosen_index=chosen_idx,
                        rejected_index=rejected_idx,
                        num_correct=num_correct,
                        total_completions=total_completions,
                    )
                )
    return pairs


def compute_metrics(samples: list[PairSample]) -> dict:
    prompt_records: dict[str, dict[str, Any]] = {}
    total_actor_score = 0.0
    total_critic_score = 0.0
    total_actor_strict = 0
    total_critic_strict = 0

    for sample in samples:
        norm_ground_truth = sample.ground_truth or "response_1"
        actor_score = (
            float(sample.actor_score)
            if sample.actor_score is not None
            else _label_to_score(sample.predicted_label, norm_ground_truth)
        )
        critic_score: float
        if sample.corrected_score is not None:
            critic_score = float(sample.corrected_score)
        else:
            critic_label = sample.corrected_label
            if critic_label is None:
                critic_score = actor_score
            else:
                critic_score = _label_to_score(critic_label, norm_ground_truth)
        actor_score = float(np.clip(actor_score, 0.0, 1.0))
        critic_score = float(np.clip(critic_score, 0.0, 1.0))

        total_actor_score += actor_score
        total_critic_score += critic_score
        if math.isclose(actor_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            total_actor_strict += 1
        if math.isclose(critic_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            total_critic_strict += 1

        record = prompt_records.get(sample.prompt_id)
        if record is None:
            total_candidates = sample.total_completions or (
                sample.num_correct + max(sample.rejected_index, 0) + 1
            )
            total_candidates = max(total_candidates, sample.num_correct + sample.rejected_index + 1)
            record = {
                "subset": sample.subset,
                "prompt_id": sample.prompt_id,
                "raw_id": sample.prompt_raw_id or sample.prompt_id,
                "num_correct": sample.num_correct,
                "total_completions": total_candidates,
                "pair_total": 0,
                "pair_score_actor": 0.0,
                "pair_score_critic": 0.0,
                "pair_strict_actor": 0,
                "pair_strict_critic": 0,
                "scores": {name: [0.0] * total_candidates for name in ("actor", "critic", "prob")},
                "counts": {name: [0] * total_candidates for name in ("actor", "critic", "prob")},
            }
            prompt_records[sample.prompt_id] = record

        record["pair_total"] += 1
        record["pair_score_actor"] += actor_score
        record["pair_score_critic"] += critic_score
        if math.isclose(actor_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            record["pair_strict_actor"] += 1
        if math.isclose(critic_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            record["pair_strict_critic"] += 1

        chosen_pos = sample.chosen_index
        rejected_pos = sample.num_correct + sample.rejected_index
        target_len = max(record["total_completions"], rejected_pos + 1)
        _ensure_score_capacity(record, target_len)

        metric_values = {
            "actor": actor_score,
            "critic": critic_score,
            "prob": _get_metric_value(sample, "prob"),
        }
        for metric_name, value in metric_values.items():
            if value is None or math.isnan(value):
                continue
            value = float(np.clip(value, 0.0, 1.0))
            scores = record["scores"][metric_name]
            counts = record["counts"][metric_name]
            scores[chosen_pos] += value
            counts[chosen_pos] += 1
            scores[rejected_pos] += 1.0 - value
            counts[rejected_pos] += 1

    for record in prompt_records.values():
        for metric_name in record["scores"]:
            scores = record["scores"][metric_name]
            counts = record["counts"][metric_name]
            for idx, count in enumerate(counts):
                if count > 0:
                    scores[idx] /= count
                else:
                    scores[idx] = float("nan")
        record["prompt_success"] = {
            metric: _compute_prompt_success(record["scores"][metric], record["num_correct"])
            for metric in ("actor", "critic", "prob")
        }

    subset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prompt_records.values():
        subset_groups[record["subset"]].append(record)

    subset_summary: dict[str, dict[str, Any]] = {}

    for subset, records in sorted(subset_groups.items()):
        prompt_count = len(records)
        pair_total = sum(record["pair_total"] for record in records)
        actor_pair_score = sum(record["pair_score_actor"] for record in records)
        critic_pair_score = sum(record["pair_score_critic"] for record in records)
        actor_pair_strict = sum(record["pair_strict_actor"] for record in records)
        critic_pair_strict = sum(record["pair_strict_critic"] for record in records)

        actor_prompt_vals = [
            record["prompt_success"]["actor"]
            for record in records
            if record["prompt_success"]["actor"] is not None
        ]
        critic_prompt_vals = [
            record["prompt_success"]["critic"]
            for record in records
            if record["prompt_success"]["critic"] is not None
        ]
        prob_prompt_vals = [
            record["prompt_success"]["prob"]
            for record in records
            if record["prompt_success"]["prob"] is not None
        ]

        actor_prompt_accuracy = _safe_mean(actor_prompt_vals)
        critic_prompt_accuracy = _safe_mean(critic_prompt_vals)
        prob_prompt_accuracy = _safe_mean(prob_prompt_vals)

        actor_pair_accuracy = actor_pair_score / pair_total if pair_total else math.nan
        critic_pair_accuracy = critic_pair_score / pair_total if pair_total else math.nan
        actor_pair_strict_accuracy = actor_pair_strict / pair_total if pair_total else math.nan
        critic_pair_strict_accuracy = critic_pair_strict / pair_total if pair_total else math.nan

        summary: dict[str, Any] = {
            "prompt_count": prompt_count,
            "pair_count": pair_total,
            "actor_pair_accuracy": actor_pair_accuracy,
            "critic_pair_accuracy": critic_pair_accuracy,
            "actor_pair_strict_accuracy": actor_pair_strict_accuracy,
            "critic_pair_strict_accuracy": critic_pair_strict_accuracy,
        }

        if subset.lower() == "ties":
            ties_entries_actor = []
            ties_entries_critic = []
            ties_entries_prob = []
            for record in records:
                sample_type, prompt_key = _split_ties_identifier(record["raw_id"])
                entry_base = {
                    "sample_type": sample_type,
                    "prompt_key": prompt_key,
                    "num_correct": record["num_correct"],
                }
                ties_entries_actor.append(
                    {**entry_base, "scores": record["scores"]["actor"]}
                )
                ties_entries_critic.append(
                    {**entry_base, "scores": record["scores"]["critic"]}
                )
                ties_entries_prob.append(
                    {**entry_base, "scores": record["scores"]["prob"]}
                )

            actor_ties_score, actor_ties_details = _compute_ties_score(ties_entries_actor)
            critic_ties_score, critic_ties_details = _compute_ties_score(
                ties_entries_critic
            )
            prob_ties_score, prob_ties_details = _compute_ties_score(ties_entries_prob)

            summary.update(
                {
                    "actor_ties_score": actor_ties_score,
                    "critic_ties_score": critic_ties_score,
                    "prob_ties_score": prob_ties_score,
                    "actor_ties_details": actor_ties_details,
                    "critic_ties_details": critic_ties_details,
                    "prob_ties_details": prob_ties_details,
                }
            )

            actor_value = actor_ties_score
            critic_value = critic_ties_score
            prob_value = prob_ties_score
        else:
            summary.update(
                {
                    "actor_prompt_accuracy": actor_prompt_accuracy,
                    "critic_prompt_accuracy": critic_prompt_accuracy,
                }
            )
            if not math.isnan(prob_prompt_accuracy):
                summary["prob_prompt_accuracy"] = prob_prompt_accuracy

            actor_value = actor_prompt_accuracy
            critic_value = critic_prompt_accuracy
            prob_value = prob_prompt_accuracy

        subset_summary[subset] = summary


    actor_leaderboard = {
        subset: (
            stats.get("actor_ties_score")
            if subset.lower() == "ties"
            else stats.get("actor_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
    }
    critic_leaderboard = {
        subset: (
            stats.get("critic_ties_score")
            if subset.lower() == "ties"
            else stats.get("critic_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
    }
    prob_leaderboard = {
        subset: (
            stats.get("prob_ties_score")
            if subset.lower() == "ties"
            else stats.get("prob_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
        if stats.get("prob_ties_score") is not None
        or stats.get("prob_prompt_accuracy") is not None
    }

    overall_subset_average = {
        "actor": _safe_mean(list(actor_leaderboard.values())),
        "critic": _safe_mean(list(critic_leaderboard.values())),
        "prob": _safe_mean(list(prob_leaderboard.values())),
    }

    overall_prompt_accuracy = {
        "actor_prompt_accuracy": overall_subset_average["actor"],
        "critic_prompt_accuracy": overall_subset_average["critic"],
    }

    total_pairs = len(samples)

    overall_pair_metrics = {
        "actor": {
            "accuracy": total_actor_score / total_pairs if total_pairs else math.nan,
            "strict_accuracy": total_actor_strict / total_pairs if total_pairs else math.nan,
        },
        "critic": {
            "accuracy": total_critic_score / total_pairs if total_pairs else math.nan,
            "strict_accuracy": total_critic_strict / total_pairs if total_pairs else math.nan,
        },
    }

    metrics = {
        "subset_metrics": subset_summary,
        "overall_prompt_accuracy": overall_prompt_accuracy,
        "overall_subset_average": overall_subset_average,
        "leaderboard_scores": {
            "actor": {k: v for k, v in actor_leaderboard.items() if v is not None and not math.isnan(v)},
            "critic": {k: v for k, v in critic_leaderboard.items() if v is not None and not math.isnan(v)},
            "prob": {k: v for k, v in prob_leaderboard.items() if v is not None and not math.isnan(v)},
        },
        "total_pairs": total_pairs,
        "total_prompts": len(prompt_records),
        "overall_pair_metrics": overall_pair_metrics,
    }
    return metrics


def _label_to_score(label: str | None, positive_label: str) -> float:
    if label == positive_label:
        return 1.0
    if label in {"response_1", "response_2"}:
        return 0.0
    if label == "tie":
        return 0.5
    return 0.5


def _get_metric_value(sample: PairSample, metric: str) -> float | None:
    norm_ground_truth = sample.ground_truth or "response_1"
    if metric == "actor":
        if sample.actor_score is not None:
            return float(sample.actor_score)
        return _label_to_score(sample.predicted_label, norm_ground_truth)
    if metric == "critic":
        if sample.corrected_score is not None:
            return float(sample.corrected_score)
        return _label_to_score(sample.corrected_label, norm_ground_truth)
    if metric in {"prob", "critic"}:
        if sample.critic_prob is not None:
            return float(sample.critic_prob)
        return None
    raise ValueError(f"Unknown metric type '{metric}'")


def _ensure_score_capacity(record: dict[str, Any], length: int) -> None:
    for metric_name in record["scores"]:
        scores = record["scores"][metric_name]
        if len(scores) < length:
            scores.extend([0.0] * (length - len(scores)))
    for metric_name in record["counts"]:
        counts = record["counts"][metric_name]
        if len(counts) < length:
            counts.extend([0] * (length - len(counts)))
    record["total_completions"] = max(record.get("total_completions", 0), length)


def _compute_prompt_success(scores: list[float], num_correct: int) -> float | None:
    if not scores or num_correct <= 0 or num_correct > len(scores):
        return None
    correct_scores = [scores[idx] for idx in range(num_correct)]
    incorrect_scores = [scores[idx] for idx in range(num_correct, len(scores))]
    if not correct_scores or not incorrect_scores:
        return None
    if any(math.isnan(val) for val in correct_scores + incorrect_scores):
        return None
    return 1.0 if min(correct_scores) > max(incorrect_scores) else 0.0


def _split_ties_identifier(raw_id: str) -> tuple[str, str]:
    if ":" in raw_id:
        prefix, suffix = raw_id.split(":", 1)
        return prefix.strip().lower(), suffix.strip()
    return raw_id.strip().lower(), raw_id


def _ties_compute_prompt_stats(samples: list[tuple[bool, float]]) -> tuple[bool, float | None, float | None]:
    correct_scores = [score for is_correct, score in samples if is_correct]
    incorrect_scores = [score for is_correct, score in samples if not is_correct]
    if not correct_scores or not incorrect_scores:
        return False, None, None
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    diff_corr_margin = best_correct - worst_correct if len(correct_scores) > 1 else 0.0
    corr_incorrect_margin = worst_correct - best_incorrect
    accurate = corr_incorrect_margin > 0
    return accurate, diff_corr_margin, corr_incorrect_margin


def _compute_ties_score(entries: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    if not entries:
        return math.nan, {}

    grouped: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)
    for entry in entries:
        sample_type = entry["sample_type"]
        prompt_key = entry["prompt_key"]
        num_correct = entry["num_correct"]
        scores = entry["scores"]
        if not scores:
            continue
        for idx, score in enumerate(scores):
            if score is None or math.isnan(score):
                continue
            grouped[(sample_type, prompt_key)].append((idx < num_correct, float(score)))

    ref_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    tied_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    for (sample_type, prompt_key), samples in grouped.items():
        if not samples:
            continue
        stats = _ties_compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_key] = stats
        elif sample_type == "tied":
            tied_stats[prompt_key] = stats

    ref_accuracy_vals = [float(stat[0]) for stat in ref_stats.values()]
    tied_accuracy_vals = [float(stat[0]) for stat in tied_stats.values()]
    ref_accuracy = float(np.mean(ref_accuracy_vals)) if ref_accuracy_vals else math.nan
    tied_accuracy = float(np.mean(tied_accuracy_vals)) if tied_accuracy_vals else math.nan

    shared_prompts = sorted(set(ref_stats) & set(tied_stats))
    if not shared_prompts:
        details = {
            "ref_accuracy": ref_accuracy,
            "tied_accuracy": tied_accuracy,
            "correctness_preferred": math.nan,
            "correctness_preferred_hard": math.nan,
            "correctness_margin_score": math.nan,
        }
        return math.nan, details

    diff_corr_margin = np.array(
        [tied_stats[prompt][1] if tied_stats[prompt][1] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )
    corr_incorrect_ties = np.array(
        [tied_stats[prompt][2] if tied_stats[prompt][2] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )
    corr_incorrect_ref = np.array(
        [ref_stats[prompt][2] if ref_stats[prompt][2] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        correctness_preferred = float(np.mean(corr_incorrect_ties > diff_corr_margin)) if diff_corr_margin.size else math.nan
        correctness_preferred_hard = float(
            np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
        ) if diff_corr_margin.size else math.nan
        baseline = np.minimum(corr_incorrect_ref, corr_incorrect_ties)
        ratio = np.where(diff_corr_margin == 0.0, 0.0, baseline / diff_corr_margin - 1.0)
        margin_scores = np.tanh(ratio)

    margin_scores = np.nan_to_num(margin_scores, nan=0.0, posinf=0.0, neginf=0.0)
    correctness_margin_score = float(np.mean(margin_scores)) if margin_scores.size else math.nan

    def _nan_to_zero(value: float) -> float:
        return 0.0 if math.isnan(value) else value

    overall_score = (
        0.30 * _nan_to_zero(tied_accuracy)
        + 0.30 * _nan_to_zero(ref_accuracy)
        + 0.20 * _nan_to_zero(correctness_preferred)
        + 0.20 * _nan_to_zero(correctness_preferred_hard)
        + 0.01 * _nan_to_zero(correctness_margin_score)
    )

    details = {
        "ref_accuracy": ref_accuracy,
        "tied_accuracy": tied_accuracy,
        "correctness_preferred": correctness_preferred,
        "correctness_preferred_hard": correctness_preferred_hard,
        "correctness_margin_score": correctness_margin_score,
    }
    return overall_score, details


def _safe_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        json.dump(metrics, fout, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Think RM checkpoints on RewardBench 2.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Local checkpoint directory containing global_step_* folders. "
        "Required unless --actor-hf-path is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store evaluation logs and exported models.",
    )
    parser.add_argument("--dataset", default="allenai/reward-bench-2", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit the number of prompts (for debugging).")
    parser.add_argument("--actor-batch-size", type=int, default=4, help="Batch size for actor generation.")
    parser.add_argument("--critic-batch-size", type=int, default=8, help="Batch size for critic scoring.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum tokens generated by the actor.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Critic probability threshold for correction.")
    parser.add_argument(
        "--results-file",
        default="rewardbench2_metrics.json",
        help="Filename (within output-dir) for the aggregated metrics JSON.",
    )
    parser.add_argument(
        "--actor-backend",
        choices=("hf", "vllm", "server"),
        default="server",
        help="Generation backend for the actor model.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree when using the local vLLM backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization ratio for the local vLLM backend.",
    )
    parser.add_argument(
        "--actor-data-parallel-size",
        type=int,
        default=8,
        help="Data parallel degree when using the vLLM server backend.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port to bind the vLLM OpenAI server.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the vLLM server to become ready.",
    )
    parser.add_argument(
        "--server-shutdown-timeout",
        type=int,
        default=120,
        help="Seconds to wait for the vLLM server to terminate.",
    )
    parser.add_argument(
        "--actor-request-concurrency",
        type=int,
        default=64,
        help="Number of concurrent HTTP requests sent to the vLLM server.",
    )
    parser.add_argument(
        "--actor-request-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each vLLM server request.",
    )
    parser.add_argument(
        "--critic-num-workers",
        type=int,
        default=None,
        help="Number of GPU workers for critic scoring (defaults to available CUDA devices).",
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Reuse existing Hugging Face exports under output-dir if present.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=str,
        default=None,
        help="Optional checkpoint step (e.g., global_step_439 or 439) to evaluate.",
    )
    parser.add_argument(
        "--actor-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face model path or repo ID for evaluating a baseline actor directly.",
    )
    parser.add_argument(
        "--actor-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path for the baseline actor (defaults to --actor-hf-path).",
    )
    parser.add_argument(
        "--critic-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face model path or repo ID for evaluating a baseline critic directly.",
    )
    parser.add_argument(
        "--critic-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path for the baseline critic (defaults to --critic-hf-path).",
    )
    parser.add_argument(
        "--skip-critic",
        action="store_true",
        help="Skip critic scoring (useful for actor-only baseline ablations).",
    )
    parser.add_argument(
        "--critic-loss-type",
        choices=("mle", "squared"),
        default="mle",
        help="Loss type used when training the critic value head (controls evaluation activation).",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional W&B project override.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group override.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name override.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled", "dryrun"),
        help="Optional W&B mode override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    actor_export: str | Path | None = args.actor_hf_path
    critic_export: str | Path | None = args.critic_hf_path
    critic_ckpt: Path | str | None = args.critic_hf_path

    if args.checkpoint_root is not None:
        checkpoint_root = args.checkpoint_root.resolve()
        if args.checkpoint_step:
            step_dir = args.checkpoint_step
            if not step_dir.startswith("global_step_"):
                step_dir = f"global_step_{step_dir}"
            checkpoint_dir = checkpoint_root / step_dir
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Specified checkpoint step not found: {checkpoint_dir}")
        else:
            checkpoint_dir = find_latest_checkpoint(checkpoint_root)

        actor_ckpt = checkpoint_dir / "actor"
        critic_ckpt = checkpoint_dir / "critic"

        export_root = output_dir / "hf_exports" / checkpoint_dir.name
        actor_export = export_root / "actor"
        critic_export = export_root / "critic"

        if not (args.reuse_export and (Path(actor_export) / "config.json").exists()):
            ensure_hf_export(actor_ckpt, Path(actor_export))
        if not args.skip_critic and critic_export is not None:
            if not (args.reuse_export and (Path(critic_export) / "config.json").exists()):
                ensure_hf_export(critic_ckpt, Path(critic_export))

    if actor_export is None:
        raise ValueError("Either --checkpoint-root or --actor-hf-path must be provided.")

    num_gpus = torch.cuda.device_count()
    dtype = torch.bfloat16 if num_gpus > 0 else torch.float32
    device = torch.device("cuda" if num_gpus > 0 else "cpu")

    actor_tokenizer_source = args.actor_hf_tokenizer or actor_export
    actor_tokenizer = AutoTokenizer.from_pretrained(actor_tokenizer_source, trust_remote_code=True)
    actor_tokenizer.padding_side = "left"

    samples = load_rewardbench_pairs(args.dataset, args.split, max_examples=args.max_examples)
    if not samples:
        raise RuntimeError("No evaluation samples were constructed.")

    actor_backend = args.actor_backend.lower()
    actor_model_ref = actor_export
    critic_outputs_logits = args.critic_loss_type == "mle"
    if actor_backend == "server":
        server_proc: subprocess.Popen | None = None
        try:
            server_proc = start_vllm_server(actor_model_ref, args.server_port, args.actor_data_parallel_size)
            wait_for_server_ready(args.server_port, args.server_startup_timeout)
            run_actor_generation_vllm_server(
                actor_tokenizer,
                samples,
                str(actor_model_ref),
                args.server_port,
                args.max_new_tokens,
                args.actor_request_concurrency,
                args.actor_request_timeout,
            )
        finally:
            shutdown_process(server_proc, args.server_shutdown_timeout)
    elif actor_backend == "vllm":
        if num_gpus == 0:
            raise RuntimeError("vLLM backend requires CUDA but no GPU was detected.")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("vLLM is not installed; re-run with --actor-backend hf or install vllm.") from exc

        dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"
        actor_llm = LLM(
            model=str(actor_model_ref),
            tokenizer=str(actor_model_ref),
            trust_remote_code=True,
            dtype=dtype_str,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
        )
        run_actor_generation_vllm_local(actor_llm, sampling_params, actor_tokenizer, samples, args.actor_batch_size)
    else:
        actor_model = AutoModelForCausalLM.from_pretrained(
            actor_model_ref,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        run_actor_generation_hf(actor_model, actor_tokenizer, samples, args.max_new_tokens, args.actor_batch_size)

    run_critic = not args.skip_critic and critic_export is not None
    critic_workers: int = 0
    if run_critic:
        critic_workers = args.critic_num_workers or 0
        if num_gpus == 0:
            critic_workers = 1
        else:
            critic_workers = critic_workers or num_gpus
            critic_workers = max(1, min(critic_workers, num_gpus))

        critic_model_ref = critic_export
        if checkpoint_dir is None and not Path(str(critic_model_ref)).exists():
            raise ValueError(
                "--critic-hf-path must point to a local directory when using baseline evaluation."
            )

        if critic_workers > 1:
            run_critic_scoring_multiproc(
                samples,
                Path(critic_model_ref),
                Path(critic_ckpt) if isinstance(critic_ckpt, (str, Path)) else None,
                critic_outputs_logits,
                dtype,
                args.critic_batch_size,
                critic_workers,
            )
        else:
            critic_model, critic_tokenizer = load_token_classifier(
                Path(critic_model_ref),
                dtype,
                Path(critic_ckpt) if isinstance(critic_ckpt, (str, Path)) else None,
            )
            critic_model.to(device)
            run_critic_scoring_single(
                critic_model,
                critic_tokenizer,
                samples,
                args.critic_batch_size,
                critic_outputs_logits,
            )

    apply_correction(samples, args.threshold)
    raw_samples = list(samples)
    samples = aggregate_pair_orientations(samples)

    actor_prediction_distribution = compute_prediction_distribution(samples, "predicted_label")
    critic_prediction_distribution = compute_prediction_distribution(samples, "corrected_label")

    metrics = compute_metrics(samples)
    metrics_config = {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "threshold": args.threshold,
        "dataset": args.dataset,
        "split": args.split,
        "max_examples": args.max_examples,
        "actor_batch_size": args.actor_batch_size,
        "critic_batch_size": args.critic_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "actor_backend": actor_backend,
        "critic_num_workers": critic_workers,
        "actor_model_ref": str(actor_model_ref),
        "skip_critic": args.skip_critic,
        "critic_model_ref": str(critic_export) if critic_export is not None else None,
        "critic_loss_type": args.critic_loss_type,
        "output_dir": str(output_dir),
    }

    metrics["prediction_distribution"] = {
        "actor": actor_prediction_distribution,
        "critic": critic_prediction_distribution,
    }

    generations_path = output_dir / f"{Path(args.results_file).stem}_generations.jsonl"
    save_request_generations(raw_samples, generations_path)

    pair_results = build_pair_level_results(raw_samples)
    pair_results_path = output_dir / f"{Path(args.results_file).stem}_pair_results.jsonl"
    save_pair_level_results(pair_results, pair_results_path)
    pair_status_summary = summarize_pair_statuses(pair_results)
    prompt_status_summary = summarize_prompt_statuses(pair_results)
    metrics["pair_status_summary"] = pair_status_summary
    metrics["prompt_status_summary"] = prompt_status_summary
    metrics_config.update(
        {
            "generation_path": str(generations_path),
            "pair_results_path": str(pair_results_path),
        }
    )

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_group:
        os.environ["WANDB_GROUP"] = args.wandb_group
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_group = os.getenv("WANDB_GROUP")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    wandb_mode_env = os.getenv("WANDB_MODE", "online").lower()

    metrics_config.update(
        {
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_name": wandb_run_name,
            "wandb_mode": wandb_mode_env,
        }
    )

    if actor_backend == "vllm":
        metrics_config.update(
            {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        )
    elif actor_backend == "server":
        metrics_config.update(
            {
                "actor_data_parallel_size": args.actor_data_parallel_size,
                "actor_request_concurrency": args.actor_request_concurrency,
                "actor_request_timeout": args.actor_request_timeout,
                "server_port": args.server_port,
            }
        )

    checkpoint_step_name: str | None = None
    if checkpoint_dir is not None:
        checkpoint_step_name = checkpoint_dir.name
    elif args.checkpoint_step:
        step_label = str(args.checkpoint_step)
        checkpoint_step_name = step_label if step_label.startswith("global_step_") else f"global_step_{step_label}"
    if checkpoint_step_name:
        metrics_config["checkpoint_step_name"] = checkpoint_step_name

    metrics["config"] = metrics_config

    output_path = output_dir / args.results_file
    save_metrics(metrics, output_path)

    if wandb and wandb_mode_env not in {"disabled", "off", "offline"}:
        try:
            run = wandb.init(
                project=wandb_project or "verl_think_rm",
                group=wandb_group,
                job_type="rewardbench2_eval",
                name=wandb_run_name,
                config=metrics["config"],
            )

            def _maybe_add(target: dict[str, float], key: str, value: Any) -> None:
                if value is None:
                    return
                if isinstance(value, float) and math.isnan(value):
                    return
                target[key] = value

            prompt_summary = metrics.get("prompt_status_summary", {})
            overall_prompt_status = prompt_summary.get("overall", {})
            actor_prompt_status = overall_prompt_status.get("actor", {})
            critic_prompt_status = overall_prompt_status.get("critic", {})

            overall_pair = metrics.get("overall_pair_metrics", {})
            actor_pair_overall = overall_pair.get("actor", {})
            critic_pair_overall = overall_pair.get("critic", {})

            pair_summary = metrics.get("pair_status_summary", {})
            actor_pair_status = pair_summary.get("actor", {})
            critic_pair_status = pair_summary.get("critic", {})

            subset_metrics = metrics.get("subset_metrics", {})
            ties_stats = next(
                (stats for subset_name, stats in subset_metrics.items() if subset_name.lower() == "ties"), None
            )

            core_payload: dict[str, float] = {}
            _maybe_add(core_payload, "rewardbench2-core/actor/prompt_strict_accuracy", actor_prompt_status.get("strict_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/actor/prompt_loose_accuracy", actor_prompt_status.get("loose_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/critic/prompt_strict_accuracy", critic_prompt_status.get("strict_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/critic/prompt_loose_accuracy", critic_prompt_status.get("loose_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/actor/pair_accuracy", actor_pair_overall.get("accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/actor/pair_strict_accuracy", actor_pair_overall.get("strict_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/critic/pair_accuracy", critic_pair_overall.get("accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/critic/pair_strict_accuracy", critic_pair_overall.get("strict_accuracy"))
            _maybe_add(core_payload, "rewardbench2-core/actor/pair_consistency_rate", actor_pair_status.get("consistency_rate"))
            _maybe_add(core_payload, "rewardbench2-core/critic/pair_consistency_rate", critic_pair_status.get("consistency_rate"))
            _maybe_add(core_payload, "rewardbench2-core/total_prompts", metrics.get("total_prompts"))
            _maybe_add(core_payload, "rewardbench2-core/total_pairs", metrics.get("total_pairs"))
            if ties_stats:
                _maybe_add(core_payload, "rewardbench2-core/actor/overall_score", ties_stats.get("actor_ties_score"))
                _maybe_add(core_payload, "rewardbench2-core/critic/overall_score", ties_stats.get("critic_ties_score"))
                _maybe_add(core_payload, "rewardbench2-core/prob/overall_score", ties_stats.get("prob_ties_score"))

            aux_payload: dict[str, float] = {}

            for subset, stats in metrics.get("subset_metrics", {}).items():
                prefix = f"rewardbench2-aux/subsets/{subset}"
                _maybe_add(aux_payload, f"{prefix}/actor_prompt_accuracy", stats.get("actor_prompt_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/critic_prompt_accuracy", stats.get("critic_prompt_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/actor_pair_accuracy", stats.get("actor_pair_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/critic_pair_accuracy", stats.get("critic_pair_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/actor_pair_strict_accuracy", stats.get("actor_pair_strict_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/critic_pair_strict_accuracy", stats.get("critic_pair_strict_accuracy"))
                _maybe_add(aux_payload, f"{prefix}/prompt_count", stats.get("prompt_count"))
                _maybe_add(aux_payload, f"{prefix}/pair_count", stats.get("pair_count"))
                if subset.lower() == "ties":
                    _maybe_add(aux_payload, f"{prefix}/actor_ties_score", stats.get("actor_ties_score"))
                    _maybe_add(aux_payload, f"{prefix}/critic_ties_score", stats.get("critic_ties_score"))
                    _maybe_add(aux_payload, f"{prefix}/prob_ties_score", stats.get("prob_ties_score"))

            prompt_subset_summary = prompt_summary.get("subsets", {})
            for subset, data in prompt_subset_summary.items():
                for mode in ("actor", "critic"):
                    mode_stats = data.get(mode, {})
                    prefix = f"rewardbench2-aux/prompt_status/{subset}/{mode}"
                    _maybe_add(aux_payload, f"{prefix}/strict_accuracy", mode_stats.get("strict_accuracy"))
                    _maybe_add(aux_payload, f"{prefix}/loose_accuracy", mode_stats.get("loose_accuracy"))
                    _maybe_add(aux_payload, f"{prefix}/strict_correct", mode_stats.get("strict_correct"))
                    _maybe_add(aux_payload, f"{prefix}/loose_correct", mode_stats.get("loose_correct"))
                _maybe_add(aux_payload, f"rewardbench2-aux/prompt_status/{subset}/total_prompts", data.get("total_prompts"))

            for mode, stats in (("actor", actor_pair_status), ("critic", critic_pair_status)):
                rates = stats.get("rates", {}) if isinstance(stats, dict) else {}
                prefix = f"rewardbench2-aux/pair_status/{mode}"
                _maybe_add(aux_payload, f"{prefix}/clear_right_rate", rates.get("clear_right"))
                _maybe_add(aux_payload, f"{prefix}/clear_wrong_rate", rates.get("clear_wrong"))
                _maybe_add(aux_payload, f"{prefix}/unclear_rate", rates.get("unclear"))
                _maybe_add(aux_payload, f"{prefix}/total_pairs", stats.get("total"))

            subset_avg = metrics.get("overall_subset_average", {})
            _maybe_add(aux_payload, "rewardbench2-aux/overall/actor_subset_average", subset_avg.get("actor"))
            _maybe_add(aux_payload, "rewardbench2-aux/overall/critic_subset_average", subset_avg.get("critic"))
            _maybe_add(aux_payload, "rewardbench2-aux/overall/prob_subset_average", subset_avg.get("prob"))

            overall_prompt_accuracy = metrics.get("overall_prompt_accuracy", {})
            _maybe_add(aux_payload, "rewardbench2-aux/overall/actor_prompt_accuracy", overall_prompt_accuracy.get("actor_prompt_accuracy"))
            _maybe_add(aux_payload, "rewardbench2-aux/overall/critic_prompt_accuracy", overall_prompt_accuracy.get("critic_prompt_accuracy"))

            run.log(core_payload)
            if aux_payload:
                run.log(aux_payload)

        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] WandB logging failed: {exc}")
        finally:
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    main()

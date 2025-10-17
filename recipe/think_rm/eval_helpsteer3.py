#!/usr/bin/env python3
"""Evaluate Think RM checkpoints on the HelpSteer3 preference dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.think_rm.eval_utils import (
    PairSample,
    apply_correction,
    build_pair_level_results,
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
from recipe.think_rm.preference_dataset_utils import build_prompt, format_context
from recipe.think_rm.prepare_helpsteer3 import _normalise_context

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


LABEL_CATEGORIES: tuple[str, ...] = ("response_1", "response_2", "tie", "unknown")


def _overall_to_label(score: Any) -> str | None:
    if isinstance(score, (int, float)):
        if score > 0:
            return "response_2"
        if score < 0:
            return "response_1"
        return "tie"
    return None


def _flip_label(label: str | None) -> str:
    if label == "response_1":
        return "response_2"
    if label == "response_2":
        return "response_1"
    return label or "unknown"


def _split_from_base(base_pair_id: str | None) -> str:
    if not base_pair_id:
        return "unknown"
    if "-" in base_pair_id:
        return base_pair_id.split("-", 1)[0]
    return base_pair_id


def load_helpsteer3_split(
    dataset_name: str,
    split: str,
    subset_name: str,
    limit: int | None,
) -> list[PairSample]:
    hf_split = split
    if limit is not None and limit >= 0:
        hf_split = f"{split}[:{limit}]"

    dataset = load_dataset(dataset_name, split=hf_split)

    samples: list[PairSample] = []
    for row_idx, row in enumerate(dataset):
        response1 = row.get("response1", "")
        response2 = row.get("response2", "")
        context_messages = _normalise_context(row.get("context"))
        domain = (row.get("domain") or "unknown").lower()
        prompt_text = format_context(context_messages)
        overall_label = _overall_to_label(row.get("overall_preference"))
        forward_truth = overall_label if overall_label in {"response_1", "response_2", "tie"} else "unknown"
        backward_truth = _flip_label(forward_truth)

        raw_id = row.get("id", row_idx)
        base_pair_id = f"{subset_name}-{raw_id}"

        if forward_truth == "response_1":
            original_chosen = response1
            original_rejected = response2
            comparison_kind = "chosen_vs_rejected"
            num_correct = 1
            forward_chosen_index = 0
            forward_rejected_index = 1
        elif forward_truth == "response_2":
            original_chosen = response2
            original_rejected = response1
            comparison_kind = "chosen_vs_rejected"
            num_correct = 1
            forward_chosen_index = 1
            forward_rejected_index = 0
        elif forward_truth == "tie":
            original_chosen = ""
            original_rejected = ""
            comparison_kind = "tie"
            num_correct = 0
            forward_chosen_index = -1
            forward_rejected_index = -1
        else:
            original_chosen = ""
            original_rejected = ""
            comparison_kind = "unknown"
            num_correct = 0
            forward_chosen_index = -1
            forward_rejected_index = -1

        forward_messages = build_prompt(context_messages, response1, response2)
        samples.append(
            PairSample(
                prompt_id=str(raw_id),
                subset=domain,
                pair_index=len(samples),
                messages=forward_messages,
                response_1=response1,
                response_2=response2,
                ground_truth=forward_truth,
                prompt_text=prompt_text,
                base_pair_id=base_pair_id,
                orientation="forward",
                original_chosen=original_chosen,
                original_rejected=original_rejected,
                prompt_raw_id=str(raw_id),
                row_index=row_idx,
                chosen_index=forward_chosen_index,
                rejected_index=forward_rejected_index,
                num_correct=num_correct,
                total_completions=2,
                response_1_pos=0,
                response_2_pos=1,
                comparison_kind=comparison_kind,
            )
        )

        backward_messages = build_prompt(context_messages, response2, response1)
        samples.append(
            PairSample(
                prompt_id=str(raw_id),
                subset=domain,
                pair_index=len(samples),
                messages=backward_messages,
                response_1=response2,
                response_2=response1,
                ground_truth=backward_truth,
                prompt_text=prompt_text,
                base_pair_id=base_pair_id,
                orientation="backward",
                original_chosen=original_chosen,
                original_rejected=original_rejected,
                prompt_raw_id=str(raw_id),
                row_index=row_idx,
                chosen_index=forward_chosen_index,
                rejected_index=forward_rejected_index,
                num_correct=num_correct,
                total_completions=2,
                response_1_pos=1,
                response_2_pos=0,
                comparison_kind=comparison_kind,
            )
        )

    return samples


def _ground_truth_distribution(samples: Iterable[PairSample]) -> dict[str, Any]:
    counts = {label: 0 for label in LABEL_CATEGORIES}
    total = 0
    for sample in samples:
        label = (sample.ground_truth or "unknown").lower()
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
        total += 1

    fractions = {label: (counts[label] / total) if total else 0.0 for label in LABEL_CATEGORIES}
    counts["total"] = total
    return {"counts": counts, "fractions": fractions}


def _serialize_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        json.dump(metrics, fout, indent=2)


def _count_pairs(samples: Iterable[PairSample]) -> int:
    return sum(1 for sample in samples if sample.comparison_kind == "chosen_vs_rejected")


def _compute_forward_accuracy(samples: Iterable[PairSample], label_attr: str) -> dict[str, Any]:
    total = 0
    evaluated = 0
    score_sum = 0.0

    for sample in samples:
        ground_truth = (sample.ground_truth or "unknown").lower()
        if ground_truth not in {"response_1", "response_2"}:
            continue
        total += 1
        predicted = getattr(sample, label_attr, None)
        if predicted is None:
            continue
        predicted = str(predicted).lower()
        if predicted == "response_1":
            score = 1.0 if ground_truth == "response_1" else 0.0
        elif predicted == "response_2":
            score = 1.0 if ground_truth == "response_2" else 0.0
        else:
            score = 0.5
        evaluated += 1
        score_sum += score

    loose_accuracy = score_sum / evaluated if evaluated else None
    coverage = evaluated / total if total else None
    return {
        "loose_total": total,
        "loose_evaluated": evaluated,
        "loose_score_sum": score_sum,
        "loose_accuracy": loose_accuracy,
        "loose_coverage": coverage,
    }


def _compute_strict_accuracy(pair_results: Iterable[dict[str, Any]], mode: str) -> dict[str, Any]:
    results = list(pair_results)
    total = len(results)
    evaluated = 0
    correct = 0

    status_key = f"{mode}_status"
    for record in results:
        status = str(record.get(status_key, "")).lower()
        if status == "clear_right":
            evaluated += 1
            correct += 1
        elif status == "clear_wrong":
            evaluated += 1

    strict_accuracy = correct / evaluated if evaluated else None
    coverage = evaluated / total if total else None
    return {
        "strict_total": total,
        "strict_evaluated": evaluated,
        "strict_correct": correct,
        "strict_accuracy": strict_accuracy,
        "strict_coverage": coverage,
    }


def _merge_accuracy(loose_stats: dict[str, Any], strict_stats: dict[str, Any]) -> dict[str, Any]:
    combined = dict(loose_stats)
    for key, value in strict_stats.items():
        combined[key] = value
    return combined


def _build_domain_reports(forward_samples: Iterable[PairSample], pair_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    forward_grouped: dict[str, list[PairSample]] = defaultdict(list)
    for sample in forward_samples:
        domain = (sample.subset or "unknown").lower()
        forward_grouped[domain].append(sample)

    pair_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_results:
        domain = str(record.get("subset", "unknown")).lower()
        pair_grouped[domain].append(record)

    domain_names = sorted(set(forward_grouped.keys()) | set(pair_grouped.keys()))
    reports: dict[str, Any] = {}
    for domain in domain_names:
        forward = forward_grouped.get(domain, [])
        pairs = pair_grouped.get(domain, [])
        actor_loose = _compute_forward_accuracy(forward, "predicted_label")
        actor_strict = _compute_strict_accuracy(pairs, "actor")
        critic_loose = _compute_forward_accuracy(forward, "corrected_label")
        critic_strict = _compute_strict_accuracy(pairs, "critic")
        reports[domain] = {
            "num_pairs": len(pairs),
            "raw_pairs": len(forward),
            "ground_truth_distribution": _ground_truth_distribution(forward),
            "prediction_distribution": {
                "actor": compute_prediction_distribution(forward, "predicted_label"),
                "critic": compute_prediction_distribution(forward, "corrected_label"),
            },
            "actor": _merge_accuracy(actor_loose, actor_strict),
            "critic": _merge_accuracy(critic_loose, critic_strict),
        }
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Think RM checkpoints on HelpSteer3.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Local checkpoint directory containing global_step_* folders. Required unless --actor-hf-path is set.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=str,
        default=None,
        help="Optional checkpoint step label (e.g., global_step_320 or 320) to evaluate.",
    )
    parser.add_argument(
        "--actor-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face path or repo ID for an exported actor (bypasses checkpoint export).",
    )
    parser.add_argument(
        "--actor-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer override for the actor (defaults to --actor-hf-path).",
    )
    parser.add_argument(
        "--critic-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face path or repo ID for an exported critic (bypasses checkpoint export).",
    )
    parser.add_argument(
        "--critic-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer override for the critic (defaults to --critic-hf-path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store evaluation logs and results.",
    )
    parser.add_argument("--results-file", default="helpsteer3_metrics.json", help="Filename for the metrics JSON.")
    parser.add_argument("--dataset", default="nvidia/HelpSteer3", help="Hugging Face dataset name.")
    parser.add_argument("--train-split", default="train", help="Dataset split used for the training subset slice.")
    parser.add_argument(
        "--train-subset-tag",
        default="train_head",
        help="Tag used in logs/metrics for the sampled training subset.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2000,
        help="Number of training split samples to evaluate (set <=0 to skip).",
    )
    parser.add_argument(
        "--validation-split",
        default="validation",
        help="Dataset split evaluated as validation.",
    )
    parser.add_argument(
        "--validation-subset-tag",
        default="validation",
        help="Tag used in logs/metrics for the validation split.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Optional limit on validation samples (defaults to the full split).",
    )
    parser.add_argument("--actor-batch-size", type=int, default=4, help="Batch size for actor generation.")
    parser.add_argument("--critic-batch-size", type=int, default=8, help="Batch size for critic scoring.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Actor generation length cap.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Critic probability threshold for correction.")
    parser.add_argument(
        "--actor-backend",
        choices=("hf", "vllm", "server"),
        default="server",
        help="Backend used for actor inference.",
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
        help="GPU memory utilization target for local vLLM.",
    )
    parser.add_argument(
        "--actor-data-parallel-size",
        type=int,
        default=8,
        help="Data parallel degree when using the vLLM server backend.",
    )
    parser.add_argument("--server-port", type=int, default=8000, help="Port to bind the vLLM OpenAI server.")
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
        help="Seconds to wait for the vLLM server to stop.",
    )
    parser.add_argument(
        "--actor-request-concurrency",
        type=int,
        default=64,
        help="Concurrency for HTTP requests when using the vLLM server backend.",
    )
    parser.add_argument(
        "--actor-request-timeout",
        type=int,
        default=120,
        help="Timeout for individual HTTP requests to the vLLM server.",
    )
    parser.add_argument(
        "--critic-num-workers",
        type=int,
        default=None,
        help="Number of GPU workers for critic scoring (defaults to device count).",
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Reuse existing Hugging Face exports if present under the output directory.",
    )
    parser.add_argument("--skip-critic", action="store_true", help="Skip critic scoring/correction.")
    parser.add_argument(
        "--critic-loss-type",
        choices=("mle", "squared"),
        default="mle",
        help="Critic loss type used during training (controls evaluation activation).",
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
    actor_export: Path | str | None = args.actor_hf_path
    critic_export: Path | str | None = args.critic_hf_path
    critic_ckpt: Path | str | None = args.critic_hf_path

    if args.checkpoint_root is not None:
        checkpoint_root = args.checkpoint_root.resolve()
        if args.checkpoint_step:
            step_dir = args.checkpoint_step
            if not step_dir.startswith("global_step_"):
                step_dir = f"global_step_{step_dir}"
            checkpoint_dir = checkpoint_root / step_dir
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint step not found: {checkpoint_dir}")
        else:
            checkpoint_dir = find_latest_checkpoint(checkpoint_root)

        actor_ckpt = checkpoint_dir / "actor"
        critic_ckpt = checkpoint_dir / "critic"

        export_root = output_dir / "hf_exports" / checkpoint_dir.name
        actor_export = export_root / "actor"
        critic_export = export_root / "critic"

        if not (args.reuse_export and (Path(actor_export) / "config.json").exists()):
            ensure_hf_export(actor_ckpt, Path(actor_export))
        if not args.skip_critic:
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

    split_specs: list[dict[str, Any]] = []

    if args.train_limit is None or args.train_limit > 0:
        split_specs.append(
            {
                "subset": args.train_subset_tag,
                "split": args.train_split,
                "limit": args.train_limit,
            }
        )

    if args.validation_split and (args.validation_limit is None or args.validation_limit > 0):
        split_specs.append(
            {
                "subset": args.validation_subset_tag,
                "split": args.validation_split,
                "limit": args.validation_limit,
            }
        )

    samples: list[PairSample] = []
    split_metadata: list[dict[str, Any]] = []

    for spec in split_specs:
        subset_name = spec["subset"]
        split_name = spec["split"]
        limit = spec["limit"]
        split_samples = load_helpsteer3_split(args.dataset, split_name, subset_name, limit)
        samples.extend(split_samples)
        split_metadata.append(
            {
                "subset": subset_name,
                "split": split_name,
                "limit": limit,
                "raw_pairs": len(split_samples) // 2,
            }
        )

    if not samples:
        raise RuntimeError("No evaluation samples were constructed. Please check split/limit arguments.")

    actor_backend = args.actor_backend.lower()
    actor_model_ref: Path | str = actor_export
    critic_outputs_logits = args.critic_loss_type == "mle"

    if actor_backend == "server":
        server_proc: subprocess.Popen | None = None
        try:
            server_proc = start_vllm_server(Path(str(actor_model_ref)), args.server_port, args.actor_data_parallel_size)
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
        sampling_params = SamplingParams(max_tokens=args.max_new_tokens)
        run_actor_generation_vllm_local(actor_llm, sampling_params, actor_tokenizer, samples, args.actor_batch_size)
    else:
        actor_model = AutoModelForCausalLM.from_pretrained(
            actor_model_ref,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        run_actor_generation_hf(actor_model, actor_tokenizer, samples, args.max_new_tokens, args.actor_batch_size)

    run_critic = not args.skip_critic and critic_export is not None
    critic_workers = 0
    if run_critic:
        critic_workers = args.critic_num_workers or 0
        if num_gpus == 0:
            critic_workers = 1
        else:
            critic_workers = critic_workers or num_gpus
            critic_workers = max(1, min(critic_workers, num_gpus))

        critic_model_ref = critic_export
        if checkpoint_dir is None and not Path(str(critic_model_ref)).exists():
            raise ValueError("--critic-hf-path must point to a local directory when evaluating without checkpoints.")

        if critic_workers > 1:
            run_critic_scoring_multiproc(
                samples,
                Path(str(critic_model_ref)),
                Path(str(critic_ckpt)) if isinstance(critic_ckpt, (str, Path)) else None,
                critic_outputs_logits,
                dtype,
                args.critic_batch_size,
                critic_workers,
            )
        else:
            critic_model, critic_tokenizer = load_token_classifier(
                Path(str(critic_model_ref)),
                dtype,
                Path(str(critic_ckpt)) if isinstance(critic_ckpt, (str, Path)) else None,
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
    forward_samples = [sample for sample in raw_samples if sample.orientation == "forward"]

    generations_path = output_dir / f"{Path(args.results_file).stem}_generations.jsonl"
    save_request_generations(raw_samples, generations_path)

    pair_results = build_pair_level_results(raw_samples)
    pair_results_path = output_dir / f"{Path(args.results_file).stem}_pair_results.jsonl"
    save_pair_level_results(pair_results, pair_results_path)
    pair_status_summary = summarize_pair_statuses(pair_results)
    prompt_status_summary = summarize_prompt_statuses(pair_results)

    split_forward_samples: dict[str, list[PairSample]] = defaultdict(list)
    for sample in forward_samples:
        split_forward_samples[_split_from_base(sample.base_pair_id)].append(sample)

    pair_results_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_results:
        split_name = _split_from_base(record.get("base_pair_id"))
        pair_results_by_split[split_name].append(record)

    split_reports: dict[str, Any] = {}
    for spec in split_metadata:
        subset = spec["subset"]
        subset_forward = split_forward_samples.get(subset, [])
        subset_pair_results = pair_results_by_split.get(subset, [])
        raw_pairs = spec.get("raw_pairs")
        if raw_pairs is None:
            raw_pairs = len(subset_forward)
        actor_report = _merge_accuracy(
            _compute_forward_accuracy(subset_forward, "predicted_label"),
            _compute_strict_accuracy(subset_pair_results, "actor"),
        )
        critic_report = _merge_accuracy(
            _compute_forward_accuracy(subset_forward, "corrected_label"),
            _compute_strict_accuracy(subset_pair_results, "critic"),
        )
        split_reports[subset] = {
            "num_pairs": len(subset_pair_results),
            "raw_pairs": raw_pairs,
            "ground_truth_distribution": _ground_truth_distribution(subset_forward),
            "prediction_distribution": {
                "actor": compute_prediction_distribution(subset_forward, "predicted_label"),
                "critic": compute_prediction_distribution(subset_forward, "corrected_label"),
            },
            "actor": actor_report,
            "critic": critic_report,
            "source_split": spec["split"],
            "limit": spec["limit"],
            "domains": _build_domain_reports(subset_forward, subset_pair_results),
        }
    overall_report = {
        "num_pairs": len(pair_results),
        "raw_pairs": len(forward_samples),
        "ground_truth_distribution": _ground_truth_distribution(forward_samples),
        "prediction_distribution": {
            "actor": compute_prediction_distribution(forward_samples, "predicted_label"),
            "critic": compute_prediction_distribution(forward_samples, "corrected_label"),
        },
        "actor": _merge_accuracy(
            _compute_forward_accuracy(forward_samples, "predicted_label"),
            _compute_strict_accuracy(pair_results, "actor"),
        ),
        "critic": _merge_accuracy(
            _compute_forward_accuracy(forward_samples, "corrected_label"),
            _compute_strict_accuracy(pair_results, "critic"),
        ),
        "domains": _build_domain_reports(forward_samples, pair_results),
    }

    metrics_config: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_step": args.checkpoint_step,
        "dataset": args.dataset,
        "train_split": args.train_split,
        "train_subset_tag": args.train_subset_tag,
        "train_limit": args.train_limit,
        "validation_split": args.validation_split,
        "validation_subset_tag": args.validation_subset_tag,
        "validation_limit": args.validation_limit,
        "actor_batch_size": args.actor_batch_size,
        "critic_batch_size": args.critic_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "actor_backend": actor_backend,
        "critic_num_workers": critic_workers,
        "actor_model_ref": str(actor_export),
        "skip_critic": args.skip_critic,
        "critic_model_ref": str(critic_export) if critic_export is not None else None,
        "critic_loss_type": args.critic_loss_type,
        "generation_path": str(generations_path),
        "pair_results_path": str(pair_results_path),
        "split_metadata": split_metadata,
    }

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

    metrics = {
        "splits": split_reports,
        "overall": overall_report,
        "pair_status_summary": pair_status_summary,
        "prompt_status_summary": prompt_status_summary,
        "config": metrics_config,
    }

    output_path = output_dir / args.results_file
    _serialize_metrics(metrics, output_path)

    if wandb and wandb_mode_env not in {"disabled", "off", "offline"}:
        run = wandb.init(
            project=wandb_project or "verl_think_rm",
            group=wandb_group,
            job_type="helpsteer3_eval",
            name=wandb_run_name,
            config=metrics_config,
        )
        try:
            overall_actor_stats = metrics["overall"]["actor"]
            overall_critic_stats = metrics["overall"]["critic"]
            overall_log: dict[str, float] = {}
            actor_loose = overall_actor_stats.get("loose_accuracy")
            actor_strict = overall_actor_stats.get("strict_accuracy")
            critic_loose = overall_critic_stats.get("loose_accuracy")
            critic_strict = overall_critic_stats.get("strict_accuracy")
            if actor_loose is not None:
                overall_log["helpsteer3/overall/actor_loose_accuracy"] = actor_loose
            if actor_strict is not None:
                overall_log["helpsteer3/overall/actor_strict_accuracy"] = actor_strict
            if critic_loose is not None:
                overall_log["helpsteer3/overall/critic_loose_accuracy"] = critic_loose
            if critic_strict is not None:
                overall_log["helpsteer3/overall/critic_strict_accuracy"] = critic_strict
            if overall_log:
                wandb.log(overall_log)

            for subset_name, report in metrics["splits"].items():
                prefix = f"helpsteer3/{subset_name}"
                actor_stats = report.get("actor", {})
                critic_stats = report.get("critic", {})
                log_payload: dict[str, float] = {}
                actor_loose = actor_stats.get("loose_accuracy")
                actor_strict = actor_stats.get("strict_accuracy")
                critic_loose = critic_stats.get("loose_accuracy")
                critic_strict = critic_stats.get("strict_accuracy")
                if actor_loose is not None:
                    log_payload[f"{prefix}/actor_loose_accuracy"] = actor_loose
                if actor_strict is not None:
                    log_payload[f"{prefix}/actor_strict_accuracy"] = actor_strict
                if critic_loose is not None:
                    log_payload[f"{prefix}/critic_loose_accuracy"] = critic_loose
                if critic_strict is not None:
                    log_payload[f"{prefix}/critic_strict_accuracy"] = critic_strict
                if log_payload:
                    wandb.log(log_payload)

            for domain_name, domain_report in metrics["overall"].get("domains", {}).items():
                prefix = f"helpsteer3/domain/{domain_name}"
                actor_stats = domain_report.get("actor", {})
                critic_stats = domain_report.get("critic", {})
                log_payload: dict[str, float] = {}
                actor_loose = actor_stats.get("loose_accuracy")
                actor_strict = actor_stats.get("strict_accuracy")
                critic_loose = critic_stats.get("loose_accuracy")
                critic_strict = critic_stats.get("strict_accuracy")
                if actor_loose is not None:
                    log_payload[f"{prefix}/actor_loose_accuracy"] = actor_loose
                if actor_strict is not None:
                    log_payload[f"{prefix}/actor_strict_accuracy"] = actor_strict
                if critic_loose is not None:
                    log_payload[f"{prefix}/critic_loose_accuracy"] = critic_loose
                if critic_strict is not None:
                    log_payload[f"{prefix}/critic_strict_accuracy"] = critic_strict
                if log_payload:
                    wandb.log(log_payload)
        finally:
            run.finish()


if __name__ == "__main__":
    main()

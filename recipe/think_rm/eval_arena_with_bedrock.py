#!/usr/bin/env python3
"""Evaluate Arena Human Preference labels against a Bedrock Claude model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import boto3
from botocore.config import Config
from datasets import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    import prepare_arena_human_preference as arena  # type: ignore  # noqa: E402
    from preference_dataset_utils import ExpandedRecord  # type: ignore  # noqa: E402
    from reward_fn import parse_preference  # type: ignore  # noqa: E402
else:  # pragma: no cover
    from . import prepare_arena_human_preference as arena
    from .preference_dataset_utils import ExpandedRecord
    from .reward_fn import parse_preference


DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

logger = logging.getLogger(__name__)


@dataclass
class OrientationSummary:
    orientation: str
    raw_predictions: list[str | None]
    mapped_predictions: list[str | None]
    completions: list[str]
    raw_responses: list[dict]
    majority: str | None


@dataclass
class EvaluationResult:
    uid: str
    ground_truth: str
    prediction: str | None
    vote_predictions: list[str | None]
    completions: list[str]
    raw_responses: list[dict]
    orientation_summaries: dict[str, OrientationSummary]


def _resolve_majority_vote(votes: Iterable[str | None]) -> str | None:
    """Return the majority label from the provided votes or None if undecided."""
    filtered = [vote for vote in votes if vote is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    top_label, top_count = counts.most_common(1)[0]
    num_with_top = sum(1 for count in counts.values() if count == top_count)
    if num_with_top > 1:
        if counts.get("tie") == top_count:
            return "tie"
        return None
    return top_label


def _select_first_n(ds: Dataset, count: int) -> Dataset:
    if count >= len(ds):
        return ds
    return ds.select(range(count))


def expand_records(split: str, count: int) -> list[ExpandedRecord]:
    raw_split = arena.load_dataset(arena.DATASET_NAME)[split]
    subset = _select_first_n(raw_split, count)
    expanded = arena._expand_split(subset, split, max_samples=count)
    return expanded[:count]


def build_bedrock_body(prompt: str, max_tokens: int, temperature: float) -> dict:
    return {
        "anthropic_version": ANTHROPIC_VERSION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 8192
        },
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def flatten_anthropic_completion(payload: dict) -> str:
    fragments: list[str] = []
    for item in payload.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            fragments.append(item.get("text", ""))
    return "".join(fragments)


ASSISTANT_A_START_MARKER = "[The Start of Assistant A's Response]\n"
ASSISTANT_A_END_MARKER = "\n[The End of Assistant A's Response]"
ASSISTANT_B_START_MARKER = "[The Start of Assistant B's Response]\n"
ASSISTANT_B_END_MARKER = "\n[The End of Assistant B's Response]"


def _swap_prompt_orientation(prompt: list[dict[str, str]]) -> list[dict[str, str]] | None:
    """Return a new prompt with Assistant A/B responses swapped, preserving formatting."""
    swapped: list[dict[str, str]] = []
    for message in prompt:
        if message.get("role") != "user":
            swapped.append(message)
            continue
        content = message.get("content", "")
        try:
            before_a, after_a_start = content.split(ASSISTANT_A_START_MARKER, 1)
            response_a, after_a_end = after_a_start.split(ASSISTANT_A_END_MARKER, 1)
            between, after_b_start = after_a_end.split(ASSISTANT_B_START_MARKER, 1)
            response_b, after_b_end = after_b_start.split(ASSISTANT_B_END_MARKER, 1)
        except ValueError:
            logger.warning("Failed to swap prompt orientation due to unexpected format.")
            return None

        swapped_content = (
            before_a
            + ASSISTANT_A_START_MARKER
            + response_b
            + ASSISTANT_A_END_MARKER
            + between
            + ASSISTANT_B_START_MARKER
            + response_a
            + ASSISTANT_B_END_MARKER
            + after_b_end
        )
        swapped.append({"role": message.get("role", "user"), "content": swapped_content})
    return swapped


def _map_backward_prediction(prediction: str | None) -> str | None:
    if prediction == "response_1":
        return "response_2"
    if prediction == "response_2":
        return "response_1"
    return prediction


def _combine_orientation_majorities(
    forward_majority: str | None,
    backward_majority: str | None,
) -> str | None:
    if backward_majority is None:
        return forward_majority
    if forward_majority is None:
        return backward_majority
    if forward_majority == backward_majority:
        return forward_majority
    return "tie"


def _collect_orientation_votes(
    client,
    model_id: str,
    prompt: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    num_votes: int,
    orientation: str,
    map_prediction: Callable[[str | None], str | None],
    record_uid: str,
) -> OrientationSummary:
    if not prompt:
        raise ValueError("Prompt must contain at least one message.")
    prompt_msg = prompt[0]["content"]
    logger.debug("Invoking model for uid=%s orientation=%s", record_uid, orientation)
    completions: list[str] = []
    raw_responses: list[dict] = []
    raw_predictions: list[str | None] = []
    mapped_predictions: list[str | None] = []
    vote_count = max(1, num_votes)
    for vote_idx in range(vote_count):
        payload, completion = invoke_bedrock(
            client,
            model_id=model_id,
            prompt=prompt_msg,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_responses.append(payload)
        completions.append(completion)
        prediction = parse_preference(completion, from_thinking_model=False)
        raw_predictions.append(prediction)
        mapped = map_prediction(prediction)
        mapped_predictions.append(mapped)
        logger.debug(
            "uid=%s orientation=%s vote=%d prediction=%s mapped=%s",
            record_uid,
            orientation,
            vote_idx,
            prediction,
            mapped,
        )
    majority = _resolve_majority_vote(mapped_predictions)
    return OrientationSummary(
        orientation=orientation,
        raw_predictions=raw_predictions,
        mapped_predictions=mapped_predictions,
        completions=completions,
        raw_responses=raw_responses,
        majority=majority,
    )


def invoke_bedrock(
    client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict, str]:
    body = build_bedrock_body(prompt, max_tokens=max_tokens, temperature=temperature)
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    payload = json.loads(response["body"].read())
    completion = flatten_anthropic_completion(payload)
    return payload, completion


def _evaluate_single_record(
    client,
    model_id: str,
    record: ExpandedRecord,
    max_tokens: int,
    temperature: float,
    num_votes: int,
    bidirectional: bool,
) -> EvaluationResult:
    forward_summary = _collect_orientation_votes(
        client=client,
        model_id=model_id,
        prompt=record.prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        num_votes=num_votes,
        orientation="forward",
        map_prediction=lambda pred: pred,
        record_uid=record.uid,
    )
    orientation_summaries: dict[str, OrientationSummary] = {"forward": forward_summary}

    combined_predictions = list(forward_summary.mapped_predictions)
    combined_completions = list(forward_summary.completions)
    combined_raw_responses = list(forward_summary.raw_responses)

    backward_summary: OrientationSummary | None = None
    if bidirectional:
        swapped_prompt = _swap_prompt_orientation(record.prompt)
        if swapped_prompt is not None:
            backward_summary = _collect_orientation_votes(
                client=client,
                model_id=model_id,
                prompt=swapped_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                num_votes=num_votes,
                orientation="backward",
                map_prediction=_map_backward_prediction,
                record_uid=record.uid,
            )
            orientation_summaries["backward"] = backward_summary
            combined_predictions.extend(backward_summary.mapped_predictions)
            combined_completions.extend(backward_summary.completions)
            combined_raw_responses.extend(backward_summary.raw_responses)
        else:
            logger.warning(
                "Falling back to forward-only evaluation for uid=%s due to prompt swap failure.",
                record.uid,
            )

    prediction = _combine_orientation_majorities(
        forward_summary.majority,
        backward_summary.majority if backward_summary is not None else None,
    )
    logger.debug(
        "uid=%s ground_truth=%s prediction=%s forward=%s backward=%s",
        record.uid,
        record.reward_model["ground_truth"],
        prediction,
        forward_summary.mapped_predictions,
        backward_summary.mapped_predictions if backward_summary else None,
    )
    return EvaluationResult(
        uid=record.uid,
        ground_truth=record.reward_model["ground_truth"],
        prediction=prediction,
        vote_predictions=combined_predictions,
        completions=combined_completions,
        raw_responses=combined_raw_responses,
        orientation_summaries=orientation_summaries,
    )


def evaluate_records(
    client,
    model_id: str,
    records: Iterable[ExpandedRecord],
    max_tokens: int,
    temperature: float,
    max_workers: int,
    num_votes: int,
    bidirectional: bool,
) -> list[EvaluationResult]:
    records = list(records)
    logger.info("Evaluating %d comparisons with model '%s'.", len(records), model_id)
    logger.info("Sampling each comparison %d times for majority voting.", max(1, num_votes))
    if bidirectional:
        logger.info("Bidirectional evaluation enabled; querying both orientations.")
    max_workers = max(1, max_workers)
    if len(records) == 0:
        return []

    if max_workers == 1:
        return [
            _evaluate_single_record(
                client,
                model_id=model_id,
                record=record,
                max_tokens=max_tokens,
                temperature=temperature,
                num_votes=num_votes,
                bidirectional=bidirectional,
            )
            for record in tqdm(records, desc="Evaluating", unit="pair")
        ]

    max_workers = min(max_workers, len(records))
    logger.info("Using up to %d concurrent workers.", max_workers)

    ordered_results: list[EvaluationResult | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_record,
                client,
                model_id,
                record,
                max_tokens,
                temperature,
                num_votes,
                bidirectional,
            ): idx
            for idx, record in enumerate(records)
        }
        with tqdm(total=len(records), desc="Evaluating", unit="pair") as progress:
            for future in as_completed(futures):
                idx = futures[future]
                ordered_results[idx] = future.result()
                progress.update(1)

    return [result for result in ordered_results if result is not None]


def summarise_results(results: Iterable[EvaluationResult]) -> dict:
    stats = Counter()
    totals = Counter()
    for res in results:
        totals["total"] += 1
        key = (res.ground_truth, res.prediction or "unknown")
        stats[key] += 1
        if res.prediction == res.ground_truth:
            totals["correct"] += 1
        if res.prediction is None:
            totals["no_parse"] += 1
    accuracy = (totals["correct"] / totals["total"]) if totals["total"] else 0.0
    return {
        "totals": dict(totals),
        "accuracy": accuracy,
        "confusion": {f"{gt}->{pred}": count for (gt, pred), count in stats.items()},
    }


def save_details(path: Path, results: Iterable[EvaluationResult]) -> None:
    serialisable = [
        {
            "uid": res.uid,
            "ground_truth": res.ground_truth,
            "prediction": res.prediction,
            "vote_predictions": res.vote_predictions,
            "completions": res.completions,
            "raw_responses": res.raw_responses,
            "vote_counts": dict(Counter([vote for vote in res.vote_predictions if vote is not None])),
            "orientations": {
                name: {
                    "raw_predictions": summary.raw_predictions,
                    "mapped_predictions": summary.mapped_predictions,
                    "majority": summary.majority,
                    "completions": summary.completions,
                    "raw_responses": summary.raw_responses,
                    "vote_counts": dict(
                        Counter([vote for vote in summary.mapped_predictions if vote is not None])
                    ),
                }
                for name, summary in res.orientation_summaries.items()
            },
        }
        for res in results
    ]
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Arena Human Preference labels with a Bedrock Claude model."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample (default: train).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of comparisons to evaluate (default: 50).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Bedrock Claude model ID (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region for Bedrock runtime (overrides AWS_REGION env).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Optional AWS credentials profile to use.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Generation cap for Claude responses (default: 10000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--details-path",
        type=Path,
        default="/tmp/claude_arena.json",
        help="Optional path to store per-sample outputs in JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent Bedrock requests (default: 4, use 1 for sequential).",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=1,
        help="Number of model samples per comparison for majority voting (default: 4).",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each comparison in both forward/backward orientations (default: enabled).",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=120,
        help="Bedrock client read timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=30,
        help="Bedrock client connect timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def make_bedrock_client(
    region: str | None,
    profile: str | None,
    read_timeout: int,
    connect_timeout: int,
):
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    config = Config(read_timeout=read_timeout, connect_timeout=connect_timeout)
    return session.client("bedrock-runtime", config=config)


def main() -> None:
    args = parse_cli()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    client = make_bedrock_client(
        args.region,
        args.profile,
        read_timeout=args.read_timeout,
        connect_timeout=args.connect_timeout,
    )
    logger.info(
        "Configured Bedrock client with connect_timeout=%ss read_timeout=%ss",
        args.connect_timeout,
        args.read_timeout,
    )

    records = expand_records(args.split, args.count)
    if not records:
        raise RuntimeError(f"No records produced for split '{args.split}'.")

    results = evaluate_records(
        client,
        model_id=args.model_id,
        records=records,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_workers=args.max_workers,
        num_votes=args.votes,
        bidirectional=args.bidirectional,
    )

    summary = summarise_results(results)
    print(json.dumps(summary, indent=2))

    if args.details_path is not None:
        save_details(args.details_path, results)
        print(f"Wrote detailed outputs to {args.details_path}")


if __name__ == "__main__":
    main()

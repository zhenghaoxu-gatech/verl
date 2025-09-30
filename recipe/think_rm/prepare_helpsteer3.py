"""Utilities for expanding HelpSteer3 preference data for generative RM training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from textwrap import dedent


# Scores are annotated per response pair; we only need the sign for binary reward.
# Negative -> response_1 preferred, positive -> response_2 preferred, zero -> tie.
def _score_to_label(score: int) -> str:
    if score > 0:
        return "response_2"
    if score < 0:
        return "response_1"
    return "tie"


def _normalise_context(raw_context: list[dict] | None) -> list[dict[str, str]]:
    if not raw_context:
        return []
    messages: list[dict[str, str]] = []
    for turn in raw_context:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append({"role": role, "content": content})
    return messages


PROMPT_INSTRUCTION = dedent(
    """
    You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the
    context displayed below. Your evaluation should be based on the following six criteria:

    - Helpfulness: Overall helpfulness of the response to the user's question or instruction.
    - Correctness: Inclusion of all pertinent facts without errors.
    - Coherence: Consistency and clarity of expression.
    - Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone
      with basic language competency or requires deep domain expertise).
    - Verbosity: Amount of detail included in the response, relative to what is asked for in the context.
    - Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.

    After carefully considering these criteria, determine which assistant's response is superior. Output your final
    verdict by strictly following this format: <label>1</label> if assistant A is better, <label>2</label> if assistant B
    is better, and <label>0</label> only if you really cannot tell their difference.
    """
).strip()


@dataclass
class ExpandedRecord:
    prompt: list[dict[str, str]]
    data_source: str
    reward_model: dict[str, object]
    extra_info: dict[str, object]
    uid: str

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "data_source": self.data_source,
            "reward_model": self.reward_model,
            "extra_info": self.extra_info,
            "uid": self.uid,
        }


def _format_context(context: list[dict[str, str]]) -> str:
    if not context:
        return "(No prior context provided.)"
    lines: list[str] = []
    for turn in context:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_prompt(context: list[dict[str, str]], response1: str, response2: str) -> list[dict[str, str]]:
    prompt = list(context)
    comparison_request = (
        f"{PROMPT_INSTRUCTION}\n\n"
        "[The Start of Context]\n"
        f"{_format_context(context)}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        f"{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        f"{response2}\n"
        "[The End of Assistant B's Response]"
    )
    prompt.append({"role": "user", "content": comparison_request})
    return prompt


def _expand_split(df: Dataset, split: str, max_samples: int | None = None) -> list[ExpandedRecord]:
    records: list[ExpandedRecord] = []
    total = len(df)

    for row_idx, row in enumerate(df):
        if max_samples is not None and len(records) >= max_samples:
            break

        base_messages = _normalise_context(row.get("context"))
        response1 = row.get("response1", "")
        response2 = row.get("response2", "")
        overall = row.get("overall_preference")
        pair_id = row.get("id", row_idx)

        annotators: Iterable[dict] = row.get("individual_preference") or []
        if not annotators:
            continue

        for annot_idx, ann in enumerate(annotators):
            score = ann.get("score")
            if not isinstance(score, (int, float)):
                continue
            label = _score_to_label(score)
            prompt = _build_prompt(base_messages, response1, response2)
            uid = f"{split}-{pair_id}-{annot_idx}"
            reward_model = {"style": "sign", "ground_truth": label}
            extra_info = {
                "annotator_index": annot_idx,
                "preference_score": float(score),
                "overall_preference": overall,
                "annotator_reasoning": ann.get("reasoning", ""),
                "sample_index": row_idx,
                "helpsteer_id": pair_id,
            }
            records.append(
                ExpandedRecord(
                    prompt=prompt,
                    data_source="helpsteer3_preference",
                    reward_model=reward_model,
                    extra_info=extra_info,
                    uid=uid,
                )
            )

    return records


def _write_records(records: list[ExpandedRecord], destination: Path) -> None:
    if not records:
        raise ValueError(f"No records to persist for {destination.name} split.")
    df = pd.DataFrame([record.as_dict() for record in records])
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination)


def prepare_dataset(output_dir: Path, splits: tuple[str, ...], max_samples: int | None = None) -> None:
    dataset: DatasetDict = load_dataset("nvidia/HelpSteer3")

    for split in splits:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not available in HelpSteer3 dataset.")

        expanded = _expand_split(dataset[split], split, max_samples=max_samples)
        target_path = output_dir / "rl" / f"{split}.parquet"
        _write_records(expanded, target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand HelpSteer3 preference annotations for RLHF training.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for generated parquet files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation"),
        help="Dataset splits to process (default: train validation).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper bound on number of expanded preference records per split",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="If set, write a JSON summary of each split alongside the parquet output.",
    )
    return parser.parse_args()


def _write_metadata(records: list[ExpandedRecord], path: Path) -> None:
    stats = {
        "num_records": len(records),
        "num_positive": sum(1 for r in records if r.reward_model["ground_truth"] == "response_2"),
        "num_negative": sum(1 for r in records if r.reward_model["ground_truth"] == "response_1"),
        "num_tie": sum(1 for r in records if r.reward_model["ground_truth"] == "tie"),
    }
    path.write_text(json.dumps(stats, indent=2))


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    dataset: DatasetDict = load_dataset("nvidia/HelpSteer3")

    for split in tuple(args.splits):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not available in HelpSteer3 dataset.")

    for split in tuple(args.splits):
        expanded = _expand_split(dataset[split], split, max_samples=args.max_samples)
        target_path = output_dir / "rl" / f"{split}.parquet"
        _write_records(expanded, target_path)
        if args.dump_metadata:
            metadata_path = target_path.with_suffix(".json")
            _write_metadata(expanded, metadata_path)


if __name__ == "__main__":
    main()

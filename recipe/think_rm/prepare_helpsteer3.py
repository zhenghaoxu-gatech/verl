"""Utilities for expanding HelpSteer3 preference data for generative RM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported when run as module
    from .preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records


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

        valid_annots: list[tuple[int, dict, str, float]] = []
        label_counts = {"response_1": 0, "response_2": 0, "tie": 0}

        for annot_idx, ann in enumerate(annotators):
            score = ann.get("score")
            if not isinstance(score, (int, float)):
                continue
            label = _score_to_label(score)
            label_counts[label] += 1
            valid_annots.append((annot_idx, ann, label, float(score)))

        total_votes = sum(label_counts.values())
        if total_votes == 0:
            continue

        prob_response_1 = label_counts["response_1"] / total_votes
        prob_response_2 = label_counts["response_2"] / total_votes
        prob_tie = label_counts["tie"] / total_votes

        for annot_idx, ann, label, score in valid_annots:
            prompt = build_prompt(base_messages, response1, response2)
            uid = f"{split}-{pair_id}-{annot_idx}"
            reward_model = {"style": "sign", "ground_truth": label}
            extra_info = {
                "annotator_index": annot_idx,
                "preference_score": score,
                "overall_preference": overall,
                "annotator_reasoning": ann.get("reasoning", ""),
                "sample_index": row_idx,
                "helpsteer_id": pair_id,
                "num_annotators": total_votes,
                "prob_response_1": prob_response_1,
                "prob_response_2": prob_response_2,
                "prob_tie": prob_tie,
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


def prepare_dataset(output_dir: Path, splits: tuple[str, ...], max_samples: int | None = None) -> None:
    dataset: DatasetDict = load_dataset("nvidia/HelpSteer3")

    for split in splits:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not available in HelpSteer3 dataset.")

        expanded = _expand_split(dataset[split], split, max_samples=max_samples)
        target_path = output_dir / "rl" / f"{split}.parquet"
        write_records(expanded, target_path)


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
        write_records(expanded, target_path)
        if args.dump_metadata:
            metadata_path = target_path.with_suffix(".json")
            dump_metadata(expanded, metadata_path)


if __name__ == "__main__":
    main()

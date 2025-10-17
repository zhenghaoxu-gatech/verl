"""Prepare Skywork reward preference data for think RM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported when run as module
    from .preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records

SKYWORK_DATASET_NAME = "Skywork/Skywork-Reward-Preference-80K-v0.2"


def _normalise_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not messages:
        return []
    normalised: list[dict[str, str]] = []
    for turn in messages:
        role = turn.get("role", "user")
        content = turn.get("content")
        normalised.append(
            {
                "role": str(role) if role is not None else "user",
                "content": content if isinstance(content, str) else ("" if content is None else str(content)),
            }
        )
    return normalised


def _split_context_and_response(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    if not messages:
        raise ValueError("Conversation is empty.")

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            response = messages[idx].get("content", "")
            context = messages[:idx]
            return context, response

    raise ValueError("Conversation does not contain an assistant response.")


def _expand_split(ds: Dataset, split: str, max_pairs: int | None = None) -> list[ExpandedRecord]:
    records: list[ExpandedRecord] = []
    processed_pairs = 0
    skipped_context_mismatch = 0

    for row_idx, row in enumerate(ds):
        if max_pairs is not None and processed_pairs >= max_pairs:
            break

        chosen_messages = _normalise_messages(row.get("chosen"))
        rejected_messages = _normalise_messages(row.get("rejected"))

        if not chosen_messages or not rejected_messages:
            continue

        try:
            context, chosen_response = _split_context_and_response(chosen_messages)
            rejected_context, rejected_response = _split_context_and_response(rejected_messages)
        except ValueError:
            continue

        source = row.get("source", "")
        uid_base = row.get("id", f"{split}-{row_idx}")
        if isinstance(uid_base, bytes):
            uid_base = uid_base.decode("utf-8", errors="ignore")
        uid_base = str(uid_base)

        if rejected_context != context:
            skipped_context_mismatch += 1
            continue

        base_extra = {
            "sample_index": row_idx,
            "skywork_source": source,
            "num_context_turns": len(context),
            "num_total_turns": len(context) + 1,
            "num_annotators": 1,
        }

        forward_prompt = build_prompt(context, chosen_response, rejected_response)
        forward = ExpandedRecord(
            prompt=forward_prompt,
            data_source="skywork_reward_preference",
            reward_model={"style": "sign", "ground_truth": "response_1"},
            extra_info={
                **base_extra,
                "orientation": "forward",
                "prob_response_1": 1.0,
                "prob_response_2": 0.0,
                "prob_tie": 0.0,
            },
            uid=f"{uid_base}-forward",
        )

        backward_prompt = build_prompt(context, rejected_response, chosen_response)
        backward = ExpandedRecord(
            prompt=backward_prompt,
            data_source="skywork_reward_preference",
            reward_model={"style": "sign", "ground_truth": "response_2"},
            extra_info={
                **base_extra,
                "orientation": "backward",
                "prob_response_1": 0.0,
                "prob_response_2": 1.0,
                "prob_tie": 0.0,
            },
            uid=f"{uid_base}-backward",
        )

        records.extend((forward, backward))
        processed_pairs += 1

    if skipped_context_mismatch:
        print(
            f"[prepare_skywork_reward] skipped {skipped_context_mismatch} pairs "
            f"with inconsistent chosen/rejected contexts in split '{split}'."
        )

    return records


def prepare_dataset(output_dir: Path, split: str, max_pairs: int | None = None) -> list[ExpandedRecord]:
    dataset: DatasetDict = load_dataset(SKYWORK_DATASET_NAME)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in {SKYWORK_DATASET_NAME}.")

    expanded = _expand_split(dataset[split], split, max_pairs=max_pairs)
    target_path = output_dir / "rl" / f"{split}.parquet"
    write_records(expanded, target_path)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand Skywork reward preference data for RLHF training.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for generated parquet files.")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to process (default: train).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional upper bound on the number of preference pairs to expand.",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="If set, write a JSON summary alongside the parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expanded = prepare_dataset(args.output_dir, args.split, max_pairs=args.max_pairs)

    if args.dump_metadata:
        target_path = args.output_dir / "rl" / f"{args.split}.parquet"
        dump_metadata(expanded, target_path.with_suffix(".json"))


if __name__ == "__main__":
    main()

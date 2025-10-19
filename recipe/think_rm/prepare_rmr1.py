"""Prepare RM-R1 preference data for think RM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported when run as module
    from .preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records

DATASET_NAME = "gaotang/RM-R1-Entire-RLVR-Train"

CLIENT_QUESTION_ANCHOR = "[Client Question]"
START_A_ANCHOR = "[The Start of Chatbot A's Response]"
END_A_ANCHOR = "[The End of Chatbot A's Response]"
START_B_ANCHOR = "[The Start of Chatbot B's Response]"
END_B_ANCHOR = "[The End of Chatbot B's Response]"


def _remove_leading_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and isinstance(messages[0], dict):
        role = messages[0].get("role")
        if isinstance(role, str) and role.lower() == "system":
            return messages[1:]
    return messages


def _combine_message_contents(messages: Iterable[dict[str, Any]]) -> str:
    contents: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            contents.append(content)
    return "\n\n".join(contents).strip()


def _extract_question_and_responses(raw_text: str) -> tuple[list[dict[str, str]], str, str]:
    question_idx = raw_text.find(CLIENT_QUESTION_ANCHOR)
    if question_idx == -1:
        raise ValueError("Client question marker not found.")
    question_start = question_idx + len(CLIENT_QUESTION_ANCHOR)

    start_a_idx = raw_text.find(START_A_ANCHOR, question_start)
    if start_a_idx == -1:
        raise ValueError("Chatbot A start marker not found.")

    question_text = raw_text[question_start:start_a_idx].strip()
    if not question_text:
        raise ValueError("Extracted question text is empty.")

    response_a_start = start_a_idx + len(START_A_ANCHOR)
    end_a_idx = raw_text.find(END_A_ANCHOR, response_a_start)
    if end_a_idx == -1:
        raise ValueError("Chatbot A end marker not found.")
    response_a_text = raw_text[response_a_start:end_a_idx].strip()
    if not response_a_text:
        raise ValueError("Chatbot A response is empty.")

    start_b_idx = raw_text.find(START_B_ANCHOR, end_a_idx + len(END_A_ANCHOR))
    if start_b_idx == -1:
        raise ValueError("Chatbot B start marker not found.")

    response_b_start = start_b_idx + len(START_B_ANCHOR)
    end_b_idx = raw_text.find(END_B_ANCHOR, response_b_start)
    if end_b_idx == -1:
        raise ValueError("Chatbot B end marker not found.")
    response_b_text = raw_text[response_b_start:end_b_idx].strip()
    if not response_b_text:
        raise ValueError("Chatbot B response is empty.")

    context_messages = [{"role": "user", "content": question_text}]
    return context_messages, response_a_text, response_b_text


def _map_winner_label(winner: Any) -> str | None:
    if not isinstance(winner, str):
        return None
    normalised = winner.strip().lower()
    if normalised == "model_a":
        return "response_1"
    if normalised == "model_b":
        return "response_2"
    if normalised in {"tie", "draw", "equal"}:
        return "tie"
    return None


def _expand_split(ds: Dataset, split: str, max_samples: int | None = None) -> list[ExpandedRecord]:
    records: list[ExpandedRecord] = []
    skipped_missing_winner = 0
    skipped_parse_errors = 0

    for row_idx, row in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break

        raw_messages = row.get("context_messages")
        if not isinstance(raw_messages, list):
            skipped_parse_errors += 1
            continue

        messages_without_system = _remove_leading_system(raw_messages)
        merged_text = _combine_message_contents(messages_without_system)
        if not merged_text:
            skipped_parse_errors += 1
            continue

        try:
            context_messages, response_a, response_b = _extract_question_and_responses(merged_text)
        except ValueError:
            skipped_parse_errors += 1
            continue

        label = _map_winner_label(row.get("winner"))
        if label is None:
            skipped_missing_winner += 1
            continue

        prompt = build_prompt(context_messages, response_a, response_b)
        extra_info = {
            "sample_index": row_idx,
            "rmr1_split": split,
        }

        uid = f"{split}-{row_idx}"
        records.append(
            ExpandedRecord(
                prompt=prompt,
                data_source="rmr1_preference",
                reward_model={"style": "sign", "ground_truth": label},
                extra_info=extra_info,
                uid=uid,
            )
        )

    if skipped_missing_winner or skipped_parse_errors:
        print(
            f"[prepare_rmr1] split='{split}' skipped {skipped_missing_winner} samples with missing winner "
            f"and {skipped_parse_errors} samples due to parsing errors.",
            file=sys.stderr,
        )

    return records


def prepare_dataset(output_dir: Path, splits: tuple[str, ...], max_samples: int | None = None) -> dict[str, list[ExpandedRecord]]:
    dataset: DatasetDict = load_dataset(DATASET_NAME)
    results: dict[str, list[ExpandedRecord]] = {}

    for split in splits:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not available in {DATASET_NAME}.")

    for split in splits:
        expanded = _expand_split(dataset[split], split, max_samples=max_samples)
        target_path = output_dir / "rl" / f"{split}.parquet"
        write_records(expanded, target_path)
        results[split] = expanded

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RM-R1 preference data for think RM training.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for generated parquet files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train",),
        help="Dataset splits to process (default: train).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper bound on number of expanded preference records per split.",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="If set, write a JSON summary alongside the parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir

    expanded_by_split = prepare_dataset(output_dir, tuple(args.splits), max_samples=args.max_samples)

    if args.dump_metadata:
        for split, records in expanded_by_split.items():
            target_path = output_dir / "rl" / f"{split}.parquet"
            dump_metadata(records, target_path.with_suffix(".json"))


if __name__ == "__main__":
    main()

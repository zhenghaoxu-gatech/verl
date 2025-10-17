"""Expand lmarena-ai/arena-human-preference-140k data for think RM training."""

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

DATASET_NAME = "lmarena-ai/arena-human-preference-140k"


def _collect_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Iterable):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
            elif isinstance(item, str) and item.strip():
                fragments.append(item.strip())
        return "\n\n".join(fragments)

    return str(content).strip()


def _normalise_role(raw_role: Any) -> str:
    if raw_role is None:
        return "assistant"
    role = str(raw_role).strip().lower()
    if not role:
        return "assistant"
    if role == "system":
        return "system"
    if role in {"prompter", "user"} or "user" in role:
        return "user"
    return "assistant"


def _normalise_conversation(conversation: Any) -> list[dict[str, str]]:
    if not isinstance(conversation, Iterable):
        return []
    processed: list[dict[str, str]] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = _normalise_role(turn.get("role"))
        text = _collect_text(turn.get("content"))
        if text:
            processed.append({"role": role, "content": text})
    return processed


def _split_context_and_response(conversation: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    for idx in range(len(conversation) - 1, -1, -1):
        role = conversation[idx].get("role")
        if role == "user":
            continue
        response_text = conversation[idx].get("content", "")
        if response_text:
            response = response_text
            context = conversation[:idx]
            return context, response
    raise ValueError("Conversation does not contain a valid assistant response.")


def _turns_equal(a: dict[str, str], b: dict[str, str]) -> bool:
    return a.get("role") == b.get("role") and a.get("content") == b.get("content")


def _contexts_equal(lhs: list[dict[str, str]], rhs: list[dict[str, str]]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(_turns_equal(a, b) for a, b in zip(lhs, rhs))


def _winner_to_label(winner: str) -> tuple[str, float, float, float]:
    canonical = winner.strip().lower()
    if canonical == "model_a":
        return "response_1", 1.0, 0.0, 0.0
    if canonical == "model_b":
        return "response_2", 0.0, 1.0, 0.0
    if canonical in {"tie", "both_bad"}:
        return "tie", 0.0, 0.0, 1.0
    raise ValueError(f"Unrecognised winner value: {winner!r}")


def _extract_from_full_conversation(full_conversation: Any) -> tuple[list[dict[str, str]], str, str] | None:
    if not isinstance(full_conversation, Iterable):
        return None

    rounds: list[dict[str, Any]] = [r for r in full_conversation if isinstance(r, dict)]
    if not rounds:
        return None

    context: list[dict[str, str]] = []
    final_round = rounds[-1]

    for entry in rounds[:-1]:
        user_struct = entry.get("user", {})
        user_text = _collect_text(user_struct.get("content"))
        if user_text:
            context.append({"role": "user", "content": user_text})

        model_a_text = _collect_text(entry.get("model_side_a", {}).get("content"))
        model_b_text = _collect_text(entry.get("model_side_b", {}).get("content"))
        combined_segments = []
        if model_a_text:
            combined_segments.append(f"Model A:\n{model_a_text}")
        if model_b_text:
            combined_segments.append(f"Model B:\n{model_b_text}")
        if combined_segments:
            context.append({"role": "assistant", "content": "\n\n".join(combined_segments)})

    user_final = _collect_text(final_round.get("user", {}).get("content"))
    response_a = _collect_text(final_round.get("model_side_a", {}).get("content"))
    response_b = _collect_text(final_round.get("model_side_b", {}).get("content"))

    if user_final:
        context.append({"role": "user", "content": user_final})

    if not response_a or not response_b:
        return None

    return context, response_a, response_b


def _expand_split(ds: Dataset, split: str, max_samples: int | None = None) -> list[ExpandedRecord]:
    records: list[ExpandedRecord] = []
    skipped_missing_winner = 0
    skipped_invalid_conversation = 0
    skipped_context_mismatch = 0
    skipped_full_conversation = 0

    for row_idx, row in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break

        winner = row.get("winner")
        if not isinstance(winner, str) or not winner.strip():
            skipped_missing_winner += 1
            continue

        context: list[dict[str, str]] | None = None
        response_a: str | None = None
        response_b: str | None = None
        source = "conversation_pair"

        full_conv = row.get("full_conversation")
        if full_conv:
            extracted = _extract_from_full_conversation(full_conv)
            if extracted is not None:
                context, response_a, response_b = extracted
                source = "full_conversation"
            else:
                skipped_full_conversation += 1

        if response_a is None or response_b is None or context is None:
            conv_a = _normalise_conversation(row.get("conversation_a"))
            conv_b = _normalise_conversation(row.get("conversation_b"))
            if not conv_a or not conv_b:
                skipped_invalid_conversation += 1
                continue

            try:
                context_a, response_a = _split_context_and_response(conv_a)
                context_b, response_b = _split_context_and_response(conv_b)
            except ValueError:
                skipped_invalid_conversation += 1
                continue

            if _contexts_equal(context_a, context_b):
                context = context_a
            else:
                min_len = min(len(context_a), len(context_b))
                prefix: list[dict[str, str]] = []
                for pos in range(min_len):
                    if _turns_equal(context_a[pos], context_b[pos]):
                        prefix.append(context_a[pos])
                    else:
                        break
                if prefix:
                    context = prefix
                else:
                    skipped_context_mismatch += 1
                    continue

        try:
            label, prob_r1, prob_r2, prob_tie = _winner_to_label(winner)
        except ValueError:
            skipped_missing_winner += 1
            continue

        uid_base = row.get("id", f"{split}-{row_idx}")
        if isinstance(uid_base, bytes):
            uid = uid_base.decode("utf-8", errors="ignore")
        else:
            uid = str(uid_base)

        prompt = build_prompt(context, response_a, response_b)
        reward_model = {"style": "sign", "ground_truth": label}

        extra_info = {
            "sample_index": row_idx,
            "evaluation_session_id": row.get("evaluation_session_id"),
            "evaluation_order": row.get("evaluation_order"),
            "prob_response_1": prob_r1,
            "prob_response_2": prob_r2,
            "prob_tie": prob_tie,
            "context_source": source,
        }

        records.append(
            ExpandedRecord(
                prompt=prompt,
                data_source="arena_human_preference",
                reward_model=reward_model,
                extra_info=extra_info,
                uid=f"{split}-{uid}",
            )
        )

    if skipped_missing_winner:
        print(f"[prepare_arena_human_preference] skipped {skipped_missing_winner} rows without valid winner labels in split '{split}'.")
    if skipped_invalid_conversation:
        print(f"[prepare_arena_human_preference] skipped {skipped_invalid_conversation} rows without valid conversations in split '{split}'.")
    if skipped_context_mismatch:
        print(f"[prepare_arena_human_preference] skipped {skipped_context_mismatch} rows due to mismatched contexts between A/B in split '{split}'.")
    if skipped_full_conversation:
        print(f"[prepare_arena_human_preference] fell back to conversation pairs {skipped_full_conversation} times because full_conversation was incomplete in split '{split}'.")

    return records


def prepare_dataset(output_dir: Path, split: str, max_samples: int | None = None) -> list[ExpandedRecord]:
    dataset: DatasetDict = load_dataset(DATASET_NAME)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in {DATASET_NAME}.")

    expanded = _expand_split(dataset[split], split, max_samples=max_samples)
    target_path = output_dir / "rl" / f"{split}.parquet"
    write_records(expanded, target_path)
    return expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand arena-human-preference-140k data for think RM training.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for generated parquet files.")
    parser.add_argument("--split", default="train", help="Dataset split to process (default: train).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper bound on the number of preference records to emit.",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="If set, write a JSON summary alongside the parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expanded = prepare_dataset(args.output_dir, args.split, max_samples=args.max_samples)

    if args.dump_metadata:
        target_path = args.output_dir / "rl" / f"{args.split}.parquet"
        dump_metadata(expanded, target_path.with_suffix(".json"))


if __name__ == "__main__":
    main()

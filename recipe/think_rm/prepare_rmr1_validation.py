"""Prepare validation data for Think RM using random orientations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Literal, cast

from datasets import Dataset, load_dataset

if __package__ is None or __package__ == "":
    from preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records, get_probs  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported when run as module
    from .preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, write_records, get_probs

LABEL_SWAP: dict[str, str] = {
    "response_1": "response_2",
    "response_2": "response_1",
    "tie": "tie",
}


def _random_orientation_flip(
    response_a: str,
    response_b: str,
    label: Literal["response_1", "response_2", "tie"],
    rng: random.Random,
) -> tuple[str, str, Literal["response_1", "response_2", "tie"], str]:
    if rng.random() < 0.5:
        return response_a, response_b, label, "forward"
    flipped_label = cast(Literal["response_1", "response_2", "tie"], LABEL_SWAP[label])
    return response_b, response_a, flipped_label, "backward"


def _normalise_context(raw_context: Iterable[dict] | None) -> list[dict[str, str]]:
    if not raw_context:
        return []
    normalised: list[dict[str, str]] = []
    for turn in raw_context:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        normalised.append({"role": role, "content": stripped})
    return normalised


def _overall_to_label(value: object) -> Literal["response_1", "response_2", "tie"] | None:
    if isinstance(value, (int, float)):
        if value < 0:
            return "response_1"
        if value > 0:
            return "response_2"
        return "tie"
    return None


def _collect_helpsteer(
    split: str,
    rng: random.Random,
    *,
    max_examples: int | None,
    num_proc: int | None,
) -> list[ExpandedRecord]:
    load_kwargs = {}
    if num_proc and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset("nvidia/HelpSteer3", split=split, **load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for HelpSteer3 split '{split}', got {type(dataset)}.")

    records: list[ExpandedRecord] = []
    for row_idx, row in enumerate(dataset):
        response1 = row.get("response1")
        response2 = row.get("response2")
        overall = row.get("overall_preference")

        if not isinstance(response1, str) or not isinstance(response2, str):
            continue

        label = _overall_to_label(overall)
        if label is None:
            continue

        context_messages = _normalise_context(row.get("context"))
        response_a, response_b, mapped_label, orientation = _random_orientation_flip(response1, response2, label, rng)
        prob_response_1, prob_response_2, prob_tie = get_probs(mapped_label)
        prompt = build_prompt(context_messages, response_a, response_b)
        row_id = row.get("id", row_idx)
        row_id_str = str(row_id)
        records.append(
            ExpandedRecord(
                prompt=prompt,
                data_source="helpsteer3_validation",
                reward_model={"style": "sign", "ground_truth": mapped_label},
                extra_info={
                    "orientation": orientation,
                    "source_split": split,
                    "prob_response_1": prob_response_1,
                    "prob_response_2": prob_response_2,
                    "prob_tie": prob_tie,
                },
                uid=f"helpsteer3-{split}-{row_id_str}",
            )
        )

        if max_examples is not None and len(records) >= max_examples:
            break

    return records


def _collect_rewardbench(
    split: str,
    rng: random.Random,
    *,
    max_examples: int | None,
    num_proc: int | None,
) -> list[ExpandedRecord]:
    load_kwargs = {}
    if num_proc and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset("allenai/reward-bench-2", split=split, **load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for RewardBench split '{split}', got {type(dataset)}.")

    records: list[ExpandedRecord] = []
    seen_prompts: set[str] = set()
    for row_idx, row in enumerate(dataset):
        prompt_text = row.get("prompt")
        chosen_list = row.get("chosen") or []
        rejected_list = row.get("rejected") or []

        if not isinstance(prompt_text, str):
            continue

        prompt_body = prompt_text.strip()
        if not prompt_body:
            continue

        chosen_candidates = [c for c in chosen_list if isinstance(c, str) and c.strip()]
        rejected_candidates = [r for r in rejected_list if isinstance(r, str) and r.strip()]
        if not chosen_candidates or not rejected_candidates:
            continue

        subset_raw = row.get("subset", "unknown")
        if not isinstance(subset_raw, str):
            subset_raw = "unknown"
        subset_clean = subset_raw.strip() or "unknown"
        subset_slug = subset_clean.lower().replace(" ", "-").replace("/", "-")

        for chosen_idx, preferred in enumerate(chosen_candidates):
            for rejected_idx, dispreferred in enumerate(rejected_candidates):
                response_a, response_b, mapped_label, orientation = _random_orientation_flip(
                    preferred,
                    dispreferred,
                    "response_1",
                    rng,
                )
                prob_response_1, prob_response_2, prob_tie = get_probs(mapped_label)
                prompt = build_prompt([{"role": "user", "content": prompt_body}], response_a, response_b)

                if prompt[0]["content"] in seen_prompts:
                    continue
                seen_prompts.add(prompt[0]["content"])
                prompt_id = row.get("id", row_idx)
                prompt_id_str = str(prompt_id)
                uid = f"rewardbench-{split}-{subset_slug}-{prompt_id_str}-c{chosen_idx}-r{rejected_idx}"
                records.append(
                    ExpandedRecord(
                        prompt=prompt,
                        data_source="rewardbench2",
                        reward_model={"style": "sign", "ground_truth": mapped_label},
                        extra_info={
                            "orientation": orientation,
                            "source_split": split,
                            "subset": subset_clean,
                            "prob_response_1": prob_response_1,
                            "prob_response_2": prob_response_2,
                            "prob_tie": prob_tie,
                        },
                        uid=uid,
                    )
                )

                if max_examples is not None and len(records) >= max_examples:
                    return records

    return records


def _collect_rmbench(
    split: str,
    rng: random.Random,
    *,
    max_examples: int | None,
    num_proc: int | None,
) -> list[ExpandedRecord]:
    load_kwargs = {}
    if num_proc and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset("THU-KEG/RM-Bench", split=split, **load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for RM-Bench split '{split}', got {type(dataset)}.")

    records: list[ExpandedRecord] = []
    for row_idx, row in enumerate(dataset):
        prompt_text = row.get("prompt")
        chosen_list = row.get("chosen") or []
        rejected_list = row.get("rejected") or []
        if not isinstance(prompt_text, str):
            continue

        prompt_body = prompt_text.strip()
        if not prompt_body:
            continue

        chosen_candidates = [c for c in chosen_list if isinstance(c, str) and c.strip()]
        rejected_candidates = [r for r in rejected_list if isinstance(r, str) and r.strip()]
        if not chosen_candidates or not rejected_candidates:
            continue

        domain = row.get("domain", "unknown")
        for chosen_idx, preferred in enumerate(chosen_candidates):
            for rejected_idx, dispreferred in enumerate(rejected_candidates):
                response_a, response_b, mapped_label, orientation = _random_orientation_flip(
                    preferred,
                    dispreferred,
                    "response_1",
                    rng,
                )
                prob_response_1, prob_response_2, prob_tie = get_probs(mapped_label)
                prompt = build_prompt([{"role": "user", "content": prompt_body}], response_a, response_b)
                prompt_id = row.get("id", row_idx)
                prompt_id_str = str(prompt_id)
                uid = f"rmbench-{split}-{prompt_id_str}-c{chosen_idx}-r{rejected_idx}"
                records.append(
                    ExpandedRecord(
                        prompt=prompt,
                        data_source="rmbench",
                        reward_model={"style": "sign", "ground_truth": mapped_label},
                        extra_info={
                            "orientation": orientation,
                            "source_split": split,
                            "prob_response_1": prob_response_1,
                            "prob_response_2": prob_response_2,
                            "prob_tie": prob_tie,
                        },
                        uid=uid,
                    )
                )

                if max_examples is not None and len(records) >= max_examples:
                    return records

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Think RM validation data with randomly oriented pairs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for generated parquet files.")
    parser.add_argument("--seed", type=int, default=0, help="Seed controlling random orientations.")
    parser.add_argument("--helpsteer-split", default="validation", help="HelpSteer3 split to use (default: validation).")
    parser.add_argument("--rewardbench-split", default="test", help="RewardBench v2 split to use (default: test).")
    parser.add_argument("--rmbench-split", default="train", help="RM-Bench split to use (default: train).")
    parser.add_argument("--helpsteer-limit", type=int, default=None, help="Optional cap on HelpSteer3 examples.")
    parser.add_argument("--rewardbench-limit", type=int, default=None, help="Optional cap on RewardBench examples.")
    parser.add_argument("--rmbench-limit", type=int, default=None, help="Optional cap on RM-Bench examples.")
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Optional multiprocessing factor passed to datasets.load_dataset.",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="If set, write a JSON summary alongside the parquet output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    records: list[ExpandedRecord] = []
    records.extend(
        _collect_helpsteer(
            args.helpsteer_split,
            rng,
            max_examples=args.helpsteer_limit,
            num_proc=args.dataset_num_proc,
        )
    )
    records.extend(
        _collect_rewardbench(
            args.rewardbench_split,
            rng,
            max_examples=args.rewardbench_limit,
            num_proc=args.dataset_num_proc,
        )
    )
    records.extend(
        _collect_rmbench(
            args.rmbench_split,
            rng,
            max_examples=args.rmbench_limit,
            num_proc=args.dataset_num_proc,
        )
    )
    random.seed(42)
    random.shuffle(records)

    destination = args.output_dir / "rl" / "validation.parquet"
    write_records(records, destination)

    if args.dump_metadata:
        dump_metadata(records, destination.with_suffix(".json"))


if __name__ == "__main__":
    main()

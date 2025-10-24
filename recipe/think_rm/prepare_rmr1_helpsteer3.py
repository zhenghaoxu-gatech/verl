"""Build a combined RM-R1 + HelpSteer3 preference dataset and optionally push to Hugging Face.

Running this script locally will:
  * expand RM-R1 preference data using the existing preparation utilities,
  * sample HelpSteer3 pairwise preferences (train split for training, validation split for held-out eval),
  * optionally persist parquet artifacts for each component and the combined dataset,
  * optionally upload the resulting DatasetDict to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset

RMR1_SPLITS: Tuple[str, ...] = ("train",)
HS3_TRAIN_SPLIT = "train"
HS3_VALIDATION_SPLIT = "validation"

if __package__ is None or __package__ == "":
    # Allow running as a standalone script.
    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from preference_dataset_utils import (  # type: ignore  # noqa: E402
        ExpandedRecord,
        build_prompt,
        dump_metadata,
        summarise_records,
        get_probs,
    )
    import prepare_rmr1 as prepare_rmr1_module  # type: ignore  # noqa: E402
    import prepare_rmr1_validation as prepare_rmr1_validation_module  # type: ignore  # noqa: E402
else:  # pragma: no cover - imported when run as module
    from .preference_dataset_utils import ExpandedRecord, build_prompt, dump_metadata, summarise_records, get_probs
    from . import prepare_rmr1 as prepare_rmr1_module
    from . import prepare_rmr1_validation as prepare_rmr1_validation_module


def _strip_extra_info(record: ExpandedRecord) -> dict:
    """Convert an ExpandedRecord into a dict without the extra_info column."""
    data = record.as_dict()
    data.pop("extra_info", None)
    return data


def _records_to_dataset(records: List[ExpandedRecord]) -> Dataset:
    """Convert ExpandedRecord objects into a Hugging Face Dataset."""
    return Dataset.from_list([record.as_dict() for record in records])


def _collapse_records(record_groups: Iterable[Iterable[ExpandedRecord]]) -> List[ExpandedRecord]:
    """Flatten multiple iterable collections of records into a single list."""
    return list(itertools.chain.from_iterable(record_groups))


def _write_records_without_extra_info(records: List[ExpandedRecord], destination: Path) -> None:
    """Persist records to parquet after removing the extra_info field."""
    if not records:
        raise ValueError(f"No records to persist for {destination.name}.")
    dataset = _records_to_dataset(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(destination))


def _prepare_rmr1_records(output_root: Path | None) -> Dict[str, List[ExpandedRecord]]:
    """Expand RM-R1 samples and optionally persist local parquet shards without extra_info."""
    dataset_dict = load_dataset(prepare_rmr1_module.DATASET_NAME)
    results: Dict[str, List[ExpandedRecord]] = {}
    for split in RMR1_SPLITS:
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not available in {prepare_rmr1_module.DATASET_NAME}.")
        records = prepare_rmr1_module._expand_split(  # pylint: disable=protected-access
            dataset_dict[split],
            split,
        )
        _normalise_extra_info(records)
        _standardise_uids(records, "rmr1")
        if output_root is not None:
            target_dir = output_root / "rmr1"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{split}.parquet"
            _write_records_without_extra_info(records, target_path)
            dump_metadata(records, target_path.with_suffix(".json"))
        results[split] = records
    return results


def _collect_helpsteer_preserve_order(
    split: str,
    *,
    max_examples: int | None = None,
    num_proc: int | None = None,
) -> List[ExpandedRecord]:
    """Collect HelpSteer3 preferences without orientation flipping."""
    load_kwargs = {}
    if num_proc and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset("nvidia/HelpSteer3", split=split, **load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for HelpSteer3 split '{split}', got {type(dataset)}.")

    records: List[ExpandedRecord] = []
    for row_idx, row in enumerate(dataset):
        if max_examples is not None and len(records) >= max_examples:
            break

        response1 = row.get("response1")
        response2 = row.get("response2")
        overall = row.get("overall_preference")

        if not isinstance(response1, str) or not isinstance(response2, str):
            continue

        label = prepare_rmr1_validation_module._overall_to_label(  # pylint: disable=protected-access
            overall
        )
        if label is None:
            continue

        context_messages = prepare_rmr1_validation_module._normalise_context(  # pylint: disable=protected-access
            row.get("context")
        )
        prompt = build_prompt(context_messages, response1, response2)
        prob_response_1, prob_response_2, prob_tie = get_probs(label)
        extra_info = {
            "prob_response_1": prob_response_1,
            "prob_response_2": prob_response_2,
            "prob_tie": prob_tie,
        }
        row_id = row.get("id", row_idx)
        row_id_str = str(row_id)
        records.append(
            ExpandedRecord(
                prompt=prompt,
                data_source=f"hs3_{split}",
                reward_model={"style": "sign", "ground_truth": label},
                extra_info=extra_info,
                uid=f"hs3-{split}-{row_id_str}",
            )
        )

    return records


def _prepare_helpsteer_records(output_root: Path | None, split: str) -> List[ExpandedRecord]:
    """Collect HelpSteer3 pairs while preserving their original orientation."""
    records = _collect_helpsteer_preserve_order(split)
    _normalise_extra_info(records)
    _standardise_uids(records, "hs3")

    if output_root is not None:
        target_dir = output_root / "hs3"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{split}.parquet"
        _write_records_without_extra_info(records, target_path)
        dump_metadata(records, target_path.with_suffix(".json"))
    return records


def _materialise_split(
    records: List[ExpandedRecord],
    output_root: Path,
    filename: str,
) -> None:
    """Persist records to parquet alongside summary metadata inside the rl directory."""
    target_dir = output_root / "rl"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    _write_records_without_extra_info(records, target_path)
    dump_metadata(records, target_path.with_suffix(".json"))


def _normalise_extra_info(records: List[ExpandedRecord]) -> None:
    """Ensure extra_info dictionaries are empty for schema consistency."""
    for record in records:
        record.extra_info = {
            "prob_response_1": record.extra_info.get("prob_response_1"),
            "prob_response_2": record.extra_info.get("prob_response_2"),
            "prob_tie": record.extra_info.get("prob_tie"),
        }


def _standardise_uids(records: List[ExpandedRecord], prefix: str) -> None:
    """Apply a consistent prefix to record identifiers."""
    for record in records:
        if record.uid.startswith(f"{prefix}-"):
            continue
        record.uid = f"{prefix}-{record.uid}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a combined RM-R1 + HelpSteer3 preference dataset and optionally push to Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where parquet artifacts will be written.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="When provided, push the resulting DatasetDict to Hugging Face.",
    )
    parser.add_argument(
        "--repo-id",
        default="zhenghaoxu/think-rm-rmr1-helpsteer3",
        help="Target Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Explicit Hugging Face token to use. Falls back to cached login when omitted.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the dataset as a private repo.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Max shard size to use when pushing to the Hub.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Expand RM-R1 split(s).
    rmr1_records_by_split = _prepare_rmr1_records(output_dir)
    rmr1_flat_records = _collapse_records(rmr1_records_by_split.values())
    if not rmr1_flat_records:
        raise ValueError("No RM-R1 records were generated; cannot build combined dataset.")

    # Collect HelpSteer3 pairs.
    helpsteer_train_records = _prepare_helpsteer_records(output_dir, HS3_TRAIN_SPLIT)
    helpsteer_validation_records = _prepare_helpsteer_records(output_dir, HS3_VALIDATION_SPLIT)

    # Prepare combined splits.
    train_records = rmr1_flat_records + helpsteer_train_records
    validation_records = helpsteer_validation_records

    if output_dir is not None:
        _materialise_split(train_records, output_dir, "train.parquet")
        _materialise_split(validation_records, output_dir, "hs3_validation.parquet")

    train_summary = summarise_records(train_records)
    validation_summary = summarise_records(validation_records)
    print(f"[combined dataset] train={train_summary} validation={validation_summary}")

    if not args.push_to_hub:
        return

    dataset_dict = DatasetDict(
        {
            "train": _records_to_dataset(train_records),
            "validation": _records_to_dataset(validation_records),
        }
    )
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        token=args.hf_token,
        max_shard_size=args.max_shard_size,
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

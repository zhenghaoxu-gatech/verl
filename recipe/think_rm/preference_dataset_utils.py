"""Shared utilities for preparing preference datasets for think RM training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd

# PROMPT_INSTRUCTION = dedent(
#     """
#     You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the
#     context displayed below. Your evaluation should be based on the following six criteria:

#     - Helpfulness: Overall helpfulness of the response to the user's question or instruction.
#     - Correctness: Inclusion of all pertinent facts without errors.
#     - Coherence: Consistency and clarity of expression.
#     - Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone
#       with basic language competency or requires deep domain expertise).
#     - Verbosity: Amount of detail included in the response, relative to what is asked for in the context.
#     - Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.

#     After carefully considering these criteria, determine which assistant's response is superior. Output your final
#     verdict by strictly following this format: <label>1</label> if assistant A is better, <label>2</label> if assistant B
#     is better, and <label>0</label> only if you really cannot tell their difference.
#     """
# ).strip()

PROMPT_INSTRUCTION = (
    "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the "
    "context displayed below. Your evaluation should be based on the following general criteria:\n"
    "\n"
    "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
    "- Correctness: Inclusion of all pertinent facts without errors.\n"
    "- Coherence: Consistency and clarity of expression.\n"
    "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone "
    "with basic language competency or requires deep domain expertise).\n"
    "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
    "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n"
    "\n"
    "Beyond the above general criteria, propose any additional context-specific criteria that would be relevant for "
    "evaluating the two responses. Justify why these criteria are important for this particular context.\n"
    "After carefully considering these criteria, determine which assistant's response is superior. Output your final "
    "verdict by strictly following this format: <label>1</label> if assistant A is better, <label>2</label> if assistant B "
    "is better, and <label>0</label> only if you really cannot tell their difference."
)


@dataclass
class ExpandedRecord:
    prompt: list[dict[str, str]]
    data_source: str
    reward_model: dict[str, Any]
    extra_info: dict[str, Any]
    uid: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "data_source": self.data_source,
            "reward_model": self.reward_model,
            "extra_info": self.extra_info,
            "uid": self.uid,
        }


def format_context(context: list[dict[str, str]]) -> str:
    if not context:
        return "(No prior context provided.)"
    lines: list[str] = []
    for turn in context:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_prompt(context: list[dict[str, str]], response1: str, response2: str) -> list[dict[str, str]]:
    comparison_request = (
        f"{PROMPT_INSTRUCTION}\n\n"
        "[The Start of Context]\n"
        f"{format_context(context)}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        f"{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        f"{response2}\n"
        "[The End of Assistant B's Response]"
    )
    return [{"role": "user", "content": comparison_request}]


def write_records(records: list[ExpandedRecord], destination: Path) -> None:
    if not records:
        raise ValueError(f"No records to persist for {destination.name}.")
    df = pd.DataFrame([record.as_dict() for record in records])
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination)


def summarise_records(records: list[ExpandedRecord]) -> dict[str, Any]:
    return {
        "num_records": len(records),
        "num_positive": sum(1 for r in records if r.reward_model["ground_truth"] == "response_2"),
        "num_negative": sum(1 for r in records if r.reward_model["ground_truth"] == "response_1"),
        "num_tie": sum(1 for r in records if r.reward_model["ground_truth"] == "tie"),
    }


def dump_metadata(records: list[ExpandedRecord], path: Path) -> None:
    path.write_text(json.dumps(summarise_records(records), indent=2))

def get_probs(mapped_label: str) -> tuple[float, float, float]:
    if mapped_label == "response_1":
        return 1.0, 0.0, 0.0
    if mapped_label == "response_2":
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0

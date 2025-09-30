"""Binary reward computation for the generative thinking reward model."""

from __future__ import annotations

import re
from typing import Optional


CHOICE_ALIASES = {
    "response 1": "response_1",
    "response1": "response_1",
    "assistant a": "response_1",
    "1": "response_1",
    "option 1": "response_1",
    "choice 1": "response_1",
    "a": "response_1",
    "response 2": "response_2",
    "response2": "response_2",
    "assistant b": "response_2",
    "2": "response_2",
    "option 2": "response_2",
    "choice 2": "response_2",
    "b": "response_2",
    "tie": "tie",
    "draw": "tie",
    "equal": "tie",
    "0": "tie",
}

FINAL_LINE_RE = re.compile(
    r"^(?:final|overall)?\s*(?:preference|decision|verdict|choice)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE,
)

FINAL_PREFERENCE_XML_RE = re.compile(
    r"<final_preference\s+label=\"(response_1|response_2|tie)\"\s*/?>",
    re.IGNORECASE,
)

LABEL_TAG_RE = re.compile(r"<label>\s*([012])\s*</label>", re.IGNORECASE)


def _normalise_choice(raw: str | None) -> Optional[str]:
    if raw is None:
        return None
    cleaned = raw.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in CHOICE_ALIASES:
        return CHOICE_ALIASES[cleaned]
    if "response" in cleaned and "1" in cleaned:
        return "response_1"
    if "response" in cleaned and "2" in cleaned:
        return "response_2"
    if "tie" in cleaned or "equal" in cleaned or "same" in cleaned:
        return "tie"
    return None


def _scan_final_line(text: str) -> Optional[str]:
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        match = FINAL_LINE_RE.search(stripped)
        if match:
            candidate = match.group(1)
            choice = _normalise_choice(candidate)
            if choice:
                return choice
        choice = _normalise_choice(stripped)
        if choice:
            return choice
    return None


def parse_preference(output: str) -> Optional[str]:
    """Extract the model's final preference label."""

    label_match = LABEL_TAG_RE.search(output)
    if label_match:
        digit = label_match.group(1)
        if digit == "1":
            return "response_1"
        if digit == "2":
            return "response_2"
        if digit == "0":
            return "tie"

    xml_match = FINAL_PREFERENCE_XML_RE.search(output)
    if xml_match:
        return xml_match.group(1).lower()

    choice = _scan_final_line(output)
    if choice:
        return choice

    tail_match = re.findall(r"response\s*[12]", output, flags=re.IGNORECASE)
    if tail_match:
        return _normalise_choice(tail_match[-1])

    if "tie" in output.lower():
        return "tie"
    return None


def compute_binary_reward(data_source, solution_str, ground_truth, extra_info):
    """Return 1.0 if the predicted label matches the annotated sign, else 0.0."""

    predicted_raw = parse_preference(solution_str)
    predicted = predicted_raw if predicted_raw is not None else "unknown"
    canonical_gt = _normalise_choice(ground_truth)

    matched = predicted_raw is not None and predicted_raw == canonical_gt
    reward = 1.0 if matched else 0.0

    result = {
        "score": reward,
        "predicted_label": predicted,
        "ground_truth": canonical_gt,
        "data_source": data_source,
    }

    pref_score = extra_info.get("preference_score")
    if pref_score is not None:
        result["preference_score"] = float(pref_score)

    return result

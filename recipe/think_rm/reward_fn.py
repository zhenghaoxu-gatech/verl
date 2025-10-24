"""Binary reward computation for the generative thinking reward model.

Behavior:
- We drop everything up to and including the last closing </think>, treating missing closers as no prediction.
- We extract the prediction ONLY from the LAST <label>...</label> (digit 0/1/2) found after that </think>.
- No other heuristics (aliases in free text, final lines, XML attributes) are used for parsing the model output.

Interfaces preserved:
- parse_preference(output: str) -> Optional[str]
- compute_binary_reward(data_source, solution_str, ground_truth, extra_info) -> dict
"""

from __future__ import annotations

import re
from typing import Optional


# --- Ground-truth normalization kept for compatibility ---
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

def _normalise_choice(raw: str | None) -> Optional[str]:
    """(Unchanged) Normalize a variety of ground-truth spellings to canonical labels."""
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


# --- Think tag helpers ---
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)

def _extract_solution_segment(text: str, exclude_think: bool = True) -> Optional[str]:
    """
    Return the portion STRICTLY after the last closing </think>.
    If no </think> is present, we treat the sample as missing a solution.
    """
    if not exclude_think:
        return text
    last_close = None
    for match in THINK_CLOSE_RE.finditer(text):
        last_close = match

    if last_close is None:
        return None

    segment = text[last_close.end():]
    # Guard against any stray closing tags in the tail.
    segment = THINK_CLOSE_RE.sub("", segment)
    return segment


# --- Parsing rule: last <label>0|1|2</label> outside <think> only ---
LABEL_TAG_RE = re.compile(r"<label>\s*([012])\s*</label>", re.IGNORECASE)

def parse_preference(output: str, from_thinking_model: bool = True) -> Optional[str]:
    """
    Extract the model's final preference label by:
      1) selecting only the text after the last </think>,
      2) taking the LAST <label>...</label> digit (0/1/2) in that segment.

    Returns one of: 'response_1' | 'response_2' | 'tie' | None
    """
    visible = _extract_solution_segment(output, exclude_think=from_thinking_model)
    if visible is None:
        return None

    last_digit = None
    for m in LABEL_TAG_RE.finditer(visible):
        last_digit = m.group(1)

    if last_digit is None:
        return None

    if last_digit == "1":
        return "response_1"
    if last_digit == "2":
        return "response_2"
    return "tie"  # digit == "0"


def compute_binary_reward(data_source, solution_str, ground_truth, extra_info):
    """Return 1.0 if the predicted label matches the annotated sign, else 0.0."""
    predicted_raw = parse_preference(solution_str)
    predicted = predicted_raw if predicted_raw is not None else "unknown"
    canonical_gt = _normalise_choice(ground_truth)

    matched = predicted_raw is not None and canonical_gt is not None and predicted_raw == canonical_gt
    if matched:
        reward = 1.0
    elif predicted == "tie":
        reward = 1.0 if canonical_gt == "tie" else 0.5
    else:
        reward = 0.0

    result = {
        "score": reward,
        "predicted_label": predicted,
        "ground_truth": canonical_gt,
        "data_source": data_source,
    }

    # Echo optional fields exactly like the original for compatibility.
    pref_score = extra_info.get("preference_score")
    if pref_score is not None:
        result["preference_score"] = float(pref_score)

    num_annotators = extra_info.get("num_annotators")
    if num_annotators is not None:
        result["num_annotators"] = float(num_annotators)

    for label in ("response_1", "response_2", "tie"):
        prob_key = f"prob_{label}"
        prob_val = extra_info.get(prob_key)
        if prob_val is not None:
            result[prob_key] = float(prob_val)

    if predicted_raw in {"response_1", "response_2", "tie"}:
        prob_key = f"prob_{predicted_raw}"
        prob_val = extra_info.get(prob_key)
        if prob_val is not None:
            result["predicted_label_prob"] = float(prob_val)

    if "predicted_label_prob" not in result:
        result["predicted_label_prob"] = float("nan")

    return result

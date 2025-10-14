"""Binary reward computation for the generative thinking reward model.

Behavior:
- We ignore anything inside <think>...</think> (case-insensitive), robust to missing closers and nested tags.
- We extract the prediction ONLY from the LAST <label>...</label> (digit 0/1/2) found OUTSIDE <think>.
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


# --- New: robust think-stripper (handles missing closers and nesting) ---
THINK_OPEN_RE  = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)

def _strip_think(text: str) -> str:
    """
    Remove <think>...</think> blocks robustly:
      - Removes all balanced pairs, supporting nested <think> blocks.
      - If an opening <think> has no matching </think>, drops everything after it.
      - Any stray </think> with no opener is stripped.
    """
    out = []
    i = 0
    n = len(text)

    while i < n:
        m_open = THINK_OPEN_RE.search(text, i)
        if not m_open:
            out.append(text[i:])
            break

        # keep visible text up to the opener
        out.append(text[i:m_open.start()])

        # find matching close, supporting nesting
        depth = 1
        pos = m_open.end()
        while depth > 0:
            m_next_open = THINK_OPEN_RE.search(text, pos)
            m_next_close = THINK_CLOSE_RE.search(text, pos)

            if not m_next_close:
                # unmatched opener: drop everything after the opener
                visible = "".join(out)
                # scrub any stray closers that might remain in the visible text
                return THINK_CLOSE_RE.sub("", visible)

            if m_next_open and m_next_open.start() < m_next_close.start():
                depth += 1
                pos = m_next_open.end()
            else:
                depth -= 1
                pos = m_next_close.end()

        # skip the entire balanced block
        i = pos

    visible = "".join(out)
    # remove any stray closing tags left in the visible part
    visible = THINK_CLOSE_RE.sub("", visible)
    return visible


# --- Parsing rule: last <label>0|1|2</label> outside <think> only ---
LABEL_TAG_RE = re.compile(r"<label>\s*([012])\s*</label>", re.IGNORECASE)

def parse_preference(output: str) -> Optional[str]:
    """
    Extract the model's final preference label by:
      1) stripping <think>...</think> (robust),
      2) taking the LAST <label>...</label> digit (0/1/2) in the remaining text.

    Returns one of: 'response_1' | 'response_2' | 'tie' | None
    """
    visible = _strip_think(output)
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
    reward = 1.0 if matched else 0.0

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

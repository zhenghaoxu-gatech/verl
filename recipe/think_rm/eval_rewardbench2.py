#!/usr/bin/env python3
"""Evaluate Think RM checkpoints on RewardBench 2 with actor/critic correction."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import os
import time
import multiprocessing as mp
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer
from verl.utils.model import build_value_head

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover - optional dependency
    safe_load_file = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification
except ImportError:  # transformers < 4.38
    AutoModelForSequenceClassification = None  # type: ignore[assignment]

from recipe.think_rm.prepare_helpsteer3 import PROMPT_INSTRUCTION
from recipe.think_rm.reward_fn import parse_preference

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PairSample:
    """Container tracking one prompt/response pair for evaluation."""

    prompt_id: str
    subset: str
    pair_index: int
    messages: list[dict[str, str]]
    response_1: str
    response_2: str
    ground_truth: str = "response_1"
    prompt_text: str = ""
    base_pair_id: str = ""
    orientation: str = "forward"
    original_chosen: str = ""
    original_rejected: str = ""
    prompt_raw_id: str = ""
    chosen_index: int = 0
    rejected_index: int = 0
    num_correct: int = 1
    total_completions: int = 1
    actor_score: float | None = None
    corrected_score: float | None = None

    actor_output: str | None = None
    predicted_label: str | None = None
    critic_prob: float | None = None
    corrected_label: str | None = None

    def pair_id(self) -> str:
        return f"{self.base_pair_id}-{self.orientation}"


@dataclass
class ValueHeadSpec:
    """Lightweight container describing the critic value head architecture."""

    prefix: str = "score"
    hidden_sizes: tuple[int, ...] = ()
    activation: str = "silu"
    dropout: float = 0.0
    force_dropout_layers: bool = False

    @property
    def is_custom(self) -> bool:
        return bool(self.hidden_sizes)


class ValueHeadLoadError(RuntimeError):
    """Raised when the critic value head weights cannot be reconstructed."""


VALUE_HEAD_ATTR_CANDIDATES: tuple[str, ...] = (
    "score",
    "value_head",
    "v_head",
    "classifier",
    "pretrained_model.score",
    "pretrained_model.value_head",
    "pretrained_model.classifier",
    "pretrained_model.v_head",
    "model.score",
    "model.value_head",
    "model.classifier",
)

WEIGHT_INDEX_FILENAMES = ("pytorch_model.bin.index.json", "model.safetensors.index.json")
WEIGHT_FILENAMES = ("pytorch_model.bin", "model.safetensors")
VALUE_HEAD_CONFIG_FILENAMES = (
    "value_head_config.json",
    "critic_value_head_config.json",
    "value_head.json",
)
VALUE_HEAD_ATTR_PATHS: tuple[tuple[str, ...], ...] = (
    ("v_head",),
    ("value_head",),
    ("classifier",),
    ("score",),
    ("pretrained_model", "v_head"),
    ("pretrained_model", "value_head"),
    ("pretrained_model", "classifier"),
    ("pretrained_model", "score"),
    ("model", "value_head"),
    ("model", "classifier"),
    ("model", "score"),
)


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file at {path}: {exc}") from exc


def _resolve_weight_files(model_dir: Path) -> tuple[list[Path], str]:
    """Return list of weight shard files and their format ('bin' or 'safetensors')."""

    for index_name in WEIGHT_INDEX_FILENAMES:
        index_path = model_dir / index_name
        if index_path.exists():
            index_data = _read_json(index_path)
            if not index_data or "weight_map" not in index_data:
                raise ValueError(f"Invalid weight index file: {index_path}")
            shard_names = sorted(set(index_data["weight_map"].values()))
            fmt = "safetensors" if "safetensors" in index_name else "bin"
            return [model_dir / shard for shard in shard_names], fmt

    for weight_name in WEIGHT_FILENAMES:
        weight_path = model_dir / weight_name
        if weight_path.exists():
            fmt = "safetensors" if weight_name.endswith(".safetensors") else "bin"
            return [weight_path], fmt

    # Fall back to auto-detect shard-style filenames such as model-00001-of-00002.safetensors.
    safetensor_shards = sorted(model_dir.glob("*.safetensors"))
    bin_shards = sorted(model_dir.glob("*.bin"))
    if safetensor_shards:
        return safetensor_shards, "safetensors"
    if bin_shards:
        return bin_shards, "bin"

    raise FileNotFoundError(f"Unable to locate model weights under {model_dir}")


def _load_state_dict_from_dir(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load a (possibly sharded) Hugging Face state dict from disk."""

    files, fmt = _resolve_weight_files(model_dir)
    state_dict: dict[str, torch.Tensor] = {}

    if fmt == "safetensors":
        if safe_load_file is None:
            raise ImportError(
                "The safetensors package is required to load critic weights stored as .safetensors."
            )
        for shard_path in files:
            shard_state = safe_load_file(str(shard_path))
            state_dict.update({key: tensor.clone() for key, tensor in shard_state.items()})
    else:
        for shard_path in files:
            shard_state = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard_state, dict):
                raise ValueError(f"Unexpected contents in weight shard {shard_path}")
            for key, tensor in shard_state.items():
                if isinstance(tensor, torch.Tensor):
                    state_dict[key] = tensor
                else:
                    raise TypeError(f"Shard {shard_path} contains non-tensor entry for key '{key}'")

    return state_dict


def _load_value_head_config_from_dirs(candidate_dirs: Sequence[Path]) -> Optional[dict]:
    """Search for a value head config JSON file in the provided directories."""

    for directory in candidate_dirs:
        if directory is None:
            continue
        for filename in VALUE_HEAD_CONFIG_FILENAMES:
            config_path = directory / filename
            config = _read_json(config_path)
            if config is not None:
                return config
    return None


def _detect_value_head_layers(
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, bool, list[tuple[int, torch.Size]]]:
    """Identify the value head prefix, whether it is custom, and per-linear layer shapes."""

    for candidate in VALUE_HEAD_ATTR_CANDIDATES:
        prefix = f"{candidate}."
        matching_keys = [key for key in state_dict if key.startswith(prefix)]
        if not matching_keys and f"{candidate}.weight" not in state_dict:
            continue

        linear_layers: list[tuple[int, torch.Size]] = []
        for key in matching_keys:
            if not key.endswith(".weight"):
                continue
            suffix = key[len(prefix) :]
            module_id = suffix.split(".", 1)[0]
            if module_id.isdigit():
                idx = int(module_id)
                linear_layers.append((idx, state_dict[key].shape))

        if linear_layers:
            linear_layers.sort()
            if len(linear_layers) >= 2:
                return candidate, True, linear_layers

        if f"{candidate}.weight" in state_dict:
            weight_shape = state_dict[f"{candidate}.weight"].shape
            return candidate, False, [(0, weight_shape)]

    return "score", False, []


def _coerce_hidden_sizes(value: object | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(
                "Failed to parse value head hidden_sizes string. Expected JSON literals such as [2560, 1024]."
            )
        return _coerce_hidden_sizes(parsed)
    return ()


def _infer_value_head_spec(
    state_dict: dict[str, torch.Tensor],
    metadata: Optional[dict] = None,
) -> ValueHeadSpec:
    """Infer the critic value head layout from the saved weights and optional metadata."""

    activation = "silu"
    dropout = 0.0
    if metadata is not None:
        activation = str(metadata.get("activation", activation)).lower()
        dropout = float(metadata.get("dropout", dropout) or 0.0)

    prefix, is_custom, linear_layers = _detect_value_head_layers(state_dict)

    hidden_sizes = ()
    if metadata and metadata.get("hidden_sizes") is not None:
        hidden_sizes = _coerce_hidden_sizes(metadata["hidden_sizes"])

    if is_custom:
        if not hidden_sizes:
            if len(linear_layers) < 2:
                raise ValueHeadLoadError("Detected custom value head but unable to infer hidden sizes.")
            hidden_sizes = tuple(int(shape[0]) for idx, shape in linear_layers[:-1])

        num_hidden = len(hidden_sizes)
        final_idx = linear_layers[-1][0] if linear_layers else 0
        expected_no_dropout = num_hidden * 2
        expected_with_dropout = num_hidden * 3

        force_dropout_layers = False
        if metadata and "dropout" in metadata:
            force_dropout_layers = dropout > 0.0
        elif num_hidden > 0 and final_idx == expected_with_dropout:
            force_dropout_layers = True
        elif num_hidden > 0 and final_idx not in (expected_no_dropout, expected_with_dropout):
            # Fallback: if indices deviate from the expected pattern, include dropout layers to preserve numbering.
            force_dropout_layers = final_idx > expected_no_dropout

        if force_dropout_layers and dropout == 0.0:
            dropout = 1e-6

        return ValueHeadSpec(
            prefix=prefix,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            force_dropout_layers=force_dropout_layers,
        )

    return ValueHeadSpec(prefix=prefix, activation=activation, dropout=dropout)


def _locate_module(root: nn.Module, path: tuple[str, ...]) -> tuple[Optional[nn.Module], Optional[nn.Module]]:
    parent: Optional[nn.Module] = None
    current: Optional[nn.Module] = root
    for name in path:
        if current is None or not hasattr(current, name):
            return None, None
        parent = current
        current = getattr(current, name)
        if not isinstance(current, nn.Module):
            return None, None
    return parent, current


def _infer_input_dim_from_module(module: nn.Module) -> Optional[int]:
    if hasattr(module, "in_features"):
        return int(getattr(module, "in_features"))
    if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
        return int(module.weight.shape[1])
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "in_features"):
            return int(getattr(child, "in_features"))
        if hasattr(child, "weight") and isinstance(getattr(child, "weight"), torch.Tensor):
            return int(child.weight.shape[1])
    return None


def _apply_value_head_spec(model: nn.Module, spec: ValueHeadSpec) -> None:
    if not spec.is_custom:
        return

    preferred_path = tuple(part for part in spec.prefix.split(".") if part)
    candidate_paths = []
    if preferred_path:
        candidate_paths.append(preferred_path)
    candidate_paths.extend(path for path in VALUE_HEAD_ATTR_PATHS if path not in candidate_paths)

    target_parent: Optional[nn.Module] = None
    target_module: Optional[nn.Module] = None
    target_path: Optional[tuple[str, ...]] = None
    for path in candidate_paths:
        parent, module = _locate_module(model, path)
        if parent is not None and module is not None:
            target_parent, target_module, target_path = parent, module, path
            break

    if target_parent is None or target_module is None or target_path is None:
        available = [name for name in dir(model) if not name.startswith("__")]
        raise ValueHeadLoadError(
            "Unable to locate critic value head module when applying Verl configuration. "
            f"Searched paths {candidate_paths} on module {model.__class__.__name__}; "
            f"available attributes (truncated): {available[:50]}"
        )

    input_dim = _infer_input_dim_from_module(target_module)
    if input_dim is None and hasattr(model, "config"):
        config = getattr(model, "config")
        input_dim = getattr(config, "hidden_size", None)
        if input_dim is None and hasattr(config, "text_config"):
            input_dim = getattr(config.text_config, "hidden_size", None)

    if input_dim is None:
        raise ValueHeadLoadError("Unable to infer input dimension for critic value head.")

    dropout = spec.dropout
    if not spec.force_dropout_layers and dropout < 0:
        dropout = 0.0
    elif spec.force_dropout_layers and dropout <= 0:
        dropout = 1e-6

    new_head = build_value_head(
        input_dim=input_dim,
        hidden_sizes=spec.hidden_sizes,
        activation=spec.activation,
        dropout=dropout,
    )
    setattr(target_parent, target_path[-1], new_head)


def _normalize_chat_tensor(
    payload: torch.Tensor | dict | Sequence,
    key: str,
    device_id: int,
) -> torch.Tensor:
    """Extract a tensor by key from tokenizer outputs, tolerating various container types."""

    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = payload[key]
    elif hasattr(payload, key):
        tensor = getattr(payload, key)
    elif isinstance(payload, (tuple, list)):
        lookup = 0 if key == "input_ids" else 1
        if len(payload) <= lookup:
            raise RuntimeError(f"Tokenizer output missing index {lookup} for key '{key}' (device {device_id}).")
        tensor = payload[lookup]
    else:
        raise TypeError(
            f"Unsupported tokenizer output type {type(payload)} for key '{key}' (device {device_id})."
        )

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.long)
    else:
        tensor = tensor.to(dtype=torch.long)
    return tensor


def _normalize_attention_tensor(
    payload: torch.Tensor | dict | Sequence,
    input_ids: torch.Tensor,
    pad_token: int,
    device_id: int,
) -> torch.Tensor:
    """Return an attention mask matching input_ids, synthesizing one if necessary."""

    mask: torch.Tensor | None = None
    if isinstance(payload, torch.Tensor):
        # Only input_ids were returned; synthesize a mask.
        mask = None
    elif isinstance(payload, dict) and "attention_mask" in payload:
        mask = payload["attention_mask"]
    elif hasattr(payload, "attention_mask"):
        mask = getattr(payload, "attention_mask")
    elif isinstance(payload, (tuple, list)):
        if len(payload) >= 2:
            mask = payload[1]

    if mask is None:
        if pad_token is None:
            mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            mask = (input_ids != pad_token).long()
    elif not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.long)
    else:
        mask = mask.to(dtype=torch.long)

    if mask.shape != input_ids.shape:
        raise RuntimeError(
            f"Attention mask shape {tuple(mask.shape)} does not match input_ids {tuple(input_ids.shape)} "
            f"(device {device_id})."
        )
    return mask


def _select_last_token_values(value_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return logits corresponding to the final non-padding token per sequence."""

    if value_logits.ndim == 3:
        if value_logits.size(-1) == 1:
            value_logits = value_logits.squeeze(-1)
        else:
            # Fall back to using the final channel when multiple label logits are present.
            value_logits = value_logits[..., -1]

    if value_logits.ndim == 1:
        if value_logits.size(0) != attention_mask.size(0):
            raise RuntimeError(
                f"Logits length {value_logits.size(0)} does not match batch size {attention_mask.size(0)}."
            )
        return value_logits

    if value_logits.ndim != 2:
        raise RuntimeError(
            f"Unsupported critic logits shape {tuple(value_logits.shape)} when gathering final token values."
        )

    last_indices = attention_mask.sum(dim=1) - 1
    last_indices = last_indices.clamp_min(0).to(device=value_logits.device, dtype=torch.long)
    max_index = value_logits.size(1) - 1
    if max_index < 0:
        raise RuntimeError("Value logits have zero sequence length.")
    if (last_indices > max_index).any():
        offending = last_indices[last_indices > max_index]
        print(
            f"[critic] Clamping gather indices; max_index={max_index}, offending_count={offending.numel()} "
            f"max_offending={int(offending.max())}",
            flush=True,
        )
        last_indices = last_indices.clamp(max=max_index)

    gather_indices = last_indices.unsqueeze(1)
    last_values = value_logits.gather(1, gather_indices).squeeze(1)

    # Illustrative debug: log the first sample's sequence and selected position once.
    if not getattr(_select_last_token_values, "_logged_debug", False) and value_logits.size(0) > 0:
        sample_idx = 0
        seq_len = value_logits.size(1)
        chosen_pos = int(last_indices[sample_idx])
        print(
            "[critic-debug] sample=0 "
            f"seq_len={seq_len} chosen_pos={chosen_pos} "
            f"attention_mask_sum={int(attention_mask[sample_idx].sum())}",
            flush=True,
        )
        setattr(_select_last_token_values, "_logged_debug", True)

    return last_values





def find_latest_checkpoint(root: Path) -> Path:
    """Return the latest checkpoint directory under ``root``."""

    marker = root / "latest_checkpointed_iteration.txt"
    if marker.exists():
        step_text = marker.read_text().strip()
        if not step_text:
            raise FileNotFoundError(f"{marker} is empty")
        step_dir = step_text if step_text.startswith("global_step_") else f"global_step_{step_text}"
        candidate = root / step_dir
        if candidate.exists():
            return candidate
    # Fallback: pick the highest global_step directory.
    candidates = sorted(root.glob("global_step_*"), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under {root}")
    return candidates[-1]


def ensure_hf_export(source_dir: Path, target_dir: Path) -> Path:
    """Convert an FSDP checkpoint to a Hugging Face format if needed."""

    config_file = target_dir / "config.json"
    if config_file.exists():
        return target_dir

    hf_config_dir = source_dir / "huggingface"
    if not hf_config_dir.exists():
        raise FileNotFoundError(f"Hugging Face config directory missing: {hf_config_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    merger_script = REPO_ROOT / "scripts" / "legacy_model_merger.py"
    cmd = [
        sys.executable,
        str(merger_script),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(source_dir),
        "--hf_model_path",
        str(hf_config_dir),
        "--target_dir",
        str(target_dir),
    ]
    subprocess.run(cmd, check=True)
    if not config_file.exists():
        raise FileNotFoundError(f"Conversion did not create {config_file}")
    return target_dir


def build_prompt_messages(prompt_text: str, response1: str, response2: str) -> list[dict[str, str]]:
    """Reproduce the HelpSteer-style prompt used during training."""

    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        prompt_text = "(No prior context provided.)"

    comparison_request = (
        f"{PROMPT_INSTRUCTION}\n\n"
        "[The Start of Context]\n"
        f"{prompt_text}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        f"{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        f"{response2}\n"
        "[The End of Assistant B's Response]"
    )

    return [{"role": "user", "content": comparison_request}]


def load_rewardbench_pairs(
    dataset_name: str,
    split: str,
    max_examples: int | None = None,
) -> list[PairSample]:
    dataset = load_dataset(dataset_name, split=split)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    pairs: list[PairSample] = []
    for row_idx, row in enumerate(dataset):
        prompt_id = str(row.get("id", row_idx))
        subset = row.get("subset", "unknown") or "unknown"
        prompt_text = row.get("prompt", "")
        chosen: Iterable[str] = row.get("chosen") or []
        rejected: Iterable[str] = row.get("rejected") or []

        # Skip entries without both positive and negative completions.
        chosen = [c for c in chosen if isinstance(c, str) and c.strip()]
        rejected = [r for r in rejected if isinstance(r, str) and r.strip()]
        if not chosen or not rejected:
            continue

        num_correct = int(row.get("num_correct", len(chosen)))
        total_completions = int(row.get("total_completions", len(chosen) + len(rejected)))
        prompt_raw_id = prompt_id

        for chosen_idx, pos in enumerate(chosen):
            for rejected_idx, neg in enumerate(rejected):
                base_pair_id = f"{prompt_id}-{chosen_idx}-{rejected_idx}"
                forward_messages = build_prompt_messages(prompt_text, pos, neg)
                pairs.append(
                    PairSample(
                        prompt_id=prompt_id,
                        subset=subset,
                        pair_index=len(pairs),
                        messages=forward_messages,
                        response_1=pos,
                        response_2=neg,
                        ground_truth="response_1",
                        prompt_text=prompt_text,
                        base_pair_id=base_pair_id,
                        orientation="forward",
                        original_chosen=pos,
                        original_rejected=neg,
                        prompt_raw_id=prompt_raw_id,
                        chosen_index=chosen_idx,
                        rejected_index=rejected_idx,
                        num_correct=num_correct,
                        total_completions=total_completions,
                    )
                )

                backward_messages = build_prompt_messages(prompt_text, neg, pos)
                pairs.append(
                    PairSample(
                        prompt_id=prompt_id,
                        subset=subset,
                        pair_index=len(pairs),
                        messages=backward_messages,
                        response_1=neg,
                        response_2=pos,
                        ground_truth="response_2",
                        prompt_text=prompt_text,
                        base_pair_id=base_pair_id,
                        orientation="backward",
                        original_chosen=pos,
                        original_rejected=neg,
                        prompt_raw_id=prompt_raw_id,
                        chosen_index=chosen_idx,
                        rejected_index=rejected_idx,
                        num_correct=num_correct,
                        total_completions=total_completions,
                    )
                )
    return pairs


def chunked(iterable: list[PairSample], batch_size: int) -> Iterable[list[PairSample]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def _build_actor_prompt(actor_tokenizer: AutoTokenizer, sample: PairSample) -> str:
    return actor_tokenizer.apply_chat_template(
        sample.messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def run_actor_generation_hf(
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: AutoTokenizer,
    samples: list[PairSample],
    max_new_tokens: int,
    batch_size: int,
) -> None:
    actor_model.eval()
    device = next(actor_model.parameters()).device

    total_batches = math.ceil(len(samples) / batch_size)
    for batch in tqdm(chunked(samples, batch_size), total=total_batches, desc="Actor generation (HF)", unit="batch"):
        prompts = [_build_actor_prompt(actor_tokenizer, sample) for sample in batch]
        inputs = actor_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            pad_token = actor_tokenizer.pad_token_id or actor_tokenizer.eos_token_id
            generated = actor_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token,
                eos_token_id=actor_tokenizer.eos_token_id,
            )

        gen_only = generated[:, inputs["input_ids"].shape[1] :]
        texts = actor_tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for sample, text in zip(batch, texts):
            sample.actor_output = text.strip()
            sample.predicted_label = parse_preference(sample.actor_output or "")


def run_actor_generation_vllm_local(
    actor_llm,
    sampling_params,
    actor_tokenizer: AutoTokenizer,
    samples: list[PairSample],
    batch_size: int,
) -> None:
    total_batches = math.ceil(len(samples) / batch_size)
    for batch in tqdm(
        chunked(samples, batch_size),
        total=total_batches,
        desc="Actor generation (vLLM local)",
        unit="batch",
    ):
        prompts = [_build_actor_prompt(actor_tokenizer, sample) for sample in batch]
        outputs = actor_llm.generate(prompts, sampling_params=sampling_params)

        for sample, output in zip(batch, outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            sample.actor_output = generated_text.strip()
            sample.predicted_label = parse_preference(sample.actor_output or "")


def start_vllm_server(
    model_path: Path,
    port: int,
    data_parallel_size: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--data-parallel-size",
        str(data_parallel_size),
        "--enforce-eager",
        "--enable-prefix-caching",
    ]
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for_server_ready(port: int, timeout: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)

    raise RuntimeError(f"Timed out waiting for vLLM server on port {port}")


def shutdown_process(proc: subprocess.Popen, timeout: int) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_actor_generation_vllm_server(
    actor_tokenizer: AutoTokenizer,
    samples: list[PairSample],
    model_name: str,
    port: int,
    max_new_tokens: int,
    concurrency: int,
    request_timeout: int,
) -> None:
    prompts = [_build_actor_prompt(actor_tokenizer, sample) for sample in samples]
    url = f"http://127.0.0.1:{port}/v1/completions"
    concurrency = max(1, min(concurrency, len(samples)))

    def submit_request(idx: int) -> tuple[int, str]:
        prompt = prompts[idx]
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=request_timeout)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                text = choice.get("text")
                if text is None and "message" in choice:
                    message = choice["message"]
                    if isinstance(message, dict):
                        text = message.get("content", "")
                return idx, (text or "")
            except Exception as exc:
                last_exc = exc
                time.sleep(2)
        raise RuntimeError(f"vLLM request failed for sample {idx}: {last_exc}") from last_exc

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(submit_request, idx) for idx in range(len(samples))]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Actor generation (vLLM server)",
            unit="pair",
        ):
            idx, text = future.result()
            sample = samples[idx]
            sample.actor_output = text.strip()
            sample.predicted_label = parse_preference(sample.actor_output or "")


def load_token_classifier(
    model_dir: Path,
    dtype: torch.dtype,
    source_ckpt_dir: Optional[Path] = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the critic model ensuring Verl's custom value head (if any) is restored."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "right"

    metadata_dirs: list[Path] = [model_dir, model_dir / "huggingface"]
    if source_ckpt_dir:
        metadata_dirs.append(source_ckpt_dir)
        metadata_dirs.append(source_ckpt_dir / "huggingface")
    metadata = _load_value_head_config_from_dirs(metadata_dirs)

    state_dict = _load_state_dict_from_dir(model_dir)
    spec = _infer_value_head_spec(state_dict, metadata)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model: Optional[torch.nn.Module] = None
    try:
        model = AutoModelForTokenClassification.from_config(config)
    except Exception:
        model = None

    if model is None:
        if AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "Unable to instantiate critic model for token classification and no sequence classification fallback."
            )
        model = AutoModelForSequenceClassification.from_config(config)

    try:
        _apply_value_head_spec(model, spec)
    except ValueHeadLoadError as exc:
        raise RuntimeError(f"Failed to construct critic value head: {exc}") from exc

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict

    if missing:
        raise RuntimeError(f"Missing parameters when loading critic weights: {sorted(missing)[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected parameters when loading critic weights: {sorted(unexpected)[:10]}")

    model.to(dtype=dtype)
    model.eval()
    return model, tokenizer


def _build_critic_inputs(sample: PairSample) -> list[dict[str, str]]:
    convo = list(sample.messages)
    output = (sample.actor_output or "").strip()
    convo.append({"role": "assistant", "content": output})
    return convo


def run_critic_scoring_single(
    critic_model: torch.nn.Module,
    critic_tokenizer: AutoTokenizer,
    samples: list[PairSample],
    batch_size: int,
    outputs_logits: bool,
) -> None:
    critic_model.eval()
    device = next(critic_model.parameters()).device

    total_batches = math.ceil(len(samples) / batch_size)
    for batch in tqdm(chunked(samples, batch_size), total=total_batches, desc="Critic scoring", unit="batch"):
        input_ids_list: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        for sample in batch:
            conv = critic_tokenizer.apply_chat_template(
                _build_critic_inputs(sample), add_generation_prompt=False, tokenize=True, return_tensors="pt"
            )
            conv_input_ids = _normalize_chat_tensor(conv, "input_ids", device_id=-1)
            conv_attention = _normalize_attention_tensor(
                conv, conv_input_ids, critic_tokenizer.pad_token_id or critic_tokenizer.eos_token_id or 0, device_id=-1
            )
            if conv_input_ids.ndim == 1:
                conv_input_ids = conv_input_ids.unsqueeze(0)
            if conv_attention.ndim == 1:
                conv_attention = conv_attention.unsqueeze(0)
            if conv_input_ids.ndim != 2 or conv_attention.ndim != 2:
                raise RuntimeError(
                    f"Unexpected critic input shapes in single process: "
                    f"input_ids {tuple(conv_input_ids.shape)}, attention_mask {tuple(conv_attention.shape)}"
                )
            input_ids_list.append(conv_input_ids.squeeze(0))
            attention_masks.append(conv_attention.squeeze(0))

        pad_token = critic_tokenizer.pad_token_id or critic_tokenizer.eos_token_id or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }

        with torch.inference_mode():
            outputs = critic_model(**inputs)

        if hasattr(outputs, "logits"):
            value_logits = outputs.logits
        elif hasattr(outputs, "value"):
            value_logits = outputs.value
        elif isinstance(outputs, tuple) and len(outputs) >= 3:
            value_logits = outputs[2]
        else:
            raise RuntimeError("Unable to locate critic value tensor in outputs")

        last_values = _select_last_token_values(value_logits, inputs["attention_mask"])
        if outputs_logits:
            probs = torch.sigmoid(last_values).detach().cpu().tolist()
        else:
            probs = last_values.clamp(0.0, 1.0).detach().cpu().tolist()

        for sample, prob in zip(batch, probs):
            sample.critic_prob = float(prob)


def _critic_worker_process(
    device_id: int,
    indices: list[int],
    samples: list[PairSample],
    model_path: str,
    source_ckpt_path: str | None,
    outputs_logits: bool,
    dtype_str: str,
    batch_size: int,
    queue: mp.Queue,
) -> None:
    try:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        outputs_logits = bool(outputs_logits)

        source_ckpt = Path(source_ckpt_path) if source_ckpt_path else None
        model, tokenizer = load_token_classifier(Path(model_path), dtype, source_ckpt)
        model.to(device)
        model.eval()

        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        debug_logged = False

        for batch_indices in chunked(indices, batch_size):
            input_ids_list: list[torch.Tensor] = []
            attention_masks: list[torch.Tensor] = []
            for idx in batch_indices:
                conv = tokenizer.apply_chat_template(
                    _build_critic_inputs(samples[idx]),
                    add_generation_prompt=False,
                    tokenize=True,
                    return_tensors="pt",
                )
                if not debug_logged:
                    repr_hint = None
                    if hasattr(conv, "keys"):
                        repr_hint = list(conv.keys())
                    elif isinstance(conv, (tuple, list)):
                        repr_hint = [type(item) for item in conv]
                    print(
                        f"[critic-worker {device_id}] chat_template type={type(conv)} hint={repr_hint}",
                        flush=True,
                    )
                conv_input_ids = _normalize_chat_tensor(conv, "input_ids", device_id)
                conv_attention = _normalize_attention_tensor(conv, conv_input_ids, pad_token, device_id)
                if conv_input_ids.ndim == 1:
                    conv_input_ids = conv_input_ids.unsqueeze(0)
                if conv_attention.ndim == 1:
                    conv_attention = conv_attention.unsqueeze(0)
                if conv_input_ids.ndim != 2 or conv_attention.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected critic input shapes: input_ids {tuple(conv_input_ids.shape)}, "
                        f"attention_mask {tuple(conv_attention.shape)}"
                    )
                if not debug_logged:
                    print(
                        f"[critic-worker {device_id}] normalized shapes: "
                        f"input_ids {tuple(conv_input_ids.shape)}, attention_mask {tuple(conv_attention.shape)}",
                        flush=True,
                    )
                    try:
                        decoded = tokenizer.decode(conv_input_ids.squeeze(0).tolist())
                    except Exception as decode_exc:  # pragma: no cover - best effort logging
                        decoded = f"<decode failed: {decode_exc}>"
                    print(
                        f"[critic-worker {device_id}] sample actor_output="
                        f"{samples[idx].actor_output!r} selected_pair={samples[idx].pair_id()}",
                        flush=True,
                    )
                    print(
                        f"[critic-worker {device_id}] chat input preview={decoded[:500]}",
                        flush=True,
                    )
                    debug_logged = True
                input_ids_list.append(conv_input_ids.squeeze(0))
                attention_masks.append(conv_attention.squeeze(0))

            if not input_ids_list:
                continue

            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token)
            attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(dtype=torch.long)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if hasattr(outputs, "logits"):
                value_logits = outputs.logits
            elif hasattr(outputs, "value"):
                value_logits = outputs.value
            elif isinstance(outputs, tuple) and len(outputs) >= 3:
                value_logits = outputs[2]
            else:
                raise RuntimeError("Unable to locate critic value tensor in outputs")

            last_values = _select_last_token_values(value_logits, attention_mask)
            if outputs_logits:
                probs = torch.sigmoid(last_values).detach().cpu().tolist()
            else:
                probs = last_values.clamp(0.0, 1.0).detach().cpu().tolist()

            for idx, prob in zip(batch_indices, probs):
                queue.put({"type": "result", "index": idx, "prob": float(prob)})
    except Exception as exc:  # pragma: no cover - sent back to main process
        queue.put(
            {
                "type": "error",
                "device": device_id,
                "message": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        queue.put({"type": "done", "device": device_id})


def run_critic_scoring_multiproc(
    samples: list[PairSample],
    model_dir: Path,
    source_ckpt_dir: Optional[Path],
    outputs_logits: bool,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
) -> None:
    if num_workers <= 1:
        raise ValueError("num_workers must be greater than 1 for multiprocess critic scoring.")

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("CUDA devices are required for multiprocess critic scoring.")
    device_ids = list(range(min(num_workers, available_gpus)))

    indices = list(range(len(samples)))
    shards: list[list[int]] = [[] for _ in device_ids]
    for idx, sample_idx in enumerate(indices):
        shards[idx % len(device_ids)].append(sample_idx)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"

    processes: list[mp.Process] = []
    for device_id, shard in zip(device_ids, shards):
        if not shard:
            continue
        proc = ctx.Process(
            target=_critic_worker_process,
            args=(
                device_id,
                shard,
                samples,
                str(model_dir),
                str(source_ckpt_dir) if source_ckpt_dir else None,
                outputs_logits,
                dtype_str,
                batch_size,
                queue,
            ),
        )
        proc.start()
        processes.append(proc)

    active_workers = len(processes)
    progress = tqdm(total=len(samples), desc="Critic scoring", unit="pair")

    try:
        finished_workers = 0
        while finished_workers < active_workers:
            message = queue.get()
            msg_type = message.get("type")
            if msg_type == "result":
                idx = message["index"]
                prob = message["prob"]
                samples[idx].critic_prob = prob
                progress.update(1)
            elif msg_type == "error":
                tb = message.get("traceback")
                err_msg = message.get("message")
                detail = f"Critic worker on device {message.get('device')} failed: {err_msg}"
                if tb:
                    detail = f"{detail}\n{tb}"
                raise RuntimeError(detail)
            elif msg_type == "done":
                finished_workers += 1
    finally:
        progress.close()
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()


def apply_correction(samples: Iterable[PairSample], threshold: float) -> None:
    for sample in samples:
        predicted = sample.predicted_label or "unknown"
        prob = sample.critic_prob
        if prob is None or predicted not in {"response_1", "response_2"}:
            sample.corrected_label = predicted
            continue
        if prob >= threshold:
            sample.corrected_label = predicted
        else:
            sample.corrected_label = "response_1" if predicted == "response_2" else "response_2"


def compute_metrics(samples: list[PairSample]) -> dict:
    prompt_records: dict[str, dict[str, Any]] = {}
    total_actor_score = 0.0
    total_corrected_score = 0.0
    total_actor_strict = 0
    total_corrected_strict = 0

    for sample in samples:
        norm_ground_truth = sample.ground_truth or "response_1"
        actor_score = (
            float(sample.actor_score)
            if sample.actor_score is not None
            else _label_to_score(sample.predicted_label, norm_ground_truth)
        )
        corrected_score: float
        if sample.corrected_score is not None:
            corrected_score = float(sample.corrected_score)
        else:
            corrected_label = sample.corrected_label
            if corrected_label is None:
                corrected_score = actor_score
            else:
                corrected_score = _label_to_score(corrected_label, norm_ground_truth)
        actor_score = float(np.clip(actor_score, 0.0, 1.0))
        corrected_score = float(np.clip(corrected_score, 0.0, 1.0))

        total_actor_score += actor_score
        total_corrected_score += corrected_score
        if math.isclose(actor_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            total_actor_strict += 1
        if math.isclose(corrected_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            total_corrected_strict += 1

        record = prompt_records.get(sample.prompt_id)
        if record is None:
            total_candidates = sample.total_completions or (
                sample.num_correct + max(sample.rejected_index, 0) + 1
            )
            total_candidates = max(total_candidates, sample.num_correct + sample.rejected_index + 1)
            record = {
                "subset": sample.subset,
                "prompt_id": sample.prompt_id,
                "raw_id": sample.prompt_raw_id or sample.prompt_id,
                "num_correct": sample.num_correct,
                "total_completions": total_candidates,
                "pair_total": 0,
                "pair_score_actor": 0.0,
                "pair_score_corrected": 0.0,
                "pair_strict_actor": 0,
                "pair_strict_corrected": 0,
                "scores": {name: [0.0] * total_candidates for name in ("actor", "corrected", "prob")},
                "counts": {name: [0] * total_candidates for name in ("actor", "corrected", "prob")},
            }
            prompt_records[sample.prompt_id] = record

        record["pair_total"] += 1
        record["pair_score_actor"] += actor_score
        record["pair_score_corrected"] += corrected_score
        if math.isclose(actor_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            record["pair_strict_actor"] += 1
        if math.isclose(corrected_score, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            record["pair_strict_corrected"] += 1

        chosen_pos = sample.chosen_index
        rejected_pos = sample.num_correct + sample.rejected_index
        target_len = max(record["total_completions"], rejected_pos + 1)
        _ensure_score_capacity(record, target_len)

        metric_values = {
            "actor": actor_score,
            "corrected": corrected_score,
            "prob": _get_metric_value(sample, "prob"),
        }
        for metric_name, value in metric_values.items():
            if value is None or math.isnan(value):
                continue
            value = float(np.clip(value, 0.0, 1.0))
            scores = record["scores"][metric_name]
            counts = record["counts"][metric_name]
            scores[chosen_pos] += value
            counts[chosen_pos] += 1
            scores[rejected_pos] += 1.0 - value
            counts[rejected_pos] += 1

    for record in prompt_records.values():
        for metric_name in record["scores"]:
            scores = record["scores"][metric_name]
            counts = record["counts"][metric_name]
            for idx, count in enumerate(counts):
                if count > 0:
                    scores[idx] /= count
                else:
                    scores[idx] = float("nan")
        record["prompt_success"] = {
            metric: _compute_prompt_success(record["scores"][metric], record["num_correct"])
            for metric in ("actor", "corrected", "prob")
        }

    subset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prompt_records.values():
        subset_groups[record["subset"]].append(record)

    subset_summary: dict[str, dict[str, Any]] = {}
    actor_subset_scores: list[float] = []
    corrected_subset_scores: list[float] = []
    prob_subset_scores: list[float] = []

    for subset, records in sorted(subset_groups.items()):
        prompt_count = len(records)
        pair_total = sum(record["pair_total"] for record in records)
        actor_pair_score = sum(record["pair_score_actor"] for record in records)
        corrected_pair_score = sum(record["pair_score_corrected"] for record in records)
        actor_pair_strict = sum(record["pair_strict_actor"] for record in records)
        corrected_pair_strict = sum(record["pair_strict_corrected"] for record in records)

        actor_prompt_vals = [
            record["prompt_success"]["actor"]
            for record in records
            if record["prompt_success"]["actor"] is not None
        ]
        corrected_prompt_vals = [
            record["prompt_success"]["corrected"]
            for record in records
            if record["prompt_success"]["corrected"] is not None
        ]
        prob_prompt_vals = [
            record["prompt_success"]["prob"]
            for record in records
            if record["prompt_success"]["prob"] is not None
        ]

        actor_prompt_accuracy = _safe_mean(actor_prompt_vals)
        corrected_prompt_accuracy = _safe_mean(corrected_prompt_vals)
        prob_prompt_accuracy = _safe_mean(prob_prompt_vals)

        actor_pair_accuracy = actor_pair_score / pair_total if pair_total else math.nan
        corrected_pair_accuracy = corrected_pair_score / pair_total if pair_total else math.nan
        actor_pair_strict_accuracy = actor_pair_strict / pair_total if pair_total else math.nan
        corrected_pair_strict_accuracy = corrected_pair_strict / pair_total if pair_total else math.nan

        summary: dict[str, Any] = {
            "prompt_count": prompt_count,
            "pair_count": pair_total,
            "actor_pair_accuracy": actor_pair_accuracy,
            "corrected_pair_accuracy": corrected_pair_accuracy,
            "actor_pair_strict_accuracy": actor_pair_strict_accuracy,
            "corrected_pair_strict_accuracy": corrected_pair_strict_accuracy,
        }

        if subset.lower() == "ties":
            ties_entries_actor = []
            ties_entries_corrected = []
            ties_entries_prob = []
            for record in records:
                sample_type, prompt_key = _split_ties_identifier(record["raw_id"])
                entry_base = {
                    "sample_type": sample_type,
                    "prompt_key": prompt_key,
                    "num_correct": record["num_correct"],
                }
                ties_entries_actor.append(
                    {**entry_base, "scores": record["scores"]["actor"]}
                )
                ties_entries_corrected.append(
                    {**entry_base, "scores": record["scores"]["corrected"]}
                )
                ties_entries_prob.append(
                    {**entry_base, "scores": record["scores"]["prob"]}
                )

            actor_ties_score, actor_ties_details = _compute_ties_score(ties_entries_actor)
            corrected_ties_score, corrected_ties_details = _compute_ties_score(
                ties_entries_corrected
            )
            prob_ties_score, prob_ties_details = _compute_ties_score(ties_entries_prob)

            summary.update(
                {
                    "actor_ties_score": actor_ties_score,
                    "corrected_ties_score": corrected_ties_score,
                    "prob_ties_score": prob_ties_score,
                    "actor_ties_details": actor_ties_details,
                    "corrected_ties_details": corrected_ties_details,
                    "prob_ties_details": prob_ties_details,
                }
            )

            actor_value = actor_ties_score
            corrected_value = corrected_ties_score
            prob_value = prob_ties_score
        else:
            summary.update(
                {
                    "actor_prompt_accuracy": actor_prompt_accuracy,
                    "corrected_prompt_accuracy": corrected_prompt_accuracy,
                }
            )
            if not math.isnan(prob_prompt_accuracy):
                summary["prob_prompt_accuracy"] = prob_prompt_accuracy

            actor_value = actor_prompt_accuracy
            corrected_value = corrected_prompt_accuracy
            prob_value = prob_prompt_accuracy

        subset_summary[subset] = summary

        if actor_value is not None and not math.isnan(actor_value):
            actor_subset_scores.append(actor_value)
        if corrected_value is not None and not math.isnan(corrected_value):
            corrected_subset_scores.append(corrected_value)
        if prob_value is not None and not math.isnan(prob_value):
            prob_subset_scores.append(prob_value)

    actor_leaderboard = {
        subset: (
            stats.get("actor_ties_score")
            if subset.lower() == "ties"
            else stats.get("actor_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
    }
    corrected_leaderboard = {
        subset: (
            stats.get("corrected_ties_score")
            if subset.lower() == "ties"
            else stats.get("corrected_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
    }
    prob_leaderboard = {
        subset: (
            stats.get("prob_ties_score")
            if subset.lower() == "ties"
            else stats.get("prob_prompt_accuracy")
        )
        for subset, stats in subset_summary.items()
        if stats.get("prob_ties_score") is not None
        or stats.get("prob_prompt_accuracy") is not None
    }

    overall_subset_average = {
        "actor": _safe_mean(list(actor_leaderboard.values())),
        "corrected": _safe_mean(list(corrected_leaderboard.values())),
        "prob": _safe_mean(list(prob_leaderboard.values())),
    }

    overall_prompt_accuracy = {
        "actor_prompt_accuracy": overall_subset_average["actor"],
        "corrected_prompt_accuracy": overall_subset_average["corrected"],
    }

    total_pairs = len(samples)

    overall_pair_metrics = {
        "actor": {
            "accuracy": total_actor_score / total_pairs if total_pairs else math.nan,
            "strict_accuracy": total_actor_strict / total_pairs if total_pairs else math.nan,
        },
        "corrected": {
            "accuracy": total_corrected_score / total_pairs if total_pairs else math.nan,
            "strict_accuracy": total_corrected_strict / total_pairs if total_pairs else math.nan,
        },
    }

    metrics = {
        "subset_metrics": subset_summary,
        "overall_prompt_accuracy": overall_prompt_accuracy,
        "overall_subset_average": overall_subset_average,
        "leaderboard_scores": {
            "actor": {k: v for k, v in actor_leaderboard.items() if v is not None and not math.isnan(v)},
            "corrected": {k: v for k, v in corrected_leaderboard.items() if v is not None and not math.isnan(v)},
            "prob": {k: v for k, v in prob_leaderboard.items() if v is not None and not math.isnan(v)},
        },
        "total_pairs": total_pairs,
        "total_prompts": len(prompt_records),
        "overall_pair_metrics": overall_pair_metrics,
    }
    return metrics


def export_pair_generations(samples: Iterable[PairSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for sample in samples:
            record = {
                "prompt_id": sample.prompt_id,
                "subset": sample.subset,
                "base_pair_id": sample.base_pair_id,
                "pair_id": sample.pair_id(),
                "orientation": sample.orientation,
                "chosen_index": sample.chosen_index,
                "rejected_index": sample.rejected_index,
                "num_correct": sample.num_correct,
                "ground_truth": sample.ground_truth,
                "predicted_label": sample.predicted_label,
                "corrected_label": sample.corrected_label,
                "actor_score": sample.actor_score,
                "corrected_score": sample.corrected_score,
                "critic_prob": sample.critic_prob,
                "actor_output": sample.actor_output,
            }
            fout.write(json.dumps(record) + "\n")


def compute_prediction_distribution(samples: Iterable[PairSample], attr: str) -> dict:
    counter: Counter[str] = Counter()
    total = 0
    for sample in samples:
        label = getattr(sample, attr) or "unknown"
        counter[label] += 1
        total += 1

    categories = ["response_1", "response_2", "tie", "unknown"]
    counts = {label: int(counter.get(label, 0)) for label in categories}
    counts["total"] = total
    fractions = {label: (counts[label] / total) if total else 0.0 for label in categories}
    return {"counts": counts, "fractions": fractions}


def _preference_score(label: str | None, orientation: str) -> float:
    if label is None:
        return 0.5
    label = label.lower()
    if label == "tie":
        return 0.5
    if label not in {"response_1", "response_2"}:
        return 0.5
    if orientation == "forward":
        return 1.0 if label == "response_1" else 0.0
    if orientation == "backward":
        return 1.0 if label == "response_2" else 0.0
    return 0.5


def _score_to_label(score: float) -> str:
    if score > 0.5:
        return "response_1"
    if score < 0.5:
        return "response_2"
    return "tie"




def _convert_critic_prob(sample: PairSample) -> float | None:
    if sample.critic_prob is None:
        return None
    label = sample.predicted_label
    if label is None or label.lower() not in {"response_1", "response_2"}:
        return 0.5
    prob = float(sample.critic_prob)
    if sample.orientation == "forward":
        return prob if label == "response_1" else 1.0 - prob
    if sample.orientation == "backward":
        return prob if label == "response_2" else 1.0 - prob
    return 0.5


def aggregate_pair_orientations(samples: list[PairSample]) -> list[PairSample]:
    grouped: dict[str, list[PairSample]] = defaultdict(list)
    for sample in samples:
        key = sample.base_pair_id or sample.pair_id()
        grouped[key].append(sample)

    aggregated: list[PairSample] = []
    for base_id in sorted(grouped.keys()):
        members = grouped[base_id]
        if not members:
            continue
        if all(member.orientation == "aggregated" for member in members):
            aggregated.extend(members)
            continue

        base = members[0]
        chosen = base.original_chosen or base.response_1
        rejected = base.original_rejected or base.response_2
        prompt_text = base.prompt_text
        messages = build_prompt_messages(prompt_text, chosen, rejected)

        actor_scores = [_preference_score(member.predicted_label, member.orientation) for member in members]
        corrected_scores = [_preference_score(member.corrected_label, member.orientation) for member in members]
        critic_scores = [score for member in members if (score := _convert_critic_prob(member)) is not None]

        actor_avg = sum(actor_scores) / len(actor_scores) if actor_scores else 0.5
        corrected_avg = sum(corrected_scores) / len(corrected_scores) if corrected_scores else actor_avg

        agg_sample = PairSample(
            prompt_id=base.prompt_id,
            subset=base.subset,
            pair_index=len(aggregated),
            messages=messages,
            response_1=chosen,
            response_2=rejected,
            ground_truth="response_1",
            prompt_text=prompt_text,
            base_pair_id=base_id,
            orientation="aggregated",
            original_chosen=chosen,
            original_rejected=rejected,
            prompt_raw_id=base.prompt_raw_id,
            chosen_index=base.chosen_index,
            rejected_index=base.rejected_index,
            num_correct=base.num_correct,
            total_completions=base.total_completions,
            actor_score=actor_avg,
            corrected_score=corrected_avg,
        )

        agg_sample.predicted_label = _score_to_label(actor_avg)
        agg_sample.corrected_label = _score_to_label(corrected_avg)
        if critic_scores:
            agg_sample.critic_prob = float(sum(critic_scores) / len(critic_scores))
        if members[0].actor_output:
            agg_sample.actor_output = members[0].actor_output

        aggregated.append(agg_sample)

    return aggregated


def _label_to_score(label: str | None, positive_label: str) -> float:
    if label == positive_label:
        return 1.0
    if label in {"response_1", "response_2"}:
        return 0.0
    if label == "tie":
        return 0.5
    return 0.5


def _get_metric_value(sample: PairSample, metric: str) -> float | None:
    norm_ground_truth = sample.ground_truth or "response_1"
    if metric == "actor":
        if sample.actor_score is not None:
            return float(sample.actor_score)
        return _label_to_score(sample.predicted_label, norm_ground_truth)
    if metric == "corrected":
        if sample.corrected_score is not None:
            return float(sample.corrected_score)
        return _label_to_score(sample.corrected_label, norm_ground_truth)
    if metric in {"prob", "critic"}:
        if sample.critic_prob is not None:
            return float(sample.critic_prob)
        return None
    raise ValueError(f"Unknown metric type '{metric}'")


def _ensure_score_capacity(record: dict[str, Any], length: int) -> None:
    for metric_name in record["scores"]:
        scores = record["scores"][metric_name]
        if len(scores) < length:
            scores.extend([0.0] * (length - len(scores)))
    for metric_name in record["counts"]:
        counts = record["counts"][metric_name]
        if len(counts) < length:
            counts.extend([0] * (length - len(counts)))
    record["total_completions"] = max(record.get("total_completions", 0), length)


def _compute_prompt_success(scores: list[float], num_correct: int) -> float | None:
    if not scores or num_correct <= 0 or num_correct > len(scores):
        return None
    correct_scores = [scores[idx] for idx in range(num_correct)]
    incorrect_scores = [scores[idx] for idx in range(num_correct, len(scores))]
    if not correct_scores or not incorrect_scores:
        return None
    if any(math.isnan(val) for val in correct_scores + incorrect_scores):
        return None
    return 1.0 if min(correct_scores) > max(incorrect_scores) else 0.0


def _split_ties_identifier(raw_id: str) -> tuple[str, str]:
    if ":" in raw_id:
        prefix, suffix = raw_id.split(":", 1)
        return prefix.strip().lower(), suffix.strip()
    return raw_id.strip().lower(), raw_id


def _ties_compute_prompt_stats(samples: list[tuple[bool, float]]) -> tuple[bool, float | None, float | None]:
    correct_scores = [score for is_correct, score in samples if is_correct]
    incorrect_scores = [score for is_correct, score in samples if not is_correct]
    if not correct_scores or not incorrect_scores:
        return False, None, None
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    diff_corr_margin = best_correct - worst_correct if len(correct_scores) > 1 else 0.0
    corr_incorrect_margin = worst_correct - best_incorrect
    accurate = corr_incorrect_margin > 0
    return accurate, diff_corr_margin, corr_incorrect_margin


def _compute_ties_score(entries: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    if not entries:
        return math.nan, {}

    grouped: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)
    for entry in entries:
        sample_type = entry["sample_type"]
        prompt_key = entry["prompt_key"]
        num_correct = entry["num_correct"]
        scores = entry["scores"]
        if not scores:
            continue
        for idx, score in enumerate(scores):
            if score is None or math.isnan(score):
                continue
            grouped[(sample_type, prompt_key)].append((idx < num_correct, float(score)))

    ref_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    tied_stats: dict[str, tuple[bool, float | None, float | None]] = {}
    for (sample_type, prompt_key), samples in grouped.items():
        if not samples:
            continue
        stats = _ties_compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_key] = stats
        elif sample_type == "tied":
            tied_stats[prompt_key] = stats

    ref_accuracy_vals = [float(stat[0]) for stat in ref_stats.values()]
    tied_accuracy_vals = [float(stat[0]) for stat in tied_stats.values()]
    ref_accuracy = float(np.mean(ref_accuracy_vals)) if ref_accuracy_vals else math.nan
    tied_accuracy = float(np.mean(tied_accuracy_vals)) if tied_accuracy_vals else math.nan

    shared_prompts = sorted(set(ref_stats) & set(tied_stats))
    if not shared_prompts:
        details = {
            "ref_accuracy": ref_accuracy,
            "tied_accuracy": tied_accuracy,
            "correctness_preferred": math.nan,
            "correctness_preferred_hard": math.nan,
            "correctness_margin_score": math.nan,
        }
        return math.nan, details

    diff_corr_margin = np.array(
        [tied_stats[prompt][1] if tied_stats[prompt][1] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )
    corr_incorrect_ties = np.array(
        [tied_stats[prompt][2] if tied_stats[prompt][2] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )
    corr_incorrect_ref = np.array(
        [ref_stats[prompt][2] if ref_stats[prompt][2] is not None else 0.0 for prompt in shared_prompts],
        dtype=float,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        correctness_preferred = float(np.mean(corr_incorrect_ties > diff_corr_margin)) if diff_corr_margin.size else math.nan
        correctness_preferred_hard = float(
            np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
        ) if diff_corr_margin.size else math.nan
        baseline = np.minimum(corr_incorrect_ref, corr_incorrect_ties)
        ratio = np.where(diff_corr_margin == 0.0, 0.0, baseline / diff_corr_margin - 1.0)
        margin_scores = np.tanh(ratio)

    margin_scores = np.nan_to_num(margin_scores, nan=0.0, posinf=0.0, neginf=0.0)
    correctness_margin_score = float(np.mean(margin_scores)) if margin_scores.size else math.nan

    def _nan_to_zero(value: float) -> float:
        return 0.0 if math.isnan(value) else value

    overall_score = (
        0.30 * _nan_to_zero(tied_accuracy)
        + 0.30 * _nan_to_zero(ref_accuracy)
        + 0.20 * _nan_to_zero(correctness_preferred)
        + 0.20 * _nan_to_zero(correctness_preferred_hard)
        + 0.01 * _nan_to_zero(correctness_margin_score)
    )

    details = {
        "ref_accuracy": ref_accuracy,
        "tied_accuracy": tied_accuracy,
        "correctness_preferred": correctness_preferred,
        "correctness_preferred_hard": correctness_preferred_hard,
        "correctness_margin_score": correctness_margin_score,
    }
    return overall_score, details


def _safe_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fout:
        json.dump(metrics, fout, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Think RM checkpoints on RewardBench 2.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Local checkpoint directory containing global_step_* folders. "
        "Required unless --actor-hf-path is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store evaluation logs and exported models.",
    )
    parser.add_argument("--dataset", default="allenai/reward-bench-2", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit the number of prompts (for debugging).")
    parser.add_argument("--actor-batch-size", type=int, default=4, help="Batch size for actor generation.")
    parser.add_argument("--critic-batch-size", type=int, default=8, help="Batch size for critic scoring.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum tokens generated by the actor.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Critic probability threshold for correction.")
    parser.add_argument(
        "--results-file",
        default="rewardbench2_metrics.json",
        help="Filename (within output-dir) for the aggregated metrics JSON.",
    )
    parser.add_argument(
        "--actor-backend",
        choices=("hf", "vllm", "server"),
        default="server",
        help="Generation backend for the actor model.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree when using the local vLLM backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization ratio for the local vLLM backend.",
    )
    parser.add_argument(
        "--actor-data-parallel-size",
        type=int,
        default=8,
        help="Data parallel degree when using the vLLM server backend.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port to bind the vLLM OpenAI server.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the vLLM server to become ready.",
    )
    parser.add_argument(
        "--server-shutdown-timeout",
        type=int,
        default=120,
        help="Seconds to wait for the vLLM server to terminate.",
    )
    parser.add_argument(
        "--actor-request-concurrency",
        type=int,
        default=64,
        help="Number of concurrent HTTP requests sent to the vLLM server.",
    )
    parser.add_argument(
        "--actor-request-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each vLLM server request.",
    )
    parser.add_argument(
        "--critic-num-workers",
        type=int,
        default=None,
        help="Number of GPU workers for critic scoring (defaults to available CUDA devices).",
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Reuse existing Hugging Face exports under output-dir if present.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=str,
        default=None,
        help="Optional checkpoint step (e.g., global_step_439 or 439) to evaluate.",
    )
    parser.add_argument(
        "--actor-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face model path or repo ID for evaluating a baseline actor directly.",
    )
    parser.add_argument(
        "--actor-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path for the baseline actor (defaults to --actor-hf-path).",
    )
    parser.add_argument(
        "--critic-hf-path",
        type=str,
        default=None,
        help="Optional Hugging Face model path or repo ID for evaluating a baseline critic directly.",
    )
    parser.add_argument(
        "--critic-hf-tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path for the baseline critic (defaults to --critic-hf-path).",
    )
    parser.add_argument(
        "--skip-critic",
        action="store_true",
        help="Skip critic scoring (useful for actor-only baseline ablations).",
    )
    parser.add_argument(
        "--critic-loss-type",
        choices=("mle", "squared"),
        default="mle",
        help="Loss type used when training the critic value head (controls evaluation activation).",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional W&B project override.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group override.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name override.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled", "dryrun"),
        help="Optional W&B mode override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    actor_export: str | Path | None = args.actor_hf_path
    critic_export: str | Path | None = args.critic_hf_path
    critic_ckpt: Path | str | None = args.critic_hf_path

    if args.checkpoint_root is not None:
        checkpoint_root = args.checkpoint_root.resolve()
        if args.checkpoint_step:
            step_dir = args.checkpoint_step
            if not step_dir.startswith("global_step_"):
                step_dir = f"global_step_{step_dir}"
            checkpoint_dir = checkpoint_root / step_dir
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Specified checkpoint step not found: {checkpoint_dir}")
        else:
            checkpoint_dir = find_latest_checkpoint(checkpoint_root)

        actor_ckpt = checkpoint_dir / "actor"
        critic_ckpt = checkpoint_dir / "critic"

        export_root = output_dir / "hf_exports" / checkpoint_dir.name
        actor_export = export_root / "actor"
        critic_export = export_root / "critic"

        if not (args.reuse_export and (Path(actor_export) / "config.json").exists()):
            ensure_hf_export(actor_ckpt, Path(actor_export))
        if not args.skip_critic and critic_export is not None:
            if not (args.reuse_export and (Path(critic_export) / "config.json").exists()):
                ensure_hf_export(critic_ckpt, Path(critic_export))

    if actor_export is None:
        raise ValueError("Either --checkpoint-root or --actor-hf-path must be provided.")

    num_gpus = torch.cuda.device_count()
    dtype = torch.bfloat16 if num_gpus > 0 else torch.float32
    device = torch.device("cuda" if num_gpus > 0 else "cpu")

    actor_tokenizer_source = args.actor_hf_tokenizer or actor_export
    actor_tokenizer = AutoTokenizer.from_pretrained(actor_tokenizer_source, trust_remote_code=True)
    actor_tokenizer.padding_side = "left"

    samples = load_rewardbench_pairs(args.dataset, args.split, max_examples=args.max_examples)
    if not samples:
        raise RuntimeError("No evaluation samples were constructed.")

    actor_backend = args.actor_backend.lower()
    actor_model_ref = actor_export
    critic_outputs_logits = args.critic_loss_type == "mle"
    if actor_backend == "server":
        server_proc: subprocess.Popen | None = None
        try:
            server_proc = start_vllm_server(actor_model_ref, args.server_port, args.actor_data_parallel_size)
            wait_for_server_ready(args.server_port, args.server_startup_timeout)
            run_actor_generation_vllm_server(
                actor_tokenizer,
                samples,
                str(actor_model_ref),
                args.server_port,
                args.max_new_tokens,
                args.actor_request_concurrency,
                args.actor_request_timeout,
            )
        finally:
            shutdown_process(server_proc, args.server_shutdown_timeout)
    elif actor_backend == "vllm":
        if num_gpus == 0:
            raise RuntimeError("vLLM backend requires CUDA but no GPU was detected.")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("vLLM is not installed; re-run with --actor-backend hf or install vllm.") from exc

        dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"
        actor_llm = LLM(
            model=str(actor_model_ref),
            tokenizer=str(actor_model_ref),
            trust_remote_code=True,
            dtype=dtype_str,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
        )
        run_actor_generation_vllm_local(actor_llm, sampling_params, actor_tokenizer, samples, args.actor_batch_size)
    else:
        actor_model = AutoModelForCausalLM.from_pretrained(
            actor_model_ref,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        run_actor_generation_hf(actor_model, actor_tokenizer, samples, args.max_new_tokens, args.actor_batch_size)

    run_critic = not args.skip_critic and critic_export is not None
    critic_workers: int = 0
    if run_critic:
        critic_workers = args.critic_num_workers or 0
        if num_gpus == 0:
            critic_workers = 1
        else:
            critic_workers = critic_workers or num_gpus
            critic_workers = max(1, min(critic_workers, num_gpus))

        critic_model_ref = critic_export
        if checkpoint_dir is None and not Path(str(critic_model_ref)).exists():
            raise ValueError(
                "--critic-hf-path must point to a local directory when using baseline evaluation."
            )

        if critic_workers > 1:
            run_critic_scoring_multiproc(
                samples,
                Path(critic_model_ref),
                Path(critic_ckpt) if isinstance(critic_ckpt, (str, Path)) else None,
                critic_outputs_logits,
                dtype,
                args.critic_batch_size,
                critic_workers,
            )
        else:
            critic_model, critic_tokenizer = load_token_classifier(
                Path(critic_model_ref),
                dtype,
                Path(critic_ckpt) if isinstance(critic_ckpt, (str, Path)) else None,
            )
            critic_model.to(device)
            run_critic_scoring_single(
                critic_model,
                critic_tokenizer,
                samples,
                args.critic_batch_size,
                critic_outputs_logits,
            )

    apply_correction(samples, args.threshold)
    raw_samples = list(samples)
    samples = aggregate_pair_orientations(samples)

    actor_prediction_distribution = compute_prediction_distribution(samples, "predicted_label")
    corrected_prediction_distribution = compute_prediction_distribution(samples, "corrected_label")

    metrics = compute_metrics(samples)
    metrics_config = {
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "threshold": args.threshold,
        "dataset": args.dataset,
        "split": args.split,
        "max_examples": args.max_examples,
        "actor_batch_size": args.actor_batch_size,
        "critic_batch_size": args.critic_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "actor_backend": actor_backend,
        "critic_num_workers": critic_workers,
        "actor_model_ref": str(actor_model_ref),
        "skip_critic": args.skip_critic,
        "critic_model_ref": str(critic_export) if critic_export is not None else None,
        "critic_loss_type": args.critic_loss_type,
        "output_dir": str(output_dir),
    }

    metrics["prediction_distribution"] = {
        "actor": actor_prediction_distribution,
        "corrected": corrected_prediction_distribution,
    }

    generations_path = output_dir / f"{Path(args.results_file).stem}_generations.jsonl"
    export_pair_generations(raw_samples, generations_path)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_group:
        os.environ["WANDB_GROUP"] = args.wandb_group
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_group = os.getenv("WANDB_GROUP")
    wandb_run_name = os.getenv("WANDB_RUN_NAME")
    wandb_mode_env = os.getenv("WANDB_MODE", "online").lower()

    metrics_config.update(
        {
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_run_name": wandb_run_name,
            "wandb_mode": wandb_mode_env,
        }
    )

    if actor_backend == "vllm":
        metrics_config.update(
            {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        )
    elif actor_backend == "server":
        metrics_config.update(
            {
                "actor_data_parallel_size": args.actor_data_parallel_size,
                "actor_request_concurrency": args.actor_request_concurrency,
                "actor_request_timeout": args.actor_request_timeout,
                "server_port": args.server_port,
            }
        )

    checkpoint_step_name: str | None = None
    if checkpoint_dir is not None:
        checkpoint_step_name = checkpoint_dir.name
    elif args.checkpoint_step:
        step_label = str(args.checkpoint_step)
        checkpoint_step_name = step_label if step_label.startswith("global_step_") else f"global_step_{step_label}"
    if checkpoint_step_name:
        metrics_config["checkpoint_step_name"] = checkpoint_step_name

    metrics["config"] = metrics_config

    output_path = output_dir / args.results_file
    save_metrics(metrics, output_path)

    if wandb and wandb_mode_env not in {"disabled", "off", "offline"}:
        try:
            run_name_suffix_parts: list[str] = []
            if checkpoint_step_name:
                run_name_suffix_parts.append(checkpoint_step_name)
            run_name_suffix_parts.append(output_dir.name)

            wandb_name = wandb_run_name
            if run_name_suffix_parts:
                suffix = "-".join(run_name_suffix_parts)
                if wandb_name:
                    wandb_name = f"{wandb_name}-{suffix}"
                else:
                    wandb_name = suffix

            run = wandb.init(
                project=wandb_project or "verl_think_rm",
                group=wandb_group,
                job_type="rewardbench2_eval",
                name=wandb_name,
                config=metrics["config"],
            )

            overall = metrics.get("overall_prompt_accuracy", {})
            log_payload = {
                "rewardbench2/overall/actor_prompt_accuracy": overall.get("actor_prompt_accuracy"),
                "rewardbench2/overall/corrected_prompt_accuracy": overall.get("corrected_prompt_accuracy"),
                "rewardbench2/total_prompts": metrics.get("total_prompts"),
                "rewardbench2/total_pairs": metrics.get("total_pairs"),
            }

            overall_pair = metrics.get("overall_pair_metrics", {})
            actor_pair_overall = overall_pair.get("actor", {})
            corrected_pair_overall = overall_pair.get("corrected", {})
            log_payload.update(
                {
                    "rewardbench2/overall/actor_pair_accuracy": actor_pair_overall.get("accuracy"),
                    "rewardbench2/overall/actor_pair_strict_accuracy": actor_pair_overall.get("strict_accuracy"),
                    "rewardbench2/overall/corrected_pair_accuracy": corrected_pair_overall.get("accuracy"),
                    "rewardbench2/overall/corrected_pair_strict_accuracy": corrected_pair_overall.get(
                        "strict_accuracy"
                    ),
                }
            )

            for subset, stats in metrics.get("subset_metrics", {}).items():
                prefix = f"rewardbench2/subsets/{subset}"
                log_payload[f"{prefix}/actor_prompt_accuracy"] = stats.get("actor_prompt_accuracy")
                log_payload[f"{prefix}/corrected_prompt_accuracy"] = stats.get("corrected_prompt_accuracy")
                log_payload[f"{prefix}/actor_pair_accuracy"] = stats.get("actor_pair_accuracy")
                log_payload[f"{prefix}/corrected_pair_accuracy"] = stats.get("corrected_pair_accuracy")
                log_payload[f"{prefix}/actor_pair_strict_accuracy"] = stats.get("actor_pair_strict_accuracy")
                log_payload[f"{prefix}/corrected_pair_strict_accuracy"] = stats.get("corrected_pair_strict_accuracy")
                log_payload[f"{prefix}/prompt_count"] = stats.get("prompt_count")
                log_payload[f"{prefix}/pair_count"] = stats.get("pair_count")
                if subset.lower() == "ties":
                    log_payload[f"{prefix}/actor_ties_score"] = stats.get("actor_ties_score")
                    log_payload[f"{prefix}/corrected_ties_score"] = stats.get("corrected_ties_score")
                    log_payload[f"{prefix}/prob_ties_score"] = stats.get("prob_ties_score")

            actor_counts = actor_prediction_distribution["counts"]
            actor_fractions = actor_prediction_distribution["fractions"]
            corrected_counts = corrected_prediction_distribution["counts"]
            corrected_fractions = corrected_prediction_distribution["fractions"]
            categories = ["response_1", "response_2", "tie", "unknown"]
            for label in categories:
                log_payload[f"rewardbench2/predictions/actor/{label}_count"] = actor_counts.get(label, 0)
                log_payload[f"rewardbench2/predictions/actor/{label}_fraction"] = actor_fractions.get(label, 0.0)
                log_payload[f"rewardbench2/predictions/corrected/{label}_count"] = corrected_counts.get(label, 0)
                log_payload[f"rewardbench2/predictions/corrected/{label}_fraction"] = corrected_fractions.get(label, 0.0)

            subset_avg = metrics.get("overall_subset_average", {})
            log_payload["rewardbench2/overall/actor_subset_average"] = subset_avg.get("actor")
            log_payload["rewardbench2/overall/corrected_subset_average"] = subset_avg.get("corrected")
            log_payload["rewardbench2/overall/prob_subset_average"] = subset_avg.get("prob")

            run.log(log_payload)

        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] WandB logging failed: {exc}")
        finally:
            if wandb.run is not None:
                wandb.finish()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

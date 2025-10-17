#!/usr/bin/env python3
"""Shared utilities for Think RM evaluation pipelines."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import requests
import torch
import torch.nn as nn
import traceback
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from verl.utils.model import build_value_head

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:  # pragma: no cover - optional dependency
    safe_load_file = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification
except ImportError:  # transformers < 4.38
    AutoModelForSequenceClassification = None  # type: ignore[assignment]

from recipe.think_rm.preference_dataset_utils import build_prompt
from recipe.think_rm.reward_fn import parse_preference


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
    row_index: int = -1
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

    request_text: str | None = None
    full_token_ids: list[int] | None = None
    response_1_pos: int = -1
    response_2_pos: int = -1
    comparison_kind: str = "chosen_vs_rejected"

    def pair_id(self) -> str:
        return f"{self.base_pair_id}-{self.orientation}"

    @property
    def prompt_key(self) -> str:
        return make_prompt_key(self.subset, self.prompt_id, self.row_index, self.base_pair_id, self.pair_index)


def make_prompt_key(
    subset: str | None,
    prompt_id: str | None,
    row_index: int | None,
    base_pair_id: str | None = None,
    pair_index: int | None = None,
) -> str:
    """Build a stable identifier for a prompt even when dataset IDs collide."""
    subset_component = (subset or "unknown").strip() or "unknown"

    if prompt_id is not None:
        prompt_component = str(prompt_id)
        return f"{subset_component}::{prompt_component}"

    if row_index is not None and row_index >= 0:
        suffix = f"row{row_index}"
    elif base_pair_id:
        suffix = f"pair{base_pair_id}"
    elif pair_index is not None and pair_index >= 0:
        suffix = f"idx{pair_index}"
    else:
        suffix = "row-1"

    return f"{subset_component}::{suffix}"


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
                    state_dict[key] = tensor.clone()
    return state_dict


def _locate_module(model: nn.Module, path: tuple[str, ...]) -> tuple[nn.Module | None, nn.Module | None]:
    parent: nn.Module | None = None
    current: nn.Module | None = model
    for name in path:
        parent = current
        if current is None:
            break
        current = getattr(current, name, None)
    return parent, current


# def _infer_value_head_spec(state_dict: dict[str, torch.Tensor], metadata: dict[str, Any] | None) -> ValueHeadSpec:
#     if metadata and "prefix" in metadata:
#         prefix = metadata["prefix"]
#     else:
#         prefix = next((candidate for candidate in VALUE_HEAD_ATTR_CANDIDATES if any(k.startswith(candidate) for k in state_dict)), "score")

#     hidden_sizes: tuple[int, ...] = ()
#     if metadata:
#         hidden_sizes = tuple(metadata.get("hidden_sizes", ()))
#     activation = "silu"
#     dropout = 0.0
#     force_dropout_layers = False
#     if metadata:
#         activation = metadata.get("activation", activation)
#         dropout = float(metadata.get("dropout", dropout))
#         force_dropout_layers = bool(metadata.get("force_dropout_layers", force_dropout_layers))

#     return ValueHeadSpec(
#         prefix=prefix,
#         hidden_sizes=hidden_sizes,
#         activation=activation,
#         dropout=dropout,
#         force_dropout_layers=force_dropout_layers,
#     )


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


def _select_last_token_values(value_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask but received shape {tuple(attention_mask.shape)}")

    if value_logits.ndim == 3 and value_logits.size(-1) == 1:
        value_logits = value_logits.squeeze(-1)
    elif value_logits.ndim != 2:
        raise ValueError(
            "Critic logits must be rank-2 (batch, seq_len) or rank-3 with singleton last dim; "
            f"received tensor with shape {tuple(value_logits.shape)}"
        )

    last_indices = attention_mask.long().sum(dim=1) - 1
    max_index = value_logits.size(1) - 1
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

    context_messages: list[dict[str, str]] = []
    cleaned_prompt = (prompt_text or "").strip()
    if cleaned_prompt:
        context_messages.append({"role": "user", "content": cleaned_prompt})
    return build_prompt(context_messages, response1, response2)


def chunked(iterable: list[PairSample], batch_size: int) -> Iterable[list[PairSample]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def _build_actor_prompt(actor_tokenizer: AutoTokenizer, sample: PairSample) -> str:
    prompt = actor_tokenizer.apply_chat_template(
        sample.messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    sample.request_text = prompt
    return prompt


def _finalize_actor_output(
    sample: PairSample,
    text: str,
    actor_tokenizer: AutoTokenizer | None,
) -> None:
    sample.actor_output = text.strip()
    sample.predicted_label = parse_preference(sample.actor_output or "")
    if actor_tokenizer is None:
        return
    convo = list(sample.messages)
    convo.append({"role": "assistant", "content": sample.actor_output})
    conv = actor_tokenizer.apply_chat_template(
        convo,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )
    conv_input_ids = _normalize_chat_tensor(conv, "input_ids", device_id=-1)
    if conv_input_ids.ndim == 2:
        conv_input_ids = conv_input_ids.squeeze(0)
    sample.full_token_ids = conv_input_ids.to(dtype=torch.long).cpu().tolist()


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
            _finalize_actor_output(sample, text, actor_tokenizer)


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
            _finalize_actor_output(sample, generated_text, actor_tokenizer)


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
            _finalize_actor_output(sample, text, actor_tokenizer)


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
            f"Attention mask shape {tuple(mask.shape)} does not match input_ids shape {tuple(input_ids.shape)} "
            f"(device {device_id})."
        )
    return mask


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
        # print('>>>>>>>>>>>> input\n', inputs, '\n>>>>>>>>>>>> critic\n', value_logits, '\n>>>>>>>>>>>>>> probs\n', probs)
        # print('>>>>>>>>>>>> input\n', inputs['input_ids'].size(), '\n>>>>>>>>>>>> critic\n', value_logits.size(), '\n>>>>>>>>>>>>>> probs\n', probs)

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

            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }

            with torch.inference_mode():
                outputs = model(**inputs)

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

            queue.put(
                {
                    "type": "result",
                    "device": device_id,
                    "indices": batch_indices,
                    "probs": [float(prob) for prob in probs],
                }
            )
            queue.put({"type": "progress", "device": device_id, "count": len(batch_indices)})

        queue.put({"type": "done", "device": device_id})
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        queue.put({"type": "error", "device": device_id, "error": str(exc), "traceback": tb})


def run_critic_scoring_multiproc(
    samples: list[PairSample],
    model_path: Path,
    source_ckpt: Path | None,
    outputs_logits: bool,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
) -> None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []

    indices = list(range(len(samples)))
    per_worker = math.ceil(len(indices) / num_workers)
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"
    progress = tqdm(total=len(indices), desc="Critic scoring (mp)", unit="pair")

    try:
        for worker_id in range(num_workers):
            start = worker_id * per_worker
            end = min(start + per_worker, len(indices))
            if start >= end:
                break
            worker_indices = indices[start:end]
            proc = ctx.Process(
                target=_critic_worker_process,
                args=(
                    worker_id,
                    worker_indices,
                    samples,
                    str(model_path),
                    str(source_ckpt) if source_ckpt is not None else None,
                    outputs_logits,
                    dtype_str,
                    batch_size,
                    queue,
                ),
            )
            proc.start()
            processes.append(proc)

        finished_workers = 0
        while finished_workers < len(processes):
            message = queue.get()
            msg_type = message.get("type")
            if msg_type == "progress":
                progress.update(message.get("count", 0))
            elif msg_type == "result":
                indices = message.get("indices", [])
                probs = message.get("probs", [])
                for idx, prob in zip(indices, probs):
                    samples[idx].critic_prob = float(prob)
            elif msg_type == "error":
                err_msg = message.get("error", "unknown error")
                tb = message.get("traceback")
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


def compute_prediction_distribution(samples: Iterable[PairSample], attr: str) -> dict:
    counter: defaultdict[str, int] = defaultdict(int)
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
        if base.comparison_kind != "chosen_vs_rejected":
            aggregated.extend(members)
            continue
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
            row_index=base.row_index,
            chosen_index=base.chosen_index,
            rejected_index=base.rejected_index,
            num_correct=base.num_correct,
            total_completions=base.total_completions,
            actor_score=actor_avg,
            corrected_score=corrected_avg,
            response_1_pos=base.response_1_pos,
            response_2_pos=base.response_2_pos,
            comparison_kind=base.comparison_kind,
        )

        agg_sample.predicted_label = _score_to_label(actor_avg)
        agg_sample.corrected_label = _score_to_label(corrected_avg)
        if critic_scores:
            agg_sample.critic_prob = float(sum(critic_scores) / len(critic_scores))
        if members[0].actor_output:
            agg_sample.actor_output = members[0].actor_output
        agg_sample.request_text = members[0].request_text
        agg_sample.full_token_ids = members[0].full_token_ids

        aggregated.append(agg_sample)

    return aggregated


def _label_to_choice(sample: PairSample, label: str | None) -> str | None:
    if label is None:
        return None
    label = label.lower()
    if label not in {"response_1", "response_2"}:
        return label if label in {"tie", "unknown"} else None
    if sample.orientation == "forward":
        return "chosen" if label == "response_1" else "rejected"
    if sample.orientation == "backward":
        return "rejected" if label == "response_1" else "chosen"
    return None


def _classify_choices(choices: list[str | None]) -> tuple[str, str | None]:
    concrete = [choice for choice in choices if choice in {"chosen", "rejected"}]
    if not choices:
        return "unclear", None
    if len(concrete) != len(choices):
        return "unclear", None
    if all(choice == "chosen" for choice in concrete):
        return "clear_right", "chosen"
    if all(choice == "rejected" for choice in concrete):
        return "clear_wrong", "rejected"
    return "unclear", None


def build_pair_level_results(samples: Iterable[PairSample]) -> list[dict[str, Any]]:
    grouped: dict[str, list[PairSample]] = defaultdict(list)
    for sample in samples:
        if sample.comparison_kind != "chosen_vs_rejected":
            continue
        if not sample.base_pair_id:
            continue
        grouped[sample.base_pair_id].append(sample)

    results: list[dict[str, Any]] = []
    for base_id, members in sorted(grouped.items()):
        if not members:
            continue
        base = members[0]
        actor_choices = [_label_to_choice(member, member.predicted_label) for member in members]
        critic_choices = [_label_to_choice(member, member.corrected_label) for member in members]
        actor_status, actor_choice = _classify_choices(actor_choices)
        critic_status, critic_choice = _classify_choices(critic_choices)

        orientation_details = []
        for member, actor_choice_single, critic_choice_single in zip(members, actor_choices, critic_choices):
            orientation_details.append(
                {
                    "pair_id": member.pair_id(),
                    "orientation": member.orientation,
                    "predicted_label": member.predicted_label,
                    "corrected_label": member.corrected_label,
                    "actor_choice": actor_choice_single,
                    "critic_choice": critic_choice_single,
                    "critic_prob": member.critic_prob,
                }
            )

        results.append(
            {
                "base_pair_id": base_id,
                "prompt_id": base.prompt_id,
                "subset": base.subset,
                "chosen_index": base.chosen_index,
                "rejected_index": base.rejected_index,
                "row_index": base.row_index,
                "original_chosen": base.original_chosen,
                "original_rejected": base.original_rejected,
                "actor_status": actor_status,
                "actor_choice": actor_choice,
                "critic_status": critic_status,
                "critic_choice": critic_choice,
                "requests": orientation_details,
            }
        )

    return results


def save_request_generations(samples: Iterable[PairSample], path: Path) -> None:
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
                "row_index": sample.row_index,
                "num_correct": sample.num_correct,
                "ground_truth": sample.ground_truth,
                "predicted_label": sample.predicted_label,
                "corrected_label": sample.corrected_label,
                "actor_score": sample.actor_score,
                "corrected_score": sample.corrected_score,
                "critic_prob": sample.critic_prob,
                "request_text": sample.request_text,
                "actor_output": sample.actor_output,
                "full_token_ids": sample.full_token_ids,
                "response_1_pos": sample.response_1_pos,
                "response_2_pos": sample.response_2_pos,
                "comparison_kind": sample.comparison_kind,
            }
            fout.write(json.dumps(record) + "\n")


def save_pair_level_results(results: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fout:
        for record in results:
            fout.write(json.dumps(record) + "\n")


def summarize_pair_statuses(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cached = results if isinstance(results, list) else list(results)
    for mode in ("actor", "critic"):
        counts: Counter[str] = Counter()
        total = 0
        for record in cached:
            status = record.get(f"{mode}_status")
            if not status:
                continue
            counts[status] += 1
            total += 1
        consistency = (counts.get("clear_right", 0) + counts.get("clear_wrong", 0)) / total if total else math.nan
        rates = {
            status: (counts.get(status, 0) / total) if total else math.nan
            for status in ("clear_right", "clear_wrong", "unclear")
        }
        summary[mode] = {
            "counts": dict(counts),
            "total": total,
            "consistency_rate": consistency,
            "rates": rates,
        }
    return summary


def summarize_prompt_statuses(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    cached = results if isinstance(results, list) else list(results)
    prompt_map: dict[str, dict[str, Any]] = {}

    for record in cached:
        prompt_id = record.get("prompt_id")
        row_index = record.get("row_index")
        base_pair_id = record.get("base_pair_id")
        if prompt_id is None and row_index is None and base_pair_id is None:
            continue
        subset = record.get("subset", "unknown")
        prompt_key = make_prompt_key(subset, prompt_id, row_index, base_pair_id)
        prompt_entry = prompt_map.setdefault(
            prompt_key,
            {
                "subset": subset,
                "prompt_id": prompt_id,
                "row_index": row_index,
                "actor": Counter(),
                "critic": Counter(),
            },
        )
        for mode in ("actor", "critic"):
            status = record.get(f"{mode}_status")
            if status:
                prompt_entry[mode][status] += 1

    total_prompts = len(prompt_map)
    overall_counts = {
        "actor": {"strict_correct": 0, "loose_correct": 0.0},
        "critic": {"strict_correct": 0, "loose_correct": 0.0},
    }
    subset_stats: dict[str, dict[str, Any]] = {}

    for prompt_data in prompt_map.values():
        subset = prompt_data["subset"]
        subset_entry = subset_stats.setdefault(
            subset,
            {
                "total_prompts": 0,
                "actor": {"strict_correct": 0, "loose_correct": 0.0},
                "critic": {"strict_correct": 0, "loose_correct": 0.0},
            },
        )
        subset_entry["total_prompts"] += 1

        for mode in ("actor", "critic"):
            counts = prompt_data[mode]
            total_pairs = sum(counts.values())
            if total_pairs == 0:
                continue
            is_strict = counts.get("clear_right", 0) == total_pairs
            has_clear_wrong = counts.get("clear_wrong", 0) > 0
            if is_strict:
                overall_counts[mode]["strict_correct"] += 1
                subset_entry[mode]["strict_correct"] += 1
            if not has_clear_wrong and total_pairs > 0:
                num_unclear = counts.get("unclear", 0)
                # Treat each unclear pair as a 50% chance of getting the prompt right
                # when only one orientation is sampled at inference time.
                loose_score = 0.5 ** num_unclear
                overall_counts[mode]["loose_correct"] += loose_score
                subset_entry[mode]["loose_correct"] += loose_score

    overall_summary = {}
    for mode in ("actor", "critic"):
        strict_correct = overall_counts[mode]["strict_correct"]
        loose_correct = overall_counts[mode]["loose_correct"]
        overall_summary[mode] = {
            "total_prompts": total_prompts,
            "strict_correct": strict_correct,
            "loose_correct": loose_correct,
            "strict_accuracy": (strict_correct / total_prompts) if total_prompts else math.nan,
            "loose_accuracy": (loose_correct / total_prompts) if total_prompts else math.nan,
        }

    subset_summary = {}
    for subset, data in subset_stats.items():
        total = data["total_prompts"]
        subset_summary[subset] = {
            "total_prompts": total,
            "actor": {
                "strict_correct": data["actor"]["strict_correct"],
                "loose_correct": data["actor"]["loose_correct"],
                "strict_accuracy": (data["actor"]["strict_correct"] / total) if total else math.nan,
                "loose_accuracy": (data["actor"]["loose_correct"] / total) if total else math.nan,
            },
            "critic": {
                "strict_correct": data["critic"]["strict_correct"],
                "loose_correct": data["critic"]["loose_correct"],
                "strict_accuracy": (data["critic"]["strict_correct"] / total) if total else math.nan,
                "loose_accuracy": (data["critic"]["loose_correct"] / total) if total else math.nan,
            },
        }

    return {
        "overall": overall_summary,
        "subsets": subset_summary,
    }

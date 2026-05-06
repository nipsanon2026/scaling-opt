# calibration_utils.py
from __future__ import annotations

import os
import json
import logging
import math
import inspect
import types
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

log = logging.getLogger(__name__)

# -------------------------
# Qwen-only layer access
# -------------------------
def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Qwen HF structure (Qwen2/Qwen2.5/Qwen3):
      model.model.layers
    """
    if not hasattr(model, "model"):
        raise RuntimeError("Model has no attribute .model (unexpected HF structure).")
    core = model.model
    if hasattr(core, "layers"):
        return core.layers
    raise RuntimeError("Qwen-only script: couldn't locate model.model.layers.")


def default_module_paths_for_qwen() -> Tuple[str, ...]:
    return (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )


def _maybe_set_pad_token(tokenizer, model: Optional[nn.Module] = None) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and hasattr(model, "config"):
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id


# -------------------------
# Activation collection
# -------------------------
class ActivationCatcher:
    """
    Activation catcher with configurable storage.
      - store_on_cpu=True: immediately move chunks to CPU (optionally pinned)
      - store_on_cpu=False: keep chunks on the same device as produced (typically GPU)

    NOTE: If store_on_cpu=False, you can easily OOM when collecting many modules/layers.
    """
    def __init__(
        self,
        max_rows: int,
        *,
        store_on_cpu: bool = True,
        pin_memory: bool = True,
    ):
        self.max_rows = int(max_rows)
        self.store_on_cpu = bool(store_on_cpu)
        self.pin_memory = bool(pin_memory)

        self.X_chunks: List[torch.Tensor] = []
        self._rows_collected: int = 0
        self._full: bool = False

    def hook(self, module, inp, out):
        if self._full:
            return

        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected input rank {x.dim()} for {type(module)}; shape={tuple(x.shape)}")

        rows_needed = self.max_rows - self._rows_collected
        if rows_needed <= 0:
            self._full = True
            return
        if x.shape[0] > rows_needed:
            x = x[:rows_needed]

        if self.store_on_cpu:
            x = x.to(device="cpu", non_blocking=True)
            if self.pin_memory:
                x = x.pin_memory()
        self.X_chunks.append(x)

        self._rows_collected += int(x.shape[0])
        if self._rows_collected >= self.max_rows:
            self._full = True

    def get_matrix(self, max_rows: int) -> torch.Tensor:
        if len(self.X_chunks) == 0:
            raise RuntimeError("No activations were collected (hook never fired?).")
        X = torch.cat(self.X_chunks, dim=0)
        if X.shape[0] > int(max_rows):
            X = X[: int(max_rows)]
        return X

def make_calib_texts(ds_split, n_rows: int) -> List[str]:
    texts = [ds_split[i]["text"] for i in range(min(n_rows, len(ds_split)))]
    texts = [t for t in texts if isinstance(t, str) and t.strip() != ""]
    if len(texts) == 0:
        raise RuntimeError("No non-empty texts found for calibration.")
    return texts


@torch.no_grad()
def make_token_batches(tokenizer, texts: List[str], batch_size: int, max_length: int) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        toks = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.cpu() for k, v in toks.items()}
        batches.append(toks)
    return batches


@torch.no_grad()
def make_fixed_length_token_batches_from_dataset(
    tokenizer,
    ds_split,
    nsamples: int,
    batch_size: int,
    seq_len: int,
    *,
    text_key: str = "text",
    max_tokens: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build calibration batches as exact-length token windows, matching the
    fixed-seqlen calibration style used by OmniQuant.

    `nsamples` counts fixed windows, not source documents.
    """
    nsamples = int(nsamples)
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    if nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")

    target_tokens = nsamples * seq_len
    if max_tokens is not None:
        target_tokens = min(target_tokens, int(max_tokens))
    if target_tokens <= 0:
        raise ValueError("target calibration token budget must be positive.")
    sep_ids = tokenizer("\n\n", add_special_tokens=False, return_tensors="pt").input_ids.view(-1).to(torch.long)
    token_chunks: List[torch.Tensor] = []
    total_tokens = 0

    for row in range(len(ds_split)):
        sample = ds_split[row]
        text = sample.get(text_key, None) if isinstance(sample, dict) else None
        if not isinstance(text, str) or text.strip() == "":
            continue

        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.view(-1).to(torch.long)
        if ids.numel() == 0:
            continue

        if token_chunks and sep_ids.numel() > 0:
            token_chunks.append(sep_ids)
            total_tokens += int(sep_ids.numel())
        token_chunks.append(ids)
        total_tokens += int(ids.numel())

        if total_tokens >= target_tokens:
            break

    if total_tokens < target_tokens:
        raise RuntimeError(
            f"Not enough calibration tokens to build target_tokens={target_tokens} "
            f"(requested nsamples={nsamples}, seq_len={seq_len}). "
            f"Collected {total_tokens} tokens."
        )

    token_stream = torch.cat(token_chunks, dim=0)[:target_tokens]
    batches: List[Dict[str, torch.Tensor]] = []
    n_full = target_tokens // seq_len
    rem = target_tokens % seq_len

    if n_full > 0:
        windows = token_stream[: n_full * seq_len].view(n_full, seq_len).contiguous()
        attn = torch.ones_like(windows, dtype=torch.long)
        for i in range(0, n_full, batch_size):
            batches.append(
                {
                    "input_ids": windows[i : i + batch_size].cpu(),
                    "attention_mask": attn[i : i + batch_size].cpu(),
                }
            )

    if rem > 0:
        tail = token_stream[n_full * seq_len :].view(1, rem).contiguous()
        tail_attn = torch.ones_like(tail, dtype=torch.long)
        batches.append(
            {
                "input_ids": tail.cpu(),
                "attention_mask": tail_attn.cpu(),
            }
        )

    return batches


def get_qwen_rotary_emb(model: nn.Module) -> Optional[nn.Module]:
    """
    Best-effort lookup for model-level rotary embedding module used by Qwen variants.
    """
    cand = getattr(getattr(model, "model", None), "rotary_emb", None)
    if cand is not None:
        return cand
    cand = getattr(model, "rotary_emb", None)
    if cand is not None:
        return cand
    base = getattr(model, "base_model", None)
    if base is not None:
        cand = getattr(getattr(base, "model", None), "rotary_emb", None)
        if cand is not None:
            return cand
    tr = getattr(model, "transformer", None)
    if tr is not None:
        cand = getattr(tr, "rotary_emb", None)
        if cand is not None:
            return cand
    return None


def _to_device_any(x: Any, device: torch.device | str):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device_any(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _to_device_any(v, device) for k, v in x.items()}
    return x


def _to_cpu_any(x: Any):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu_any(t) for t in x)
    if isinstance(x, dict):
        return {k: _to_cpu_any(v) for k, v in x.items()}
    return x


def _layer_accepts_kw(layer: nn.Module, kw: str) -> bool:
    try:
        sig = inspect.signature(layer.forward)
        return kw in sig.parameters
    except Exception:
        return True


def _ensure_bsh(x: torch.Tensor, *, B_hint: Optional[int] = None) -> torch.Tensor:
    """
    Normalize hidden states to (B,S,H).
    """
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        if B_hint is not None and B_hint > 0 and (x.shape[0] % B_hint == 0):
            S = x.shape[0] // B_hint
            return x.view(B_hint, S, x.shape[1])
        return x.unsqueeze(0)
    return x


def _ensure_position_ids(x_bsh: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
    B, S, _ = x_bsh.shape
    dev = x_bsh.device
    if position_ids is not None and torch.is_tensor(position_ids):
        pid = position_ids.to(dev, non_blocking=True)
        if pid.ndim == 1:
            pid = pid.view(1, -1).expand(B, -1)
        return pid
    return torch.arange(S, device=dev, dtype=torch.long).view(1, S).expand(B, S)


def _compute_qwen_position_embeddings(
    rotary_emb: Optional[nn.Module],
    hidden_states_bsh: torch.Tensor,
    position_ids: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if rotary_emb is None or (not callable(rotary_emb)):
        return None

    for call in (
        lambda: rotary_emb(hidden_states_bsh, position_ids),
        lambda: rotary_emb(hidden_states_bsh, position_ids, hidden_states_bsh.shape[1]),
        lambda: rotary_emb(position_ids),
    ):
        try:
            out = call()
            if (
                isinstance(out, (tuple, list))
                and len(out) == 2
                and torch.is_tensor(out[0])
                and torch.is_tensor(out[1])
            ):
                return out[0], out[1]
        except TypeError:
            continue
        except Exception:
            continue
    return None


def _build_replay_kwargs_for_layer(
    *,
    layer: nn.Module,
    x_bsh: torch.Tensor,
    attention_mask: Any,
    position_ids_cached: Any,
    device: torch.device,
    rotary_emb: Optional[nn.Module],
) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}

    if _layer_accepts_kw(layer, "use_cache"):
        kw["use_cache"] = False
    if _layer_accepts_kw(layer, "past_key_values"):
        kw["past_key_values"] = None

    if attention_mask is not None and _layer_accepts_kw(layer, "attention_mask"):
        kw["attention_mask"] = _to_device_any(attention_mask, device)

    pid_cached_t = position_ids_cached if torch.is_tensor(position_ids_cached) else None
    pid = _ensure_position_ids(x_bsh, pid_cached_t)
    if _layer_accepts_kw(layer, "position_ids"):
        kw["position_ids"] = pid

    if _layer_accepts_kw(layer, "cache_position"):
        _, S, _ = x_bsh.shape
        kw["cache_position"] = torch.arange(S, device=x_bsh.device, dtype=torch.long)

    if _layer_accepts_kw(layer, "position_embeddings"):
        pe = _compute_qwen_position_embeddings(rotary_emb, x_bsh, pid)
        if pe is not None:
            kw["position_embeddings"] = pe

    return kw


@torch.no_grad()
def capture_layer0_replay_cache(
    model: nn.Module,
    token_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    max_rows: int,
    *,
    pin_cpu_memory: bool = True,
) -> List[Tuple[torch.Tensor, Any, Any]]:
    """
    Capture decoder-layer-0 inputs once, for sequential layer replay.
    Returns a list of tuples: (hidden_states_bsh_cpu, attention_mask_cpu_any, position_ids_cpu_any).
    """
    layers = get_decoder_layers(model)
    if len(layers) == 0:
        raise RuntimeError("No decoder layers found.")
    blk0 = layers[0]

    class _StopReplayCapture(Exception):
        pass

    cache: List[Tuple[torch.Tensor, Any, Any]] = []
    rows = 0
    max_rows = int(max(1, max_rows))

    def _capture_pre_hook(module, args, kwargs):
        nonlocal rows
        if rows >= max_rows:
            raise _StopReplayCapture()

        x = _ensure_bsh(args[0].detach())
        attn = kwargs.get("attention_mask", None)
        pos_ids = kwargs.get("position_ids", None)

        rows_needed = max_rows - rows
        B, S, H = x.shape
        if rows_needed <= 0:
            raise _StopReplayCapture()

        if B * S > rows_needed:
            full_batches = rows_needed // S
            if full_batches > 0:
                x = x[:full_batches, :, :]
                if torch.is_tensor(attn) and attn.shape[0] == B:
                    attn = attn[:full_batches, ...]
                if torch.is_tensor(pos_ids) and pos_ids.ndim >= 2 and pos_ids.shape[0] == B:
                    pos_ids = pos_ids[:full_batches, ...]
            else:
                keep_s = min(S, rows_needed)
                x = x[:1, :keep_s, :]
                if torch.is_tensor(attn):
                    if attn.ndim >= 2 and attn.shape[0] == B and attn.shape[1] == S:
                        attn = attn[:1, :keep_s, ...]
                    elif attn.ndim >= 4 and attn.shape[0] == B:
                        attn = attn[:1, :, :keep_s, :keep_s]
                    elif attn.shape[0] == B:
                        attn = attn[:1, ...]
                if torch.is_tensor(pos_ids):
                    if pos_ids.ndim >= 2 and pos_ids.shape[0] == B and pos_ids.shape[1] == S:
                        pos_ids = pos_ids[:1, :keep_s, ...]
                    elif pos_ids.shape[0] == B:
                        pos_ids = pos_ids[:1, ...]

        x_cpu = x.to(device="cpu", non_blocking=True).contiguous()
        if pin_cpu_memory:
            x_cpu = x_cpu.pin_memory()

        attn_cpu = _to_cpu_any(attn)
        pos_cpu = _to_cpu_any(pos_ids)

        cache.append((x_cpu, attn_cpu, pos_cpu))
        rows += int(x_cpu.shape[0]) * int(x_cpu.shape[1])
        raise _StopReplayCapture()

    h = blk0.register_forward_pre_hook(_capture_pre_hook, with_kwargs=True)
    model.eval()
    try:
        with torch.inference_mode():
            for toks_cpu in token_batches:
                if rows >= max_rows:
                    break
                toks = {k: v.to(device, non_blocking=True) for k, v in toks_cpu.items()}
                try:
                    model(**toks)
                except _StopReplayCapture:
                    pass
                finally:
                    del toks
    finally:
        h.remove()

    if len(cache) == 0:
        raise RuntimeError("Failed to capture layer-0 replay cache; cache is empty.")
    return cache


@torch.no_grad()
def collect_layer_XW_from_replay_cache(
    model: nn.Module,
    replay_cache: List[Tuple[torch.Tensor, Any, Any]],
    layer_id: int,
    module_paths: Tuple[str, ...],
    device: torch.device,
    max_rows_X: int,
    *,
    store_activations_on_cpu: bool = False,
    pin_cpu_memory: bool = True,
    return_W_on_cpu: bool = False,
    rotary_emb: Optional[nn.Module] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect (X, W) for one decoder layer using cached layer inputs (sequential replay path).
    """
    layers = get_decoder_layers(model)
    blk = layers[layer_id]

    def _try_get_linear(mp: str) -> Optional[nn.Linear]:
        try:
            mod = get_submodule_by_path(blk, mp)
        except Exception:
            return None
        return mod if isinstance(mod, nn.Linear) else None

    group_candidates = [
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("mlp.gate_proj", "mlp.up_proj"),
    ]

    requested = [mp for mp in module_paths]
    groups: List[Tuple[str, ...]] = []
    grouped_set = set()
    for g in group_candidates:
        g_req = tuple([mp for mp in g if mp in requested])
        if len(g_req) >= 2:
            groups.append(g_req)
            grouped_set.update(g_req)
    solo_paths = [mp for mp in requested if mp not in grouped_set]

    catchers: Dict[str, ActivationCatcher] = {}
    mod_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _register(mp: str, catcher: ActivationCatcher) -> None:
        mod = _try_get_linear(mp)
        if mod is None:
            return
        h = mod.register_forward_hook(catcher.hook)
        mod_hooks.append(h)
        catchers[mp] = catcher

    for g in groups:
        first = g[0]
        if _try_get_linear(first) is None:
            continue
        catcher = ActivationCatcher(
            max_rows=max_rows_X,
            store_on_cpu=store_activations_on_cpu,
            pin_memory=pin_cpu_memory,
        )
        _register(first, catcher)
        for mp in g[1:]:
            if _try_get_linear(mp) is not None:
                catchers[mp] = catcher

    for mp in solo_paths:
        if _try_get_linear(mp) is None:
            continue
        _register(
            mp,
            ActivationCatcher(
                max_rows=max_rows_X,
                store_on_cpu=store_activations_on_cpu,
                pin_memory=pin_cpu_memory,
            ),
        )

    if not catchers:
        for h in mod_hooks:
            h.remove()
        return {}

    model.eval()
    try:
        with torch.inference_mode():
            for x_cpu, attn_cpu, pos_cpu in replay_cache:
                x = x_cpu.to(device, non_blocking=True)
                x = _ensure_bsh(x)
                kw = _build_replay_kwargs_for_layer(
                    layer=blk,
                    x_bsh=x,
                    attention_mask=attn_cpu,
                    position_ids_cached=pos_cpu,
                    device=device,
                    rotary_emb=rotary_emb,
                )
                _ = blk(x, **kw)
                del x, kw
    finally:
        for h in mod_hooks:
            h.remove()

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    X_cache: Dict[int, torch.Tensor] = {}
    for mp, catcher in catchers.items():
        if not catcher.X_chunks:
            continue

        cid = id(catcher)
        if cid not in X_cache:
            X = catcher.get_matrix(max_rows=max_rows_X).to(dtype=torch.float32)
            if not store_activations_on_cpu:
                X = X.to(device=device, non_blocking=False)
            X_cache[cid] = X.contiguous()

        mod = _try_get_linear(mp)
        if mod is None:
            continue

        W = mod.weight.detach().to(dtype=torch.float32).T.contiguous()
        if return_W_on_cpu:
            W = W.cpu()
        else:
            W = W.to(device=device, non_blocking=False)
        out[mp] = (X_cache[cid], W)

    return out


@torch.no_grad()
def replay_decoder_layer_from_cache(
    model: nn.Module,
    replay_cache: List[Tuple[torch.Tensor, Any, Any]],
    layer_id: int,
    device: torch.device,
    *,
    pin_cpu_memory: bool = True,
    rotary_emb: Optional[nn.Module] = None,
) -> List[Tuple[torch.Tensor, Any, Any]]:
    """
    Run one decoder layer on cached inputs and return next-layer inputs on CPU.
    """
    layers = get_decoder_layers(model)
    blk = layers[layer_id]

    out_cache: List[Tuple[torch.Tensor, Any, Any]] = []
    model.eval()
    with torch.inference_mode():
        for x_cpu, attn_cpu, pos_cpu in replay_cache:
            x = x_cpu.to(device, non_blocking=True)
            x = _ensure_bsh(x)
            kw = _build_replay_kwargs_for_layer(
                layer=blk,
                x_bsh=x,
                attention_mask=attn_cpu,
                position_ids_cached=pos_cpu,
                device=device,
                rotary_emb=rotary_emb,
            )
            y = blk(x, **kw)
            y0 = y[0] if isinstance(y, (tuple, list)) else y
            y0 = _ensure_bsh(y0, B_hint=x.shape[0])

            y_cpu = y0.detach().to("cpu", non_blocking=True).contiguous()
            if pin_cpu_memory:
                y_cpu = y_cpu.pin_memory()
            out_cache.append((y_cpu, attn_cpu, pos_cpu))
            del x, kw, y, y0, y_cpu

    return out_cache


@torch.no_grad()
def collect_layer_XW_with_early_stop(
    model: nn.Module,
    token_batches: List[Dict[str, torch.Tensor]],
    layer_id: int,
    module_paths: Tuple[str, ...],
    device: torch.device,
    max_rows_X: int,
    *,
    store_activations_on_cpu: bool = False,
    pin_cpu_memory: bool = True,
    return_W_on_cpu: bool = False,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect (X, W) pairs for requested module_paths in a decoder block,
    aborting each forward immediately after the block fires.

    Options:
      - store_activations_on_cpu: if True, ActivationCatcher moves chunks to CPU immediately.
      - pin_cpu_memory: if storing on CPU, pin the resulting CPU chunks.
      - return_W_on_cpu: return weight matrix W on CPU fp32 (recommended). If False, returns on `device`.

    Returned X:
      - if store_activations_on_cpu=True  => CPU (possibly pinned) tensor
      - else                             => typically GPU tensor
    """
    layers = get_decoder_layers(model)
    blk = layers[layer_id]

    def _try_get_linear(mp: str) -> Optional[nn.Linear]:
        try:
            mod = get_submodule_by_path(blk, mp)
        except Exception:
            return None
        return mod if isinstance(mod, nn.Linear) else None

    group_candidates = [
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("mlp.gate_proj", "mlp.up_proj"),
    ]

    requested = [mp for mp in module_paths]

    groups: List[Tuple[str, ...]] = []
    grouped_set = set()
    for g in group_candidates:
        g_req = tuple([mp for mp in g if mp in requested])
        if len(g_req) >= 2:
            groups.append(g_req)
            grouped_set.update(g_req)

    solo_paths = [mp for mp in requested if mp not in grouped_set]

    catchers: Dict[str, ActivationCatcher] = {}
    mod_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _register(mp: str, catcher: ActivationCatcher) -> None:
        mod = _try_get_linear(mp)
        if mod is None:
            return
        h = mod.register_forward_hook(catcher.hook)
        mod_hooks.append(h)
        catchers[mp] = catcher

    # Grouped paths share one catcher
    for g in groups:
        first = g[0]
        mod_first = _try_get_linear(first)
        if mod_first is None:
            continue
        catcher = ActivationCatcher(
            max_rows=max_rows_X,
            store_on_cpu=store_activations_on_cpu,
            pin_memory=pin_cpu_memory,
        )
        _register(first, catcher)
        for mp in g[1:]:
            if _try_get_linear(mp) is not None:
                catchers[mp] = catcher

    # Solo paths each get their own catcher
    for mp in solo_paths:
        if _try_get_linear(mp) is None:
            continue
        _register(
            mp,
            ActivationCatcher(
                max_rows=max_rows_X,
                store_on_cpu=store_activations_on_cpu,
                pin_memory=pin_cpu_memory,
            ),
        )

    if not catchers:
        for h in mod_hooks:
            h.remove()
        return {}

    class _Stop(Exception):
        pass

    def early_stop_hook(module, inp, out):
        raise _Stop()

    stop_hook = blk.register_forward_hook(early_stop_hook)

    model.eval()
    with torch.inference_mode():
        for toks_cpu in token_batches:
            # stop when all unique catchers are full
            all_full = True
            seen = set()
            for c in catchers.values():
                cid = id(c)
                if cid in seen:
                    continue
                seen.add(cid)
                if not c._full:
                    all_full = False
                    break
            if all_full:
                break

            toks = {k: v.to(device, non_blocking=True) for k, v in toks_cpu.items()}
            try:
                model(**toks)
            except _Stop:
                pass
            finally:
                del toks

    stop_hook.remove()
    for h in mod_hooks:
        h.remove()

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    X_cache: Dict[int, torch.Tensor] = {}

    for mp, catcher in catchers.items():
        if not catcher.X_chunks:
            continue

        cid = id(catcher)
        if cid not in X_cache:
            X = catcher.get_matrix(max_rows=max_rows_X).to(dtype=torch.float32)
            # If activations are on GPU and you want to keep them there, leave as-is.
            # If activations are on CPU, keep them contiguous.
            if not store_activations_on_cpu:
                X = X.to(device=device, non_blocking=False)
            X_cache[cid] = X.contiguous()


        mod = _try_get_linear(mp)
        if mod is None:
            continue

        W = mod.weight.detach().to(dtype=torch.float32).T.contiguous()
        if return_W_on_cpu:
            W = W.cpu()
        else:
            W = W.to(device=device, non_blocking=False)

        out[mp] = (X, W)

    return out

@torch.no_grad()
def restore_state_inplace(
    model: nn.Module,
    cpu_sd: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """
    Restore a CPU snapshot state_dict into an already-allocated model on `device`
    WITHOUT constructing a temporary GPU state_dict.

    This avoids a large transient peak from:
        {k: v.to(device) for k,v in cpu_sd.items()}
    and is usually both lower-peak-memory and faster.
    """
    model.eval()

    # Parameters
    for name, p in model.named_parameters():
        if name not in cpu_sd:
            continue
        src = cpu_sd[name]
        # copy_ keeps storage; avoids allocating a second full model
        p.copy_(src.to(device=device, dtype=p.dtype, non_blocking=True))

    # Buffers (some models use buffers for e.g. rotary caches, etc.)
    # Only restore those present in cpu_sd (most HF models have few/no buffers).
    for name, b in model.named_buffers():
        if name not in cpu_sd:
            continue
        src = cpu_sd[name]
        b.copy_(src.to(device=device, dtype=b.dtype, non_blocking=True))


# -------------------------
# HF token (optional)
# -------------------------
def _maybe_set_hf_token(hf_token: str) -> None:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


# -------------------------
# Utilities: dotted module paths
# -------------------------
def get_submodule_by_path(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for name in path.split("."):
        if not hasattr(cur, name):
            raise AttributeError(f"{type(cur).__name__} has no attribute '{name}' (path='{path}')")
        cur = getattr(cur, name)
    return cur


def set_submodule_by_path(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent_path, leaf = ".".join(parts[:-1]), parts[-1]
    parent = get_submodule_by_path(root, parent_path) if parent_path else root
    if not hasattr(parent, leaf):
        raise AttributeError(f"{type(parent).__name__} has no attribute '{leaf}' (path='{path}')")
    setattr(parent, leaf, new_module)


class _EarlyStop(Exception):
    """Raised by a hook to abort a forward pass immediately after the target layer fires."""
    pass

def _parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def print_layerwise_summary(stats: Dict[str, Dict[str, float]], title: str) -> None:
    keys = sorted(stats.keys())
    print(f"\n=== {title} ===")
    print("name | N | Ci | Co | f_base | f_sq | f_used | f_init | f_alt")
    for k in keys:
        s = stats[k]
        f_base = float(s.get("f_base", s.get("f_base_sum_group", float("nan"))))
        f_sq = float(s.get("f_sq", s.get("f_sq_sum_group", float("nan"))))
        f_used = float(s.get("f_used", s.get("f_used_sum_group", float("nan"))))
        f_init = float(s.get("f_init", s.get("f_init_sum_group", float("nan"))))
        f_alt = float(s.get("f_alt", s.get("f_alt_sum_group", float("nan"))))
        print(
            f"{k} | {int(s['N'])} | {int(s['Ci'])} | {int(s['Co'])} | "
            f"{f_base:.3e} | {f_sq:.3e} | {f_used:.3e} | "
            f"{f_init:.3e} | {f_alt:.3e}"
        )


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def save_coord_progress_plot(
    rel_obj: List[float],
    module_name: str,
    sweep_idx: int,
    out_path_base: Path,
) -> None:
    if len(rel_obj) == 0:
        return
    out_path_base.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, len(rel_obj) + 1))
    plt.figure()
    plt.plot(x, rel_obj, marker="o", linewidth=1)
    plt.xlabel("Coordinate optimized (rank by screening score)")
    plt.ylabel("Objective relative to initial (start of sweep)")
    plt.title(f"{module_name}\ncoord progress (sweep {sweep_idx})")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_x_over_s_surface_plot(
    X: torch.Tensor,
    s: torch.Tensor,
    module_name: str,
    sweep_idx: int,
    out_path_base: Path,
    *,
    selected_channels: Optional[List[int]] = None,
    updated_channels: Optional[List[int]] = None,
    max_tokens_plot: int = 256,
    max_in_channels_plot: int = 1024,
    elev: float = 25.0,
    azim: float = -60.0,
    dpi: int = 220,
    p_lo: float = 5.0,
    p_hi: float = 99.0,
    alpha: float = 0.85,
    cmap_name: str = "coolwarm",
    shade: bool = False,
) -> None:
    if X.ndim != 2 or s.ndim != 1 or X.shape[1] != s.shape[0]:
        raise ValueError(f"Expected X:(N,C) and s:(C,), got X={tuple(X.shape)} s={tuple(s.shape)}")

    T = min(int(max_tokens_plot), int(X.shape[0]))
    if selected_channels is not None and len(selected_channels) > 0:
        ch_idx = []
        seen = set()
        for jj in selected_channels:
            j = int(jj)
            if 0 <= j < int(X.shape[1]) and j not in seen:
                ch_idx.append(j)
                seen.add(j)
        if len(ch_idx) == 0:
            return
    else:
        C = min(int(max_in_channels_plot), int(X.shape[1]))
        ch_idx = list(range(C))
    C = len(ch_idx)
    if T <= 0 or C <= 0:
        return

    updated_count: Optional[int] = None
    if updated_channels is not None:
        updated_set = set()
        for jj in updated_channels:
            j = int(jj)
            if 0 <= j < int(X.shape[1]):
                updated_set.add(j)
        updated_count = sum(1 for j in ch_idx if j in updated_set)
    extra_count = None if updated_count is None else max(0, len(ch_idx) - int(updated_count))

    ch_idx_t = torch.tensor(ch_idx, device=X.device, dtype=torch.long)
    Xs = X[:T, :].index_select(1, ch_idx_t).to(torch.float32)
    ss = s.index_select(0, ch_idx_t).clamp_min(1e-8).to(torch.float32)
    Z = (Xs / ss.unsqueeze(0)).abs().detach().cpu().numpy()

    if Z.size == 0:
        return

    x_ch = np.asarray(ch_idx, dtype=np.float32)
    y_tok = np.arange(T, dtype=np.float32)
    Xgrid = np.repeat(x_ch.reshape(1, -1), T, axis=0)
    Ygrid = np.repeat(y_tok.reshape(-1, 1), C, axis=1)

    vmin = float(np.percentile(Z, p_lo))
    vmax = float(np.percentile(Z, p_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(Z))
        vmax = float(np.max(Z) + 1e-12)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Xgrid,
        Ygrid,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        cmap=cmap,
        norm=norm,
        shade=bool(shade),
    )
    surf.set_alpha(float(alpha))
    if updated_count is None:
        title_suffix = f"sweep {sweep_idx}, channels={len(ch_idx)}"
    else:
        title_suffix = (
            f"sweep {sweep_idx}, updated={int(updated_count)}, "
            f"extra={int(extra_count)}, total={len(ch_idx)}"
        )
    ax.set_title(f"{module_name}\n|X diag(s)^-1| ({title_suffix})", pad=12)
    ax.set_xlabel("In-Channel")
    ax.set_ylabel("Token")
    ax.set_zlabel("|X / s|")
    ax.view_init(elev=float(elev), azim=float(azim))
    cbar = fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("|X / s|")
    plt.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".pdf"), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    seq_len: int = 256,
    max_tokens: int = 50_000,
    batch_size: int = 4,
) -> float:
    clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        raise RuntimeError("No non-empty texts available for PPL evaluation.")

    # Match standard PTQ eval scripts more closely by evaluating one continuous
    # token stream rather than tokenizing each sample independently and
    # concatenating token ids afterward.
    joined = "\n\n".join(clean_texts)
    testenc = tokenizer(joined, return_tensors="pt")
    input_ids = testenc.input_ids
    if input_ids.numel() < seq_len + 2:
        raise RuntimeError("Not enough tokens to evaluate PPL (try increasing max_tokens or using more texts).")

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(device)

    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(f"Expected testenc.input_ids shape (1, T), got {tuple(input_ids.shape)}")

    total_tokens = input_ids.shape[1]
    if max_tokens is not None:
        total_tokens = min(total_tokens, int(max_tokens))
        input_ids = input_ids[:, :total_tokens]

    nsamples = input_ids.numel() // seq_len
    if nsamples == 0:
        raise ValueError(f"Not enough tokens ({input_ids.numel()}) for seq_len={seq_len}")

    input_ids = input_ids[:, : nsamples * seq_len].view(nsamples, seq_len)

    nll_sum = 0.0
    n_chunk_sum = 0

    batch_size = max(1, int(batch_size))

    def _extract_hidden_states(outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        if isinstance(outputs, list):
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        raise RuntimeError(f"Unable to extract hidden states from output type {type(outputs)!r}")

    def _forward_logits_reference_style(m: nn.Module, batch_ids: torch.Tensor) -> torch.Tensor:
        # Mirror the common PTQ evaluation path more closely by running the base
        # decoder/transformer and applying lm_head manually where possible.
        if hasattr(m, "model"):
            core = m.model
            if hasattr(core, "decoder"):
                outputs = core.decoder(batch_ids)
                hidden_states = _extract_hidden_states(outputs)
                return m.lm_head(hidden_states)
            try:
                outputs = core(batch_ids)
            except TypeError:
                outputs = core(input_ids=batch_ids)
            hidden_states = _extract_hidden_states(outputs)
            return m.lm_head(hidden_states)

        if hasattr(m, "transformer"):
            core = m.transformer
            try:
                outputs = core(batch_ids)
            except TypeError:
                outputs = core(input_ids=batch_ids)
            hidden_states = _extract_hidden_states(outputs)
            return m.lm_head(hidden_states)

        outputs = m(input_ids=batch_ids)
        if hasattr(outputs, "logits"):
            return outputs.logits
        raise RuntimeError(f"Unable to extract logits from output type {type(outputs)!r}")

    try:
        for start in tqdm(range(0, nsamples, batch_size), desc="(Eval) Batches", leave=False):
            batch = input_ids[start : start + batch_size].to(device)

            logits = _forward_logits_reference_style(model, batch)

            if not torch.isfinite(logits).all():
                bad = (~torch.isfinite(logits)).nonzero(as_tuple=False)[0].tolist()
                raise RuntimeError(f"Non-finite logits at eval batch starting {start}, example idx {bad}")

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].to(shift_logits.device).contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

            cur_chunks = int(batch.size(0))
            nll_sum += float(loss.float().item()) * float(cur_chunks * seq_len)
            n_chunk_sum += cur_chunks

        ppl = torch.exp(torch.tensor(nll_sum / max(1, n_chunk_sum * seq_len))).item()
    finally:
        model.config.use_cache = use_cache

    log.info("PPL: %.3f", ppl)
    return ppl

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SQ_DIR = ROOT / "SQ_Project" / "SQ_improved"
if str(SQ_DIR) not in sys.path:
    sys.path.insert(0, str(SQ_DIR))

import argparse
import json
import math
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quantization_utils import (
    quantize_per_token_maxabs_fp32,
    quantize_weights_per_out_channel_fp32,
)


def _maybe_set_pad_token(tokenizer, model=None) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and hasattr(model, "config"):
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id


def _load_config_without_stale_pre_rotation_metadata(
    model_ref: str,
    manifest: dict,
    trust_remote_code: bool,
):
    pre_rotations = manifest.get("pre_rotations", {})
    if not (isinstance(pre_rotations, dict) and bool(pre_rotations.get("applied", False))):
        return None

    config = AutoConfig.from_pretrained(
        model_ref,
        trust_remote_code=bool(trust_remote_code),
    )
    removed: list[str] = []
    for attr in ("quantization_config", "compression_config"):
        if not hasattr(config, attr):
            continue
        try:
            delattr(config, attr)
            removed.append(attr)
        except Exception:
            try:
                setattr(config, attr, None)
                removed.append(attr)
            except Exception:
                pass

    if removed:
        print(
            "[info] ignoring stale pre-rotation compression metadata during load: "
            + ", ".join(removed),
            flush=True,
        )
    return config


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _pick_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _parse_store_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


class SimQuantLinear(nn.Module):
    def __init__(
        self,
        *,
        weight_q_out_in: torch.Tensor,
        bias: torch.Tensor | None,
        act_bits: int,
        act_mode: str,
        eps: float,
        act_chunk_cols: int,
    ):
        super().__init__()
        self.act_bits = int(act_bits)
        self.act_mode = str(act_mode)
        self.eps = float(eps)
        self.act_chunk_cols = int(act_chunk_cols)
        self.register_buffer("weight_q", weight_q_out_in.contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.contiguous())

    @staticmethod
    @torch.no_grad()
    def from_linear(
        lin: nn.Linear,
        *,
        act_bits: int,
        act_mode: str,
        w_bits: int,
        w_mode: str,
        eps: float,
        act_chunk_cols: int,
        w_chunk_rows: int | None,
        store_dtype: torch.dtype,
    ) -> "SimQuantLinear":
        # quantization_utils expects (Ci, Co), while nn.Linear stores (Co, Ci)
        w_in_out = lin.weight.detach().to(torch.float32).T.contiguous()
        wq_in_out = quantize_weights_per_out_channel_fp32(
            w_in_out,
            bits=int(w_bits),
            eps=float(eps),
            chunk_rows=w_chunk_rows,
            mode=str(w_mode),
        )
        wq_out_in = wq_in_out.T.contiguous().to(store_dtype)
        b = None if lin.bias is None else lin.bias.detach().to(store_dtype)
        return SimQuantLinear(
            weight_q_out_in=wq_out_in,
            bias=b,
            act_bits=int(act_bits),
            act_mode=str(act_mode),
            eps=float(eps),
            act_chunk_cols=int(act_chunk_cols),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        dev = x.device

        x32 = x.to(torch.float32)
        xq32 = quantize_per_token_maxabs_fp32(
            x32,
            bits=self.act_bits,
            eps=self.eps,
            chunk_cols=self.act_chunk_cols,
            mode=self.act_mode,
        )
        w = self.weight_q.to(device=dev)
        xq = xq32.to(dtype=w.dtype)
        b = self.bias.to(device=dev) if (self.bias is not None) else None
        y = F.linear(xq, w, b)
        return y.to(x_dtype)


@torch.no_grad()
def simulate_quantize_linears_inplace(
    model: nn.Module,
    *,
    act_bits: int,
    act_mode: str,
    w_bits: int,
    w_mode: str,
    eps: float,
    act_chunk_cols: int,
    w_chunk_rows: int | None,
    store_dtype: torch.dtype,
    include_lm_head: bool,
) -> int:
    targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if (not include_lm_head) and name == "lm_head":
            continue
        targets.append(name)

    for name in targets:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        lin = getattr(parent, child_name)
        if not isinstance(lin, nn.Linear):
            continue
        qlin = SimQuantLinear.from_linear(
            lin,
            act_bits=int(act_bits),
            act_mode=str(act_mode),
            w_bits=int(w_bits),
            w_mode=str(w_mode),
            eps=float(eps),
            act_chunk_cols=int(act_chunk_cols),
            w_chunk_rows=w_chunk_rows,
            store_dtype=store_dtype,
        )
        setattr(parent, child_name, qlin)
    return len(targets)


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    seq_len: int = 256,
    max_tokens: int = 50_000,
) -> float:
    model.eval()

    ids_all: List[int] = []
    for t in texts:
        if not t or not t.strip():
            continue
        ids_all.extend(tokenizer(t, return_tensors=None)["input_ids"])
        if len(ids_all) >= max_tokens + seq_len + 1:
            break

    if len(ids_all) < seq_len + 2:
        raise RuntimeError("Not enough tokens to evaluate PPL.")

    ids_all = ids_all[: max_tokens + seq_len + 1]
    input_ids = torch.tensor(ids_all, device=device, dtype=torch.long)

    nll = 0.0
    n_tokens = 0

    for start in range(0, input_ids.numel() - 1, seq_len):
        end = min(start + seq_len, input_ids.numel() - 1)
        x = input_ids[start:end].unsqueeze(0)
        y = input_ids[start + 1 : end + 1].unsqueeze(0)

        out = model(input_ids=x)
        logits = out.logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        nll += float(loss.item())
        n_tokens += y.numel()
        if n_tokens >= max_tokens:
            break

    return math.exp(nll / max(n_tokens, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to saved model dir or HF model id")
    ap.add_argument("--base_model", type=str, default="", help="Base HF model id/path for patch-based loading")
    ap.add_argument("--patches_pt", type=str, default="", help="Optional patch file (.pt) containing SmoothQuant params")
    ap.add_argument("--tokenizer", type=str, default="", help="Tokenizer source path/id; defaults to --model")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=50_000)
    ap.add_argument("--dataset_name", type=str, default="wikitext")
    ap.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--simulate-quant", action="store_true", help="Fake-quantize nn.Linear layers during eval.")
    ap.add_argument("--sim_act_bits", type=int, default=8)
    ap.add_argument("--sim_w_bits", type=int, default=8)
    ap.add_argument(
        "--sim_act_mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric", "sym", "asym", "affine"],
        help="Activation quantization mode.",
    )
    ap.add_argument(
        "--sim_w_mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric", "sym", "asym", "affine"],
        help="Weight quantization mode.",
    )
    ap.add_argument("--sim_eps", type=float, default=1e-8)
    ap.add_argument("--sim_act_chunk_cols", type=int, default=1024)
    ap.add_argument(
        "--sim_w_chunk_rows",
        type=int,
        default=0,
        help="Row chunk size for weight quantization (0 = all rows).",
    )
    ap.add_argument(
        "--sim_store_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Storage dtype for quantized weights/bias in simulation wrappers.",
    )
    ap.add_argument(
        "--sim_include_lm_head",
        action="store_true",
        help="Also quantize lm_head in simulation mode.",
    )
    ap.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model/tokenizer code from HF repos (required by some Qwen models).",
    )
    args = ap.parse_args()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)
    print(f"[info] device={device} dtype={dtype}")

    model_ref = str(args.model)
    model_dir = Path(model_ref)
    manifest = {}
    if model_dir.is_dir():
        manifest_path = model_dir / "final_model_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

    base_model = str(args.base_model).strip() or str(manifest.get("base_model", "")).strip()

    patches_pt = str(args.patches_pt).strip()
    if not patches_pt:
        manifest_patch = str(manifest.get("final_patches_path", "")).strip()
        if manifest_patch:
            mp = Path(manifest_patch)
            if mp.exists():
                patches_pt = str(mp)
            elif model_dir.is_dir():
                mp2 = model_dir / Path(manifest_patch).name
                if mp2.exists():
                    patches_pt = str(mp2)

        if not patches_pt and model_dir.is_dir():
            local_patch = model_dir / "final_patches.pt"
            if local_patch.exists():
                patches_pt = str(local_patch)

    tokenizer_source = str(args.tokenizer).strip() or model_ref
    print(f"[info] loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    _maybe_set_pad_token(tokenizer, None)

    print("[info] loading model")
    t0 = time.time()
    if patches_pt:
        if not base_model:
            raise RuntimeError(
                "Patch-based load requested but base model is unknown. "
                "Pass --base_model or provide final_model_manifest.json with base_model."
            )
        print(f"[info] patch-based load: base_model={base_model} patches={patches_pt}")
        from utils import load_model_with_patches

        model = load_model_with_patches(
            model_name=base_model,
            patches_pt=patches_pt,
            device=device,
            torch_dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
    else:
        print(f"[info] standard load from {model_ref}")
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": bool(args.trust_remote_code),
        }
        config = _load_config_without_stale_pre_rotation_metadata(
            model_ref,
            manifest=manifest,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if config is not None:
            load_kwargs["config"] = config
        model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs).to(device)
        model.eval()
        model.config.use_cache = False

    if hasattr(model, "config"):
        model.config.use_cache = False
    _maybe_set_pad_token(tokenizer, model)

    if bool(args.simulate_quant):
        w_chunk_rows = None if int(args.sim_w_chunk_rows) <= 0 else int(args.sim_w_chunk_rows)
        n_wrapped = simulate_quantize_linears_inplace(
            model=model,
            act_bits=int(args.sim_act_bits),
            act_mode=str(args.sim_act_mode),
            w_bits=int(args.sim_w_bits),
            w_mode=str(args.sim_w_mode),
            eps=float(args.sim_eps),
            act_chunk_cols=int(args.sim_act_chunk_cols),
            w_chunk_rows=w_chunk_rows,
            store_dtype=_parse_store_dtype(str(args.sim_store_dtype)),
            include_lm_head=bool(args.sim_include_lm_head),
        )
        print(
            f"[info] simulate-quant enabled: wrapped {n_wrapped} Linear modules "
            f"(a{int(args.sim_act_bits)}:{args.sim_act_mode} "
            f"w{int(args.sim_w_bits)}:{args.sim_w_mode} "
            f"lm_head={int(bool(args.sim_include_lm_head))})"
        )

    print(f"[info] model loaded in {time.time() - t0:.1f}s")

    print(f"[info] loading {args.dataset_name}/{args.dataset_config} test split")
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import `datasets`. Install/repair Hugging Face datasets + pyarrow in this environment."
        ) from e

    ds_test = load_dataset(args.dataset_name, args.dataset_config, split="test")
    test_texts = [x["text"] for x in ds_test if isinstance(x.get("text", None), str)]
    if not test_texts:
        raise RuntimeError("No valid text rows found in test split.")

    ppl = eval_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=device,
        seq_len=int(args.seq_len),
        max_tokens=int(args.max_tokens),
    )
    print(f"[result] wikitext-2 test ppl = {ppl:.6f}")


if __name__ == "__main__":
    main()

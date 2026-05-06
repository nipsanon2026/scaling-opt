# utils.py
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from calibration_utils import (
    default_module_paths_for_qwen as default_module_paths_for_qwen_like,
    get_decoder_layers,
    get_submodule_by_path,
    set_submodule_by_path,
)
from quantization_utils import (
    canonicalize_quant_mode,
    quantize_per_token_maxabs_fp32,
    quantize_weights_per_out_channel_fp32,
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class PatchRecord:
    layer_id: int
    module_path: str
    act_bits: int
    w_bits: int
    s: torch.Tensor
    U_in_out: torch.Tensor


@dataclass
class SQCfg:
    act_bits: int = 4
    w_bits: int = 4
    act_quant_mode: str = "symmetric"
    w_quant_mode: str = "symmetric"

    pre_rotations: Tuple[str, ...] = ()
    pre_rotation_block_size: Optional[int] = None
    pre_rotation_type: str = "hadamard"
    pre_rotation_randomize: bool = False

    alt_iters: int = 1
    s_sweeps: int = 1
    golden_max_iter: int = 20
    s_bounds_mult: float = 10.0
    s_eps: float = 1e-8

    ls_ridge: float = 1e-4

    accept_tol: float = 0.0
    coord_backtrack_steps: int = 3
    outer_accept: bool = True

    alt_s_init: str = "heuristic"
    alt_step_order: str = "s_then_u"

    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01
    gptq_actorder: bool = False

    gptq_cholesky_max_tries: int = 12
    gptq_cholesky_damp_growth: float = 10.0
    gptq_cholesky_damp_add: float = 0.0
    gptq_cholesky_symmetrize: bool = True
    gptq_cholesky_verbose: bool = False

    show_tqdm: bool = True
    show_coord_tqdm: bool = False

    screen_gamma: float = 1.0
    screen_ste_rows: int = 8192
    screen_ste_rows_max: int = 200_000
    screen_ste_rows_min: int = 512
    screen_ste_center_grad: bool = True
    screen_ste_oom_retry: bool = True

    log_alt_objective_breakdown: bool = True

    ppl_after_each_module: bool = True
    ppl_after_each_layer: bool = True
    ppl_seq_len: int = 2048
    ppl_max_tokens: int = 50_000
    ppl_batch_size: int = 4

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    u_update: str = "gptq"

    unstable_rows_chunk_cols: int = 256
    unstable_rows_eval_row_chunk: int = 1024
    unstable_rows_eval_chunk_cols: int = 512
    unstable_rows_fused_branch_gemm: bool = True
    unstable_rows_prepack_groups: bool = True
    unstable_rows_prepack_max_mb: float = 512.0
    unstable_rows_fullcol_precompute: bool = True
    unstable_rows_fullcol_max_mb: float = 1536.0
    unstable_rows_eval_gather_once: bool = True
    unstable_rows_eval_gather_max_mb: float = 1024.0

    # calibration pipeline
    pipeline_use_sequential_replay: bool = True
    collect_X_on_cpu: bool = True
    collect_W_on_cpu: bool = True
    collect_pin_cpu_memory: bool = True
    pipeline_empty_cache_each_group: bool = False
    pipeline_empty_cache_each_layer: bool = False

    # memory knobs
    act_quant_chunk_cols: int = 512
    w_quant_chunk_rows: int = 2048
    act_top2_chunk_cols: int = 512
    w_top2_chunk_rows: int = 2048
    cB_row_chunk: int = 1024
    G_row_chunk: int = 1024

    unstable_rows_commit_row_chunk: int = 256
    unstable_rows_quant_col_chunk: int = 512
    G_tile: int = 1024
    G64_tile: int = 1024
    hard_free_between_commit_chunks: bool = False
    hard_free_after_commit: bool = False

    # correctness-first coordinate acceptance:
    # if True, use full streaming objective for per-iteration accept/reject checks
    coord_accept_use_exact_obj: bool = True


# ============================================================
# Patch I/O
# ============================================================

def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").to(torch.float32).contiguous()


def _save_torch_payload_robust(payload: Dict[str, Any], path: Path, *, try_zip_first: bool) -> None:
    tmp_root = path.parent if path.parent.exists() else Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    tmp_root.mkdir(parents=True, exist_ok=True)

    fd, local_tmp_str = tempfile.mkstemp(prefix="sq_patch_", suffix=".pt.tmp", dir=str(tmp_root))
    os.close(fd)
    local_tmp = Path(local_tmp_str)
    dst_tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if dst_tmp.exists():
            dst_tmp.unlink()
        if try_zip_first:
            try:
                torch.save(payload, local_tmp)
            except Exception as e_zip:
                if local_tmp.exists():
                    local_tmp.unlink()
                fd2, local_tmp_str2 = tempfile.mkstemp(prefix="sq_patch_", suffix=".pt.tmp", dir=str(tmp_root))
                os.close(fd2)
                local_tmp = Path(local_tmp_str2)
                print(
                    "[warn] patch save with zip serialization failed; "
                    "retrying with legacy torch.save format: "
                    f"{e_zip}"
                )
                torch.save(payload, local_tmp, _use_new_zipfile_serialization=False)
        else:
            torch.save(payload, local_tmp, _use_new_zipfile_serialization=False)
        shutil.copy2(local_tmp, dst_tmp)
        dst_tmp.replace(path)
    finally:
        if local_tmp.exists():
            local_tmp.unlink()
        if dst_tmp.exists():
            dst_tmp.unlink()

def save_patches_pt(
    out_dir: Path,
    run_id: str,
    method: str,
    model_name: str,
    module_paths: Tuple[str, ...],
    cfg: SQCfg,
    patch_records: List[PatchRecord],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}__patches__{method}.pt"
    shard_dir = out_dir / f"{run_id}__patches__{method}.d"

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    patch_entries: List[Dict[str, Any]] = []
    for idx, pr in enumerate(patch_records):
        shard_name = f"patch_{idx:04d}.pt"
        shard_payload = {
            "layer_id": pr.layer_id,
            "module_path": pr.module_path,
            "act_bits": pr.act_bits,
            "w_bits": pr.w_bits,
            "s": pr.s,
            "U_in_out": pr.U_in_out,
        }
        shard_path = shard_dir / shard_name
        _save_torch_payload_robust(shard_payload, shard_path, try_zip_first=False)
        patch_entries.append(
            {
                "layer_id": pr.layer_id,
                "module_path": pr.module_path,
                "act_bits": pr.act_bits,
                "w_bits": pr.w_bits,
                "shard_file": shard_name,
            }
        )

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "model_name": model_name,
        "module_paths": list(module_paths),
        "cfg": asdict(cfg),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patch_format": "sharded_v1",
        "patches_dir": shard_dir.name,
        "patches": patch_entries,
    }

    _save_torch_payload_robust(payload, path, try_zip_first=True)
    return path


def load_patches_payload(patches_pt: str | Path) -> Dict[str, Any]:
    ppath = Path(patches_pt)
    if not ppath.is_file():
        raise FileNotFoundError(f"Patches payload not found: {ppath}")
    try:
        payload = torch.load(ppath, map_location="cpu")
    except Exception as e:
        size_bytes = ppath.stat().st_size if ppath.exists() else -1
        raise RuntimeError(
            f"Failed to load patches payload '{ppath}' ({size_bytes} bytes). "
            "The file is likely truncated, corrupt, or was only partially copied."
        ) from e
    if not isinstance(payload, dict):
        raise RuntimeError("Bad patches payload: expected a dict.")

    if str(payload.get("patch_format", "")).strip() != "sharded_v1":
        return payload

    patches_dir_name = str(payload.get("patches_dir", "")).strip()
    if not patches_dir_name:
        raise RuntimeError("Bad sharded patches payload: missing 'patches_dir'.")
    patches_dir = ppath.parent / patches_dir_name
    if not patches_dir.is_dir():
        raise RuntimeError(f"Missing sharded patches directory: {patches_dir}")

    patch_entries = payload.get("patches", [])
    if not isinstance(patch_entries, list):
        raise RuntimeError("Bad sharded patches payload: expected list in key 'patches'.")

    materialized: List[Dict[str, Any]] = []
    for entry in patch_entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Bad sharded patches payload: each patch entry must be a dict.")
        shard_file = str(entry.get("shard_file", "")).strip()
        if not shard_file:
            raise RuntimeError("Bad sharded patches payload: missing 'shard_file'.")
        shard_path = patches_dir / shard_file
        try:
            shard_payload = torch.load(shard_path, map_location="cpu")
        except Exception as e:
            size_bytes = shard_path.stat().st_size if shard_path.exists() else -1
            raise RuntimeError(
                f"Failed to load patch shard '{shard_path}' ({size_bytes} bytes). "
                "The shard is likely truncated, corrupt, or was only partially copied."
            ) from e
        if not isinstance(shard_payload, dict):
            raise RuntimeError(f"Bad patch shard payload at {shard_path}")
        merged = dict(entry)
        merged.update(shard_payload)
        materialized.append(merged)

    out = dict(payload)
    out["patches"] = materialized
    return out


# ============================================================
# Model-family and grouping helpers
# ============================================================

def infer_model_family(model_name: str) -> str:
    name = str(model_name).lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    return "generic"


def default_module_paths_for_model_family(model_family: str) -> Tuple[str, ...]:
    fam = str(model_family).lower().strip()
    if fam in ("qwen", "llama", "generic"):
        return default_module_paths_for_qwen_like()
    raise ValueError(f"Unsupported model_family='{model_family}'.")


def default_module_groups_for_decoder(module_paths: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    remaining = set(module_paths)
    groups: List[Tuple[str, ...]] = []

    qkv = tuple(
        p for p in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj") if p in remaining
    )
    if qkv:
        groups.append(qkv)
        remaining -= set(qkv)

    gate_up = tuple(p for p in ("mlp.gate_proj", "mlp.up_proj") if p in remaining)
    if gate_up:
        groups.append(gate_up)
        remaining -= set(gate_up)

    for module_path in sorted(remaining):
        groups.append((module_path,))
    return groups


def _pick_shared_X(
    layer_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    group: Sequence[str],
) -> torch.Tensor:
    X0, _ = layer_data[group[0]]
    for module_path in group[1:]:
        Xp, _ = layer_data[module_path]
        if Xp.shape != X0.shape:
            raise RuntimeError(f"Group X shapes differ: {group[0]}={X0.shape}, {module_path}={Xp.shape}")
    return X0


def _runtime_w_chunk_rows(cfg: SQCfg) -> Optional[int]:
    rr = int(getattr(cfg, "runtime_w_quant_chunk_rows", getattr(cfg, "w_quant_chunk_rows", 0)))
    return rr if rr > 0 else None


# ============================================================
# Generic math
# ============================================================

def golden_section_search_1d(f, a: float, b: float, max_iter: int = 25) -> float:
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0

    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


# ============================================================
# Low-level tensor builders
# ============================================================

@torch.no_grad()
def build_Xq_from_s(
    X: torch.Tensor,
    s: torch.Tensor,
    act_bits: int,
    eps: float,
    act_chunk_cols: int = 1024,
    act_quant_mode: str = "symmetric",
) -> torch.Tensor:
    X_scaled = X / s.unsqueeze(0)
    return quantize_per_token_maxabs_fp32(
        X_scaled,
        bits=act_bits,
        eps=eps,
        chunk_cols=int(act_chunk_cols),
        mode=act_quant_mode,
    )


@torch.no_grad()
def build_U_from_s_What(
    W_hat: torch.Tensor,
    s: torch.Tensor,
    w_bits: int,
    eps: float,
    w_chunk_rows: Optional[int] = None,
    w_quant_mode: str = "symmetric",
) -> torch.Tensor:
    Z = W_hat * s.unsqueeze(1)
    return quantize_weights_per_out_channel_fp32(
        Z,
        bits=w_bits,
        eps=eps,
        chunk_rows=w_chunk_rows,
        mode=w_quant_mode,
    )


@torch.no_grad()
def compute_GU_chunked(G: torch.Tensor, U: torch.Tensor, chunk_cols: int = 256) -> torch.Tensor:
    Ci, Co = U.shape
    out = torch.empty((Ci, Co), device=U.device, dtype=torch.float32)
    cc = int(max(1, chunk_cols))
    for o0 in range(0, Co, cc):
        o1 = min(o0 + cc, Co)
        out[:, o0:o1] = G @ U[:, o0:o1]
    return out


# ============================================================
# Streaming objectives
# ============================================================

@torch.no_grad()
def obj_from_X_Xq_W_U_streaming(
    X: torch.Tensor,          # (N,Ci) fp32
    Xq: torch.Tensor,         # (N,Ci) fp32
    W_fp: torch.Tensor,       # (Ci,Co) fp32
    U: torch.Tensor,          # (Ci,Co) fp32
    *,
    row_chunk: int = 256,
) -> float:
    assert X.dtype == torch.float32 and Xq.dtype == torch.float32
    assert W_fp.dtype == torch.float32 and U.dtype == torch.float32
    N = X.shape[0]
    rc = int(max(1, row_chunk))

    acc = torch.zeros((), device=X.device, dtype=torch.float64)
    for r0 in range(0, N, rc):
        r1 = min(r0 + rc, N)
        Xb = X[r0:r1, :]
        Xqb = Xq[r0:r1, :]
        Yb = Xb @ W_fp
        Pb = Xqb @ U
        R = (Yb - Pb).to(torch.float64)
        acc += torch.sum(R * R)
        del Xb, Xqb, Yb, Pb, R
    return float(acc.item())


@torch.no_grad()
def compute_c_and_B_streaming(
    X: torch.Tensor,          # (N,Ci) fp32
    Xq: torch.Tensor,         # (N,Ci) fp32
    W_fp: torch.Tensor,       # (Ci,Co) fp32
    *,
    row_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert X.dtype == torch.float32 and Xq.dtype == torch.float32 and W_fp.dtype == torch.float32
    assert X.shape == Xq.shape
    N, Ci = X.shape
    Co = W_fp.shape[1]
    assert W_fp.shape[0] == Ci

    dev = X.device
    B64 = torch.zeros((Ci, Co), device=dev, dtype=torch.float64)
    c64 = torch.zeros((), device=dev, dtype=torch.float64)

    rc = int(max(1, row_chunk))
    for r0 in range(0, N, rc):
        r1 = min(r0 + rc, N)
        Xb = X[r0:r1, :]
        Xqb = Xq[r0:r1, :]
        Yb = Xb @ W_fp
        Yb64 = Yb.to(torch.float64)
        c64 += torch.sum(Yb64 * Yb64)
        B64 += Xqb.to(torch.float64).T @ Yb64
        del Xb, Xqb, Yb, Yb64

    return c64, B64


# ============================================================
# Tiled Gram builders
# ============================================================

@torch.no_grad()
def gram64_add_tiled_(
    G64: torch.Tensor,
    X64: torch.Tensor,
    *,
    tile: int = 1024,
    alpha: float = 1.0,
) -> None:
    assert G64.dtype == torch.float64 and X64.dtype == torch.float64
    assert G64.device == X64.device
    rb, Ci = X64.shape
    assert G64.shape == (Ci, Ci)

    t = int(max(64, tile))
    a = float(alpha)

    for i0 in range(0, Ci, t):
        i1 = min(i0 + t, Ci)
        Xi = X64[:, i0:i1]
        for j0 in range(i0, Ci, t):
            j1 = min(j0 + t, Ci)
            Xj = X64[:, j0:j1]

            blk = Xi.T @ Xj
            if a != 1.0:
                blk.mul_(a)

            G64[i0:i1, j0:j1].add_(blk)
            if j0 != i0:
                G64[j0:j1, i0:i1].add_(blk.T)
            del blk


@torch.no_grad()
def build_G64_streaming_tiled(
    Xq: torch.Tensor,
    *,
    row_chunk: int = 1024,
    tile: int = 1024,
) -> torch.Tensor:
    assert Xq.dtype == torch.float32
    N, Ci = Xq.shape
    dev = Xq.device

    G64 = torch.zeros((Ci, Ci), device=dev, dtype=torch.float64)

    rc = int(max(1, row_chunk))
    for r0 in range(0, N, rc):
        r1 = min(r0 + rc, N)
        Xqb64 = Xq[r0:r1, :].to(torch.float64)
        gram64_add_tiled_(G64, Xqb64, tile=int(tile), alpha=1.0)
        del Xqb64

    return G64


@torch.no_grad()
def gram32_add_tiled_(
    G32: torch.Tensor,
    X32: torch.Tensor,
    *,
    tile: int = 1024,
    alpha: float = 1.0,
) -> None:
    assert G32.dtype == torch.float32 and X32.dtype == torch.float32
    assert G32.device == X32.device
    rb, Ci = X32.shape
    assert G32.shape == (Ci, Ci)

    t = int(max(64, tile))
    a = float(alpha)

    for i0 in range(0, Ci, t):
        i1 = min(i0 + t, Ci)
        Xi = X32[:, i0:i1]
        for j0 in range(i0, Ci, t):
            j1 = min(j0 + t, Ci)
            Xj = X32[:, j0:j1]

            blk = Xi.T @ Xj
            if a != 1.0:
                blk.mul_(a)

            G32[i0:i1, j0:j1].add_(blk)
            if j0 != i0:
                G32[j0:j1, i0:i1].add_(blk.T)
            del blk


# ============================================================
# Patch module and deployment helpers
# ============================================================

class AltSQU(nn.Module):
    def __init__(
        self,
        *,
        s: torch.Tensor,
        U_in_out: torch.Tensor,
        act_bits: int,
        act_quant_mode: str,
        name: str,
        bias: torch.Tensor | None = None,
        store_dtype: torch.dtype = torch.bfloat16,
        debug: bool = False,
    ):
        super().__init__()
        self.act_bits = int(act_bits)
        self.act_quant_mode = canonicalize_quant_mode(act_quant_mode)
        self.name = str(name)
        self.debug = bool(debug)

        self.register_buffer("s", s.detach().to(torch.float32))
        self.register_buffer("U", U_in_out.detach().to(torch.float32).T.contiguous().to(store_dtype))

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(store_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        dev = x.device

        x32 = x.to(torch.float32)
        s32 = self.s.to(device=dev, dtype=torch.float32)

        if self.debug and (not torch.isfinite(x32).all()):
            raise RuntimeError(f"[{self.name}] non-finite input before quant")

        xq32 = quantize_per_token_maxabs_fp32(
            x32 / s32,
            bits=self.act_bits,
            eps=1e-8,
            chunk_cols=1024,
            mode=self.act_quant_mode,
        )

        if self.debug and (not torch.isfinite(xq32).all()):
            raise RuntimeError(f"[{self.name}] non-finite xq after quant")

        U = self.U.to(device=dev)
        xq = xq32.to(dtype=U.dtype)

        b = self.bias.to(device=dev) if (self.bias is not None) else None
        y = F.linear(xq, U, b)

        if self.debug and (not torch.isfinite(y).all()):
            raise RuntimeError(f"[{self.name}] non-finite output after linear")

        return y.to(x_dtype)


def patch_one_linear_altU(
    model: nn.Module,
    layer_id: int,
    module_path: str,
    s: torch.Tensor,
    U: torch.Tensor,
    cfg: SQCfg,
    debug: bool = False,
    store_dtype: torch.dtype = torch.bfloat16,
) -> None:
    layers = get_decoder_layers(model)
    blk = layers[layer_id]
    lin = get_submodule_by_path(blk, module_path)

    if not isinstance(lin, nn.Linear):
        raise RuntimeError(f"Expected nn.Linear at layers[{layer_id}].{module_path} to patch, got {type(lin)}")

    bias = lin.bias

    new_mod = AltSQU(
        s=s,
        U_in_out=U,
        act_bits=int(cfg.act_bits),
        act_quant_mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
        name=f"layers.{layer_id}.{module_path}",
        bias=bias,
        store_dtype=store_dtype,
        debug=debug,
    )
    set_submodule_by_path(blk, module_path, new_mod)


@torch.no_grad()
def apply_patches_to_model(model: nn.Module, patches_payload: Dict[str, Any]) -> None:
    patches = patches_payload.get("patches", [])
    if not isinstance(patches, list):
        raise RuntimeError("Bad patches payload: expected list in key 'patches'.")

    cfg_dict = patches_payload.get("cfg", {})
    act_bits_default = int(cfg_dict.get("act_bits", 4))

    for p in patches:
        layer_id = int(p["layer_id"])
        module_path = str(p["module_path"])
        act_bits = int(p.get("act_bits", act_bits_default))
        s = p["s"]
        U_in_out = p["U_in_out"]

        if not isinstance(s, torch.Tensor) or not isinstance(U_in_out, torch.Tensor):
            raise RuntimeError("Bad patches payload: s/U tensors missing or wrong type.")

        tmp_cfg = SQCfg(**cfg_dict) if isinstance(cfg_dict, dict) else SQCfg()
        tmp_cfg.act_bits = act_bits

        patch_one_linear_altU(
            model=model,
            layer_id=layer_id,
            module_path=module_path,
            s=s.to(next(model.parameters()).device),
            U=U_in_out.to(next(model.parameters()).device),
            cfg=tmp_cfg,
            debug=False,
        )


def load_model_with_patches(model_name: str, patches_pt: str, device: torch.device) -> nn.Module:
    payload = load_patches_payload(patches_pt)
    return load_model_with_patch_payload(model_name=model_name, payload=payload, device=device)


def load_model_with_patch_payload(model_name: str, payload: Dict[str, Any], device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    model.to(torch.float32)
    apply_patches_to_model(model, payload)
    return model

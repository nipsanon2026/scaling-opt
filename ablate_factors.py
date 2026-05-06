#!/usr/bin/env python3
"""
One-block ablation runner for SQ alternating coordinate updates.

This script intentionally calls functions from SQ_alternating.py so runtime/memory
reflect the real implementation path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import SQ_alternating as sq
from calibration_utils import (
    _maybe_set_hf_token,
    _maybe_set_pad_token,
    collect_layer_XW_with_early_stop,
    make_calib_texts,
    make_token_batches,
)
from utils import SQCfg, obj_from_X_Xq_W_U_streaming


def _max_rss_gb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 ** 3)
    return (rss * 1024.0) / (1024.0 ** 3)


def _cuda_peak_stats_gb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3),
        float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 3),
    )


def _reset_cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def _group_objective_sum(
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    s: torch.Tensor,
    U_list: Sequence[torch.Tensor],
    cfg: SQCfg,
) -> float:
    Xq = sq.build_Xq_from_s_streaming(
        X,
        s,
        act_bits=int(cfg.act_bits),
        eps=float(cfg.s_eps),
        act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
        mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
    )
    f = 0.0
    for W_fp, U in zip(W_fp_list, U_list):
        f += obj_from_X_Xq_W_U_streaming(
            X=X,
            Xq=Xq,
            W_fp=W_fp,
            U=U,
            row_chunk=int(getattr(cfg, "obj_row_chunk", 256)),
        )
    return float(f)


@torch.no_grad()
def _initial_U_list(
    W_fp_list: Sequence[torch.Tensor],
    s: torch.Tensor,
    cfg: SQCfg,
) -> List[torch.Tensor]:
    w_chunk_rows = sq._runtime_w_chunk_rows(cfg)
    return [
        sq.build_U_from_s_What(
            W_fp,
            s,
            w_bits=cfg.w_bits,
            eps=cfg.s_eps,
            w_chunk_rows=w_chunk_rows,
            w_quant_mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
        )
        for W_fp in W_fp_list
    ]


@torch.no_grad()
def _run_case(
    case_name: str,
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    cfg_base: SQCfg,
    *,
    disable_active_set: bool,
    disable_instability_screening: bool,
    disable_efficient_objective: bool,
    s_init_mode: str,
    max_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = SQCfg(**asdict(cfg_base))
    cfg.ablate_disable_active_set = bool(disable_active_set)
    cfg.ablate_disable_instability_screening = bool(disable_instability_screening)
    cfg.ablate_disable_efficient_objective = bool(disable_efficient_objective)
    cfg.ablate_dense_precompute_y = bool(getattr(cfg_base, "ablate_dense_precompute_y", False))
    cfg.s_cd_max_steps = int(max_steps)

    eps = float(cfg.s_eps)
    if s_init_mode == "heuristic":
        s0 = sq.smoothquant_heuristic_s_group(X, W_fp_list, alpha=0.5, normalize_geom_mean=True)
    else:
        s0 = torch.ones((X.shape[1],), device=X.device, dtype=torch.float32)
    s0 = s0.clamp_min(eps)

    W_hat_list = [W_fp.detach().clone().to(torch.float32) for W_fp in W_fp_list]

    U0_list = _initial_U_list(W_fp_list=W_fp_list, s=s0, cfg=cfg)
    obj_start = _group_objective_sum(X=X, W_fp_list=W_fp_list, s=s0, U_list=U0_list, cfg=cfg)

    rss_before = _max_rss_gb()
    _reset_cuda_peaks(device)
    t0 = time.perf_counter()

    s_out, U_out_list = sq.update_s_coordinate_descent_golden_both_quant_group(
        X=X,
        W_fp_list=W_fp_list,
        s=s0,
        W_hat_list=W_hat_list,
        cfg=cfg,
        plots_dir=None,
        module_name=f"ablation::{case_name}",
    )

    elapsed = float(time.perf_counter() - t0)
    peak_alloc_gb, peak_reserved_gb = _cuda_peak_stats_gb(device)
    rss_after = _max_rss_gb()

    obj_end = _group_objective_sum(X=X, W_fp_list=W_fp_list, s=s_out, U_list=U_out_list, cfg=cfg)

    mode = "optimized"
    if bool(disable_efficient_objective):
        mode = "exact_no_eff_recompute"
    elif bool(disable_instability_screening):
        mode = "exact_streaming_recompute"

    coord_stats = getattr(cfg, "ablate_last_coord_stats", {})
    attempted_steps = int(coord_stats.get("attempted_total", 0))
    active_init_total = int(coord_stats.get("active_init_total", 0))
    active_final_total = int(coord_stats.get("active_final_total", 0))
    sweeps = int(coord_stats.get("sweeps", int(cfg.s_sweeps)))
    ci = int(coord_stats.get("ci", int(X.shape[1])))
    active_max_size = int(coord_stats.get("active_max_size", int(ci)))
    stopped_on_active_cap = bool(coord_stats.get("stopped_on_active_cap", False))
    active_max_frac = float(coord_stats.get("active_max_frac", float(getattr(cfg, "screen_active_max_frac", 0.10))))
    if attempted_steps <= 0:
        attempted_steps = int(max_steps) if int(max_steps) > 0 else int(ci) * max(1, sweeps)
    expected_full = int(ci) * max(1, sweeps)
    if bool(disable_active_set) and attempted_steps != expected_full:
        raise RuntimeError(
            f"disable_active_set expected full traversal (attempted={attempted_steps}, expected={expected_full})."
        )

    return {
        "case": case_name,
        "disable_active_set": bool(disable_active_set),
        "disable_instability_screening": bool(disable_instability_screening),
        "disable_efficient_objective": bool(disable_efficient_objective),
        "mode": mode,
        "attempted_steps": attempted_steps,
        "active_init_total": active_init_total,
        "active_final_total": active_final_total,
        "sweeps": sweeps,
        "ci": ci,
        "expected_full_steps": expected_full,
        "active_max_size": active_max_size,
        "active_max_frac": active_max_frac,
        "stopped_on_active_cap": stopped_on_active_cap,
        "seconds_total": elapsed,
        "seconds_per_step": elapsed / float(max(1, attempted_steps)),
        "peak_cuda_alloc_gb": float(peak_alloc_gb),
        "peak_cuda_reserved_gb": float(peak_reserved_gb),
        "max_rss_gb": float(rss_after),
        "max_rss_delta_gb": float(max(0.0, rss_after - rss_before)),
        "obj_start": float(obj_start),
        "obj_end": float(obj_end),
        "obj_delta": float(obj_end - obj_start),
    }


def _build_cases(args: argparse.Namespace) -> List[Tuple[str, bool, bool, bool]]:
    if bool(args.run_all_combos):
        out = []
        for das, dis, deo in itertools.product([False, True], repeat=3):
            out.append((f"as{int(not das)}_instab{int(not dis)}_effobj{int(not deo)}", das, dis, deo))
        return out

    if bool(args.run_cumulative):
        return [
            ("everything", False, False, False),
            ("w_o_active_set", True, False, False),
            ("w_o_active_set_instability", True, True, False),
            ("w_o_active_set_instability_effobj", True, True, True),
        ]

    if bool(args.run_matrix):
        return [
            ("baseline", False, False, False),
            ("no_active_set", True, False, False),
            ("no_instability_screening", False, True, False),
            ("no_efficient_objective", False, False, True),
        ]

    return [(
        "custom",
        bool(args.disable_active_set),
        bool(args.disable_instability_screening),
        bool(args.disable_efficient_objective),
    )]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen", "llama", "generic"])

    ap.add_argument("--layer_id", type=int, default=0)
    ap.add_argument("--group_index", type=int, default=0)

    ap.add_argument("--calib_rows", type=int, default=512)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_length", type=int, default=128)
    ap.add_argument("--calib_max_rows_X", type=int, default=80_000)

    ap.add_argument("--act_bits", type=int, default=4)
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--s_sweeps", type=int, default=1)
    ap.add_argument("--golden_max_iter", type=int, default=20)
    ap.add_argument("--coord_backtrack_steps", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0, help="Cap attempted coordinate updates per sweep (0 = no cap).")
    ap.add_argument("--screen_active_max_frac", type=float, default=0.10)
    ap.add_argument(
        "--dense_precompute_y",
        action="store_true",
        help="For no_efficient_objective mode, precompute Y=X@W once (faster, less strict ablation).",
    )
    ap.add_argument(
        "--no_eff_row_chunk",
        type=int,
        default=64,
        help="Row chunk for no_efficient_objective exact recompute path (smaller = slower).",
    )
    ap.add_argument(
        "--no_eff_no_dense_materialize",
        action="store_true",
        help="Disable dense buffer materialization overhead in no_efficient_objective mode.",
    )

    ap.add_argument("--s_init", type=str, default="heuristic", choices=["heuristic", "ones"])
    ap.add_argument("--hf_token", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="./ablation_one_block.csv")

    ap.add_argument("--disable_active_set", action="store_true")
    ap.add_argument("--disable_instability_screening", action="store_true")
    ap.add_argument("--disable_efficient_objective", action="store_true")
    ap.add_argument("--run_matrix", action="store_true")
    ap.add_argument("--run_cumulative", action="store_true")
    ap.add_argument("--run_all_combos", action="store_true")

    args = ap.parse_args()

    _maybe_set_hf_token(args.hf_token)
    # Delayed imports so --help remains usable if optional deps are broken.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    cfg_base = SQCfg(
        act_bits=int(args.act_bits),
        w_bits=int(args.w_bits),
        s_sweeps=int(args.s_sweeps),
        golden_max_iter=int(args.golden_max_iter),
        coord_backtrack_steps=int(args.coord_backtrack_steps),
        show_tqdm=False,
        show_coord_tqdm=False,
        ppl_after_each_module=False,
    )
    cfg_base.screen_active_max_frac = float(args.screen_active_max_frac)
    cfg_base.ablate_dense_precompute_y = bool(args.dense_precompute_y)
    cfg_base.ablate_no_eff_row_chunk = int(args.no_eff_row_chunk)
    cfg_base.ablate_no_eff_materialize_dense = not bool(args.no_eff_no_dense_materialize)

    model_name = str(args.model)
    model_family = sq.infer_model_family(model_name) if args.model_family == "auto" else str(args.model_family)
    module_paths = sq.default_module_paths_for_model_family(model_family)
    groups = sq.default_module_groups_for_decoder(module_paths)

    gi = int(args.group_index)
    if gi < 0 or gi >= len(groups):
        raise SystemExit(f"group_index out of range: {gi} (num_groups={len(groups)})")
    group = groups[gi]

    print(f"[info] device={device}")
    print(f"[info] model={model_name}")
    print(f"[info] model_family={model_family}")
    print(f"[info] layer_id={int(args.layer_id)} group_index={gi} group={group}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _maybe_set_pad_token(tokenizer, None)

    ds_train = load_dataset(cfg_base.dataset_name, cfg_base.dataset_config, split="train")
    calib_texts = make_calib_texts(ds_train, n_rows=int(args.calib_rows))
    token_batches = make_token_batches(
        tokenizer=tokenizer,
        texts=calib_texts,
        batch_size=int(args.calib_batch_size),
        max_length=int(args.calib_max_length),
    )

    print("[info] Loading model ...")
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    _maybe_set_pad_token(tokenizer, model)
    print(f"[info] model loaded in {time.perf_counter() - t_load:.1f}s")

    n_params = sum(int(p.numel()) for p in model.parameters())
    if n_params > 30_000_000_000:
        cfg_base.runtime_w_quant_chunk_rows = int(max(1, getattr(cfg_base, "w_quant_chunk_rows", 2048)))
    else:
        cfg_base.runtime_w_quant_chunk_rows = 0

    print("[info] Collecting one-block X/W ...")
    layer_data = collect_layer_XW_with_early_stop(
        model=model,
        token_batches=token_batches,
        layer_id=int(args.layer_id),
        module_paths=tuple(group),
        device=device,
        max_rows_X=int(args.calib_max_rows_X),
        store_activations_on_cpu=True,
        pin_cpu_memory=True,
        return_W_on_cpu=True,
    )

    if not all(p in layer_data for p in group):
        missing = [p for p in group if p not in layer_data]
        raise RuntimeError(f"Missing collected modules in group: {missing}")

    X_host = sq._pick_shared_X(layer_data, tuple(group))
    W_host_list = [layer_data[p][1] for p in group]
    X = X_host.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    W_fp_list = [W.to(device=device, dtype=torch.float32, non_blocking=True).contiguous() for W in W_host_list]

    print(f"[info] X shape={tuple(X.shape)} Ci={X.shape[1]} group_size={len(group)}")

    cases = _build_cases(args)
    rows: List[Dict[str, Any]] = []

    for case_name, das, dis, deo in cases:
        print("\n============================================================")
        print(f"[case] {case_name}")
        print(
            f"  disable_active_set={das} "
            f"disable_instability_screening={dis} "
            f"disable_efficient_objective={deo}"
        )

        row = _run_case(
            case_name=case_name,
            X=X,
            W_fp_list=W_fp_list,
            cfg_base=cfg_base,
            disable_active_set=das,
            disable_instability_screening=dis,
            disable_efficient_objective=deo,
            s_init_mode=str(args.s_init),
            max_steps=int(args.max_steps),
            device=device,
        )
        rows.append(row)
        print(
            f"[result] secs={row['seconds_total']:.3f} "
            f"secs/step={row['seconds_per_step']:.6f} "
            f"attempted={row['attempted_steps']} "
            f"active_init_total={row['active_init_total']} "
            f"active_final_total={row['active_final_total']} "
            f"active_cap={row['active_max_size']}({row['active_max_frac']:.3f}*Ci) "
            f"stopped_on_cap={int(bool(row['stopped_on_active_cap']))} "
            f"peak_alloc={row['peak_cuda_alloc_gb']:.3f}GB "
            f"peak_reserved={row['peak_cuda_reserved_gb']:.3f}GB "
            f"obj_delta={row['obj_delta']:.3e}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case",
        "disable_active_set",
        "disable_instability_screening",
        "disable_efficient_objective",
        "mode",
        "attempted_steps",
        "active_init_total",
        "active_final_total",
        "sweeps",
        "ci",
        "expected_full_steps",
        "active_max_size",
        "active_max_frac",
        "stopped_on_active_cap",
        "seconds_total",
        "seconds_per_step",
        "peak_cuda_alloc_gb",
        "peak_cuda_reserved_gb",
        "max_rss_gb",
        "max_rss_delta_gb",
        "obj_start",
        "obj_end",
        "obj_delta",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print("\n============================================================")
    print(f"[saved] {out_path}")
    print("[summary]")
    for r in rows:
        print(
            f"  {r['case']:>24s} | mode={r['mode']} | t={r['seconds_total']:.3f}s "
            f"| attempted={r['attempted_steps']} "
            f"| cap={r['active_max_size']} "
            f"| cap_stop={int(bool(r['stopped_on_active_cap']))} "
            f"| t/step={r['seconds_per_step']:.6f}s "
            f"| peak_alloc={r['peak_cuda_alloc_gb']:.3f}GB"
        )


if __name__ == "__main__":
    main()

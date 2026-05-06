#!/usr/bin/env python3
"""
Grouped alternating SmoothQuant driver.

This version keeps the file focused on the high-level alternating flow:
- model/module grouping
- per-group calibration and patching
- outer alternating optimization orchestration
- CLI entrypoint

Low-level active-set, Gram/objective maintenance, and efficient commit logic
now live in dedicated helper modules.
"""

from __future__ import annotations

import argparse
import builtins
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from active_set import gd_init_s_ste_group, smoothquant_heuristic_s_group
from calibration_utils import (
    _ensure_out_dir,
    _maybe_set_hf_token,
    _maybe_set_pad_token,
    _parse_csv_list,
    _safe_name,
    _write_json,
    capture_layer0_replay_cache,
    collect_layer_XW_from_replay_cache,
    collect_layer_XW_with_early_stop,
    eval_ppl,
    get_decoder_layers,
    get_qwen_rotary_emb as get_decoder_rotary_emb,
    get_submodule_by_path,
    make_fixed_length_token_batches_from_dataset,
    print_layerwise_summary,
    replay_decoder_layer_from_cache,
    restore_state_inplace,
)
from cuda_peak_tracker import CUDAPeakTracker
from data_utils import get_c4, get_ptb, get_wikitext2
from efficient_updates import (
    update_s_coordinate_descent_golden_both_quant_group,
    update_U_from_current_group,
)
from eval_utils import ppl_eval as benchmark_ppl_eval, zeroshot_eval
from gptq import gptq_quantize_U_from_H
from gram_utils import (
    build_G64_streaming_tiled,
    compute_c_and_B_streaming,
    group_obj_from_gram_stats_fp64,
    obj_from_gram_stats_fp32,
    ridge_ls_from_gram_fp32,
)
from quantization_utils import build_Xq_from_s_streaming
from utils import (
    PatchRecord,
    SQCfg,
    _pick_shared_X,
    _runtime_w_chunk_rows,
    _to_cpu_f32,
    build_U_from_s_What,
    default_module_groups_for_decoder,
    default_module_paths_for_model_family,
    infer_model_family,
    obj_from_X_Xq_W_U_streaming,
    patch_one_linear_altU,
    save_patches_pt,
)

def _parse_optional_positive_int(raw: str) -> int | None:
    value = str(raw).strip()
    if value == "" or value.lower() in {"auto", "none", "null"}:
        return None

    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "expected a positive integer or one of: auto, none"
        ) from e

    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _enable_hf_dataset_remote_code() -> None:
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    try:
        import datasets

        try:
            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        except Exception:
            pass
    except Exception:
        pass


@contextmanager
def _auto_accept_remote_code_prompts(enabled: bool = True):
    if not enabled:
        yield
        return

    original_input = builtins.input

    def _patched_input(prompt: str = "") -> str:
        prompt_l = str(prompt).lower()
        if (
            "trust_remote_code" in prompt_l
            or "custom code" in prompt_l
            or "do you wish to run the custom code" in prompt_l
            or "repository for" in prompt_l
        ):
            print("[info] auto-accepting remote-code prompt with 'y'", flush=True)
            return "y"
        print("[info] intercepted input prompt; replying 'y'", flush=True)
        return "y"

    builtins.input = _patched_input
    try:
        yield
    finally:
        builtins.input = original_input


def _run_final_ppl_suite(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    *,
    seq_len: int,
    batch_size: int,
    which: Sequence[str],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in which:
        key = str(name).strip().lower()
        try:
            if key == "wikitext2":
                print("[info] loading WikiText-2 eval set", flush=True)
                testenc = get_wikitext2(tokenizer, seqlen=seq_len, eval_mode=True)
            elif key == "c4":
                print("[info] loading C4 eval set", flush=True)
                testenc = get_c4(tokenizer, seqlen=seq_len, eval_mode=True)
            elif key == "ptb":
                print("[info] loading PTB eval set", flush=True)
                testenc = get_ptb(tokenizer, seqlen=seq_len, eval_mode=True)
            else:
                raise ValueError(f"Unsupported ppl dataset '{name}'. Use wikitext2, c4, or ptb.")

            ppl = benchmark_ppl_eval(model, testenc, seqlen=seq_len, batch_size=batch_size, dev=device)
            results[f"{key}_ppl"] = float(ppl)
            print(f"[result] {key} ppl = {ppl:.6f}", flush=True)
        except Exception as e:
            print(f"[warn] skipping {key} ppl eval: {e}", flush=True)
    return results


def _get_pre_rotation_list(cfg: SQCfg) -> Tuple[str, ...]:
    raw = getattr(cfg, "pre_rotations", ())
    if isinstance(raw, str):
        items = _parse_csv_list(raw)
    else:
        items = [str(x).strip() for x in raw if str(x).strip()]
    return tuple(items)


@torch.no_grad()
def _maybe_apply_pre_rotations(
    model: nn.Module,
    tokenizer,
    cfg: SQCfg,
) -> bool:
    rotations = _get_pre_rotation_list(cfg)
    if not rotations:
        return False

    try:
        try:
            from llmcompressor import oneshot
        except ImportError:
            from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.transform import SpinQuantModifier
    except ImportError as e:
        raise RuntimeError(
            "Pre-rotations were requested, but llmcompressor is not installed. "
            "Install llmcompressor to use --pre_rotations."
        ) from e

    transform_block_size = getattr(cfg, "pre_rotation_block_size", None)
    if transform_block_size is not None:
        transform_block_size = int(transform_block_size)
    modifier_kwargs = {
        "rotations": list(rotations),
        "transform_block_size": transform_block_size,
        "transform_type": str(getattr(cfg, "pre_rotation_type", "hadamard")),
    }
    if bool(getattr(cfg, "pre_rotation_randomize", False)):
        modifier_kwargs["randomize"] = True

    prev_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if prev_use_cache is not None:
        model.config.use_cache = False

    print(
        "[info] Applying SpinQuant pre-rotations: "
        f"rotations={list(rotations)} "
        f"block_size={modifier_kwargs['transform_block_size'] if modifier_kwargs['transform_block_size'] is not None else 'auto'} "
        f"type={modifier_kwargs['transform_type']} "
        f"randomize={bool(modifier_kwargs.get('randomize', False))}",
        flush=True,
    )
    t0 = time.time()
    try:
        try:
            oneshot(
                model=model,
                tokenizer=tokenizer,
                recipe=[SpinQuantModifier(**modifier_kwargs)],
                pipeline="datafree",
            )
        except TypeError as e:
            if "tokenizer" not in str(e):
                raise
            oneshot(
                model=model,
                recipe=[SpinQuantModifier(**modifier_kwargs)],
                pipeline="datafree",
            )
    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache

    print(f"[info] SpinQuant pre-rotations applied in {time.time() - t0:.2f}s", flush=True)
    return True


def _strip_transformers_compression_metadata(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

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
            "[info] Removed pre-rotation compression metadata before saving: "
            + ", ".join(removed),
            flush=True,
        )


def _default_hf_repo_id(model_name: str, run_id: str, final_method: str) -> str:
    model_slug = _safe_name(Path(str(model_name)).name or str(model_name))
    return f"{HF_OWNER}/{model_slug}__{_safe_name(run_id)}__{_safe_name(final_method)}"


def _push_final_model_dir_to_hub(
    final_model_dir: Path,
    *,
    repo_id: str,
    hf_token: str,
    commit_message: str,
) -> str:
    if not hf_token:
        raise RuntimeError("Hugging Face token is empty; cannot upload final model.")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed in this environment; cannot upload final model."
        ) from e

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(final_model_dir),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def _make_temp_artifact_dir(run_id: str, final_method: str) -> Path:
    tmp_root = Path(
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    tmp_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"{_safe_name(run_id)}__{_safe_name(final_method)}__",
            dir=str(tmp_root),
        )
    )


@torch.no_grad()
def alternating_optimize_s_and_What_group(
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    s_init: torch.Tensor,
    cfg: SQCfg,
    plots_dir: Optional[Path] = None,
    module_name: str = "",
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], float, float]:
    """High-level grouped alternating loop with configurable step order."""
    eps = cfg.s_eps
    s = s_init.detach().clone().clamp_min(eps)

    W_hat_list = [W_fp.detach().clone().to(torch.float32) for W_fp in W_fp_list]

    act_chunk = int(getattr(cfg, "act_quant_chunk_cols", 512))
    w_chunk_rows = _runtime_w_chunk_rows(cfg)
    Xq = build_Xq_from_s_streaming(
        X,
        s,
        act_bits=cfg.act_bits,
        eps=eps,
        act_chunk_cols=act_chunk,
        mode=cfg.act_quant_mode,
    )
    U_list = [
        build_U_from_s_What(
            W_hat,
            s,
            w_bits=cfg.w_bits,
            eps=eps,
            w_chunk_rows=w_chunk_rows,
            w_quant_mode=cfg.w_quant_mode,
        )
        for W_hat in W_hat_list
    ]

    obj_cur = 0.0
    for W_fp, U in zip(W_fp_list, U_list):
        obj_cur += obj_from_X_Xq_W_U_streaming(
            X=X, Xq=Xq, W_fp=W_fp, U=U, row_chunk=int(getattr(cfg, "obj_row_chunk", 256))
        )
    init_obj = float(obj_cur)

    best_obj = float(obj_cur)
    best_s = s.clone()
    best_W_hat_list = [w.clone() for w in W_hat_list]
    best_U_list = [u.clone() for u in U_list]

    it_range = range(cfg.alt_iters)
    pbar = tqdm(it_range, desc="Alternating (outer, group)", leave=False) if cfg.show_tqdm else None
    if pbar is not None:
        it_range = pbar

    def _recompute_obj_cur(
        X_cur: torch.Tensor,
        Xq_cur: torch.Tensor,
        U_cur_list: Sequence[torch.Tensor],
    ) -> float:
        obj = 0.0
        for W_fp, U_cur in zip(W_fp_list, U_cur_list):
            obj += obj_from_X_Xq_W_U_streaming(
                X=X_cur, Xq=Xq_cur, W_fp=W_fp, U=U_cur, row_chunk=int(getattr(cfg, "obj_row_chunk", 256))
            )
        return float(obj)

    def _run_s_step() -> Tuple[float, float, bool]:
        nonlocal s, Xq, U_list, obj_cur
        obj_before = float(obj_cur)
        s_old = s.clone()
        Xq_old = Xq
        U_old_list = [u.clone() for u in U_list]

        s_cand, U_cand_list = update_s_coordinate_descent_golden_both_quant_group(
            X, W_fp_list, s.clone(), W_hat_list, cfg, plots_dir=plots_dir, module_name=module_name
        )
        Xq_cand = build_Xq_from_s_streaming(
            X,
            s_cand,
            act_bits=cfg.act_bits,
            eps=eps,
            act_chunk_cols=act_chunk,
            mode=cfg.act_quant_mode,
        )

        obj_s = _recompute_obj_cur(X, Xq_cand, U_cand_list)
        s_accepted = bool((not cfg.outer_accept) or ((obj_s + cfg.accept_tol) < obj_before))
        if s_accepted:
            s, Xq, U_list, obj_cur = s_cand, Xq_cand, U_cand_list, float(obj_s)
        else:
            s, Xq, U_list = s_old, Xq_old, U_old_list
            obj_cur = _recompute_obj_cur(X, Xq, U_list)
        return obj_before, float(obj_cur), bool(s_accepted)

    def _run_u_step() -> Tuple[float, float, bool]:
        nonlocal W_hat_list, U_list, obj_cur
        obj_before = float(obj_cur)
        g_row_chunk = int(getattr(cfg, "G_row_chunk", getattr(cfg, "cB_row_chunk", 1024)))
        g64_tile = int(getattr(cfg, "G64_tile", 1024))
        G64 = build_G64_streaming_tiled(Xq, row_chunk=g_row_chunk, tile=g64_tile)
        G = G64.to(torch.float32)
        branch_stats = [
            compute_c_and_B_streaming(
                X,
                Xq,
                W_fp,
                row_chunk=int(getattr(cfg, "cB_row_chunk", 1024)),
            )
            for W_fp in W_fp_list
        ]
        W_hat_new_list, U_new_list, obj_U, accU = update_U_from_current_group(
            G=G,
            G64=G64,
            branch_stats=branch_stats,
            s=s,
            W_hat_list=W_hat_list,
            cfg=cfg,
            u_update=cfg.u_update,
        )
        if accU:
            W_hat_list, U_list, obj_cur = W_hat_new_list, U_new_list, float(obj_U)
        else:
            U_list = [
                build_U_from_s_What(
                    W_hat,
                    s,
                    w_bits=cfg.w_bits,
                    eps=eps,
                    w_chunk_rows=w_chunk_rows,
                    w_quant_mode=cfg.w_quant_mode,
                )
                for W_hat in W_hat_list
            ]
            obj_cur = group_obj_from_gram_stats_fp64(
                G64,
                branch_stats,
                U_list,
                col_chunk=int(getattr(cfg, "gu_obj_chunk_cols", 256)),
            )
        return obj_before, float(obj_cur), bool(accU)

    for _outer_it in it_range:
        outer_it = int(_outer_it) + 1
        step_order = str(getattr(cfg, "alt_step_order", "s_then_u")).strip().lower()
        if step_order == "u_then_s":
            obj_before_u, obj_after_u, accU = _run_u_step()
            obj_before_s, obj_after_s, s_accepted = _run_s_step()
        elif step_order == "s_then_u":
            obj_before_s, obj_after_s, s_accepted = _run_s_step()
            obj_before_u, obj_after_u, accU = _run_u_step()
        else:
            raise ValueError("alt_step_order must be one of {'s_then_u', 'u_then_s'}")

        if float(obj_cur) < best_obj:
            best_obj = float(obj_cur)
            best_s = s.clone()
            best_W_hat_list = [w.clone() for w in W_hat_list]
            best_U_list = [u.clone() for u in U_list]

        if bool(getattr(cfg, "log_alt_objective_breakdown", True)):
            print(
                f"[alt-outer {outer_it}/{int(cfg.alt_iters)}] "
                f"order={step_order} "
                f"before_s={obj_before_s:.3e} after_s={obj_after_s:.3e} s_acc={int(s_accepted)} "
                f"before_u={obj_before_u:.3e} after_u={obj_after_u:.3e} u_acc={int(accU)} best={best_obj:.3e}"
            )

        if pbar is not None:
            pbar.set_postfix(obj=f"{float(obj_cur):.3e}", best=f"{best_obj:.3e}", accU=int(accU))

    if pbar is not None:
        pbar.close()

    return best_s, best_W_hat_list, best_U_list, float(best_obj), float(init_obj)

@torch.no_grad()
def calibrate_and_patch_all_linear_method_grouped(
    model: nn.Module,
    token_batches: List[Dict[str, torch.Tensor]],
    cfg: SQCfg,
    device: torch.device,
    method: str,
    calib_max_rows_X: int,
    module_paths: Tuple[str, ...],
    ppl_seq_len: int,
    ppl_max_tokens: int,
    tokenizer=None,
    ppl_texts_train: Optional[List[str]] = None,
    ppl_texts_test: Optional[List[str]] = None,
    run_id: str = "",
) -> Tuple[Dict[str, Dict[str, float]], List[PatchRecord]]:
    """
    Group-aware version of your calibrate_and_patch_all_linear_method.

    Behavior:
      - collects layer_data for module_paths (unchanged)
      - forms groups (qkv, gate/up, rest singletons)
      - for each group, uses one shared X and optimizes shared s with summed objective
      - patches each module in group with the SAME s and its own U
    """
    if method not in ("s1", "s1_gptq", "heuristic", "heuristic_gptq", "alt"):
        raise ValueError("method must be one of {'s1','s1_gptq','heuristic','heuristic_gptq','alt'}")

    if cfg.alt_s_init not in ("heuristic", "ones", "GD"):
        raise ValueError("cfg.alt_s_init must be one of {'heuristic','ones'} for this grouped file")

    do_ppl_each = bool(getattr(cfg, "ppl_after_each_module", False))
    do_ppl_each_layer = bool(getattr(cfg, "ppl_after_each_layer", False))
    if do_ppl_each and (tokenizer is None or ppl_texts_train is None or ppl_texts_test is None):
        raise ValueError("Per-module PPL enabled but tokenizer/train_texts/test_texts not provided.")
    if do_ppl_each_layer and (tokenizer is None or ppl_texts_test is None):
        raise ValueError("Per-layer test PPL enabled but tokenizer/test_texts not provided.")

    plots_dir = Path("./plots") / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, float]] = {}
    patch_records: List[PatchRecord] = []

    layers = get_decoder_layers(model)
    n_layers = len(layers)

    outer = tqdm(range(n_layers), desc=f"Calibrate+patch layers [{method} grouped]", leave=True) if cfg.show_tqdm else range(n_layers)
    patched_count = 0

    groups = default_module_groups_for_decoder(module_paths)
    use_replay = bool(getattr(cfg, "pipeline_use_sequential_replay", True))
    replay_cache: Optional[List[Tuple[torch.Tensor, Any, Any]]] = None
    rotary_emb: Optional[nn.Module] = None
    peak_tracker = CUDAPeakTracker(device)

    def _accumulate_layer_peak(
        layer_peak_alloc: int,
        layer_peak_reserved: int,
    ) -> Tuple[int, int, int, int]:
        peak_alloc, peak_reserved = peak_tracker.snapshot()
        return (
            max(int(layer_peak_alloc), int(peak_alloc)),
            max(int(layer_peak_reserved), int(peak_reserved)),
            int(peak_alloc),
            int(peak_reserved),
        )

    def _maybe_trim_cuda_cache(tag: str) -> Tuple[int, int]:
        if device.type != "cuda" or (not torch.cuda.is_available()):
            return 0, 0
        cur_alloc, cur_reserved = peak_tracker.current()
        slack_bytes = int(cur_reserved) - int(cur_alloc)
        trim_slack_mb = float(getattr(cfg, "pipeline_trim_cache_slack_mb", 4096.0))
        if slack_bytes >= int(trim_slack_mb * 1024.0 * 1024.0):
            torch.cuda.empty_cache()
            cur_alloc, cur_reserved = peak_tracker.current()
            print(
                f"[cuda-cache-trim] {tag} "
                f"{CUDAPeakTracker.format_current(cur_alloc, cur_reserved)}"
            )
        return int(cur_alloc), int(cur_reserved)

    if use_replay:
        t_cache0 = time.time()
        try:
            replay_cache = capture_layer0_replay_cache(
                model=model,
                token_batches=token_batches,
                device=device,
                max_rows=int(calib_max_rows_X),
                pin_cpu_memory=bool(getattr(cfg, "collect_pin_cpu_memory", True)),
            )
            rotary_emb = get_decoder_rotary_emb(model)
            cache_rows = sum(int(x.shape[0]) * int(x.shape[1]) for x, _, _ in replay_cache)
            print(
                f"[replay-cache] captured {len(replay_cache)} batches "
                f"({cache_rows} rows) in {time.time() - t_cache0:.2f}s"
            )
        except Exception as e:
            replay_cache = None
            rotary_emb = None
            print(f"[warn] sequential replay cache setup failed; falling back to per-layer collection: {e}")

    for layer_id in outer:
        blk = layers[layer_id]
        layer_peak_alloc = 0
        layer_peak_reserved = 0
        peak_tracker.reset()

        t_collect = time.time()
        if replay_cache is not None:
            try:
                layer_data = collect_layer_XW_from_replay_cache(
                    model=model,
                    replay_cache=replay_cache,
                    layer_id=layer_id,
                    module_paths=module_paths,
                    device=device,
                    max_rows_X=calib_max_rows_X,
                    store_activations_on_cpu=bool(getattr(cfg, "collect_X_on_cpu", True)),
                    pin_cpu_memory=bool(getattr(cfg, "collect_pin_cpu_memory", True)),
                    return_W_on_cpu=bool(getattr(cfg, "collect_W_on_cpu", True)),
                    rotary_emb=rotary_emb,
                )
            except Exception as e:
                print(f"[warn] replay collect failed at layer {layer_id}; falling back to per-layer full forward: {e}")
                replay_cache = None
                rotary_emb = None
                layer_data = collect_layer_XW_with_early_stop(
                    model=model,
                    token_batches=token_batches,
                    layer_id=layer_id,
                    module_paths=module_paths,
                    device=device,
                    max_rows_X=calib_max_rows_X,
                    store_activations_on_cpu=bool(getattr(cfg, "collect_X_on_cpu", True)),
                    pin_cpu_memory=bool(getattr(cfg, "collect_pin_cpu_memory", True)),
                    return_W_on_cpu=bool(getattr(cfg, "collect_W_on_cpu", True)),
                )
        else:
            layer_data = collect_layer_XW_with_early_stop(
                model=model,
                token_batches=token_batches,
                layer_id=layer_id,
                module_paths=module_paths,
                device=device,
                max_rows_X=calib_max_rows_X,
                store_activations_on_cpu=bool(getattr(cfg, "collect_X_on_cpu", True)),
                pin_cpu_memory=bool(getattr(cfg, "collect_pin_cpu_memory", True)),
                return_W_on_cpu=bool(getattr(cfg, "collect_W_on_cpu", True)),
            )
        if layer_data:
            print(f"\n[layer {layer_id}] collected activations for {len(layer_data)} modules in {time.time()-t_collect:.2f}s")
        layer_peak_alloc, layer_peak_reserved, collect_peak_alloc, collect_peak_reserved = _accumulate_layer_peak(
            layer_peak_alloc,
            layer_peak_reserved,
        )
        print(
            f"[layer {layer_id}] collection CUDA memory "
            f"{CUDAPeakTracker.format_snapshot(collect_peak_alloc, collect_peak_reserved)} "
            f"{CUDAPeakTracker.format_current(*peak_tracker.current())}"
        )

        # Process GROUPS
        for group in groups:
            # skip group if none of its paths were collected
            if not all(p in layer_data for p in group):
                # if some missing, fall back to processing the ones that exist as singletons
                # (safe behavior; avoids silently dropping)
                for p in group:
                    if p in layer_data:
                        group = (p,)
                        break
                else:
                    continue

            # ensure all modules exist and are Linear
            is_ok = True
            for module_path in group:
                mod = get_submodule_by_path(blk, module_path)
                if not isinstance(mod, nn.Linear):
                    is_ok = False
                    break
            if not is_ok:
                continue

            group_name = f"layers.{layer_id}." + ("{" + ",".join(group) + "}" if len(group) > 1 else group[0])
            print(f"\n--- {group_name} [{method}] ---")
            peak_tracker.reset()

            X_host = _pick_shared_X(layer_data, group)
            W_fp_host_list = [layer_data[p][1] for p in group]

            # Compute one group at a time on the target device to reduce peak GPU memory.
            X = X_host.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            W_fp_list = [W.to(device=device, dtype=torch.float32, non_blocking=True).contiguous() for W in W_fp_host_list]
            print(f"[layer {layer_id}] X shape: {tuple(X.shape)}")
            for module_path, W_fp in zip(group, W_fp_list):
                print(f"[layer {layer_id}] W shape ({module_path}): {tuple(W_fp.shape)}")
            w_chunk_rows = _runtime_w_chunk_rows(cfg)

            need_sq = method in ("heuristic", "heuristic_gptq", "alt")
            need_base = method in ("s1", "s1_gptq", "alt")

            # group heuristic s
            s_h = None
            if need_sq:
                s_h = smoothquant_heuristic_s_group(X, W_fp_list, alpha=0.5, normalize_geom_mean=True)

            # baseline / heuristic objectives (summed)
            f_base = float("nan")
            if need_base:
                # baseline s=1 summed
                s1 = torch.ones((X.shape[1],), device=device, dtype=torch.float32)
                Xq1 = build_Xq_from_s_streaming(
                    X, s1, act_bits=cfg.act_bits, eps=cfg.s_eps,
                    act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
                    mode=cfg.act_quant_mode,
                )
                f_b = 0.0
                for W_fp in W_fp_list:
                    U1 = build_U_from_s_What(
                        W_fp,
                        s1,
                        cfg.w_bits,
                        cfg.s_eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=cfg.w_quant_mode,
                    )
                    f_b += obj_from_X_Xq_W_U_streaming(
                        X=X, Xq=Xq1, W_fp=W_fp, U=U1, row_chunk=int(getattr(cfg, "obj_row_chunk", 256))
                    )
                f_base = float(f_b)

            f_sq = float("nan")
            if need_sq and s_h is not None:
                Xq_h = build_Xq_from_s_streaming(
                    X, s_h, act_bits=cfg.act_bits, eps=cfg.s_eps,
                    act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
                    mode=cfg.act_quant_mode,
                )
                f_h = 0.0
                for W_fp in W_fp_list:
                    U_h = build_U_from_s_What(
                        W_fp,
                        s_h,
                        cfg.w_bits,
                        cfg.s_eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=cfg.w_quant_mode,
                    )
                    f_h += obj_from_X_Xq_W_U_streaming(
                        X=X, Xq=Xq_h, W_fp=W_fp, U=U_h, row_chunk=int(getattr(cfg, "obj_row_chunk", 256))
                    )
                f_sq = float(f_h)

            f_init = float("nan")
            f_alt = float("nan")

            if method == "heuristic":
                assert s_h is not None
                s_use = s_h
                U_list = [
                    build_U_from_s_What(
                        W_fp,
                        s_use,
                        cfg.w_bits,
                        cfg.s_eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=cfg.w_quant_mode,
                    )
                    for W_fp in W_fp_list
                ]
                f_use = f_sq

            elif method == "s1":
                s_use = torch.ones((X.shape[1],), device=device, dtype=torch.float32)
                U_list = [
                    build_U_from_s_What(
                        W_fp,
                        s_use,
                        cfg.w_bits,
                        cfg.s_eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=cfg.w_quant_mode,
                    )
                    for W_fp in W_fp_list
                ]
                f_use = f_base

            elif method == "s1_gptq":
                s_use = torch.ones((X.shape[1],), device=device, dtype=torch.float32)
                Xq = build_Xq_from_s_streaming(
                    X, s_use, act_bits=cfg.act_bits, eps=cfg.s_eps,
                    act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
                    mode=cfg.act_quant_mode,
                )
                G = (Xq.T @ Xq).to(torch.float32)
                H = (2.0 * G).to(torch.float32)

                U_list = []
                f_use_sum = 0.0
                for W_fp in W_fp_list:
                    c, B = compute_c_and_B_streaming(X, Xq, W_fp, row_chunk=int(getattr(cfg, "cB_row_chunk", 1024)))
                    U_ls = ridge_ls_from_gram_fp32(G, B.to(torch.float32), cfg)
                    U_use = gptq_quantize_U_from_H(H=H, U_init=U_ls, cfg=cfg, desc=f"GPTQ(U) {group_name} s=1")
                    U_list.append(U_use)
                    f_use_sum += obj_from_gram_stats_fp32(
                        G,
                        B,
                        c,
                        U_use,
                        col_chunk=int(getattr(cfg, "gu_obj_chunk_cols", 256)),
                    )
                f_use = float(f_use_sum)

            elif method == "heuristic_gptq":
                assert s_h is not None
                s_use = s_h
                Xq = build_Xq_from_s_streaming(
                    X, s_use, act_bits=cfg.act_bits, eps=cfg.s_eps,
                    act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
                    mode=cfg.act_quant_mode,
                )
                G = (Xq.T @ Xq).to(torch.float32)
                H = (2.0 * G).to(torch.float32)

                U_list = []
                f_use_sum = 0.0
                for W_fp in W_fp_list:
                    c, B = compute_c_and_B_streaming(X, Xq, W_fp, row_chunk=int(getattr(cfg, "cB_row_chunk", 1024)))
                    U_ls = ridge_ls_from_gram_fp32(G, B.to(torch.float32), cfg)
                    U_use = gptq_quantize_U_from_H(H=H, U_init=U_ls, cfg=cfg, desc=f"GPTQ(U) {group_name} s_h")
                    U_list.append(U_use)
                    f_use_sum += obj_from_gram_stats_fp32(
                        G,
                        B,
                        c,
                        U_use,
                        col_chunk=int(getattr(cfg, "gu_obj_chunk_cols", 256)),
                    )
                f_use = float(f_use_sum)

            else:  # alt (group)
                assert s_h is not None
                if cfg.alt_s_init == "heuristic":
                    s0 = s_h
                elif cfg.alt_s_init == "ones":
                    s0 = torch.ones((X.shape[1],), device=device, dtype=torch.float32)
                else:  # GD
                    s0 = gd_init_s_ste_group(
                        X=X,
                        W_fp_list=W_fp_list,
                        cfg=cfg,
                        gd_print=False,
                        gd_desc=group_name,
                    )

                print(f"[alt-group] s-init = {cfg.alt_s_init} group={group}")
                s_star, W_hat_list, U_star_list, f_alt, f_init = alternating_optimize_s_and_What_group(
                    X, W_fp_list, s0, cfg, plots_dir=plots_dir, module_name=group_name
                )
                s_use = s_star
                U_list = U_star_list
                f_use = f_alt

            # logging
            print(f"N={X.shape[0]}  Ci={X.shape[1]}  group_size={len(group)}")
            if need_base:
                print(f"baseline(sum)     f_base : {f_base:.6e}")
            if need_sq:
                print(f"heuristic(sum)    f_sq   : {f_sq:.6e}")
            if method == "alt":
                print(f"init(sum)         f_init : {f_init:.6e}")
                print(f"alt(sum)          f_alt  : {f_alt:.6e}")
            print(f"USED       method={method}  f_used : {float(f_use):.6e}")

            # record & patch each module in group with SAME s and its own U
            for module_path, U_use in zip(group, U_list):
                name = f"layers.{layer_id}.{module_path}"

                patch_records.append(
                    PatchRecord(
                        layer_id=int(layer_id),
                        module_path=str(module_path),
                        act_bits=int(cfg.act_bits),
                        w_bits=int(cfg.w_bits),
                        s=_to_cpu_f32(s_use),
                        U_in_out=_to_cpu_f32(U_use),
                    )
                )

                patch_one_linear_altU(model, layer_id, module_path, s_use, U_use, cfg, debug=False)
                patched_count += 1

                if do_ppl_each:
                    seq_len_ppl = int(ppl_seq_len)
                    max_toks = int(ppl_max_tokens)
                    ppl_bs = int(getattr(cfg, "ppl_batch_size", 4))
                    ppl_tr = eval_ppl(
                        model, tokenizer, ppl_texts_train, device,
                        seq_len=seq_len_ppl, max_tokens=max_toks, batch_size=ppl_bs,
                    )
                    ppl_te = eval_ppl(
                        model, tokenizer, ppl_texts_test, device,
                        seq_len=seq_len_ppl, max_tokens=max_toks, batch_size=ppl_bs,
                    )
                    print(
                        f"[PPL after patch {patched_count:04d}] {name} "
                        f"train={ppl_tr:.4f} test={ppl_te:.4f} "
                        f"(seq_len={seq_len_ppl}, max_tokens={max_toks})"
                    )

                stats[name] = {
                    # Compatibility keys expected by print_layerwise_summary()
                    "f_base": float(f_base),
                    "f_sq": float(f_sq),
                    "f_used": float(f_use),
                    "f_init": float(f_init) if method == "alt" else float("nan"),
                    "f_alt": float(f_alt) if method == "alt" else float("nan"),
                    # Group-aware aliases kept for downstream analysis
                    "f_base_sum_group": float(f_base),
                    "f_sq_sum_group": float(f_sq),
                    "f_used_sum_group": float(f_use),
                    "f_init_sum_group": float(f_init) if method == "alt" else float("nan"),
                    "f_alt_sum_group": float(f_alt) if method == "alt" else float("nan"),
                    "Ci": float(X.shape[1]),
                    "Co": float(U_use.shape[1]),
                    "N": float(X.shape[0]),
                    "group": ",".join(group),
                }

                try:
                    del layer_data[module_path]
                except Exception:
                    pass

            layer_peak_alloc, layer_peak_reserved, group_peak_alloc, group_peak_reserved = _accumulate_layer_peak(
                layer_peak_alloc,
                layer_peak_reserved,
            )
            try:
                del Xq1
            except Exception:
                pass
            try:
                del Xq_h
            except Exception:
                pass
            try:
                del Xq
            except Exception:
                pass
            try:
                del G
            except Exception:
                pass
            try:
                del G64
            except Exception:
                pass
            try:
                del H
            except Exception:
                pass
            try:
                del s1
            except Exception:
                pass
            try:
                del s_h
            except Exception:
                pass
            try:
                del s0
            except Exception:
                pass
            try:
                del s_star
            except Exception:
                pass
            try:
                del s_use
            except Exception:
                pass
            try:
                del U_list
            except Exception:
                pass
            try:
                del U_star_list
            except Exception:
                pass
            try:
                del W_hat_list
            except Exception:
                pass
            try:
                del X_host, W_fp_host_list
            except Exception:
                pass
            del X, W_fp_list
            cur_alloc, cur_reserved = _maybe_trim_cuda_cache(group_name)
            print(
                f"[group-memory] {group_name} "
                f"{CUDAPeakTracker.format_snapshot(group_peak_alloc, group_peak_reserved)} "
                f"{CUDAPeakTracker.format_current(cur_alloc, cur_reserved)}"
            )

            if bool(getattr(cfg, "pipeline_empty_cache_each_group", False)) and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if do_ppl_each_layer:
            seq_len_ppl = int(ppl_seq_len)
            max_toks = int(ppl_max_tokens)
            ppl_bs = int(getattr(cfg, "ppl_batch_size", 4))
            ppl_te = eval_ppl(
                model, tokenizer, ppl_texts_test, device,
                seq_len=seq_len_ppl, max_tokens=max_toks, batch_size=ppl_bs,
            )
            if ppl_texts_train is not None:
                ppl_tr = eval_ppl(
                    model, tokenizer, ppl_texts_train, device,
                    seq_len=seq_len_ppl, max_tokens=max_toks, batch_size=ppl_bs,
                )
                print(
                    f"[PPL after layer {layer_id:04d}] "
                    f"train={ppl_tr:.4f} test={ppl_te:.4f} "
                    f"(seq_len={seq_len_ppl}, max_tokens={max_toks})"
                )
            else:
                print(
                    f"[PPL after layer {layer_id:04d}] "
                    f"test={ppl_te:.4f} "
                    f"(seq_len={seq_len_ppl}, max_tokens={max_toks})"
                )

        if replay_cache is not None:
            t_rp0 = time.time()
            peak_tracker.reset()
            try:
                replay_cache = replay_decoder_layer_from_cache(
                    model=model,
                    replay_cache=replay_cache,
                    layer_id=layer_id,
                    device=device,
                    pin_cpu_memory=bool(getattr(cfg, "collect_pin_cpu_memory", True)),
                    rotary_emb=rotary_emb,
                )
                print(f"[layer {layer_id}] replayed quantized outputs in {time.time() - t_rp0:.2f}s")
            except Exception as e:
                replay_cache = None
                rotary_emb = None
                print(f"[warn] replay output pass failed at layer {layer_id}; disabling replay for remaining layers: {e}")
            layer_peak_alloc, layer_peak_reserved, replay_peak_alloc, replay_peak_reserved = _accumulate_layer_peak(
                layer_peak_alloc,
                layer_peak_reserved,
            )
            print(
                f"[layer {layer_id}] replay CUDA memory "
                f"{CUDAPeakTracker.format_snapshot(replay_peak_alloc, replay_peak_reserved)} "
                f"{CUDAPeakTracker.format_current(*peak_tracker.current())}"
            )

        del layer_data
        cur_alloc, cur_reserved = _maybe_trim_cuda_cache(f"layer {layer_id}")
        print(
            f"[layer {layer_id}] peak CUDA memory "
            f"{CUDAPeakTracker.format_snapshot(layer_peak_alloc, layer_peak_reserved)} "
            f"{CUDAPeakTracker.format_current(cur_alloc, cur_reserved)}"
        )
        if bool(getattr(cfg, "pipeline_empty_cache_each_layer", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[plots] wrote plots under {plots_dir}")
    return stats, patch_records


# ============================================================
# Run one method (unchanged except uses grouped calibrator)
# ============================================================

@torch.no_grad()
def alternating_alg(
    model: nn.Module,
    clean_state_dict: Dict[str, torch.Tensor],
    module_paths: Tuple[str, ...],
    device: torch.device,
    tokenizer,
    train_texts: List[str],
    test_texts: List[str],
    token_batches: List[Dict[str, torch.Tensor]],
    cfg: SQCfg,
    calib_max_rows_X: int,
    ppl_seq_len: int,
    ppl_max_tokens: int,
    save_patches: bool,
    out_dir: Path,
    run_id: str,
    run_final_eval_ppl_suite: bool,
    final_eval_ppl_datasets: Sequence[str],
    run_final_zeroshot: bool,
    final_zeroshot_tasks: Optional[Sequence[str]],
    final_eval_batch_size: int,
    yes_to_remote_code: bool,
    skip_initial_eval: bool,
) -> Tuple[Dict[str, object], List[PatchRecord]]:
    t0 = time.time()
    method = "alt"

    restore_state_inplace(model, clean_state_dict, device)
    model.eval()

    out: Dict[str, object] = {"method": method}
    patches_path: Optional[str] = None
    init_eval_results: Dict[str, object] = {}
    final_eval_results: Dict[str, object] = {}
    init_eval_results_path: Optional[str] = None
    final_eval_results_path: Optional[str] = None

    ppl_bs = int(getattr(cfg, "ppl_batch_size", 4))
    if bool(skip_initial_eval):
        print("[info] skipping initial pre-quantization evaluation", flush=True)
        out["ppl_train_init"] = None
        out["ppl_test_init"] = None
    else:
        ppl_tr_init = eval_ppl(
            model, tokenizer, train_texts, device,
            seq_len=ppl_seq_len, max_tokens=None, batch_size=ppl_bs,
        )
        ppl_te_init = eval_ppl(
            model, tokenizer, test_texts, device,
            seq_len=ppl_seq_len, max_tokens=None, batch_size=ppl_bs,
        )
        out["ppl_train_init"] = float(ppl_tr_init)
        out["ppl_test_init"] = float(ppl_te_init)
        print(f"[PPL before quantization] train={ppl_tr_init:.4f} test={ppl_te_init:.4f} (full dataset)")

        if run_final_eval_ppl_suite:
            print("[info] running pre-quantization PPL benchmark suite", flush=True)
            out["init_eval_ppl"] = _run_final_ppl_suite(
                model=model,
                tokenizer=tokenizer,
                device=device,
                seq_len=int(ppl_seq_len),
                batch_size=int(final_eval_batch_size),
                which=final_eval_ppl_datasets,
            )
            init_eval_results["ppl"] = dict(out["init_eval_ppl"])

        if run_final_zeroshot:
            print("[info] running pre-quantization 0-shot evaluation", flush=True)
            if yes_to_remote_code:
                _enable_hf_dataset_remote_code()
            with _auto_accept_remote_code_prompts(enabled=bool(yes_to_remote_code)):
                avg_acc_init, per_task_init = zeroshot_eval(
                    model,
                    tokenizer,
                    batch_size=int(final_eval_batch_size),
                    tasks=list(final_zeroshot_tasks) if final_zeroshot_tasks else None,
                )
            out["init_eval_zeroshot_avg_acc"] = float(avg_acc_init)
            out["init_eval_zeroshot_per_task"] = {str(k): float(v) for k, v in per_task_init.items()}
            init_eval_results["zeroshot_avg_acc"] = float(avg_acc_init)
            init_eval_results["zeroshot_per_task"] = dict(out["init_eval_zeroshot_per_task"])
            print(f"[result] pre-quantization zeroshot avg acc = {avg_acc_init:.6f}", flush=True)

    if init_eval_results:
        init_eval_results_path = str(out_dir / f"{run_id}__init_eval_results.json")
        _write_json(Path(init_eval_results_path), init_eval_results)
        out["init_eval_results_path"] = init_eval_results_path

    stats, patch_records = calibrate_and_patch_all_linear_method_grouped(
        model=model,
        token_batches=token_batches,
        cfg=cfg,
        device=device,
        method=method,
        calib_max_rows_X=calib_max_rows_X,
        module_paths=module_paths,
        ppl_seq_len=ppl_seq_len,
        ppl_max_tokens=ppl_max_tokens,
        tokenizer=tokenizer,
        ppl_texts_train=train_texts,
        ppl_texts_test=test_texts,
        run_id=run_id,
    )
    print_layerwise_summary(stats, title=f"Layer-wise objectives ({method}) [grouped]")

    ppl_tr = eval_ppl(
        model, tokenizer, train_texts, device,
        seq_len=ppl_seq_len, max_tokens=None, batch_size=ppl_bs,
    )
    ppl_te = eval_ppl(
        model, tokenizer, test_texts, device,
        seq_len=ppl_seq_len, max_tokens=None, batch_size=ppl_bs,
    )
    out["ppl_train"] = float(ppl_tr)
    out["ppl_test"] = float(ppl_te)
    print(
        f"[PPL after quantization] train={ppl_tr:.4f} "
        f"test={ppl_te:.4f} (full dataset, seq_len={ppl_seq_len})"
    )
    out["ppl_train_full"] = float(ppl_tr)
    out["ppl_test_full"] = float(ppl_te)

    if run_final_eval_ppl_suite:
        out["final_eval_ppl"] = _run_final_ppl_suite(
            model=model,
            tokenizer=tokenizer,
            device=device,
            seq_len=int(ppl_seq_len),
            batch_size=int(final_eval_batch_size),
            which=final_eval_ppl_datasets,
        )
        final_eval_results["ppl"] = dict(out["final_eval_ppl"])

    if run_final_zeroshot:
        print("[info] running 0-shot evaluation", flush=True)
        if yes_to_remote_code:
            _enable_hf_dataset_remote_code()
        with _auto_accept_remote_code_prompts(enabled=bool(yes_to_remote_code)):
            avg_acc, per_task = zeroshot_eval(
                model,
                tokenizer,
                batch_size=int(final_eval_batch_size),
                tasks=list(final_zeroshot_tasks) if final_zeroshot_tasks else None,
            )
        out["final_eval_zeroshot_avg_acc"] = float(avg_acc)
        out["final_eval_zeroshot_per_task"] = {str(k): float(v) for k, v in per_task.items()}
        final_eval_results["zeroshot_avg_acc"] = float(avg_acc)
        final_eval_results["zeroshot_per_task"] = dict(out["final_eval_zeroshot_per_task"])
        print(f"[result] zeroshot avg acc = {avg_acc:.6f}", flush=True)

    if final_eval_results:
        final_eval_results_path = str(out_dir / f"{run_id}__final_eval_results.json")
        _write_json(Path(final_eval_results_path), final_eval_results)
        out["final_eval_results_path"] = final_eval_results_path

    out["layer_stats"] = stats

    if save_patches:
        ppath = save_patches_pt(
            out_dir=out_dir,
            run_id=run_id,
            method=method,
            model_name="",
            module_paths=module_paths,
            cfg=cfg,
            patch_records=patch_records,
        )
        patches_path = str(ppath)
    out["patches_path"] = patches_path

    out["seconds"] = float(time.time() - t0)
    return out, patch_records


# ============================================================
# main (same CLI)
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen", "llama", "generic"])
    ap.add_argument("--module_paths", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="run")

    ap.add_argument("--calib_rows", type=int, default=2000)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_length", type=int, default=128)
    ap.add_argument("--calib_max_rows_X", type=int, default=80_000)

    ap.add_argument("--ppl_seq_len", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=50_000)

    ap.add_argument("--act_bits", type=int, default=4)
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--act_quant_mode", type=str, default="symmetric", choices=["symmetric", "asymmetric"])
    ap.add_argument("--w_quant_mode", type=str, default="symmetric", choices=["symmetric", "asymmetric"])
    ap.add_argument("--alt_iters", type=int, default=1)
    ap.add_argument("--s_sweeps", type=int, default=1)
    ap.add_argument("--golden_max_iter", type=int, default=20)
    ap.add_argument("--coord_backtrack_steps", type=int, default=3)

    ap.add_argument("--screen_gamma", type=float, default=1.0)
    ap.add_argument("--screen_ste_rows", type=int, default=8192)
    ap.add_argument("--screen_ste_rows_max", type=int, default=200000)
    ap.add_argument("--screen_ste_rows_min", type=int, default=512)
    ap.add_argument(
        "--screen_ste_center_grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--screen_ste_oom_retry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--gptq_blocksize", type=int, default=128)
    ap.add_argument("--gptq_percdamp", type=float, default=0.01)
    ap.add_argument("--gptq_actorder", action="store_true")

    ap.add_argument("--ppl_after_each_module", action="store_true")
    ap.add_argument(
        "--ppl_after_each_layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated: per-layer PPL evaluation is disabled by default.",
    )
    ap.add_argument("--ppl_batch_size", type=int, default=4)
    ap.add_argument(
        "--skip_initial_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the initial pre-quantization evaluation pass.",
    )
    ap.add_argument(
        "--run_final_eval_ppl_suite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run final WikiText-2/C4/PTB perplexity evaluation after quantization.",
    )
    ap.add_argument(
        "--final_eval_ppl_datasets",
        type=str,
        default="wikitext2,c4,ptb",
        help="Comma-separated final PPL datasets from {wikitext2,c4,ptb}.",
    )
    ap.add_argument(
        "--run_final_zeroshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run final 0-shot lm-eval-harness evaluation after quantization.",
    )
    ap.add_argument(
        "--final_zeroshot_tasks",
        type=str,
        default="",
        help="Optional comma-separated 0-shot task list. Empty uses eval_utils defaults.",
    )
    ap.add_argument("--final_eval_batch_size", type=int, default=4)
    ap.add_argument(
        "--yes_to_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-accept remote-code prompts during final zeroshot eval.",
    )

    ap.add_argument("--u_update", type=str, default="gptq", choices=["gptq", "ls_project"])
    ap.add_argument(
        "--alt_step_order",
        type=str,
        default="s_then_u",
        choices=["s_then_u", "u_then_s"],
        help="Outer alternating step order.",
    )

    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", HF_TOKEN_HARDCODED))

    ap.add_argument("--save_patches", action="store_true")
    ap.add_argument("--no_save_patches", action="store_true")
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--no_save_final_model", action="store_true")
    ap.add_argument("--final_model_dir", type=str, default="")
    ap.add_argument("--alt_s_init", type=str, default="heuristic", choices=["heuristic", "ones", "GD"])
    ap.add_argument(
        "--pre_rotations",
        type=str,
        default="",
        help="Optional comma-separated SpinQuant pre-rotations to apply before alternating SQ, e.g. R1,R2,R4.",
    )
    ap.add_argument(
        "--pre_rotation_block_size",
        type=_parse_optional_positive_int,
        default=None,
        help=(
            "SpinQuant transform block size for pre-rotations. "
            "Omit or pass 'auto' to let SpinQuant choose per-rotation defaults."
        ),
    )
    ap.add_argument(
        "--pre_rotation_type",
        type=str,
        default="hadamard",
        help="SpinQuant transform type for pre-rotations.",
    )
    ap.add_argument(
        "--pre_rotation_randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SpinQuant randomize=True when applying pre-rotations.",
    )

    args = ap.parse_args()

    _maybe_set_hf_token(args.hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model_name = args.model
    model_family = infer_model_family(model_name) if args.model_family == "auto" else str(args.model_family)
    print(f"[info] model_family={model_family} (arg={args.model_family})")
    method = "alt"

    module_paths_list = _parse_csv_list(args.module_paths)
    module_paths = (
        tuple(module_paths_list)
        if module_paths_list
        else default_module_paths_for_model_family(model_family)
    )

    out_dir = _ensure_out_dir(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tag}__{stamp}"
    out_json = out_dir / f"{run_id}.json"

    # Upload-only flow: avoid persisting large patch artifacts under out_dir.
    save_patches_flag = False

    save_final_model_flag = True
    if bool(args.no_save_final_model):
        save_final_model_flag = False
    elif bool(args.save_final_model):
        save_final_model_flag = True

    cfg = SQCfg(
        act_bits=args.act_bits,
        w_bits=args.w_bits,
        act_quant_mode=str(args.act_quant_mode),
        w_quant_mode=str(args.w_quant_mode),
        pre_rotations=tuple(_parse_csv_list(args.pre_rotations)),
        pre_rotation_block_size=args.pre_rotation_block_size,
        pre_rotation_type=str(args.pre_rotation_type),
        pre_rotation_randomize=bool(args.pre_rotation_randomize),
        alt_iters=args.alt_iters,
        s_sweeps=args.s_sweeps,
        golden_max_iter=args.golden_max_iter,
        coord_backtrack_steps=args.coord_backtrack_steps,
        screen_gamma=args.screen_gamma,
        screen_ste_rows=args.screen_ste_rows,
        screen_ste_rows_max=args.screen_ste_rows_max,
        screen_ste_rows_min=args.screen_ste_rows_min,
        screen_ste_center_grad=bool(args.screen_ste_center_grad),
        screen_ste_oom_retry=bool(args.screen_ste_oom_retry),
        gptq_blocksize=args.gptq_blocksize,
        gptq_percdamp=args.gptq_percdamp,
        gptq_actorder=bool(args.gptq_actorder),
        ppl_after_each_module=bool(args.ppl_after_each_module),
        ppl_after_each_layer=bool(args.ppl_after_each_layer),
        ppl_seq_len=args.ppl_seq_len,
        ppl_max_tokens=args.ppl_max_tokens,
        ppl_batch_size=args.ppl_batch_size,
        alt_s_init=str(args.alt_s_init),
        show_tqdm=True,
        show_coord_tqdm=True,
        u_update=str(args.u_update),
        alt_step_order=str(args.alt_step_order),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _maybe_set_pad_token(tokenizer, None)

    tok_dir = out_dir / f"{run_id}__tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    ds_train = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    ds_test = load_dataset(cfg.dataset_name, cfg.dataset_config, split="test")
    train_texts = [x["text"] for x in ds_train if isinstance(x.get("text", None), str)]
    test_texts = [x["text"] for x in ds_test if isinstance(x.get("text", None), str)]

    requested_calib_tokens = int(args.calib_rows) * int(args.calib_max_length)
    effective_calib_tokens = min(requested_calib_tokens, int(args.calib_max_rows_X))
    token_batches = make_fixed_length_token_batches_from_dataset(
        tokenizer=tokenizer,
        ds_split=ds_train,
        nsamples=args.calib_rows,
        batch_size=args.calib_batch_size,
        seq_len=args.calib_max_length,
        max_tokens=effective_calib_tokens,
    )
    print(
        f"[info] Built calibration token windows: "
        f"requested_tokens={requested_calib_tokens}, "
        f"effective_tokens={effective_calib_tokens}, "
        f"seq_len={args.calib_max_length}"
    )

    print(f"[info] Loading model {model_name} in bfloat16 ...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    _maybe_set_pad_token(tokenizer, model)
    print(f"[info] Model loaded in {time.time()-t_load:.1f}s")

    pre_rotations_applied = _maybe_apply_pre_rotations(model, tokenizer, cfg)

    n_params = sum(int(p.numel()) for p in model.parameters())
    n_params_b = float(n_params) / 1e9
    high_memory_fast_path_max_params = 30_000_000_000
    if n_params > high_memory_fast_path_max_params:
        cfg.runtime_w_quant_chunk_rows = int(max(1, getattr(cfg, "w_quant_chunk_rows", 2048)))
        cfg.pipeline_empty_cache_each_group = True
        cfg.screen_ste_rows = min(int(getattr(cfg, "screen_ste_rows", 8192)), 2048)
        cfg.screen_ste_rows_max = min(
            int(getattr(cfg, "screen_ste_rows_max", 200_000)),
            max(int(cfg.screen_ste_rows), int(getattr(cfg, "screen_ste_rows_min", 512))),
        )
        print(
            f"[info] model params={n_params_b:.2f}B "
            f"(>{high_memory_fast_path_max_params / 1e9:.0f}B): "
            f"enabling low-memory weight quant chunking (rows={cfg.runtime_w_quant_chunk_rows}) "
            f"and per-group CUDA cache flush; screening rows={cfg.screen_ste_rows}"
        )
    else:
        cfg.runtime_w_quant_chunk_rows = 0
        # On smaller models, spend more memory to keep the exact unstable-row
        # evaluator on its fused/prepacked path more often.
        cfg.unstable_rows_prepack_groups = True
        cfg.unstable_rows_prepack_max_mb = 2048.0
        cfg.unstable_rows_fullcol_precompute = True
        cfg.unstable_rows_fullcol_max_mb = 4096.0
        cfg.unstable_rows_eval_gather_once = True
        cfg.unstable_rows_eval_gather_max_mb = 2048.0
        cfg.unstable_rows_eval_row_chunk = max(int(getattr(cfg, "unstable_rows_eval_row_chunk", 1024)), 2048)
        cfg.unstable_rows_eval_chunk_cols = max(int(getattr(cfg, "unstable_rows_eval_chunk_cols", 512)), 1024)
        cfg.unstable_rows_quant_col_chunk = max(int(getattr(cfg, "unstable_rows_quant_col_chunk", 512)), 1024)
        cfg.act_quant_chunk_cols = max(int(getattr(cfg, "act_quant_chunk_cols", 512)), 1024)
        cfg.cB_row_chunk = max(int(getattr(cfg, "cB_row_chunk", 1024)), 2048)
        cfg.G_row_chunk = max(int(getattr(cfg, "G_row_chunk", cfg.cB_row_chunk)), 2048)
        cfg.G64_tile = max(int(getattr(cfg, "G64_tile", 1024)), 2048)
        print(
            f"[info] model params={n_params_b:.2f}B "
            f"(<= {high_memory_fast_path_max_params / 1e9:.0f}B): "
            "enabling higher-memory exact CD fast paths "
            f"(prepack={cfg.unstable_rows_prepack_max_mb:.0f}MB, "
            f"fullcol={cfg.unstable_rows_fullcol_max_mb:.0f}MB, "
            f"gather={cfg.unstable_rows_eval_gather_max_mb:.0f}MB)"
        )

    print("[info] Snapshotting clean model state dict on CPU ...")
    clean_state_dict: Dict[str, torch.Tensor] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[info] Snapshot done ({sum(v.numel() for v in clean_state_dict.values()) * 2 / 1e9:.2f} GB bfloat16)")

    run_meta = {
        "run_id": run_id,
        "model": model_name,
        "model_family": model_family,
        "algorithm": method,
        "pre_rotations_applied": bool(pre_rotations_applied),
        "module_paths": list(module_paths),
        "args": vars(args),
        "cfg": asdict(cfg),
        "artifacts": {
            "tokenizer_dir": str(tok_dir),
            "final_model_dir": None,
            "final_model_method": None,
            "final_model_manifest": None,
            "final_model_patches": None,
            "hf_repo_id": None,
            "hf_repo_url": None,
        },
        "result": None,
    }
    _write_json(out_json, run_meta)

    print("\n============================================================")
    print("=== Alternating SQ ===")
    print("============================================================")
    res, patch_records = alternating_alg(
        model=model,
        clean_state_dict=clean_state_dict,
        module_paths=module_paths,
        device=device,
        tokenizer=tokenizer,
        train_texts=train_texts,
        test_texts=test_texts,
        token_batches=token_batches,
        cfg=cfg,
        calib_max_rows_X=args.calib_max_rows_X,
        ppl_seq_len=args.ppl_seq_len,
        ppl_max_tokens=args.ppl_max_tokens,
        save_patches=save_patches_flag,
        out_dir=out_dir,
        run_id=run_id,
        run_final_eval_ppl_suite=bool(args.run_final_eval_ppl_suite),
        final_eval_ppl_datasets=[x.strip() for x in str(args.final_eval_ppl_datasets).split(",") if x.strip()],
        run_final_zeroshot=bool(args.run_final_zeroshot),
        final_zeroshot_tasks=[x.strip() for x in str(args.final_zeroshot_tasks).split(",") if x.strip()] or None,
        final_eval_batch_size=int(args.final_eval_batch_size),
        yes_to_remote_code=bool(args.yes_to_remote_code),
        skip_initial_eval=bool(args.skip_initial_eval),
    )
    run_meta["result"] = res
    _write_json(out_json, run_meta)

    if res.get("patches_path", None):
        print(f"[saved patches] {res['patches_path']}")
    print(f"\n[PPL] {method:14s} train={res['ppl_train']:.4f} test={res['ppl_test']:.4f} time={res['seconds']:.1f}s")

    if save_final_model_flag:
        final_method = method
        upload_only_mode = not bool(args.final_model_dir)
        if args.final_model_dir:
            final_model_dir = Path(args.final_model_dir)
            cleanup_final_model_dir = False
        else:
            final_model_dir = _make_temp_artifact_dir(run_id=run_id, final_method=final_method)
            cleanup_final_model_dir = True
        final_model_dir.mkdir(parents=True, exist_ok=True)
        if upload_only_mode:
            print(f"[save] Using temporary artifact directory for upload: {final_model_dir}")

        # Break any unexpected shared storage in custom SQ buffers before HF safe serialization.
        # Grouped patching can intentionally reuse one `s` object across modules.
        for mod in model.modules():
            try:
                if "s" in getattr(mod, "_buffers", {}) and torch.is_tensor(mod._buffers["s"]):
                    mod._buffers["s"] = mod._buffers["s"].detach().clone()
            except Exception:
                pass

        print(f"[save] Saving final model ({final_method}) -> {final_model_dir}")
        t_save = time.time()
        _strip_transformers_compression_metadata(model)
        try:
            model.save_pretrained(final_model_dir, safe_serialization=True)
        except RuntimeError as e:
            msg = str(e)
            if "shared tensors" in msg:
                print(
                    "[warn] safe serialization failed due to shared tensors; "
                    "retrying with safe_serialization=False"
                )
                model.save_pretrained(final_model_dir, safe_serialization=False)
            else:
                raise
        tokenizer.save_pretrained(final_model_dir)
        print(f"[save] Final model saved in {time.time() - t_save:.1f}s")

        final_patches_dst = None
        if patch_records:
            ppatch = save_patches_pt(
                out_dir=final_model_dir,
                run_id=run_id,
                method=final_method,
                model_name=model_name,
                module_paths=module_paths,
                cfg=cfg,
                patch_records=patch_records,
            )
            final_patches_dst = ppatch.name

        manifest = {
            "run_id": run_id,
            "base_model": model_name,
            "model_family": model_family,
            "final_method": final_method,
            "final_model_dir": "" if upload_only_mode else str(final_model_dir),
            "final_patches_path": final_patches_dst,
        }
        manifest_path = final_model_dir / "final_model_manifest.json"
        _write_json(manifest_path, manifest)
        print(f"[save] Final manifest -> {manifest_path}")

        run_meta["artifacts"]["final_model_dir"] = None if upload_only_mode else str(final_model_dir)
        run_meta["artifacts"]["final_model_method"] = str(final_method)
        run_meta["artifacts"]["final_model_manifest"] = str(manifest_path)
        run_meta["artifacts"]["final_model_patches"] = final_patches_dst

        hf_repo_id = _default_hf_repo_id(model_name=model_name, run_id=run_id, final_method=final_method)
        print(f"[upload] Pushing final model to https://huggingface.co/{hf_repo_id}")
        hf_repo_url = _push_final_model_dir_to_hub(
            final_model_dir,
            repo_id=hf_repo_id,
            hf_token=args.hf_token,
            commit_message=f"Upload final SQ checkpoint for {run_id}",
        )
        print(f"[upload] Final model uploaded -> {hf_repo_url}")
        run_meta["artifacts"]["hf_repo_id"] = hf_repo_id
        run_meta["artifacts"]["hf_repo_url"] = hf_repo_url
        _write_json(out_json, run_meta)

        if cleanup_final_model_dir:
            shutil.rmtree(final_model_dir)
            print(f"[save] Removed temporary artifact directory: {final_model_dir}")

    del model
    del clean_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[saved] {out_json}")
    print(f"[plots] ./plots/{run_id}/  (one PDF per group per sweep)")
    print("Done.")


if __name__ == "__main__":
    main()

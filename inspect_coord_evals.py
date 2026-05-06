#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from SQ_alternating import alternating_optimize_s_and_What_group
from active_set import gd_init_s_ste_group, smoothquant_heuristic_s_group
from calibration_utils import (
    _maybe_set_pad_token,
    collect_layer_XW_with_early_stop,
    make_calib_texts,
    make_token_batches,
)
from utils import (
    SQCfg,
    _pick_shared_X,
    default_module_groups_for_decoder,
    default_module_paths_for_model_family,
    infer_model_family,
)


def _parse_csv_list(s: str) -> List[str]:
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _maybe_set_hf_token(hf_token: str) -> None:
    tok = str(hf_token).strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGINGFACE_HUB_TOKEN"] = tok


def _plot_topk_coord_traces(
    traces: List[Dict[str, object]],
    *,
    layer_idx: int,
    module_name: str,
    top_k: int,
    out_base: Path,
) -> None:
    usable: List[Dict[str, object]] = []
    for tr in traces:
        evals = tr.get("evals", [])
        if not isinstance(evals, list) or len(evals) == 0:
            continue
        usable.append(tr)
    if not usable:
        return

    usable.sort(key=lambda x: float(x.get("rel_obj_after", float("inf"))))
    top = usable[: max(1, min(int(top_k), len(usable)))]

    n = len(top)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.8 * nrows))
    if not isinstance(axes, (list, tuple)):
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = list(axes)
    if hasattr(axes, "__len__") and not isinstance(axes, list):
        axes = list(axes)

    for ax, tr in zip(axes, top):
        curve_color = "#2F6C8F"
        evals = tr["evals"]
        pts = [
            (
                ii,
                float(e["s"]),
                float(e.get("rel_obj", e.get("rel_obj_sweep", 1.0))),
                str(e.get("source", "")),
            )
            for ii, e in enumerate(evals)
            if isinstance(e, dict)
        ]
        pts_by_s = sorted(pts, key=lambda x: x[1])
        xs = [p[1] for p in pts_by_s]
        ys = [p[2] for p in pts_by_s]
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3.0, alpha=0.8, color=curve_color)

        order_x = [p[1] for p in pts]
        order_y = [p[2] for p in pts]

        s0 = float(tr.get("s_before", xs[0]))
        rel0 = float(tr.get("rel_obj_before", 1.0))
        best_s = float(tr.get("s_best_pred", s0))
        best_rel = float(tr.get("rel_obj_best_pred", min(ys)))
        final_rel = float(tr.get("rel_obj_after", ys[-1]))

        ax.scatter(
            [s0],
            [rel0],
            facecolors="white",
            edgecolors="#C25B39",
            linewidths=1.2,
            s=34,
            zorder=4,
        )
        ax.scatter([best_s], [best_rel], color="green", s=34, zorder=4)

        ax2 = ax.inset_axes([0.58, 0.55, 0.36, 0.32])
        ax2.plot(range(len(order_y)), order_y, marker="o", linewidth=1.0, markersize=2.4, color="tab:purple")
        if order_y:
            start_idx = min(
                range(len(order_y)),
                key=lambda ii: abs(order_x[ii] - s0) + abs(order_y[ii] - rel0),
            )
            best_idx = min(
                range(len(order_y)),
                key=lambda ii: abs(order_x[ii] - best_s) + abs(order_y[ii] - best_rel),
            )
            ax2.scatter(
                [start_idx],
                [order_y[start_idx]],
                facecolors="white",
                edgecolors="#C25B39",
                linewidths=1.0,
                s=24,
                zorder=4,
            )
            ax2.scatter([best_idx], [order_y[best_idx]], color="green", s=24, zorder=4)
        ax2.set_title("eval order", fontsize=7)
        ax2.grid(True, linewidth=0.4, alpha=0.4)
        ax2.tick_params(axis="both", labelsize=6, length=2)
        ax2.set_ylim(0.8, 2.5)

        ax.set_ylim(0.8, 2.5)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.set_xlabel("s_j", fontsize=8)
        ax.set_ylabel("rel obj", fontsize=8)
        ax.set_title(
            f"layer={layer_idx} j={int(tr['coord'])} final={final_rel:.3f}",
            fontsize=9,
        )
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Top-{n} coords for {module_name} (objective normalized by coordinate-start objective)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_xinv_grid_payload(
    X: torch.Tensor,
    sweep_records: List[Dict[str, object]],
    *,
    module_name: str,
    max_tokens_plot: int,
    max_stage_panels: int = 5,
) -> Dict[str, Any]:
    if not sweep_records:
        return {}

    T = min(int(max_tokens_plot), int(X.shape[0]))
    if T <= 0:
        return {}
    X_base = X[:T, :].detach().to(torch.float32).cpu()

    def _normalize_channels(channels: object) -> List[int]:
        out: List[int] = []
        seen = set()
        if isinstance(channels, list):
            for jj in channels:
                j = int(jj)
                if 0 <= j < int(X_base.shape[1]) and j not in seen:
                    out.append(j)
                    seen.add(j)
        return out

    panels: List[Dict[str, object]] = []
    first_channels = _normalize_channels(sweep_records[0].get("selected_channels", []))
    if not first_channels:
        first_channels = list(range(min(int(X_base.shape[1]), 64)))
    original_panel = (
        {
            "title": r"Original $|\mathbf{X}|$",
            "channels": first_channels,
            "updated": [],
            "s": None,
        }
    )
    stage_panels: List[Dict[str, object]] = []
    for rec in sweep_records:
        channels = _normalize_channels(rec.get("selected_channels", []))
        if not channels:
            continue
        stage = int(rec.get("stage", len(panels)))
        stage_panels.append(
            {
                "title": rf"$|\mathbf{{X}}/s|$ after Active Set Update {stage}",
                "channels": channels,
                "s": rec.get("s"),
            }
        )
    panels.append(original_panel)
    max_stage_panels = max(1, int(max_stage_panels))
    if len(stage_panels) <= max_stage_panels:
        panels.extend(stage_panels)
    else:
        panels.extend(stage_panels[: max_stage_panels - 1])
        final_panel = dict(stage_panels[-1])
        final_panel["title"] = f"{final_panel['title']} [final]"
        panels.append(final_panel)
    if not panels:
        return {}

    panel_data: List[Dict[str, object]] = []
    for panel in panels:
        channels = panel["channels"]
        ch_idx_t = torch.tensor(channels, dtype=torch.long)
        Xs = X_base.index_select(1, ch_idx_t)
        s_panel = panel["s"]
        if isinstance(s_panel, torch.Tensor):
            ss = s_panel.detach().to(torch.float32).cpu().index_select(0, ch_idx_t).clamp_min(1e-8)
            Z = (Xs / ss.unsqueeze(0)).abs().numpy()
            z_label = r"$|\mathbf{X}/s|$"
        else:
            Z = Xs.abs().numpy()
            z_label = r"$|\mathbf{X}|$"
        panel_data.append(
            {
                "title": panel["title"],
                "x_ch": np.asarray(channels, dtype=np.float32),
                "y_tok": np.arange(T, dtype=np.float32),
                "Z": torch.from_numpy(Z.copy()),
                "z_label": z_label,
            }
        )
    return {
        "module_name": str(module_name),
        "max_tokens_plot": int(T),
        "panel_data": panel_data,
    }


def plot_xinv_grid_from_payload(
    payload: Dict[str, Any],
    *,
    out_base: Path,
    elev: float,
    azim: float,
    dpi: int,
    p_lo: float,
    p_hi: float,
    alpha: float,
    cmap_name: str,
    shade: bool,
) -> None:
    panel_data = payload.get("panel_data", [])
    if not isinstance(panel_data, list) or len(panel_data) == 0:
        return

    def _clean_panel_title(title: str, panel_idx: int) -> str:
        if panel_idx == 0:
            return r"Original $|\mathbf{X}|$"
        title = str(title or "")
        m = re.search(r"(?:update|stage)\s+(\d+)", title, flags=re.IGNORECASE)
        update_idx = int(m.group(1)) if m is not None else int(panel_idx)
        return rf"After Active Set Update {update_idx}"

    z_values: List[np.ndarray] = []
    normalized_panels: List[Dict[str, Any]] = []
    for panel_idx, panel in enumerate(panel_data):
        if not isinstance(panel, dict):
            continue
        x_ch = np.asarray(panel.get("x_ch", []), dtype=np.float32)
        y_tok = np.asarray(panel.get("y_tok", []), dtype=np.float32)
        Z_raw = panel.get("Z")
        if isinstance(Z_raw, torch.Tensor):
            Z = Z_raw.detach().cpu().to(torch.float32).numpy()
        else:
            Z = np.asarray(Z_raw, dtype=np.float32)
        if x_ch.size == 0 or y_tok.size == 0 or Z.size == 0:
            continue
        Xgrid = np.repeat(x_ch.reshape(1, -1), y_tok.shape[0], axis=0)
        Ygrid = np.repeat(y_tok.reshape(-1, 1), x_ch.shape[0], axis=1)
        normalized_panels.append(
            {
                "title": _clean_panel_title(str(panel.get("title", "")), panel_idx),
                "Xgrid": Xgrid,
                "Ygrid": Ygrid,
                "Z": Z,
                "z_label": (
                    r"$|\mathbf{X}|$"
                    if panel_idx == 0
                    else r"$|\mathbf{X}\operatorname{diag}(\mathbf{s})^{-1}|$"
                ),
            }
        )
        z_values.append(Z.reshape(-1))
    if not normalized_panels:
        return

    z_all = np.concatenate(z_values, axis=0) if z_values else np.zeros((1,), dtype=np.float32)
    vmin = float(np.percentile(z_all, p_lo))
    vmax = float(np.percentile(z_all, p_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(z_all))
        vmax = float(np.max(z_all) + 1e-12)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)

    n = len(normalized_panels)
    ncols = 3
    nrows = 2
    fig = plt.figure(figsize=(18.0, 10.8))
    axes: List[object] = []
    surf_last = None
    for idx, panel in enumerate(normalized_panels, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        axes.append(ax)
        surf = ax.plot_surface(
            panel["Xgrid"],
            panel["Ygrid"],
            panel["Z"],
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            cmap=cmap,
            norm=norm,
            shade=bool(shade),
        )
        surf.set_alpha(float(alpha))
        surf_last = surf
        ax.set_title(str(panel["title"]), pad=10, fontsize=12, fontweight="bold")
        ax.set_xlabel("In-Channel")
        ax.set_ylabel("Token")
        ax.set_zlabel(str(panel["z_label"]))
        ax.view_init(elev=float(elev), azim=float(azim))

    for idx in range(len(normalized_panels) + 1, nrows * ncols + 1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        ax.set_axis_off()

    fig.subplots_adjust(left=0.03, right=0.90, top=0.93, bottom=0.04, wspace=0.02, hspace=0.12)
    if surf_last is not None:
        cax = fig.add_axes([0.915, 0.20, 0.016, 0.60])
        cbar = fig.colorbar(surf_last, cax=cax)
        cbar.set_label("Activation magnitude")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_xinv_sweep_grid(
    X: torch.Tensor,
    sweep_records: List[Dict[str, object]],
    *,
    module_name: str,
    out_base: Path,
    max_tokens_plot: int,
    elev: float,
    azim: float,
    dpi: int,
    p_lo: float,
    p_hi: float,
    alpha: float,
    cmap_name: str,
    shade: bool,
) -> Dict[str, Any]:
    payload = build_xinv_grid_payload(
        X=X,
        sweep_records=sweep_records,
        module_name=module_name,
        max_tokens_plot=max_tokens_plot,
    )
    if payload:
        plot_xinv_grid_from_payload(
            payload,
            out_base=out_base,
            elev=elev,
            azim=azim,
            dpi=dpi,
            p_lo=p_lo,
            p_hi=p_hi,
            alpha=alpha,
            cmap_name=cmap_name,
            shade=shade,
        )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser("Inspect coordinate evaluation traces for one layer/group.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_family", type=str, default="auto", choices=["auto", "qwen", "llama", "generic"])
    ap.add_argument("--layer_idx", type=int, required=True)
    ap.add_argument("--group_idx", type=int, default=0)
    ap.add_argument("--module_paths", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="coord_eval_inspect")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--calib_rows", type=int, default=256)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_length", type=int, default=128)
    ap.add_argument("--calib_max_rows_X", type=int, default=32768)
    ap.add_argument("--act_bits", type=int, default=4)
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--act_quant_mode", type=str, default="symmetric", choices=["symmetric", "asymmetric"])
    ap.add_argument("--w_quant_mode", type=str, default="symmetric", choices=["symmetric", "asymmetric"])
    ap.add_argument("--alt_iters", type=int, default=1)
    ap.add_argument("--s_sweeps", type=int, default=1)
    ap.add_argument("--golden_max_iter", type=int, default=20)
    ap.add_argument("--coord_backtrack_steps", type=int, default=5)
    ap.add_argument("--alt_s_init", type=str, default="GD", choices=["heuristic", "ones", "GD"])
    ap.add_argument("--screen_init_k", type=int, default=20)
    ap.add_argument("--screen_append_k", type=int, default=20)
    ap.add_argument("--screen_check_every", type=int, default=20)
    ap.add_argument("--screen_active_max_frac", type=float, default=0.10)
    ap.add_argument("--top_k_plot", type=int, default=20)
    ap.add_argument("--xinv_max_tokens_plot", type=int, default=256)
    ap.add_argument("--xinv_max_in_channels_plot", type=int, default=1024)
    ap.add_argument("--xinv_extra_channels_plot", type=int, default=64)
    ap.add_argument("--xinv_elev", type=float, default=25.0)
    ap.add_argument("--xinv_azim", type=float, default=-60.0)
    ap.add_argument("--xinv_dpi", type=int, default=220)
    ap.add_argument("--xinv_p_lo", type=float, default=5.0)
    ap.add_argument("--xinv_p_hi", type=float, default=99.0)
    ap.add_argument("--xinv_alpha", type=float, default=0.85)
    ap.add_argument("--xinv_cmap", type=str, default="coolwarm")
    ap.add_argument("--xinv_shade", action="store_true")
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

    model_name = str(args.model)
    model_family = infer_model_family(model_name) if args.model_family == "auto" else str(args.model_family)
    module_paths_list = _parse_csv_list(args.module_paths)
    module_paths = tuple(module_paths_list) if module_paths_list else default_module_paths_for_model_family(model_family)
    groups = default_module_groups_for_decoder(module_paths)
    if not (0 <= int(args.group_idx) < len(groups)):
        raise ValueError(f"group_idx={args.group_idx} out of range; available groups={groups}")
    group = tuple(groups[int(args.group_idx)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.tag}__layer{int(args.layer_idx):02d}__group{int(args.group_idx):02d}__{stamp}"
    module_name = f"layers.{int(args.layer_idx)}." + ("{" + ",".join(group) + "}" if len(group) > 1 else group[0])

    cfg = SQCfg(
        act_bits=int(args.act_bits),
        w_bits=int(args.w_bits),
        act_quant_mode=str(args.act_quant_mode),
        w_quant_mode=str(args.w_quant_mode),
        alt_iters=int(args.alt_iters),
        s_sweeps=int(args.s_sweeps),
        golden_max_iter=int(args.golden_max_iter),
        coord_backtrack_steps=int(args.coord_backtrack_steps),
        alt_s_init=str(args.alt_s_init),
        show_tqdm=True,
        show_coord_tqdm=True,
    )
    cfg.screen_init_k = int(args.screen_init_k)
    cfg.screen_append_k = int(args.screen_append_k)
    cfg.screen_check_every = int(args.screen_check_every)
    cfg.screen_active_max_frac = float(args.screen_active_max_frac)
    cfg.collect_coord_eval_traces = True
    cfg.plot_xinv_each_sweep = False
    cfg.xinv_max_tokens_plot = int(args.xinv_max_tokens_plot)
    cfg.xinv_max_in_channels_plot = int(args.xinv_max_in_channels_plot)
    cfg.xinv_extra_channels_plot = int(args.xinv_extra_channels_plot)
    cfg.xinv_elev = float(args.xinv_elev)
    cfg.xinv_azim = float(args.xinv_azim)
    cfg.xinv_dpi = int(args.xinv_dpi)
    cfg.xinv_p_lo = float(args.xinv_p_lo)
    cfg.xinv_p_hi = float(args.xinv_p_hi)
    cfg.xinv_alpha = float(args.xinv_alpha)
    cfg.xinv_cmap = str(args.xinv_cmap)
    cfg.xinv_shade = bool(args.xinv_shade)

    print(f"[info] loading tokenizer/model for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _maybe_set_pad_token(tokenizer, None)
    ds_train = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    calib_texts = make_calib_texts(ds_train, n_rows=int(args.calib_rows))
    token_batches = make_token_batches(
        tokenizer=tokenizer,
        texts=calib_texts,
        batch_size=int(args.calib_batch_size),
        max_length=int(args.calib_max_length),
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    _maybe_set_pad_token(tokenizer, model)

    layer_data = collect_layer_XW_with_early_stop(
        model=model,
        token_batches=token_batches,
        layer_id=int(args.layer_idx),
        module_paths=module_paths,
        device=device,
        max_rows_X=int(args.calib_max_rows_X),
        store_activations_on_cpu=True,
        pin_cpu_memory=True,
        return_W_on_cpu=True,
    )
    if not all(p in layer_data for p in group):
        raise RuntimeError(f"Requested group {group} not fully collected; available={sorted(layer_data.keys())}")

    X_host = _pick_shared_X(layer_data, group)
    W_fp_host_list = [layer_data[p][1] for p in group]
    X = X_host.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    W_fp_list = [w.to(device=device, dtype=torch.float32, non_blocking=True).contiguous() for w in W_fp_host_list]

    s_h = smoothquant_heuristic_s_group(X, W_fp_list, alpha=0.5, normalize_geom_mean=True)
    if cfg.alt_s_init == "heuristic":
        s0 = s_h
    elif cfg.alt_s_init == "ones":
        s0 = torch.ones((X.shape[1],), device=device, dtype=torch.float32)
    else:
        s0 = gd_init_s_ste_group(X=X, W_fp_list=W_fp_list, cfg=cfg)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] running alternating optimizer for {module_name}")
    s_star, W_hat_list, U_list, f_final, f_init = alternating_optimize_s_and_What_group(
        X=X,
        W_fp_list=W_fp_list,
        s_init=s0,
        cfg=cfg,
        plots_dir=plots_dir,
        module_name=module_name,
    )
    trace_payload = getattr(cfg, "ablate_last_coord_eval_traces", {})
    trace_payload = dict(trace_payload) if isinstance(trace_payload, dict) else {}
    xinv_payload = getattr(cfg, "ablate_last_xinv_sweeps", {})
    xinv_payload = dict(xinv_payload) if isinstance(xinv_payload, dict) else {}
    trace_payload.update(
        {
            "run_id": run_id,
            "model": model_name,
            "model_family": model_family,
            "layer_idx": int(args.layer_idx),
            "group_idx": int(args.group_idx),
            "group": list(group),
            "module_name": module_name,
            "f_init": float(f_init),
            "f_final": float(f_final),
            "s_final": s_star.detach().cpu(),
            "config": {
                "act_bits": int(args.act_bits),
                "w_bits": int(args.w_bits),
                "act_quant_mode": str(args.act_quant_mode),
                "w_quant_mode": str(args.w_quant_mode),
                "alt_iters": int(args.alt_iters),
                "s_sweeps": int(args.s_sweeps),
                "golden_max_iter": int(args.golden_max_iter),
                "coord_backtrack_steps": int(args.coord_backtrack_steps),
                "alt_s_init": str(args.alt_s_init),
            },
        }
    )
    if xinv_payload:
        trace_payload["xinv_sweeps"] = xinv_payload

    traces_out = out_dir / f"{run_id}__coord_eval_traces.pt"
    torch.save(trace_payload, traces_out)
    print(f"[save] raw traces -> {traces_out}")

    trace_list = trace_payload.get("coord_eval_sweeps", [])
    if not isinstance(trace_list, list):
        trace_list = []
    plot_base = out_dir / f"{run_id}__top_coords"
    _plot_topk_coord_traces(
        trace_list,
        layer_idx=int(args.layer_idx),
        module_name=module_name,
        top_k=int(args.top_k_plot),
        out_base=plot_base,
    )
    print(f"[save] plots -> {plot_base.with_suffix('.png')} and {plot_base.with_suffix('.pdf')}")
    xinv_sweeps = xinv_payload.get("stages", []) if isinstance(xinv_payload, dict) else []
    if isinstance(xinv_sweeps, list) and len(xinv_sweeps) > 0:
        xinv_base = out_dir / f"{run_id}__activation_sweeps"
        xinv_grid_payload = _plot_xinv_sweep_grid(
            X=X,
            sweep_records=xinv_sweeps,
            module_name=module_name,
            out_base=xinv_base,
            max_tokens_plot=int(args.xinv_max_tokens_plot),
            elev=float(args.xinv_elev),
            azim=float(args.xinv_azim),
            dpi=int(args.xinv_dpi),
            p_lo=float(args.xinv_p_lo),
            p_hi=float(args.xinv_p_hi),
            alpha=float(args.xinv_alpha),
            cmap_name=str(args.xinv_cmap),
            shade=bool(args.xinv_shade),
        )
        print(f"[save] activation sweep grid -> {xinv_base.with_suffix('.png')}")
        xinv_payload_out = out_dir / f"{run_id}__activation_sweeps_data.pt"
        torch.save(
            {
                "run_id": run_id,
                "module_name": module_name,
                "plot_config": {
                    "elev": float(args.xinv_elev),
                    "azim": float(args.xinv_azim),
                    "dpi": int(args.xinv_dpi),
                    "p_lo": float(args.xinv_p_lo),
                    "p_hi": float(args.xinv_p_hi),
                    "alpha": float(args.xinv_alpha),
                    "cmap_name": str(args.xinv_cmap),
                    "shade": bool(args.xinv_shade),
                },
                "grid_payload": xinv_grid_payload,
            },
            xinv_payload_out,
        )
        print(f"[save] activation grid data -> {xinv_payload_out}")
    else:
        print("[warn] no activation sweep snapshots were recorded")

    del W_hat_list, U_list, X, W_fp_list, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

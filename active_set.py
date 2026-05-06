from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from quantization_utils import (
    quantize_per_token_maxabs_fp32,
    quantize_per_token_maxabs_ste,
    quantize_weights_per_out_channel_fp32,
    quantize_weights_per_out_channel_ste,
)
from utils import SQCfg


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def smoothquant_heuristic_s_group(
    X: torch.Tensor,
    W_list: Sequence[torch.Tensor],
    alpha: float = 0.5,
    eps: float = 1e-8,
    normalize_geom_mean: bool = True,
) -> torch.Tensor:
    """
    SmoothQuant heuristic for a group that shares the same activation X.

    For single module you used:
      s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    Here we aggregate the weight scale across modules by taking max over modules:
      w_max_j = max_i max_out(|W_i[j, :]|)  (i.e., max over output dims, then max over i)
    """
    x_max = torch.clamp(X.abs().amax(dim=0), min=eps)  # (Ci,)
    w_max_all = None
    for W in W_list:
        w_max = torch.clamp(W.abs().amax(dim=1), min=eps)  # (Ci,)
        w_max_all = w_max if w_max_all is None else torch.maximum(w_max_all, w_max)
    assert w_max_all is not None
    s = (x_max ** alpha) / (w_max_all ** (1.0 - alpha))
    s = torch.clamp(s, min=eps)
    if normalize_geom_mean:
        s = s / torch.exp(torch.mean(torch.log(s)))
    return s



def gd_init_s_ste_group(
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    cfg: "SQCfg",
    *,
    gd_steps: int = 100,
    gd_lr: float = 0.01,
    gd_rows: int = 100_000,
    gd_opt: str = "adam",
    gd_print: bool = False,
    gd_minibatch_frac: float = 0.25,
    gd_minibatch_rows_min: int = 8192,
    gd_minibatch_rows_max: int = 200_000,
    gd_shuffle_each_step: bool = True,
    gd_desc: str = "",
) -> torch.Tensor:
    """
    Grouped version of gd_init_s_ste:
      - one shared ell (log s) for the group,
      - loss summed across modules:
          sum_i || X W_fp_i - Q_a(X/s) Q_w(W_fp_i*s) ||_F^2
    Notes:
      - Uses STE quantizers (same as your single-module GD init).
      - Uses W_fp_i directly on the weight side (same choice as your single-module GD init).
    """
    assert X.dtype == torch.float32
    device = X.device
    eps = float(getattr(cfg, "s_eps", 1e-8))
    if len(W_fp_list) == 0:
        raise ValueError("gd_init_s_ste_group: W_fp_list must be non-empty.")
    Ci = int(X.shape[1])
    for W_fp in W_fp_list:
        assert W_fp.dtype == torch.float32
        assert W_fp.shape[0] == Ci

    # row subsampling for GD init (same as your single-module init)
    if int(gd_rows) > 0 and X.shape[0] > int(gd_rows):
        idx = torch.randperm(X.shape[0], device=device)[: int(gd_rows)]
        Xg = X.index_select(0, idx)
    else:
        Xg = X
    Ng = int(Xg.shape[0])

    ell = torch.zeros((Ci,), device=device, dtype=torch.float32, requires_grad=True)

    m = float(getattr(cfg, "s_bounds_mult", 10.0))
    ell_lo = math.log(max(1.0 / m, eps))
    ell_hi = math.log(max(m, eps))

    opt_name = str(gd_opt).lower().strip()
    if opt_name == "sgd":
        opt = torch.optim.SGD([ell], lr=float(gd_lr), momentum=0.0)
    elif opt_name == "adam":
        opt = torch.optim.Adam([ell], lr=float(gd_lr), betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError("gd_opt must be one of {'sgd','adam'}")

    frac = float(gd_minibatch_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError("gd_minibatch_frac must be in (0,1].")

    mb = int(round(frac * Ng))
    mb = max(mb, int(gd_minibatch_rows_min))
    mb = min(mb, Ng)
    mb = min(mb, int(gd_minibatch_rows_max)) if int(gd_minibatch_rows_max) > 0 else mb

    def sample_minibatch_indices() -> torch.Tensor:
        if mb == Ng:
            return torch.arange(Ng, device=device)
        if gd_shuffle_each_step:
            return torch.randperm(Ng, device=device)[:mb]
        return torch.arange(mb, device=device)

    if gd_print:
        desc = f" {gd_desc}" if gd_desc else ""
        print(
            f"[GD-init-group{desc}] start steps={int(max(0, gd_steps))} "
            f"lr={float(gd_lr):.3e} opt={opt_name} rows={Ng} mb={mb}/{Ng} ({mb/max(1, Ng):.2f})"
        )

    with torch.enable_grad():
        for t in range(int(max(0, gd_steps))):
            opt.zero_grad(set_to_none=True)

            idx_mb = sample_minibatch_indices()
            Xmb = Xg.index_select(0, idx_mb)

            s = torch.exp(ell).clamp_min(eps)

            # shared activation quantization
            Xq = quantize_per_token_maxabs_ste(
                Xmb / s.unsqueeze(0),
                bits=int(cfg.act_bits),
                eps=eps,
                mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
            )

            # summed loss across branches
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for W_fp in W_fp_list:
                Ymb = Xmb @ W_fp
                Uq = quantize_weights_per_out_channel_ste(
                    W_fp * s.unsqueeze(1),
                    bits=int(cfg.w_bits),
                    eps=eps,
                    mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
                )
                P = Xq @ Uq
                R = Ymb - P
                loss = loss + torch.sum(R * R)

            loss.backward()
            grad_norm = float(ell.grad.norm().item()) if ell.grad is not None else float("nan")
            opt.step()

            with torch.no_grad():
                ell.clamp_(ell_lo, ell_hi)
                ell -= torch.mean(ell)

            if gd_print:
                with torch.no_grad():
                    s_dbg = torch.exp(ell).clamp_min(eps)
                    s_dbg = s_dbg / torch.exp(torch.mean(torch.log(s_dbg)))
                    s_min = float(s_dbg.min().item())
                    s_max = float(s_dbg.max().item())
                    s_mean = float(s_dbg.mean().item())
                print(
                    f"[GD-init-group{desc}] step {t+1:03d}/{gd_steps} "
                    f"loss_mb={float(loss.item()):.6e} grad_norm={grad_norm:.6e} "
                    f"s[min,max,mean]=({s_min:.6e}, {s_max:.6e}, {s_mean:.6e})"
                )

    with torch.no_grad():
        s = torch.exp(ell).clamp_min(eps)
        s = s / torch.exp(torch.mean(torch.log(s)))
        if gd_print:
            loss_full = torch.zeros((), device=device, dtype=torch.float32)
            eval_row_chunk = int(getattr(cfg, "obj_row_chunk", 256))
            act_chunk_cols = int(getattr(cfg, "act_quant_chunk_cols", 1024))
            w_chunk_rows = int(getattr(cfg, "w_quant_chunk_rows", 0))
            Uq_list = [
                quantize_weights_per_out_channel_fp32(
                    W_fp * s.unsqueeze(1),
                    bits=int(cfg.w_bits),
                    eps=eps,
                    chunk_rows=w_chunk_rows,
                    mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
                )
                for W_fp in W_fp_list
            ]
            inv_s = torch.reciprocal(s)
            for r0 in range(0, Ng, eval_row_chunk):
                r1 = min(r0 + eval_row_chunk, Ng)
                X_chunk = Xg[r0:r1, :]
                Xq_chunk = quantize_per_token_maxabs_fp32(
                    X_chunk * inv_s.unsqueeze(0),
                    bits=int(cfg.act_bits),
                    eps=eps,
                    chunk_cols=act_chunk_cols,
                    mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
                )
                for W_fp, Uq in zip(W_fp_list, Uq_list):
                    Y_chunk = X_chunk @ W_fp
                    R = Y_chunk - (Xq_chunk @ Uq)
                    loss_full = loss_full + torch.sum(R * R)
            print(
                f"[GD-init-group{desc}] done final_loss_rows={float(loss_full.item()):.6e} "
                f"s[min,max,mean]=({float(s.min().item()):.6e}, {float(s.max().item()):.6e}, {float(s.mean().item()):.6e})"
            )
        return s.detach()
        
# ============================================================
# STE screening for GROUPS (sum of losses)
# ============================================================

def ste_grad_scores_for_active_set_group(
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    W_hat_list: Sequence[torch.Tensor],
    s: torch.Tensor,
    cfg: "SQCfg",
    *,
    rows: int = 8192,
    rows_max: int = 200_000,
    use_abs: bool = True,
    subtract_mean: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Same as your ste_grad_scores_for_active_set but sum reconstruction losses across modules.

    Loss(s) = sum_i || X W_fp_i - Q_a(X/s) Q_w(W_hat_i*s) ||_F^2
    """
    assert X.dtype == torch.float32 and s.dtype == torch.float32
    dev = X.device
    eps = float(getattr(cfg, "s_eps", 1e-8))

    N, Ci = X.shape
    assert s.numel() == Ci
    assert len(W_fp_list) == len(W_hat_list) and len(W_fp_list) > 0
    for W_fp, W_hat in zip(W_fp_list, W_hat_list):
        assert W_fp.dtype == torch.float32 and W_hat.dtype == torch.float32
        assert W_fp.shape[0] == Ci and W_hat.shape == W_fp.shape

    min_rows = int(max(1, getattr(cfg, "screen_ste_rows_min", 512)))
    allow_oom_retry = bool(getattr(cfg, "screen_ste_oom_retry", True))

    def _compute_once(r: int) -> torch.Tensor:
        r = int(max(1, r))
        r = min(r, N)
        r = min(r, int(rows_max)) if int(rows_max) > 0 else r

        if r < N:
            if generator is None:
                idx = torch.randperm(N, device=dev)[:r]
            else:
                idx = torch.randperm(N, device=dev, generator=generator)[:r]
            Xb = X.index_select(0, idx)
        else:
            Xb = X

        ell = torch.log(s.clamp_min(eps)).detach().clone().requires_grad_(True)

        with torch.enable_grad():
            sb = torch.exp(ell).clamp_min(eps)
            Xq = quantize_per_token_maxabs_ste(
                Xb / sb.unsqueeze(0),
                bits=int(cfg.act_bits),
                eps=eps,
                mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
            )

            loss = torch.zeros((), device=dev, dtype=torch.float32)
            for W_fp, W_hat in zip(W_fp_list, W_hat_list):
                with torch.no_grad():
                    Yb = Xb @ W_fp
                Uq = quantize_weights_per_out_channel_ste(
                    W_hat * sb.unsqueeze(1),
                    bits=int(cfg.w_bits),
                    eps=eps,
                    mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
                )
                P = Xq @ Uq
                loss = loss + torch.sum((Yb - P) * (Yb - P))

            g = torch.autograd.grad(loss, ell, retain_graph=False, create_graph=False)[0]
            return g.detach()

    r = int(max(1, rows))
    r = min(r, N)
    r = min(r, int(rows_max)) if int(rows_max) > 0 else r

    while True:
        try:
            g = _compute_once(r)
            break
        except Exception as e:
            if dev.type != "cuda" or (not allow_oom_retry) or (not _is_cuda_oom_exception(e)):
                raise

            next_r = max(min_rows, r // 2)
            if next_r >= r:
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[screen-oom] CUDA OOM during STE screening; retrying with fewer rows "
                f"({r} -> {next_r})",
                flush=True,
            )
            r = next_r

    if subtract_mean:
        g = g - torch.mean(g)
    return g.abs() if use_abs else g

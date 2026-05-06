# gptq.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def qmax_signed(bits: int) -> int:
    return (1 << (bits - 1)) - 1


class _SymmetricPerRowQuantizer:
    """
    Quantize each row (i.e., each output channel) with a symmetric maxabs scale.
    Intended for GPTQ-style per-row quant of weight matrices shaped (Co, Ci).
    """

    def __init__(self, bits: int, eps: float = 1e-8):
        self.bits = int(bits)
        self.eps = float(eps)
        self.maxq = qmax_signed(self.bits)
        self.scale: Optional[torch.Tensor] = None  # (Co,1)

    @torch.no_grad()
    def find_params(self, W: torch.Tensor) -> None:
        # W: (Co, Ci)
        maxabs = W.abs().amax(dim=1, keepdim=True).clamp_min(self.eps)
        self.scale = (maxabs / float(self.maxq)).clamp_min(self.eps)


@torch.no_grad()
def _quantize_per_row_symmetric(W: torch.Tensor, scale: torch.Tensor, maxq: int) -> torch.Tensor:
    q = torch.round(W / scale).clamp(-maxq, maxq)
    return q * scale


@torch.no_grad()
def gptq_quantize_U_from_H(
    H: torch.Tensor,
    U_init: torch.Tensor,
    *,
    cfg: Optional[Any] = None,
    w_bits: Optional[int] = None,
    eps: float = 1e-8,
    blocksize: int = 128,
    percdamp: float = 0.01,
    actorder: bool = False,
    cholesky_max_tries: int = 12,
    cholesky_damp_growth: float = 10.0,
    cholesky_damp_add: float = 0.0,
    cholesky_symmetrize: bool = True,
    cholesky_verbose: bool = False,
    show_tqdm: bool = False,
    desc: str = "GPTQ(U)",
) -> torch.Tensor:
    """
    GPTQ-style quantization for U (Ci,Co) given Hessian H = 2 Xq^T Xq.

    Returns Q (Ci,Co) quantized.

    Notes:
    - Internally works on W = U^T (Co,Ci) with per-row scales.
    """
    dev = H.device
    if cfg is not None:
        if w_bits is None:
            w_bits = int(getattr(cfg, "w_bits"))
        eps = float(getattr(cfg, "s_eps", eps))
        blocksize = int(getattr(cfg, "gptq_blocksize", blocksize))
        percdamp = float(getattr(cfg, "gptq_percdamp", percdamp))
        actorder = bool(getattr(cfg, "gptq_actorder", actorder))
        cholesky_max_tries = int(getattr(cfg, "gptq_cholesky_max_tries", cholesky_max_tries))
        cholesky_damp_growth = float(getattr(cfg, "gptq_cholesky_damp_growth", cholesky_damp_growth))
        cholesky_damp_add = float(getattr(cfg, "gptq_cholesky_damp_add", cholesky_damp_add))
        cholesky_symmetrize = bool(getattr(cfg, "gptq_cholesky_symmetrize", cholesky_symmetrize))
        cholesky_verbose = bool(getattr(cfg, "gptq_cholesky_verbose", cholesky_verbose))
        show_tqdm = bool(getattr(cfg, "show_tqdm", show_tqdm))
    if w_bits is None:
        raise ValueError("gptq_quantize_U_from_H requires w_bits when cfg is not provided.")
    eps = float(eps)

    if H.dtype != torch.float32:
        H = H.to(torch.float32)
    if U_init.dtype != torch.float32:
        U_init = U_init.to(torch.float32)

    Ci = U_init.shape[0]
    Co = U_init.shape[1]
    assert U_init.shape == (Ci, Co)
    assert H.shape == (Ci, Ci)

    # Work with (Co, Ci)
    W = U_init.T.contiguous()  # (Co, Ci)

    # Handle dead directions
    dead = torch.diag(H) == 0
    if dead.any():
        H = H.clone()
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

    # Optional activation ordering
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)
        H = H[perm][:, perm]
        W = W[:, perm]
    else:
        invperm = None

    # Per-row symmetric quantizer
    quantizer = _SymmetricPerRowQuantizer(bits=int(w_bits), eps=eps)
    quantizer.find_params(W)
    assert quantizer.scale is not None

    # Damping for Cholesky
    base_damp = float(percdamp) * torch.mean(torch.diag(H)).clamp_min(eps)
    damp = float(base_damp.item() if torch.is_tensor(base_damp) else base_damp)

    max_tries = int(cholesky_max_tries)
    growth = float(cholesky_damp_growth)
    additive = float(cholesky_damp_add)
    symmetrize = bool(cholesky_symmetrize)
    verbose = bool(cholesky_verbose)

    H_base = H
    if symmetrize:
        H_base = 0.5 * (H_base + H_base.T)

    diag_idx = torch.arange(Ci, device=dev)

    last_err = None
    Hinv_chol = None
    for t in range(max_tries):
        try:
            H_try = H_base.clone()
            H_try[diag_idx, diag_idx] += damp
            Hc = torch.linalg.cholesky(H_try)
            Hi = torch.cholesky_inverse(Hc)
            Hinv_chol = torch.linalg.cholesky(Hi, upper=True)
            last_err = None
            if verbose and t > 0:
                print(f"[{desc}] Cholesky succeeded after {t} retries; final damp={damp:.3e}")
            break
        except torch._C._LinAlgError as e:
            last_err = e
            if verbose:
                print(f"[{desc}] Cholesky failed (try {t+1}/{max_tries}) damp={damp:.3e}: {e}")
            damp = damp * growth + additive

    if last_err is not None or Hinv_chol is None:
        raise last_err

    Q = torch.zeros_like(W)  # (Co,Ci)

    blocksize = int(blocksize)
    blocks = range(0, Ci, blocksize)
    pbar = tqdm(blocks, desc=desc, leave=False) if show_tqdm else None
    it = pbar if pbar is not None else blocks

    for i1 in it:
        i2 = min(i1 + blocksize, Ci)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        Hinv1 = Hinv_chol[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i].clamp_min(eps)

            q = _quantize_per_row_symmetric(w.unsqueeze(1), quantizer.scale, quantizer.maxq).flatten()
            Q1[:, i] = q

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        if i2 < Ci:
            W[:, i2:] -= Err1.matmul(Hinv_chol[i1:i2, i2:])

    if pbar is not None:
        pbar.close()

    if invperm is not None:
        Q = Q[:, invperm]

    return Q.T.contiguous()  # (Ci,Co)


@torch.no_grad()
def gptq_quantize_U_from_Xq(
    Xq: torch.Tensor,
    U_init: torch.Tensor,
    *,
    cfg: Optional[Any] = None,
    w_bits: Optional[int] = None,
    eps: float = 1e-8,
    blocksize: int = 128,
    percdamp: float = 0.01,
    actorder: bool = False,
    cholesky_max_tries: int = 12,
    cholesky_damp_growth: float = 10.0,
    cholesky_damp_add: float = 0.0,
    cholesky_symmetrize: bool = True,
    cholesky_verbose: bool = False,
    show_tqdm: bool = False,
    desc: str = "GPTQ(U)",
) -> torch.Tensor:
    """
    Backward-compatible wrapper that forms H from Xq for existing callers.
    """
    if Xq.dtype != torch.float32:
        Xq = Xq.to(torch.float32)
    H = 2.0 * (Xq.T @ Xq).to(torch.float32)
    return gptq_quantize_U_from_H(
        H,
        U_init,
        cfg=cfg,
        w_bits=w_bits,
        eps=eps,
        blocksize=blocksize,
        percdamp=percdamp,
        actorder=actorder,
        cholesky_max_tries=cholesky_max_tries,
        cholesky_damp_growth=cholesky_damp_growth,
        cholesky_damp_add=cholesky_damp_add,
        cholesky_symmetrize=cholesky_symmetrize,
        cholesky_verbose=cholesky_verbose,
        show_tqdm=show_tqdm,
        desc=desc,
    )

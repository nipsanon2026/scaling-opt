from __future__ import annotations

from typing import Sequence, Tuple

import torch

from utils import SQCfg


@torch.no_grad()
def ridge_ls_from_gram_fp32(G: torch.Tensor, XtY: torch.Tensor, cfg: SQCfg) -> torch.Tensor:
    assert G.dtype == torch.float32 and XtY.dtype == torch.float32
    Ci = G.shape[0]
    diag_mean = torch.mean(torch.diag(G)).clamp_min(cfg.s_eps)
    lam = float(cfg.ls_ridge) * float(diag_mean.item())
    G_reg = G + lam * torch.eye(Ci, device=G.device, dtype=torch.float32)
    return torch.linalg.solve(G_reg, XtY)


@torch.no_grad()
def compute_c_and_B_streaming(
    X: torch.Tensor,
    Xq: torch.Tensor,
    W_fp: torch.Tensor,
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


@torch.no_grad()
def obj_from_gram_stats_fp64(
    G64: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    U: torch.Tensor,
    *,
    col_chunk: int = 256,
) -> float:
    """
    Evaluate ||XW - XqU||_F^2 from sufficient statistics:
      c = ||XW||_F^2, B = Xq^T (XW), G = Xq^T Xq.
    """
    assert G64.dtype == torch.float64 and U.dtype == torch.float32
    assert G64.dim() == 2 and U.dim() == 2 and G64.shape[0] == G64.shape[1] == U.shape[0]
    assert B.shape == U.shape

    B64 = B.to(torch.float64)
    U64 = U.to(torch.float64)
    term1 = torch.sum(U64 * B64)

    cc = int(max(1, col_chunk))
    term2 = torch.zeros((), device=U.device, dtype=torch.float64)
    for o0 in range(0, U.shape[1], cc):
        o1 = min(o0 + cc, U.shape[1])
        Uc64 = U[:, o0:o1].to(torch.float64)
        GUc64 = G64 @ Uc64
        term2 += torch.sum(Uc64 * GUc64)
        del Uc64, GUc64

    c64 = c.to(torch.float64) if torch.is_tensor(c) else torch.tensor(float(c), device=U.device, dtype=torch.float64)
    return float((c64 - 2.0 * term1 + term2).item())


@torch.no_grad()
def group_obj_from_gram_stats_fp64(
    G64: torch.Tensor,
    branch_stats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    U_list: Sequence[torch.Tensor],
    *,
    col_chunk: int = 256,
) -> float:
    assert len(branch_stats) == len(U_list)
    obj = 0.0
    for (c64, B64), U in zip(branch_stats, U_list):
        obj += obj_from_gram_stats_fp64(G64, B64, c64, U, col_chunk=col_chunk)
    return float(obj)


@torch.no_grad()
def obj_from_gram_stats_fp32(
    G: torch.Tensor,
    B: torch.Tensor,
    c: torch.Tensor,
    U: torch.Tensor,
    *,
    col_chunk: int = 256,
) -> float:
    return obj_from_gram_stats_fp64(G.to(torch.float64), B, c, U, col_chunk=col_chunk)


@torch.no_grad()
def group_obj_from_gram_stats_fp32(
    G: torch.Tensor,
    branch_stats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    U_list: Sequence[torch.Tensor],
    *,
    col_chunk: int = 256,
) -> float:
    return group_obj_from_gram_stats_fp64(G.to(torch.float64), branch_stats, U_list, col_chunk=col_chunk)


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
    _, Ci = X64.shape
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
    _, Ci = X32.shape
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

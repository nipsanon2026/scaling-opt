from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from active_set import ste_grad_scores_for_active_set_group
from calibration_utils import _safe_name, save_coord_progress_plot, save_x_over_s_surface_plot
from gptq import gptq_quantize_U_from_H
from gram_utils import (
    build_G64_streaming_tiled,
    compute_c_and_B_streaming,
    group_obj_from_gram_stats_fp64,
    gram32_add_tiled_,
    gram64_add_tiled_,
    ridge_ls_from_gram_fp32,
)
from quantization_utils import (
    build_Xq_from_s_streaming,
    canonicalize_quant_mode,
    compute_per_out_channel_quant_params,
    compute_per_token_quant_params,
    qmax_signed,
    quant_int_bounds,
    quantize_per_token_maxabs_fp32,
    quantize_scalar_with_delta,
    quantize_with_params,
    quantize_weights_per_out_channel_fp32,
)
from utils import (
    SQCfg,
    _runtime_w_chunk_rows,
    build_U_from_s_What,
    compute_GU_chunked,
    golden_section_search_1d,
    obj_from_X_Xq_W_U_streaming,
)


def _accum_unstable_rows_chunked(
    X_old: torch.Tensor,         # (R, Ci)
    X_new: torch.Tensor,         # (R, Ci)
    Y_rows: torch.Tensor,        # (R, Co)
    U: torch.Tensor,             # (Ci, Co)
    j: int,
    du: torch.Tensor,            # (Co,)
    dU_cols: Optional[torch.Tensor],
    cols_u: Optional[torch.Tensor],
    chunk_cols: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert X_old.dtype == torch.float32 and X_new.dtype == torch.float32
    assert Y_rows.dtype == torch.float32 and U.dtype == torch.float32 and du.dtype == torch.float32
    R, Ci = X_old.shape
    _, Co = Y_rows.shape
    assert U.shape == (Ci, Co)

    add_term1 = torch.zeros((), device=U.device, dtype=torch.float32)
    add_term2 = torch.zeros((), device=U.device, dtype=torch.float32)

    use_dU = (dU_cols is not None) and (cols_u is not None) and (cols_u.numel() > 0)
    if use_dU:
        cols_u = cols_u.to(device=U.device)

    xold_j = X_old[:, j].unsqueeze(1)
    xnew_j = X_new[:, j].unsqueeze(1)

    cc = int(max(1, chunk_cols))
    for c0 in range(0, Co, cc):
        c1 = min(c0 + cc, Co)

        Uc = U[:, c0:c1]
        P_old_c = X_old @ Uc
        P_new_c = X_new @ Uc

        duc = du[c0:c1].unsqueeze(0)
        P_old_c = P_old_c + xold_j * duc
        P_new_c = P_new_c + xnew_j * duc

        if use_dU:
            in_chunk = (cols_u >= c0) & (cols_u < c1)
            if bool(in_chunk.any()):
                cols_chunk = cols_u[in_chunk]
                local = (cols_chunk - c0).to(torch.long)
                pos = in_chunk.nonzero(as_tuple=False).squeeze(1)
                dUc = dU_cols.index_select(1, pos)

                add_old = X_old @ dUc
                add_new = X_new @ dUc

                P_old_c.index_add_(1, local, add_old)
                P_new_c.index_add_(1, local, add_new)

        Yc = Y_rows[:, c0:c1]
        dP_c = (P_new_c - P_old_c)

        add_term1 = add_term1 + torch.sum(dP_c * Yc)
        add_term2 = add_term2 + (torch.sum(P_new_c * P_new_c) - torch.sum(P_old_c * P_old_c))

    return add_term1, add_term2


@torch.no_grad()
def project_U_ls_rtn(U_ls: torch.Tensor, cfg: SQCfg) -> torch.Tensor:
    return quantize_weights_per_out_channel_fp32(
        U_ls.to(torch.float32),
        bits=int(cfg.w_bits),
        eps=float(cfg.s_eps),
        chunk_rows=_runtime_w_chunk_rows(cfg),
        mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
    )


# ============================================================
# GROUP evaluator
# ============================================================

@dataclass
class CandidateCacheBranch:
    """
    Per-branch pieces needed for commit.
    """
    zrow_new: torch.Tensor
    urowj_new: torch.Tensor
    cols_u: Optional[torch.Tensor]
    # For unstable cols correction inside eval
    U_new_cols: Optional[torch.Tensor]
    U_old_cols: Optional[torch.Tensor]
    dU_cols: Optional[torch.Tensor]
    # deltaB_j is per-branch (depends on W_fp_i)
    deltaB_j: torch.Tensor
    # row-j delta in U
    du: torch.Tensor
    # objective terms at candidate (fp64 scalars)
    term1_new_64: torch.Tensor
    term2_new_64: torch.Tensor


@dataclass
class CandidateCacheGroup:
    j: int
    sj_new: float
    a_new: torch.Tensor
    xqj_new: torch.Tensor
    idx_u_rows: Optional[torch.Tensor]
    deltaG_j: torch.Tensor
    deltaG_jj: float
    branches: List[CandidateCacheBranch]
    fast_path: bool
    obj: float


@dataclass
class CoordEvalBranchContext:
    branch_idx: int
    W_hat_rowj: torch.Tensor
    urowj_old: torch.Tensor
    B_rowj_64: torch.Tensor
    GU_rowj_64: torch.Tensor
    max_excl_rowj: Optional[torch.Tensor]
    min_excl_rowj: Optional[torch.Tensor]
    w_scale_row: Optional[torch.Tensor]
    w_zp_row: Optional[torch.Tensor]


@dataclass
class CoordEvalContext:
    j: int
    x_col_j: torch.Tensor
    xqj_old: torch.Tensor
    gcol: torch.Tensor
    Gjj: float
    act_max_excl_j: Optional[torch.Tensor]
    act_min_excl_j: Optional[torch.Tensor]
    branches: List[CoordEvalBranchContext]


class SCoordCandidateEvaluatorGramGroup:
    """
    Group version of your SCoordCandidateEvaluatorGram.

    Shared across group:
      - X, s, A, Xq
      - G64/G (Gram of Xq), activation top2 stats, delta_x
    Per-branch i:
      - W_fp_i, W_hat_i, Z_i, U_i
      - B64_i, c64_i
      - GU_i
      - weight top2 stats (delta_w_i, w_top1/2_i)
      - scalar term1_i = <U_i, B_i>
      - scalar term2_i = <U_i, G U_i> (computed in fp64 via G64)
    Objective:
      f = sum_i (c_i - 2 term1_i + term2_i)
    """

    @torch.no_grad()
    def __init__(
        self,
        X: torch.Tensor,
        W_fp_list: Sequence[torch.Tensor],
        s: torch.Tensor,
        W_hat_list: Sequence[torch.Tensor],
        cfg: SQCfg,
    ):
        assert X.dtype == torch.float32 and s.dtype == torch.float32
        assert len(W_fp_list) == len(W_hat_list) and len(W_fp_list) > 0

        self.X = X
        self.cfg = cfg
        self.eps = float(getattr(cfg, "s_eps", 1e-8))
        self.s = s.clone()
        self.act_quant_mode = canonicalize_quant_mode(str(getattr(cfg, "act_quant_mode", "symmetric")))
        self.w_quant_mode = canonicalize_quant_mode(str(getattr(cfg, "w_quant_mode", "symmetric")))
        self.act_symmetric = self.act_quant_mode == "symmetric"
        self.w_symmetric = self.w_quant_mode == "symmetric"

        self.N, self.Ci = X.shape
        self.act_qmin, self.act_qmax = quant_int_bounds(int(cfg.act_bits), self.act_quant_mode)
        self.w_qmin, self.w_qmax = quant_int_bounds(int(cfg.w_bits), self.w_quant_mode)
        self.act_qrange = int(self.act_qmax - self.act_qmin)
        self.w_qrange = int(self.w_qmax - self.w_qmin)
        self.act_qmin_f = float(self.act_qmin)
        self.w_qmin_f = float(self.w_qmin)
        self.qx = qmax_signed(int(cfg.act_bits))
        self.qw = qmax_signed(int(cfg.w_bits))
        self.inv_s = torch.reciprocal(self.s.clamp_min(self.eps))

        self.act_top2_chunk_cols = int(getattr(cfg, "act_top2_chunk_cols", 512))
        self.w_top2_chunk_rows = int(getattr(cfg, "w_top2_chunk_rows", 2048))

        # Shared activation side (A = X/s not stored; computed on demand to save GPU memory)
        self.Xq = build_Xq_from_s_streaming(
            self.X,
            self.s,
            act_bits=int(cfg.act_bits),
            eps=self.eps,
            act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
            mode=self.act_quant_mode,
        )

        # Shared Gram
        g_row_chunk = int(getattr(cfg, "G_row_chunk", getattr(cfg, "cB_row_chunk", 1024)))
        g64_tile = int(getattr(cfg, "G64_tile", 1024))
        self.G64 = build_G64_streaming_tiled(self.Xq, row_chunk=g_row_chunk, tile=g64_tile)
        self.G = self.G64.to(torch.float32)

        # Activation top2 stats
        self._recompute_top2_stats_activation()

        # Per-branch state containers
        self.branches: List[Dict[str, Any]] = []
        w_chunk_rows = _runtime_w_chunk_rows(cfg)
        for W_fp, W_hat in zip(W_fp_list, W_hat_list):
            assert W_fp.dtype == torch.float32 and W_hat.dtype == torch.float32
            assert W_fp.shape[0] == self.Ci and W_hat.shape == W_fp.shape

            b: Dict[str, Any] = {}
            b["W_fp"] = W_fp
            b["W_hat"] = W_hat
            b["Co"] = int(W_fp.shape[1])

            # Z and U
            b["Z"] = b["W_hat"] * self.s.unsqueeze(1)
            b["U"] = quantize_weights_per_out_channel_fp32(
                b["Z"],
                bits=int(cfg.w_bits),
                eps=self.eps,
                chunk_rows=w_chunk_rows,
                mode=self.w_quant_mode,
            )

            # c_i, B_i
            c64, B64 = compute_c_and_B_streaming(
                self.X, self.Xq, b["W_fp"], row_chunk=int(getattr(cfg, "cB_row_chunk", 1024))
            )
            b["c64"] = c64.to(torch.float64)
            b["B64"] = B64.to(torch.float64)

            # weight top2 stats
            self._recompute_top2_stats_weight(b)

            # GU, terms
            self._recompute_GU_and_terms_branch(b)

            self.branches.append(b)

        # Branch output-width groups are static across coordinate proposals.
        self.co_groups: Dict[int, List[int]] = {}
        for bi, b in enumerate(self.branches):
            co = int(b["Co"])
            if co not in self.co_groups:
                self.co_groups[co] = []
            self.co_groups[co].append(bi)
        self.group_eval_packs: List[Dict[str, Any]] = []
        self.branch_pack_pos: List[Tuple[int, int]] = [(-1, -1) for _ in range(len(self.branches))]
        self._build_group_eval_packs()

    @torch.no_grad()
    def _build_group_eval_packs(self) -> None:
        cfg = self.cfg
        col_chunk = int(getattr(cfg, "unstable_rows_eval_chunk_cols", getattr(cfg, "unstable_rows_chunk_cols", 256)))
        col_chunk = max(1, col_chunk)

        allow_prepack = bool(getattr(cfg, "unstable_rows_prepack_groups", True))
        prepack_max_mb = float(getattr(cfg, "unstable_rows_prepack_max_mb", 512.0))
        use_fused_branch_gemm = bool(getattr(cfg, "unstable_rows_fused_branch_gemm", True))

        packs: List[Dict[str, Any]] = []
        for co, idxs in self.co_groups.items():
            bg = len(idxs)
            est_prepack_bytes = (2 * bg * self.Ci * co) * 4
            can_prepack = allow_prepack and (
                est_prepack_bytes <= int(prepack_max_mb * 1024.0 * 1024.0)
            )
            use_fused = bool(use_fused_branch_gemm and can_prepack and bg > 1)

            Wg = None
            Ug = None
            Wg_tc = None
            Ug_tc = None
            if can_prepack:
                if use_fused:
                    # (Ci,bg,Co) layout enables one large GEMM across branches.
                    Wg_tc = torch.stack([self.branches[ii]["W_fp"] for ii in idxs], dim=1).contiguous()
                    Ug_tc = torch.empty(
                        (self.Ci, bg, int(co)),
                        device=Wg_tc.device,
                        dtype=self.branches[idxs[0]]["U"].dtype,
                    )
                else:
                    Wg = torch.stack([self.branches[ii]["W_fp"] for ii in idxs], dim=0).contiguous()  # (bg,Ci,Co)
                    Ug = torch.empty(
                        (bg, self.Ci, int(co)),
                        device=Wg.device,
                        dtype=self.branches[idxs[0]]["U"].dtype,
                    )

            chunk_ranges: List[Tuple[int, int]] = []
            for c0 in range(0, int(co), col_chunk):
                c1 = min(c0 + col_chunk, int(co))
                chunk_ranges.append((c0, c1))

            pack_idx = len(packs)
            for local_idx, bi in enumerate(idxs):
                self.branch_pack_pos[bi] = (pack_idx, local_idx)

            packs.append(
                {
                    "co": int(co),
                    "bg": int(bg),
                    "idxs": idxs,
                    "Wg": Wg,
                    "Ug": Ug,
                    "Wg_tc": Wg_tc,
                    "Ug_tc": Ug_tc,
                    "can_prepack": bool(can_prepack),
                    "use_fused": bool(use_fused),
                    "chunk_ranges": chunk_ranges,
                }
            )
        self.group_eval_packs = packs
        self._refresh_group_eval_pack_u_views()

    @torch.no_grad()
    def _refresh_group_eval_pack_u_views(self) -> None:
        for gp in self.group_eval_packs:
            if not bool(gp["can_prepack"]):
                continue
            idxs = gp["idxs"]
            if bool(gp["use_fused"]) and gp["Ug_tc"] is not None:
                Ug_tc = gp["Ug_tc"]
                for local_idx, bi in enumerate(idxs):
                    Ug_tc[:, local_idx, :].copy_(self.branches[bi]["U"])
            elif gp["Ug"] is not None:
                Ug = gp["Ug"]
                for local_idx, bi in enumerate(idxs):
                    Ug[local_idx].copy_(self.branches[bi]["U"])

    @torch.no_grad()
    def prepare_coord_eval_context(self, j: int) -> CoordEvalContext:
        j = int(j)
        if self.act_symmetric:
            act_max_excl_j = torch.where(self.top1_idx != j, self.top1_val, self.top2_val)
            act_min_excl_j = None
        else:
            act_max_excl_j = torch.where(self.act_max1_idx != j, self.act_max1_val, self.act_max2_val)
            act_min_excl_j = torch.where(self.act_min1_idx != j, self.act_min1_val, self.act_min2_val)

        branch_ctxs: List[CoordEvalBranchContext] = []
        for bi, b in enumerate(self.branches):
            if self.w_symmetric:
                max_excl_rowj = torch.where(b["w_top1_idx"] != j, b["w_top1_val"], b["w_top2_val"])
                min_excl_rowj = None
                w_scale_row = None
                w_zp_row = None
            else:
                max_excl_rowj = torch.where(b["w_max1_idx"] != j, b["w_max1_val"], b["w_max2_val"])
                min_excl_rowj = torch.where(b["w_min1_idx"] != j, b["w_min1_val"], b["w_min2_val"])
                w_scale_row = b["w_scale"].squeeze(0)
                w_zp_row = b["w_zp"].squeeze(0)

            branch_ctxs.append(
                CoordEvalBranchContext(
                    branch_idx=int(bi),
                    W_hat_rowj=b["W_hat"][j, :],
                    urowj_old=b["U"][j, :],
                    B_rowj_64=b["B64"][j, :].to(torch.float64),
                    GU_rowj_64=b["GU"][j, :].to(torch.float64),
                    max_excl_rowj=max_excl_rowj,
                    min_excl_rowj=min_excl_rowj,
                    w_scale_row=w_scale_row,
                    w_zp_row=w_zp_row,
                )
            )

        return CoordEvalContext(
            j=j,
            x_col_j=self.X[:, j],
            xqj_old=self.Xq[:, j],
            gcol=self.G[:, j],
            Gjj=float(self.G[j, j].item()),
            act_max_excl_j=act_max_excl_j,
            act_min_excl_j=act_min_excl_j,
            branches=branch_ctxs,
        )

    @torch.no_grad()
    def _quantized_dX_rows_for_candidate(
        self,
        X_fp_rows: torch.Tensor,   # (rb,Ci)
        X_old_rows: torch.Tensor,  # (rb,Ci)
        *,
        j: int,
        a_new_rows: torch.Tensor,  # (rb,)
        delta_rows: torch.Tensor,  # (rb,)
        bits: int,
        col_chunk: int,
    ) -> torch.Tensor:
        qm = qmax_signed(int(bits))
        rb, Ci = X_fp_rows.shape
        assert X_old_rows.shape == (rb, Ci)
        assert a_new_rows.shape == (rb,)
        assert delta_rows.shape == (rb,)

        if not self.act_symmetric:
            A_rows = torch.div(X_fp_rows, self.s.unsqueeze(0))
            A_rows[:, j] = a_new_rows
            X_new = quantize_per_token_maxabs_fp32(
                A_rows,
                bits=int(bits),
                eps=self.eps,
                chunk_cols=int(max(1, col_chunk)),
                mode=self.act_quant_mode,
            )
            return X_new - X_old_rows

        dX = torch.empty_like(X_old_rows)
        delta_col = delta_rows.unsqueeze(1)  # (rb,1)
        cc = int(max(1, col_chunk))

        for c0 in range(0, Ci, cc):
            c1 = min(c0 + cc, Ci)
            A_chunk = torch.div(X_fp_rows[:, c0:c1], self.s[c0:c1].unsqueeze(0))
            if c0 <= j < c1:
                A_chunk[:, int(j - c0)] = a_new_rows
            q = torch.round(torch.div(A_chunk, delta_col)).clamp(-qm, qm)
            dX[:, c0:c1] = q.mul(delta_col) - X_old_rows[:, c0:c1]
            del A_chunk, q

        return dX
    # ---------- top2 stats (activation shared) ----------

    @torch.no_grad()
    def _recompute_top2_stats_activation(self) -> None:
        eps = float(self.eps)
        N, Ci = self.N, self.Ci
        cc = int(max(16, self.act_top2_chunk_cols))
        dev = self.X.device

        if not self.act_symmetric:
            act_max1_val = torch.full((N,), -float("inf"), device=dev, dtype=torch.float32)
            act_max1_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
            act_max2_val = torch.full((N,), -float("inf"), device=dev, dtype=torch.float32)
            act_max2_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
            act_min1_val = torch.full((N,), float("inf"), device=dev, dtype=torch.float32)
            act_min1_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
            act_min2_val = torch.full((N,), float("inf"), device=dev, dtype=torch.float32)
            act_min2_idx = torch.full((N,), -1, device=dev, dtype=torch.long)

            def update_max2(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
                nonlocal act_max1_val, act_max1_idx, act_max2_val, act_max2_idx
                better1 = cand_val > act_max1_val
                if bool(better1.any()):
                    old_idx = act_max1_idx
                    act_max2_val = torch.where(better1, act_max1_val, act_max2_val)
                    act_max2_idx = torch.where(better1, old_idx, act_max2_idx)
                    act_max1_val = torch.where(better1, cand_val, act_max1_val)
                    act_max1_idx = torch.where(better1, cand_idx, act_max1_idx)
                better2 = (~better1) & (cand_val > act_max2_val)
                if bool(better2.any()):
                    act_max2_val = torch.where(better2, cand_val, act_max2_val)
                    act_max2_idx = torch.where(better2, cand_idx, act_max2_idx)

            def update_min2(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
                nonlocal act_min1_val, act_min1_idx, act_min2_val, act_min2_idx
                better1 = cand_val < act_min1_val
                if bool(better1.any()):
                    old_idx = act_min1_idx
                    act_min2_val = torch.where(better1, act_min1_val, act_min2_val)
                    act_min2_idx = torch.where(better1, old_idx, act_min2_idx)
                    act_min1_val = torch.where(better1, cand_val, act_min1_val)
                    act_min1_idx = torch.where(better1, cand_idx, act_min1_idx)
                better2 = (~better1) & (cand_val < act_min2_val)
                if bool(better2.any()):
                    act_min2_val = torch.where(better2, cand_val, act_min2_val)
                    act_min2_idx = torch.where(better2, cand_idx, act_min2_idx)

            for c0 in range(0, Ci, cc):
                c1 = min(c0 + cc, Ci)
                A_chunk = self.X[:, c0:c1] / self.s[c0:c1].unsqueeze(0)
                k = 2 if (c1 - c0) >= 2 else 1
                max_vals, max_idx = torch.topk(A_chunk, k=k, dim=1, largest=True)
                min_vals, min_idx = torch.topk(A_chunk, k=k, dim=1, largest=False)
                update_max2(max_vals[:, 0].contiguous(), (max_idx[:, 0].to(torch.long) + c0).contiguous())
                update_min2(min_vals[:, 0].contiguous(), (min_idx[:, 0].to(torch.long) + c0).contiguous())
                if k == 2:
                    update_max2(max_vals[:, 1].contiguous(), (max_idx[:, 1].to(torch.long) + c0).contiguous())
                    update_min2(min_vals[:, 1].contiguous(), (min_idx[:, 1].to(torch.long) + c0).contiguous())
                del A_chunk, max_vals, max_idx, min_vals, min_idx

            self.act_max1_val = act_max1_val.contiguous()
            self.act_max1_idx = act_max1_idx.contiguous()
            self.act_max2_val = act_max2_val.contiguous()
            self.act_max2_idx = act_max2_idx.contiguous()
            self.act_min1_val = act_min1_val.contiguous()
            self.act_min1_idx = act_min1_idx.contiguous()
            self.act_min2_val = act_min2_val.contiguous()
            self.act_min2_idx = act_min2_idx.contiguous()
            self.act_scale = ((self.act_max1_val - self.act_min1_val) / float(self.act_qrange)).clamp_min(eps)
            self.act_zp = torch.round(self.act_qmin_f - self.act_min1_val / self.act_scale).clamp(
                self.act_qmin, self.act_qmax
            )
            return

        top1_val = torch.full((N,), -float("inf"), device=dev, dtype=torch.float32)
        top1_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
        top2_val = torch.full((N,), -float("inf"), device=dev, dtype=torch.float32)
        top2_idx = torch.full((N,), -1, device=dev, dtype=torch.long)

        def update_top2(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
            nonlocal top1_val, top1_idx, top2_val, top2_idx
            better1 = cand_val > top1_val
            if bool(better1.any()):
                old_top1_idx = top1_idx
                top2_val = torch.where(better1, top1_val, top2_val)
                top2_idx = torch.where(better1, old_top1_idx, top2_idx)
                top1_val = torch.where(better1, cand_val, top1_val)
                top1_idx = torch.where(better1, cand_idx, top1_idx)
            better2 = (~better1) & (cand_val > top2_val)
            if bool(better2.any()):
                top2_val = torch.where(better2, cand_val, top2_val)
                top2_idx = torch.where(better2, cand_idx, top2_idx)

        for c0 in range(0, Ci, cc):
            c1 = min(c0 + cc, Ci)
            chunk_abs = (self.X[:, c0:c1] / self.s[c0:c1].unsqueeze(0)).abs()
            vals2, idx2 = torch.topk(chunk_abs, k=2, dim=1, largest=True)
            del chunk_abs

            v1 = vals2[:, 0].contiguous()
            i1 = (idx2[:, 0].to(torch.long) + c0).contiguous()
            v2 = vals2[:, 1].contiguous()
            i2 = (idx2[:, 1].to(torch.long) + c0).contiguous()

            update_top2(v1, i1)
            update_top2(v2, i2)

            del vals2, idx2, v1, v2, i1, i2

        self.top1_val = top1_val.contiguous()
        self.top1_idx = top1_idx.contiguous()
        self.top2_val = top2_val.contiguous()
        self.top2_idx = top2_idx.contiguous()
        self.delta_x = (self.top1_val / float(self.qx)).clamp_min(eps)

    # ---------- top2 stats (per-branch weight) ----------

    @torch.no_grad()
    def _recompute_top2_stats_weight(self, b: Dict[str, Any]) -> None:
        eps = float(self.eps)
        Z = b["Z"]  # (Ci,Co)
        Ci2, Co = Z.shape
        rr = int(max(64, self.w_top2_chunk_rows))

        if not self.w_symmetric:
            w_max1_val = torch.full((Co,), -float("inf"), device=Z.device, dtype=torch.float32)
            w_max1_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)
            w_max2_val = torch.full((Co,), -float("inf"), device=Z.device, dtype=torch.float32)
            w_max2_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)
            w_min1_val = torch.full((Co,), float("inf"), device=Z.device, dtype=torch.float32)
            w_min1_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)
            w_min2_val = torch.full((Co,), float("inf"), device=Z.device, dtype=torch.float32)
            w_min2_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)

            def update_max2_cols(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
                nonlocal w_max1_val, w_max1_idx, w_max2_val, w_max2_idx
                better1 = cand_val > w_max1_val
                if bool(better1.any()):
                    old_idx = w_max1_idx
                    w_max2_val = torch.where(better1, w_max1_val, w_max2_val)
                    w_max2_idx = torch.where(better1, old_idx, w_max2_idx)
                    w_max1_val = torch.where(better1, cand_val, w_max1_val)
                    w_max1_idx = torch.where(better1, cand_idx, w_max1_idx)
                better2 = (~better1) & (cand_val > w_max2_val)
                if bool(better2.any()):
                    w_max2_val = torch.where(better2, cand_val, w_max2_val)
                    w_max2_idx = torch.where(better2, cand_idx, w_max2_idx)

            def update_min2_cols(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
                nonlocal w_min1_val, w_min1_idx, w_min2_val, w_min2_idx
                better1 = cand_val < w_min1_val
                if bool(better1.any()):
                    old_idx = w_min1_idx
                    w_min2_val = torch.where(better1, w_min1_val, w_min2_val)
                    w_min2_idx = torch.where(better1, old_idx, w_min2_idx)
                    w_min1_val = torch.where(better1, cand_val, w_min1_val)
                    w_min1_idx = torch.where(better1, cand_idx, w_min1_idx)
                better2 = (~better1) & (cand_val < w_min2_val)
                if bool(better2.any()):
                    w_min2_val = torch.where(better2, cand_val, w_min2_val)
                    w_min2_idx = torch.where(better2, cand_idx, w_min2_idx)

            for r0 in range(0, Ci2, rr):
                r1 = min(r0 + rr, Ci2)
                Z_chunk = Z[r0:r1, :]
                k = 2 if (r1 - r0) >= 2 else 1
                max_vals, max_idx = torch.topk(Z_chunk, k=k, dim=0, largest=True)
                min_vals, min_idx = torch.topk(Z_chunk, k=k, dim=0, largest=False)
                update_max2_cols(max_vals[0, :].contiguous(), (max_idx[0, :].to(torch.long) + r0).contiguous())
                update_min2_cols(min_vals[0, :].contiguous(), (min_idx[0, :].to(torch.long) + r0).contiguous())
                if k == 2:
                    update_max2_cols(max_vals[1, :].contiguous(), (max_idx[1, :].to(torch.long) + r0).contiguous())
                    update_min2_cols(min_vals[1, :].contiguous(), (min_idx[1, :].to(torch.long) + r0).contiguous())
                del Z_chunk, max_vals, max_idx, min_vals, min_idx

            b["w_max1_val"] = w_max1_val.contiguous()
            b["w_max1_idx"] = w_max1_idx.contiguous()
            b["w_max2_val"] = w_max2_val.contiguous()
            b["w_max2_idx"] = w_max2_idx.contiguous()
            b["w_min1_val"] = w_min1_val.contiguous()
            b["w_min1_idx"] = w_min1_idx.contiguous()
            b["w_min2_val"] = w_min2_val.contiguous()
            b["w_min2_idx"] = w_min2_idx.contiguous()
            b["w_scale"], b["w_zp"], _, _ = compute_per_out_channel_quant_params(
                Z,
                bits=int(self.cfg.w_bits),
                eps=eps,
                mode=self.w_quant_mode,
            )
            return

        w_top1_val = torch.full((Co,), -float("inf"), device=Z.device, dtype=torch.float32)
        w_top1_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)
        w_top2_val = torch.full((Co,), -float("inf"), device=Z.device, dtype=torch.float32)
        w_top2_idx = torch.full((Co,), -1, device=Z.device, dtype=torch.long)

        def update_top2_cols(cand_val: torch.Tensor, cand_idx: torch.Tensor) -> None:
            nonlocal w_top1_val, w_top1_idx, w_top2_val, w_top2_idx
            better1 = cand_val > w_top1_val
            if bool(better1.any()):
                old_top1_idx = w_top1_idx
                w_top2_val = torch.where(better1, w_top1_val, w_top2_val)
                w_top2_idx = torch.where(better1, old_top1_idx, w_top2_idx)
                w_top1_val = torch.where(better1, cand_val, w_top1_val)
                w_top1_idx = torch.where(better1, cand_idx, w_top1_idx)
            better2 = (~better1) & (cand_val > w_top2_val)
            if bool(better2.any()):
                w_top2_val = torch.where(better2, cand_val, w_top2_val)
                w_top2_idx = torch.where(better2, cand_idx, w_top2_idx)

        for r0 in range(0, Ci2, rr):
            r1 = min(r0 + rr, Ci2)
            chunk_abs = Z[r0:r1, :].abs()
            vals2, idx2 = torch.topk(chunk_abs, k=2, dim=0, largest=True)
            del chunk_abs

            v1 = vals2[0, :].contiguous()
            i1 = (idx2[0, :].to(torch.long) + r0).contiguous()
            v2 = vals2[1, :].contiguous()
            i2 = (idx2[1, :].to(torch.long) + r0).contiguous()

            update_top2_cols(v1, i1)
            update_top2_cols(v2, i2)

            del vals2, idx2, v1, v2, i1, i2

        b["w_top1_val"] = w_top1_val.contiguous()
        b["w_top1_idx"] = w_top1_idx.contiguous()
        b["w_top2_val"] = w_top2_val.contiguous()
        b["w_top2_idx"] = w_top2_idx.contiguous()
        b["delta_w"] = (w_top1_val / float(self.qw)).clamp_min(eps)

    @torch.no_grad()
    def _refresh_top2_stats_activation_for_coord(self, j: int, old_sj: float, new_sj: float) -> None:
        """
        Exact incremental refresh after changing only s[j].
        Re-scan only affected rows.
        """
        j = int(j)
        eps = float(self.eps)

        if not self.act_symmetric:
            old_a_j = self.X[:, j] / max(float(old_sj), eps)
            new_a_j = self.X[:, j] / max(float(new_sj), eps)
            impacted = (
                (self.act_max1_idx == j)
                | (self.act_max2_idx == j)
                | (self.act_min1_idx == j)
                | (self.act_min2_idx == j)
                | (new_a_j >= self.act_max2_val)
                | (old_a_j >= self.act_max2_val)
                | (new_a_j <= self.act_min2_val)
                | (old_a_j <= self.act_min2_val)
            )
            if bool(impacted.any()):
                rows = impacted.nonzero(as_tuple=False).squeeze(1)
                A = self.X.index_select(0, rows) / self.s.unsqueeze(0)
                k = 2 if self.Ci >= 2 else 1
                max_vals, max_idx = torch.topk(A, k=k, dim=1, largest=True)
                min_vals, min_idx = torch.topk(A, k=k, dim=1, largest=False)
                self.act_max1_val.index_copy_(0, rows, max_vals[:, 0].contiguous())
                self.act_max1_idx.index_copy_(0, rows, max_idx[:, 0].to(torch.long).contiguous())
                self.act_min1_val.index_copy_(0, rows, min_vals[:, 0].contiguous())
                self.act_min1_idx.index_copy_(0, rows, min_idx[:, 0].to(torch.long).contiguous())
                if k == 2:
                    self.act_max2_val.index_copy_(0, rows, max_vals[:, 1].contiguous())
                    self.act_max2_idx.index_copy_(0, rows, max_idx[:, 1].to(torch.long).contiguous())
                    self.act_min2_val.index_copy_(0, rows, min_vals[:, 1].contiguous())
                    self.act_min2_idx.index_copy_(0, rows, min_idx[:, 1].to(torch.long).contiguous())
                del rows, A, max_vals, max_idx, min_vals, min_idx

            self.act_scale = ((self.act_max1_val - self.act_min1_val) / float(self.act_qrange)).clamp_min(eps)
            self.act_zp = torch.round(self.act_qmin_f - self.act_min1_val / self.act_scale).clamp(
                self.act_qmin, self.act_qmax
            )
            return

        old_abs_j = (self.X[:, j] / max(float(old_sj), eps)).abs()
        new_abs_j = (self.X[:, j] / max(float(new_sj), eps)).abs()

        impacted = (
            (self.top1_idx == j)
            | (self.top2_idx == j)
            | (new_abs_j >= self.top2_val)
            | (old_abs_j >= self.top2_val)
        )
        if bool(impacted.any()):
            rows = impacted.nonzero(as_tuple=False).squeeze(1)
            A_abs = (self.X.index_select(0, rows) / self.s.unsqueeze(0)).abs()
            vals2, idx2 = torch.topk(A_abs, k=2, dim=1, largest=True)
            self.top1_val.index_copy_(0, rows, vals2[:, 0].contiguous())
            self.top1_idx.index_copy_(0, rows, idx2[:, 0].to(torch.long).contiguous())
            self.top2_val.index_copy_(0, rows, vals2[:, 1].contiguous())
            self.top2_idx.index_copy_(0, rows, idx2[:, 1].to(torch.long).contiguous())
            del rows, A_abs, vals2, idx2

        self.delta_x = (self.top1_val / float(self.qx)).clamp_min(eps)

    @torch.no_grad()
    def _refresh_top2_stats_weight_for_coord(self, b: Dict[str, Any], j: int, old_sj: float, new_sj: float) -> None:
        """
        Exact incremental refresh after changing only Z[j,:] = W_hat[j,:] * s[j].
        Re-scan only affected columns.
        """
        j = int(j)
        eps = float(self.eps)

        if not self.w_symmetric:
            old_z_j = b["W_hat"][j, :] * float(old_sj)
            new_z_j = b["W_hat"][j, :] * float(new_sj)
            impacted = (
                (b["w_max1_idx"] == j)
                | (b["w_max2_idx"] == j)
                | (b["w_min1_idx"] == j)
                | (b["w_min2_idx"] == j)
                | (new_z_j >= b["w_max2_val"])
                | (old_z_j >= b["w_max2_val"])
                | (new_z_j <= b["w_min2_val"])
                | (old_z_j <= b["w_min2_val"])
            )
            if bool(impacted.any()):
                cols = impacted.nonzero(as_tuple=False).squeeze(1)
                Z = b["Z"].index_select(1, cols)
                k = 2 if Z.shape[0] >= 2 else 1
                max_vals, max_idx = torch.topk(Z, k=k, dim=0, largest=True)
                min_vals, min_idx = torch.topk(Z, k=k, dim=0, largest=False)
                b["w_max1_val"].index_copy_(0, cols, max_vals[0, :].contiguous())
                b["w_max1_idx"].index_copy_(0, cols, max_idx[0, :].to(torch.long).contiguous())
                b["w_min1_val"].index_copy_(0, cols, min_vals[0, :].contiguous())
                b["w_min1_idx"].index_copy_(0, cols, min_idx[0, :].to(torch.long).contiguous())
                if k == 2:
                    b["w_max2_val"].index_copy_(0, cols, max_vals[1, :].contiguous())
                    b["w_max2_idx"].index_copy_(0, cols, max_idx[1, :].to(torch.long).contiguous())
                    b["w_min2_val"].index_copy_(0, cols, min_vals[1, :].contiguous())
                    b["w_min2_idx"].index_copy_(0, cols, min_idx[1, :].to(torch.long).contiguous())
                del cols, Z, max_vals, max_idx, min_vals, min_idx

            b["w_scale"] = ((b["w_max1_val"] - b["w_min1_val"]) / float(self.w_qrange)).clamp_min(eps).unsqueeze(0)
            b["w_zp"] = torch.round(self.w_qmin_f - b["w_min1_val"].unsqueeze(0) / b["w_scale"]).clamp(
                self.w_qmin, self.w_qmax
            )
            return

        old_abs_j = (b["W_hat"][j, :] * float(old_sj)).abs()
        new_abs_j = (b["W_hat"][j, :] * float(new_sj)).abs()

        impacted = (
            (b["w_top1_idx"] == j)
            | (b["w_top2_idx"] == j)
            | (new_abs_j >= b["w_top2_val"])
            | (old_abs_j >= b["w_top2_val"])
        )
        if bool(impacted.any()):
            cols = impacted.nonzero(as_tuple=False).squeeze(1)
            Z_abs = b["Z"].index_select(1, cols).abs()
            vals2, idx2 = torch.topk(Z_abs, k=2, dim=0, largest=True)
            b["w_top1_val"].index_copy_(0, cols, vals2[0, :].contiguous())
            b["w_top1_idx"].index_copy_(0, cols, idx2[0, :].to(torch.long).contiguous())
            b["w_top2_val"].index_copy_(0, cols, vals2[1, :].contiguous())
            b["w_top2_idx"].index_copy_(0, cols, idx2[1, :].to(torch.long).contiguous())
            del cols, Z_abs, vals2, idx2

        b["delta_w"] = (b["w_top1_val"] / float(self.qw)).clamp_min(eps)

    # ---------- recompute per-branch GU and scalar terms ----------

    @torch.no_grad()
    def _recompute_GU_and_terms_branch(self, b: Dict[str, Any]) -> None:
        dev = self.X.device
        Co = int(b["U"].shape[1])
        cc = int(max(1, int(getattr(self.cfg, "unstable_rows_chunk_cols", 256))))

        # term1 = <U, B>
        b["term1_64"] = torch.dot(
            b["U"].reshape(-1).to(torch.float64),
            b["B64"].reshape(-1),
        )

        # GU in fp32
        b["GU"] = compute_GU_chunked(self.G, b["U"], chunk_cols=cc)

        # term2 in fp64 via G64
        term2 = torch.zeros((), device=dev, dtype=torch.float64)
        for o0 in range(0, Co, cc):
            o1 = min(o0 + cc, Co)
            Uc64 = b["U"][:, o0:o1].to(torch.float64)
            GUc64 = self.G64 @ Uc64
            term2 += torch.sum(Uc64 * GUc64)
            del Uc64, GUc64
        b["term2_64"] = term2

    # ---------- objective ----------

    @torch.no_grad()
    def current_obj(self) -> float:
        f64 = torch.zeros((), device=self.X.device, dtype=torch.float64)
        for b in self.branches:
            f64 = f64 + (b["c64"] - 2.0 * b["term1_64"] + b["term2_64"])
        f = float(f64.item())
        if f < 0.0:
            # clamp tiny numerical noise
            scale = max(1.0, abs(f))
            tol = 1e-8 * scale
            if f > -tol:
                f = 0.0
        return f

    # ---------- quantize helper ----------

    @torch.no_grad()
    def _quantize_rows_with_given_delta(self, A_rows: torch.Tensor, delta_rows: torch.Tensor, bits: int) -> torch.Tensor:
        if not self.act_symmetric:
            return quantize_per_token_maxabs_fp32(
                A_rows,
                bits=int(bits),
                eps=self.eps,
                chunk_cols=int(getattr(self.cfg, "act_quant_chunk_cols", 512)),
                mode=self.act_quant_mode,
            )
        qm = qmax_signed(int(bits))
        q = torch.round(A_rows / delta_rows).clamp(-qm, qm)
        return q * delta_rows

    # ---------- eval candidate (SUM across branches) ----------

    @torch.no_grad()
    def eval_candidate(
        self,
        j: int,
        sj_new: float,
        coord_ctx: Optional[CoordEvalContext] = None,
    ) -> Tuple[float, float, float, CandidateCacheGroup]:
        cfg = self.cfg
        eps = self.eps
        j = int(j)
        sj_new = float(max(float(sj_new), eps))
        ctx = coord_ctx if coord_ctx is not None else self.prepare_coord_eval_context(j)

        # -----------------------
        # Activation side (shared)
        # -----------------------
        a_new = ctx.x_col_j / sj_new

        if self.act_symmetric:
            assert ctx.act_max_excl_j is not None
            new_max = torch.maximum(ctx.act_max_excl_j, a_new.abs())
            delta_x_new = (new_max / float(self.qx)).clamp_min(eps)
            stable_rows = (delta_x_new == self.delta_x)
            xqj_new = quantize_scalar_with_delta(a_new, delta_x_new, bits=int(cfg.act_bits))
        else:
            assert ctx.act_max_excl_j is not None and ctx.act_min_excl_j is not None
            act_max_new = torch.maximum(ctx.act_max_excl_j, a_new)
            act_min_new = torch.minimum(ctx.act_min_excl_j, a_new)
            stable_rows = (act_max_new == self.act_max1_val) & (act_min_new == self.act_min1_val)
            act_scale_new = ((act_max_new - act_min_new) / float(self.act_qrange)).clamp_min(eps)
            act_zp_new = torch.round(self.act_qmin_f - act_min_new / act_scale_new).clamp(
                self.act_qmin, self.act_qmax
            )
            delta_x_new = act_scale_new
            xqj_new = quantize_with_params(
                a_new,
                act_scale_new,
                act_zp_new,
                bits=int(cfg.act_bits),
                mode=self.act_quant_mode,
            )

        unstable_rows = ~stable_rows
        n_unstable_rows = int(unstable_rows.sum().item())
        frac_unstable_rows = float(n_unstable_rows) / float(self.N)
        xqj_old = ctx.xqj_old
        idx_u_rows = unstable_rows.nonzero(as_tuple=False).squeeze(1) if n_unstable_rows > 0 else None

        # stable-row deltas for Gram updates
        dx = (xqj_new - xqj_old)
        dx_stable = dx * stable_rows.to(dx.dtype)

        # deltaG_j depends only on activation side
        deltaG_j_raw = (dx_stable.unsqueeze(0) @ self.Xq).squeeze(0)  # (Ci,) fp32
        deltaG_jj = float(torch.sum((xqj_old + dx_stable) ** 2 - (xqj_old ** 2)).item())
        deltaG_off = deltaG_j_raw.clone()
        deltaG_off[j] = 0.0

        # We will sum objective updates across branches
        obj_sum_64 = torch.zeros((), device=self.X.device, dtype=torch.float64)

        # Keep per-branch temporary state, then finalize caches after unstable-row pass.
        branch_states: List[Dict[str, Any]] = []

        # Track max unstable cols fraction across branches (for logging only)
        max_frac_unstable_cols = 0.0

        # -----------------------
        # Branch loop
        # -----------------------
        # Shared across branches: deltaB_j uses only activation-side change.
        v = (dx_stable.unsqueeze(0) @ self.X).squeeze(0)  # (Ci,) fp32
        for bctx in ctx.branches:
            b = self.branches[bctx.branch_idx]
            W_fp = b["W_fp"]
            U = b["U"]
            Z = b["Z"]
            Co = int(b["Co"])

            # -----------------------
            # Weight side (per branch)
            # -----------------------
            zrow_new = bctx.W_hat_rowj * sj_new
            urowj_old = bctx.urowj_old
            if self.w_symmetric:
                urowj_new = quantize_scalar_with_delta(zrow_new, b["delta_w"], bits=int(cfg.w_bits))
                assert bctx.max_excl_rowj is not None
                new_col_max = torch.maximum(bctx.max_excl_rowj, zrow_new.abs())
                delta_w_new = (new_col_max / float(self.qw)).clamp_min(eps)
                unstable_cols = (delta_w_new != b["delta_w"])
            else:
                assert bctx.w_scale_row is not None and bctx.w_zp_row is not None
                assert bctx.max_excl_rowj is not None and bctx.min_excl_rowj is not None
                urowj_new = quantize_with_params(
                    zrow_new,
                    bctx.w_scale_row,
                    bctx.w_zp_row,
                    bits=int(cfg.w_bits),
                    mode=self.w_quant_mode,
                )
                w_max_new = torch.maximum(bctx.max_excl_rowj, zrow_new)
                w_min_new = torch.minimum(bctx.min_excl_rowj, zrow_new)
                unstable_cols = (w_max_new != b["w_max1_val"]) | (w_min_new != b["w_min1_val"])
            n_unstable_cols = int(unstable_cols.sum().item())
            frac_unstable_cols = float(n_unstable_cols) / float(Co)
            max_frac_unstable_cols = max(max_frac_unstable_cols, frac_unstable_cols)

            cols_u = None
            U_new_cols = None
            U_old_cols = None
            dU_cols = None

            if n_unstable_cols > 0:
                cols_u = unstable_cols.nonzero(as_tuple=False).squeeze(1)
                cols = cols_u

                Z_sub = Z.index_select(1, cols).clone()
                Z_sub[j, :] = zrow_new.index_select(0, cols)
                U_new_cols = quantize_weights_per_out_channel_fp32(
                    Z_sub,
                    bits=int(cfg.w_bits),
                    eps=eps,
                    mode=self.w_quant_mode,
                )
                U_old_cols = U.index_select(1, cols)

                urowj_final = urowj_new.clone()
                urowj_final.index_copy_(0, cols, U_new_cols[j, :])
                urowj_new = urowj_final

                dU_cols = (U_new_cols - U_old_cols)
                dU_cols[j, :] = 0.0

            du = (urowj_new - urowj_old)  # fp32

            # -----------------------
            # deltaB_j for this branch: (dx^T X) W_fp
            # -----------------------
            deltaB_j = (v.unsqueeze(0) @ W_fp).squeeze(0)         # (Co,) fp32

            # -----------------------
            # Scalar update in fp64
            # -----------------------
            term1_new = b["term1_64"].to(torch.float64)
            term2_new = b["term2_64"].to(torch.float64)
            c64 = b["c64"].to(torch.float64)
            Bmat = b["B64"]

            Bj_old_64 = bctx.B_rowj_64
            du64 = du.to(torch.float64)
            urowj_new64 = urowj_new.to(torch.float64)
            deltaB_j64 = deltaB_j.to(torch.float64)

            # term1 updates
            term1_new = term1_new + torch.dot(du64, Bj_old_64)
            term1_new = term1_new + torch.dot(urowj_new64, deltaB_j64)

            # term2 updates from changing U row j (G fixed)
            Gjj = ctx.Gjj
            term2_new = term2_new + (
                2.0 * torch.dot(du64, bctx.GU_rowj_64)
                + Gjj * torch.dot(du64, du64)
            )

            # term2 updates from deltaG (stable rows only)
            if dU_cols is None or cols_u is None:
                # fully stable weight side: <U_new, dG U_new> via r = U^T d
                r = U.transpose(0, 1).matmul(deltaG_off)  # (Co,) fp32
                term2_new = term2_new + (
                    2.0 * torch.dot(urowj_new64, r.to(torch.float64))
                    + float(deltaG_jj) * torch.dot(urowj_new64, urowj_new64)
                )
            else:
                # unstable cols: keep exact correction
                Sj = torch.sum(U * urowj_new.unsqueeze(0), dim=1)  # fp32
                Sj = Sj + dU_cols.matmul(urowj_new.index_select(0, cols_u))
                Sj[j] = torch.dot(urowj_new, urowj_new)
                term2_new = term2_new + (
                    2.0 * torch.sum((deltaG_off * Sj).to(torch.float64))
                    + float(deltaG_jj) * float(Sj[j].item())
                )

            # -----------------------
            # Unstable cols correction (exact) - per branch
            # -----------------------
            if dU_cols is not None and cols_u is not None and U_new_cols is not None and U_old_cols is not None:
                cols = cols_u

                B_cols = Bmat.index_select(1, cols).to(torch.float64)
                term1_new = term1_new + torch.dot(dU_cols.reshape(-1).to(torch.float64), B_cols.reshape(-1))

                gcol = ctx.gcol
                du_cols = du.index_select(0, cols)

                GU_base_cols = b["GU"].index_select(1, cols) + gcol.unsqueeze(1) * du_cols.unsqueeze(0)

                U_base_cols = U_old_cols.clone()
                U_base_cols[j, :] = urowj_new.index_select(0, cols)

                dot_base = torch.dot(
                    U_base_cols.reshape(-1).to(torch.float64),
                    GU_base_cols.reshape(-1).to(torch.float64),
                )

                GU_new_cols = self.G @ U_new_cols
                dot_new = torch.dot(
                    U_new_cols.reshape(-1).to(torch.float64),
                    GU_new_cols.reshape(-1).to(torch.float64),
                )

                term2_new = term2_new - dot_base + dot_new

            branch_states.append(
                {
                    "W_fp": W_fp,
                    "U": U,
                    "du": du,
                    "dU_cols": dU_cols,
                    "cols_u": cols_u,
                    "term1_new": term1_new,
                    "term2_new": term2_new,
                    "c64": c64,
                    "zrow_new": zrow_new,
                    "urowj_new": urowj_new,
                    "U_new_cols": U_new_cols,
                    "U_old_cols": U_old_cols,
                    "deltaB_j": deltaB_j,
                }
            )

        # -----------------------
        # Unstable rows correction (exact) - shared row chunks across all branches
        # -----------------------
        if idx_u_rows is not None and idx_u_rows.numel() > 0:
            row_chunk = int(getattr(cfg, "unstable_rows_eval_row_chunk", getattr(cfg, "unstable_rows_row_chunk", 1024)))
            row_chunk = max(1, row_chunk)
            use_fullcol_precompute = bool(getattr(cfg, "unstable_rows_fullcol_precompute", True))
            fullcol_max_mb = float(getattr(cfg, "unstable_rows_fullcol_max_mb", 1536.0))
            q_col_chunk = int(getattr(cfg, "unstable_rows_quant_col_chunk", getattr(cfg, "act_quant_chunk_cols", 512)))
            q_col_chunk = max(1, q_col_chunk)
            gather_once = bool(getattr(cfg, "unstable_rows_eval_gather_once", True))
            gather_max_mb = float(getattr(cfg, "unstable_rows_eval_gather_max_mb", 1024.0))

            packed_groups: List[Dict[str, Any]] = []
            for gp in self.group_eval_packs:
                idxs = gp["idxs"]
                dug = torch.stack([branch_states[ii]["du"] for ii in idxs], dim=0).contiguous()  # (bg,Co)

                has_any_corr = False
                for bi in idxs:
                    cols_u = branch_states[bi]["cols_u"]
                    if cols_u is not None and cols_u.numel() > 0:
                        has_any_corr = True
                        break

                chunk_corrs: Optional[List[Dict[str, Any]]] = None
                if has_any_corr:
                    chunk_corrs = []
                    for c0, c1 in gp["chunk_ranges"]:
                        corr_meta: List[Tuple[int, torch.Tensor, int, int]] = []
                        corr_dUc_list: List[torch.Tensor] = []
                        corr_off = 0
                        for li, bi in enumerate(idxs):
                            dU_cols = branch_states[bi]["dU_cols"]
                            cols_u = branch_states[bi]["cols_u"]
                            if dU_cols is None or cols_u is None or cols_u.numel() == 0:
                                continue

                            in_chunk = (cols_u >= c0) & (cols_u < c1)
                            if bool(in_chunk.any()):
                                cols_chunk = cols_u[in_chunk]
                                local = (cols_chunk - c0).to(torch.long)
                                pos = in_chunk.nonzero(as_tuple=False).squeeze(1)
                                dUc = dU_cols.index_select(1, pos).contiguous()
                                ncols = int(dUc.shape[1])
                                corr_dUc_list.append(dUc)
                                corr_meta.append((li, local, corr_off, ncols))
                                corr_off += ncols

                        dUc_cat = torch.cat(corr_dUc_list, dim=1).contiguous() if len(corr_dUc_list) > 0 else None
                        chunk_corrs.append({"dUc_cat": dUc_cat, "meta": corr_meta})

                g = dict(gp)
                g["dug"] = dug
                g["has_any_corr"] = bool(has_any_corr)
                g["chunk_corrs"] = chunk_corrs
                packed_groups.append(g)

            R = int(idx_u_rows.numel())
            X_old_u_all = None
            X_fp_u_all = None
            a_u_all = None
            delta_u_all = None
            if gather_once:
                est_gather_mb = (2.0 * float(R) * float(self.Ci) * 4.0) / (1024.0 * 1024.0)
                if est_gather_mb <= gather_max_mb:
                    X_old_u_all = self.Xq.index_select(0, idx_u_rows).contiguous()
                    X_fp_u_all = self.X.index_select(0, idx_u_rows).contiguous()
                    a_u_all = a_new.index_select(0, idx_u_rows).contiguous()
                    delta_u_all = delta_x_new.index_select(0, idx_u_rows).contiguous()
            for r0 in range(0, R, row_chunk):
                r1 = min(r0 + row_chunk, R)
                if X_old_u_all is not None and X_fp_u_all is not None and a_u_all is not None and delta_u_all is not None:
                    X_old = X_old_u_all[r0:r1]
                    X_fp_b = X_fp_u_all[r0:r1]
                    a_b = a_u_all[r0:r1]
                    d_b = delta_u_all[r0:r1]
                else:
                    idx_b = idx_u_rows[r0:r1]
                    X_old = self.Xq.index_select(0, idx_b)  # (rb, Ci)
                    X_fp_b = self.X.index_select(0, idx_b)  # (rb, Ci)
                    a_b = a_new.index_select(0, idx_b)
                    d_b = delta_x_new.index_select(0, idx_b)

                dX = self._quantized_dX_rows_for_candidate(
                    X_fp_rows=X_fp_b,
                    X_old_rows=X_old,
                    j=j,
                    a_new_rows=a_b,
                    delta_rows=d_b,
                    bits=int(cfg.act_bits),
                    col_chunk=q_col_chunk,
                )
                X_old_b = X_old.unsqueeze(0)  # (1,rb,Ci)
                dX_b = dX.unsqueeze(0)  # (1,rb,Ci)
                X_fp_b_b = X_fp_b.unsqueeze(0)  # (1,rb,Ci)
                xold_j = X_old[:, j].unsqueeze(0).unsqueeze(2)  # (1,rb,1)
                dx_j = dX[:, j].unsqueeze(0).unsqueeze(2)  # (1,rb,1)
                rb = int(X_old.shape[0])

                for g in packed_groups:
                    idxs = g["idxs"]
                    Wg = g["Wg"]
                    Ug = g["Ug"]
                    Wg_tc = g["Wg_tc"]
                    Ug_tc = g["Ug_tc"]
                    dug = g["dug"]
                    can_prepack = bool(g["can_prepack"])
                    use_fused = bool(g["use_fused"])
                    chunk_ranges = g["chunk_ranges"]
                    chunk_corrs = g["chunk_corrs"]
                    has_any_corr = bool(g["has_any_corr"])
                    bg = len(idxs)
                    add1_g = torch.zeros((bg,), device=self.X.device, dtype=torch.float32)
                    add2_g = torch.zeros((bg,), device=self.X.device, dtype=torch.float32)

                    co_g = int(dug.shape[1]) if dug is not None else int(branch_states[idxs[0]]["U"].shape[1])
                    do_full_precompute = bool(
                        can_prepack
                        and use_fused
                        and (Wg_tc is not None)
                        and (Ug_tc is not None)
                        and (dug is not None)
                        and use_fullcol_precompute
                    )
                    if do_full_precompute:
                        est_fullcol_mb = (3.0 * float(bg) * float(rb) * float(co_g) * 4.0) / (1024.0 * 1024.0)
                        do_full_precompute = bool(est_fullcol_mb <= fullcol_max_mb)

                    P_old_full = None
                    dP_full = None
                    Y_full = None
                    if do_full_precompute and Wg_tc is not None and Ug_tc is not None and dug is not None:
                        Uall2 = Ug_tc.reshape(self.Ci, bg * co_g)
                        Wall2 = Wg_tc.reshape(self.Ci, bg * co_g)

                        P_old_full = (X_old @ Uall2).view(rb, bg, co_g).permute(1, 0, 2).contiguous()
                        dP_full = (dX @ Uall2).view(rb, bg, co_g).permute(1, 0, 2).contiguous()
                        Y_full = (X_fp_b @ Wall2).view(rb, bg, co_g).permute(1, 0, 2).contiguous()

                        P_old_full = P_old_full + xold_j * dug.unsqueeze(1)
                        dP_full = dP_full + dx_j * dug.unsqueeze(1)

                    if do_full_precompute and (not has_any_corr) and P_old_full is not None and dP_full is not None and Y_full is not None:
                        add1_g = add1_g + torch.sum(dP_full * Y_full, dim=(1, 2))
                        add2_g = add2_g + (
                            2.0 * torch.sum(P_old_full * dP_full, dim=(1, 2))
                            + torch.sum(dP_full * dP_full, dim=(1, 2))
                        )
                    else:
                        for ci, (c0, c1) in enumerate(chunk_ranges):
                            cc = int(c1 - c0)
                            if do_full_precompute and P_old_full is not None and dP_full is not None and Y_full is not None:
                                P_old = P_old_full[:, :, c0:c1]
                                dP = dP_full[:, :, c0:c1]
                                Yc = Y_full[:, :, c0:c1]
                            elif can_prepack and use_fused and Wg_tc is not None and Ug_tc is not None and dug is not None:
                                duc = dug[:, c0:c1]  # (bg,cc)
                                # One GEMM per term across all branches: (rb,Ci) @ (Ci,bg*cc).
                                Uc2 = Ug_tc[:, :, c0:c1].reshape(self.Ci, bg * cc)
                                Wc2 = Wg_tc[:, :, c0:c1].reshape(self.Ci, bg * cc)

                                P_old = (X_old @ Uc2).view(rb, bg, cc).permute(1, 0, 2)
                                dP = (dX @ Uc2).view(rb, bg, cc).permute(1, 0, 2)
                                Yc = (X_fp_b @ Wc2).view(rb, bg, cc).permute(1, 0, 2)

                                P_old = P_old + xold_j * duc.unsqueeze(1)
                                dP = dP + dx_j * duc.unsqueeze(1)
                            else:
                                if can_prepack and Wg is not None and Ug is not None and dug is not None:
                                    Wc = Wg[:, :, c0:c1]  # (bg,Ci,cc)
                                    Uc = Ug[:, :, c0:c1]  # (bg,Ci,cc)
                                    duc = dug[:, c0:c1]   # (bg,cc)
                                else:
                                    Wc = torch.stack([branch_states[ii]["W_fp"][:, c0:c1] for ii in idxs], dim=0)
                                    Uc = torch.stack([branch_states[ii]["U"][:, c0:c1] for ii in idxs], dim=0)
                                    duc = torch.stack([branch_states[ii]["du"][c0:c1] for ii in idxs], dim=0)

                                # Exact base predictions for all branches in the group.
                                P_old = torch.matmul(X_old_b, Uc)
                                dP = torch.matmul(dX_b, Uc)
                                P_old = P_old + xold_j * duc.unsqueeze(1)
                                dP = dP + dx_j * duc.unsqueeze(1)
                                # Exact Y rows for all branches in the group.
                                Yc = torch.matmul(X_fp_b_b, Wc)

                            if chunk_corrs is not None:
                                # Exact sparse dU_cols corrections (branch-specific).
                                corr_pack = chunk_corrs[ci]
                                dUc_cat = corr_pack["dUc_cat"]
                                if dUc_cat is not None:
                                    add_old_cat = X_old @ dUc_cat
                                    d_add_cat = dX @ dUc_cat
                                    for li, local, off, ncols in corr_pack["meta"]:
                                        add_old = add_old_cat[:, off:(off + ncols)]
                                        d_add = d_add_cat[:, off:(off + ncols)]
                                        P_old[li].index_add_(1, local, add_old)
                                        dP[li].index_add_(1, local, d_add)

                            add1_g = add1_g + torch.sum(dP * Yc, dim=(1, 2))
                            add2_g = add2_g + (
                                2.0 * torch.sum(P_old * dP, dim=(1, 2))
                                + torch.sum(dP * dP, dim=(1, 2))
                            )

                    for li, bi in enumerate(idxs):
                        st = branch_states[bi]
                        st["term1_new"] = st["term1_new"] + add1_g[li].to(torch.float64)
                        st["term2_new"] = st["term2_new"] + add2_g[li].to(torch.float64)

                    del P_old_full, dP_full, Y_full

                del X_old, X_fp_b, a_b, d_b, dX, X_old_b, dX_b, X_fp_b_b

            del X_old_u_all, X_fp_u_all, a_u_all, delta_u_all

        branch_caches: List[CandidateCacheBranch] = []
        for st in branch_states:
            f64 = st["c64"] - 2.0 * st["term1_new"] + st["term2_new"]
            obj_sum_64 = obj_sum_64 + f64
            branch_caches.append(
                CandidateCacheBranch(
                    zrow_new=st["zrow_new"],
                    urowj_new=st["urowj_new"],
                    cols_u=st["cols_u"],
                    U_new_cols=st["U_new_cols"],
                    U_old_cols=st["U_old_cols"],
                    dU_cols=st["dU_cols"],
                    deltaB_j=st["deltaB_j"],
                    du=st["du"],
                    term1_new_64=st["term1_new"],
                    term2_new_64=st["term2_new"],
                )
            )

        obj = float(obj_sum_64.item())
        if obj < 0.0:
            scale = max(1.0, abs(obj))
            tol = 1e-8 * scale
            if obj > -tol:
                obj = 0.0

        fast_path = (n_unstable_rows == 0) and (max_frac_unstable_cols == 0.0)

        cache = CandidateCacheGroup(
            j=j,
            sj_new=float(sj_new),
            a_new=a_new,
            xqj_new=xqj_new,
            idx_u_rows=idx_u_rows,
            deltaG_j=deltaG_j_raw,
            deltaG_jj=float(deltaG_jj),
            branches=branch_caches,
            fast_path=bool(fast_path),
            obj=float(obj),
        )
        return float(obj), float(frac_unstable_rows), float(max_frac_unstable_cols), cache

    # ---------- commit candidate (updates shared activation and per-branch weights) ----------

    @torch.no_grad()
    def commit(self, cache: CandidateCacheGroup) -> None:
        cfg = self.cfg
        eps = self.eps
        j = int(cache.j)
        sj_new = float(max(cache.sj_new, eps))
        sj_old = float(self.s[j].item())
        fully_stable = bool(cache.fast_path)
        gcol_old = self.G[:, j].clone() if fully_stable else None

        # ---- update s ----
        self.s[j] = torch.tensor(sj_new, device=self.s.device, dtype=torch.float32)
        self.inv_s[j] = torch.reciprocal(self.s[j].clamp_min(self.eps))

        # ---- per-branch: update Z/U row j, and unstable cols ----
        for b, bc in zip(self.branches, cache.branches):
            b["Z"][j, :] = bc.zrow_new
            b["U"][j, :] = bc.urowj_new
            # Stable-row activation delta updates only row j of B.
            b["B64"][j, :] += bc.deltaB_j.to(torch.float64)

            # unstable cols: requantize those columns exactly
            if bc.cols_u is not None and bc.cols_u.numel() > 0:
                cols = bc.cols_u
                Z_sub = b["Z"].index_select(1, cols)
                U_sub = quantize_weights_per_out_channel_fp32(
                    Z_sub,
                    bits=int(cfg.w_bits),
                    eps=eps,
                    mode=self.w_quant_mode,
                )
                b["U"][:, cols] = U_sub
                del Z_sub, U_sub

        # ---- activation side: update Xq and (G, G64) consistently ----
        if cache.idx_u_rows is not None and cache.idx_u_rows.numel() > 0:
            idx_all = cache.idx_u_rows

            # Apply stable-row contribution (rows where delta_x did not change) exactly.
            # In mixed stable/unstable updates, stable rows can still change xq[:,j].
            dG = cache.deltaG_j
            dGjj = float(cache.deltaG_jj)
            if int(idx_all.numel()) < int(self.N):
                stable_mask = torch.ones((self.N,), device=self.Xq.device, dtype=torch.bool)
                stable_mask.index_fill_(0, idx_all, False)
                idx_s = stable_mask.nonzero(as_tuple=False).squeeze(1)
                if idx_s.numel() > 0:
                    self.Xq[idx_s, j] = cache.xqj_new.index_select(0, idx_s)
                del stable_mask, idx_s

            if j > 0:
                self.G[j, :j] += dG[:j]
                self.G[:j, j] += dG[:j]
            if j + 1 < self.Ci:
                self.G[j, j + 1:] += dG[j + 1:]
                self.G[j + 1:, j] += dG[j + 1:]
            self.G[j, j] += dGjj

            dG64 = dG.to(torch.float64)
            if j > 0:
                self.G64[j, :j] += dG64[:j]
                self.G64[:j, j] += dG64[:j]
            if j + 1 < self.Ci:
                self.G64[j, j + 1:] += dG64[j + 1:]
                self.G64[j + 1:, j] += dG64[j + 1:]
            self.G64[j, j] += float(dGjj)

            tile32 = int(getattr(cfg, "G_tile", 1024))
            tile64 = int(getattr(cfg, "G64_tile", 1024))

            rc = int(getattr(cfg, "unstable_rows_commit_row_chunk", 512))
            rc = max(1, rc)

            hard_free = bool(getattr(cfg, "hard_free_between_commit_chunks", True))
            # Reuse fp64 views across row chunks; avoids repeated cast costs.
            w_fp64_list = [b["W_fp"].to(torch.float64) for b in self.branches]

            R = int(idx_all.numel())
            for r0 in range(0, R, rc):
                r1 = min(r0 + rc, R)
                idx = idx_all[r0:r1]

                X_old = self.Xq.index_select(0, idx)

                A_rows = self.X.index_select(0, idx) / self.s.unsqueeze(0)
                maxabs = A_rows.abs().amax(dim=1, keepdim=True).clamp_min(self.eps)
                delta = (maxabs / float(self.qx)).clamp_min(self.eps)
                X_new = self._quantize_rows_with_given_delta(A_rows, delta, bits=int(cfg.act_bits))

                # Update B64 for unstable rows via DeltaC @ W.
                dX64 = (X_new - X_old).to(torch.float64)           # (rb, Ci)
                X_fp64 = self.X.index_select(0, idx).to(torch.float64)  # (rb, Ci)
                deltaC64 = dX64.transpose(0, 1) @ X_fp64           # (Ci, Ci)
                for b, w_fp64 in zip(self.branches, w_fp64_list):
                    b["B64"] += deltaC64 @ w_fp64

                self.Xq.index_copy_(0, idx, X_new)

                gram32_add_tiled_(self.G, X_new, tile=tile32, alpha=+1.0)
                gram32_add_tiled_(self.G, X_old, tile=tile32, alpha=-1.0)

                Xn64 = X_new.to(torch.float64)
                gram64_add_tiled_(self.G64, Xn64, tile=tile64, alpha=+1.0)
                del Xn64

                Xo64 = X_old.to(torch.float64)
                gram64_add_tiled_(self.G64, Xo64, tile=tile64, alpha=-1.0)
                del Xo64

                del idx, X_old, A_rows, maxabs, delta, X_new, dX64, X_fp64, deltaC64

                if hard_free and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del w_fp64_list
        else:
            # stable rows: only column j changes
            self.Xq[:, j] = cache.xqj_new

            dG = cache.deltaG_j
            dGjj = float(cache.deltaG_jj)

            if j > 0:
                self.G[j, :j] += dG[:j]
                self.G[:j, j] += dG[:j]
            if j + 1 < self.Ci:
                self.G[j, j + 1:] += dG[j + 1:]
                self.G[j + 1:, j] += dG[j + 1:]
            self.G[j, j] += dGjj

            dG64 = dG.to(torch.float64)
            if j > 0:
                self.G64[j, :j] += dG64[:j]
                self.G64[:j, j] += dG64[:j]
            if j + 1 < self.Ci:
                self.G64[j, j + 1:] += dG64[j + 1:]
                self.G64[j + 1:, j] += dG64[j + 1:]
            self.G64[j, j] += float(dGjj)

        # ---- update GU/terms per branch ----
        if fully_stable and gcol_old is not None:
            d_off = cache.deltaG_j.clone()
            d_off[j] = 0.0
            dGjj = float(cache.deltaG_jj)
            for b, bc in zip(self.branches, cache.branches):
                GU = b["GU"]
                du = bc.du
                u_new = bc.urowj_new

                # G*U row-update contribution: G[:,j] * du^T using pre-update G.
                GU.add_(gcol_old.unsqueeze(1) * du.unsqueeze(0))
                # dG*U contribution for dG = e_j d^T + d e_j^T + dGjj e_j e_j^T.
                GU.add_(d_off.unsqueeze(1) * u_new.unsqueeze(0))
                r = b["U"].transpose(0, 1).matmul(d_off)
                GU[j, :] += r + (dGjj * u_new)

                b["term1_64"] = bc.term1_new_64
                b["term2_64"] = bc.term2_new_64
        else:
            for b in self.branches:
                self._recompute_GU_and_terms_branch(b)

        # ---- refresh top2 stats incrementally (exact for single-coordinate change) ----
        self._refresh_top2_stats_activation_for_coord(j=j, old_sj=sj_old, new_sj=sj_new)
        for b in self.branches:
            self._refresh_top2_stats_weight_for_coord(b=b, j=j, old_sj=sj_old, new_sj=sj_new)
        self._refresh_group_eval_pack_u_views()

        if bool(getattr(cfg, "hard_free_after_commit", True)) and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Group CD update for s (same logic as yours; evaluator changes)
# ============================================================

@torch.no_grad()
def update_s_coordinate_descent_golden_both_quant_group(
    X: torch.Tensor,
    W_fp_list: Sequence[torch.Tensor],
    s: torch.Tensor,
    W_hat_list: Sequence[torch.Tensor],
    cfg: "SQCfg",
    plots_dir: Optional[Path] = None,
    module_name: str = "",
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    eps = float(getattr(cfg, "s_eps", 1e-8))
    s = s.clone().clamp_min(eps)
    Ci = X.shape[1]

    rel_slack = 1e-7
    abs_slack = 1e-12

    ste_rows = int(getattr(cfg, "screen_ste_rows", 8192))
    ste_rows_max = int(getattr(cfg, "screen_ste_rows_max", 200_000))
    ste_center = bool(getattr(cfg, "screen_ste_center_grad", True))
    gamma = float(getattr(cfg, "screen_gamma", 1.0))

    init_k = int(getattr(cfg, "screen_init_k", 50))
    check_every = int(getattr(cfg, "screen_check_every", 50))
    append_k = int(getattr(cfg, "screen_append_k", 50))
    max_steps = int(getattr(cfg, "s_cd_max_steps", 0))
    active_max_frac = float(getattr(cfg, "screen_active_max_frac", 0.1))

    tol_rel = float(getattr(cfg, "screen_stationary_rel", 1e-4))
    # tol_rel = float(getattr(cfg, "screen_stationary_rel", 1e-2))
    tol_abs = float(getattr(cfg, "screen_stationary_abs", 0.0))
    obj_eps = float(getattr(cfg, "screen_obj_eps", 1e-12))
    print_rel = bool(getattr(cfg, "screen_print_rel_stats", True))
    disable_active_set = bool(getattr(cfg, "ablate_disable_active_set", False))
    disable_instability_screen = bool(getattr(cfg, "ablate_disable_instability_screening", False))
    disable_efficient_obj = bool(getattr(cfg, "ablate_disable_efficient_objective", False))
    asymmetric_quant = (
        canonicalize_quant_mode(str(getattr(cfg, "act_quant_mode", "symmetric"))) != "symmetric"
        or canonicalize_quant_mode(str(getattr(cfg, "w_quant_mode", "symmetric"))) != "symmetric"
    )
    use_exact_recompute = bool(disable_instability_screen or disable_efficient_obj)
    use_dense_exact_obj = bool(disable_efficient_obj)
    dense_precompute_y = bool(getattr(cfg, "ablate_dense_precompute_y", False))
    no_eff_row_chunk = int(getattr(cfg, "ablate_no_eff_row_chunk", 64))
    no_eff_row_chunk = max(1, no_eff_row_chunk)
    no_eff_materialize_dense = bool(getattr(cfg, "ablate_no_eff_materialize_dense", True))

    min_rel_improve = float(getattr(cfg, "screen_append_min_rel_improve", 1e-3))
    # min_rel_improve = float(getattr(cfg, "screen_append_min_rel_improve", 1e-1))
    collect_coord_eval_traces = bool(getattr(cfg, "collect_coord_eval_traces", False))

    if check_every <= 0:
        check_every = 20
    init_k = max(1, min(init_k, Ci))
    append_k = max(1, min(append_k, Ci))
    if disable_active_set:
        init_k = Ci
        # Active-set ablation should traverse all coordinates unless explicitly overridden elsewhere.
        max_steps = 0
        max_active_size = int(Ci)
    else:
        max_active_size = int(math.ceil(max(0.0, active_max_frac) * float(Ci)))
        max_active_size = max(1, min(int(Ci), int(max_active_size)))
        init_k = min(int(init_k), int(max_active_size))

    def _stats_1d(x: torch.Tensor) -> Tuple[float, float, float, float]:
        if x.numel() == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        x_mean = float(x.mean().item())
        return x_mean, x_max, x_min, (x_max - x_min)

    sweep_stats: List[Dict[str, int]] = []
    sweep_coord_eval_traces: List[Dict[str, Any]] = []
    xinv_stage_records: List[Dict[str, Any]] = []
    stop_all_sweeps = False

    def _select_xinv_plot_channels(
        ci: int,
        updated_channels: Sequence[int],
        extra_channels: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        updated_unique: List[int] = []
        seen = set()
        for jj in updated_channels:
            j = int(jj)
            if 0 <= j < int(ci) and j not in seen:
                updated_unique.append(j)
                seen.add(j)

        remaining = [j for j in range(int(ci)) if j not in seen]
        k = min(max(0, int(extra_channels)), len(remaining))
        if k == 0:
            return updated_unique, [], sorted(updated_unique)
        if k >= len(remaining):
            extra_unique = remaining
            return updated_unique, extra_unique, sorted(updated_unique + extra_unique)

        chosen_pos: List[int] = []
        used_pos = set()
        if k == 1:
            mid = len(remaining) // 2
            chosen_pos.append(mid)
            used_pos.add(mid)
        else:
            denom = max(1, k - 1)
            for ii in range(k):
                pos = int(round(ii * (len(remaining) - 1) / denom))
                pos = max(0, min(pos, len(remaining) - 1))
                if pos in used_pos:
                    continue
                chosen_pos.append(pos)
                used_pos.add(pos)
            fill_pos = 0
            while len(chosen_pos) < k and fill_pos < len(remaining):
                if fill_pos not in used_pos:
                    chosen_pos.append(fill_pos)
                    used_pos.add(fill_pos)
                fill_pos += 1

        extra_unique = [remaining[pos] for pos in sorted(chosen_pos)]
        return updated_unique, extra_unique, sorted(updated_unique + extra_unique)

    for sweep in range(int(getattr(cfg, "s_sweeps", 1))):
        sweep_hit_active_cap = False
        evaluator = SCoordCandidateEvaluatorGramGroup(
            X=X, W_fp_list=W_fp_list, s=s, W_hat_list=W_hat_list, cfg=cfg
        )

        f0 = float(evaluator.current_obj())
        base = f0 if f0 != 0.0 else 1.0

        if disable_active_set:
            active = list(range(Ci))
            in_active = torch.ones((Ci,), device=X.device, dtype=torch.bool)
            scores0 = None
        else:
            scores0 = ste_grad_scores_for_active_set_group(
                X=X,
                W_fp_list=W_fp_list,
                W_hat_list=W_hat_list,
                s=evaluator.s,
                cfg=cfg,
                rows=ste_rows,
                rows_max=ste_rows_max,
                use_abs=True,
                subtract_mean=ste_center,
            )
            if gamma != 1.0:
                scores0 = scores0.clamp_min(0.0) ** gamma

            order0 = torch.argsort(scores0, descending=True)
            active = order0[:init_k].tolist()
            in_active = torch.zeros((Ci,), device=X.device, dtype=torch.bool)
            in_active[order0[:init_k]] = True
        initial_active_for_sweep = [int(x) for x in active]

        Y_targets_dense: Optional[List[torch.Tensor]] = None
        if use_exact_recompute and use_dense_exact_obj and dense_precompute_y:
            Y_targets_dense = [(X @ W_fp).contiguous() for W_fp in W_fp_list]

        def _eval_candidate_exact_obj(j: int, sj_try: float) -> float:
            s_try = evaluator.s.clone()
            s_try[int(j)] = float(max(float(sj_try), eps))
            Xq_try = build_Xq_from_s_streaming(
                X,
                s_try,
                act_bits=int(cfg.act_bits),
                eps=eps,
                act_chunk_cols=int(getattr(cfg, "act_quant_chunk_cols", 512)),
                mode=str(getattr(cfg, "act_quant_mode", "symmetric")),
            )
            w_chunk_rows = _runtime_w_chunk_rows(cfg)
            obj = 0.0
            if use_dense_exact_obj:
                for bi, (W_fp, W_hat) in enumerate(zip(W_fp_list, W_hat_list)):
                    U_try = build_U_from_s_What(
                        W_hat,
                        s_try,
                        w_bits=cfg.w_bits,
                        eps=eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
                    )
                    # Optional dense materialization to reflect non-efficient buffer-heavy path.
                    if no_eff_materialize_dense:
                        Y_ref = Y_targets_dense[bi] if Y_targets_dense is not None else (X @ W_fp)
                        P = Xq_try @ U_try
                        R = (Y_ref - P)
                        # Touch dense residual so this path reflects dense-buffer cost.
                        _ = float(R.abs().mean().item())
                        del P, R, Y_ref

                    # Exact objective recompute without rank-one updates; intentionally less optimized.
                    obj += obj_from_X_Xq_W_U_streaming(
                        X=X,
                        Xq=Xq_try,
                        W_fp=W_fp,
                        U=U_try,
                        row_chunk=no_eff_row_chunk,
                    )
                    del U_try
            else:
                for W_fp, W_hat in zip(W_fp_list, W_hat_list):
                    U_try = build_U_from_s_What(
                        W_hat,
                        s_try,
                        w_bits=cfg.w_bits,
                        eps=eps,
                        w_chunk_rows=w_chunk_rows,
                        w_quant_mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
                    )
                    obj += obj_from_X_Xq_W_U_streaming(
                        X=X, Xq=Xq_try, W_fp=W_fp, U=U_try, row_chunk=int(getattr(cfg, "obj_row_chunk", 256))
                    )
                    del U_try
            return float(obj)

        if disable_active_set:
            print(f"[screen-init] disabled active-set screening; active={len(active)}/{Ci}")
        else:
            assert scores0 is not None
            top1 = float(scores0.max().item()) if Ci > 0 else float("nan")
            print(
                f"[screen-init] f0={f0:.3e} score_top1={top1:.3e} active={len(active)} "
                f"active_cap={max_active_size}/{Ci} "
                f"tol_abs={tol_abs:.2e} tol_rel={tol_rel:.2e} min_rel_improve={min_rel_improve:.2e} rows={min(ste_rows, X.shape[0])}"
            )
        if asymmetric_quant and sweep == 0:
            print("[info] asymmetric quantization selected; using incremental min/max endpoint tracking")

        rel_obj_curve: List[float] = []
        accepted_coords_for_stage: List[int] = []
        f_now = float(f0)
        attempted = 0
        idx_ptr = 0
        active_init = int(len(active))
        stage_attempted = 0
        stage_index = 0

        def _record_xinv_stage() -> None:
            nonlocal accepted_coords_for_stage, stage_attempted, stage_index
            if stage_attempted <= 0:
                return
            stage_index += 1
            updated_plot_channels, extra_plot_channels, selected_plot_channels = _select_xinv_plot_channels(
                ci=int(Ci),
                updated_channels=accepted_coords_for_stage,
                extra_channels=int(getattr(cfg, "xinv_extra_channels_plot", 64)),
            )
            xinv_stage_records.append(
                {
                    "sweep": int(sweep + 1),
                    "stage": int(stage_index),
                    "stage_attempted": int(stage_attempted),
                    "s": evaluator.s.detach().cpu().clone(),
                    "selected_channels": list(selected_plot_channels),
                    "updated_channels": list(updated_plot_channels),
                    "extra_channels": list(extra_plot_channels),
                }
            )
            accepted_coords_for_stage = []
            stage_attempted = 0

        use_tqdm = bool(getattr(cfg, "show_tqdm", True)) and bool(getattr(cfg, "show_coord_tqdm", False))
        pbar = tqdm(total=max(1, len(active)), desc=f"s-update(group) sweep {sweep+1}", leave=False) if use_tqdm else None

        pending_batch_remaining = 0
        pending_batch_start_obj = None
        pending_batch_size = 0

        def _maybe_check_append_effectiveness() -> bool:
            nonlocal pending_batch_remaining, pending_batch_start_obj, pending_batch_size
            if pending_batch_remaining != 0:
                return False
            if pending_batch_start_obj is None or pending_batch_size <= 0:
                return False

            obj_after = float(evaluator.current_obj())
            obj_before = float(pending_batch_start_obj)
            denom = max(obj_before, obj_eps)
            rel_improve = (obj_before - obj_after) / denom

            if rel_improve < min_rel_improve:
                print(
                    f"[append-stop] batch_size={pending_batch_size} obj_before={obj_before:.3e} obj_after={obj_after:.3e} "
                    f"rel_improve={rel_improve:.3e} < {min_rel_improve:.3e} => stop"
                )
                return True
            else:
                print(
                    f"[append-ok] batch_size={pending_batch_size} obj_before={obj_before:.3e} obj_after={obj_after:.3e} "
                    f"rel_improve={rel_improve:.3e} >= {min_rel_improve:.3e}"
                )
                pending_batch_start_obj = None
                pending_batch_size = 0
                return False

        def _stationarity_check_and_append_batch() -> bool:
            nonlocal active, in_active, pending_batch_remaining, pending_batch_start_obj, pending_batch_size, sweep_hit_active_cap
            if (not disable_active_set) and (len(active) >= int(max_active_size)):
                sweep_hit_active_cap = True
                print(
                    f"[stop] sweep {sweep+1}: active-set cap reached "
                    f"({len(active)}/{int(max_active_size)}; Ci={Ci}); stopping CD."
                )
                return True

            scores = ste_grad_scores_for_active_set_group(
                X=X,
                W_fp_list=W_fp_list,
                W_hat_list=W_hat_list,
                s=evaluator.s,
                cfg=cfg,
                rows=ste_rows,
                rows_max=ste_rows_max,
                use_abs=True,
                subtract_mean=ste_center,
            )
            if gamma != 1.0:
                scores = scores.clamp_min(0.0) ** gamma

            outside = ~in_active
            if not bool(outside.any()):
                return True

            outside_scores = scores.masked_select(outside)
            if outside_scores.numel() == 0:
                return True

            obj_cur = float(evaluator.current_obj())
            obj_scale = max(obj_cur, obj_eps)
            thr = float(tol_abs + tol_rel * obj_scale)

            max_out = float(outside_scores.max().item())
            mean_out, _, min_out, rng_out = _stats_1d(outside_scores)

            if print_rel:
                outside_rel = outside_scores / obj_scale
                mean_rel, max_rel, min_rel, rng_rel = _stats_1d(outside_rel)
                print(
                    f"[screen] obj={obj_cur:.3e} thr={thr:.3e} "
                    f"outside_abs: mean={mean_out:.3e} max={max_out:.3e} min={min_out:.3e} range={rng_out:.3e} "
                    f"outside_rel: mean={mean_rel:.3e} max={max_rel:.3e} min={min_rel:.3e} range={rng_rel:.3e}"
                )
            else:
                print(
                    f"[screen] obj={obj_cur:.3e} thr={thr:.3e} "
                    f"outside_abs: mean={mean_out:.3e} max={max_out:.3e} min={min_out:.3e} range={rng_out:.3e}"
                )

            if max_out <= thr:
                _record_xinv_stage()
                return True

            viol = outside & (scores > thr)
            n_viol = int(viol.sum().item())
            if n_viol == 0:
                return True

            viol_idx = viol.nonzero(as_tuple=False).squeeze(1)
            viol_scores = scores.index_select(0, viol_idx)

            slots_left = int(max_active_size) - int(len(active))
            if slots_left <= 0:
                sweep_hit_active_cap = True
                print(
                    f"[stop] sweep {sweep+1}: active-set cap reached "
                    f"({len(active)}/{int(max_active_size)}; Ci={Ci}); stopping CD."
                )
                return True

            k = min(int(append_k), int(viol_idx.numel()), int(slots_left))
            topk_local = torch.topk(viol_scores, k=k, largest=True).indices
            pick = viol_idx.index_select(0, topk_local)

            pick_scores = scores.index_select(0, pick)
            sort_local = torch.argsort(pick_scores, descending=True)
            pick = pick.index_select(0, sort_local)

            _record_xinv_stage()
            new_list = pick.tolist()
            added = 0
            for jj in new_list:
                jj = int(jj)
                if not bool(in_active[jj].item()):
                    active.append(jj)
                    in_active[jj] = True
                    added += 1

            if added > 0:
                pending_batch_start_obj = float(evaluator.current_obj())
                pending_batch_remaining = int(added)
                pending_batch_size = int(added)

            print(
                f"[stationarity] max_outside={max_out:.3e} thr={thr:.3e} "
                f"(tol_abs={tol_abs:.1e}, tol_rel={tol_rel:.1e}, obj={obj_cur:.3e}) appended={added} active_now={len(active)}/{Ci}"
            )
            if (not disable_active_set) and (len(active) >= int(max_active_size)):
                sweep_hit_active_cap = True
                print(
                    f"[stop] sweep {sweep+1}: active-set cap reached "
                    f"({len(active)}/{int(max_active_size)}; Ci={Ci}); stopping CD."
                )
                return True
            return False

        while idx_ptr < len(active):
            if max_steps > 0 and attempted >= max_steps:
                break

            j = int(active[idx_ptr])
            idx_ptr += 1
            attempted += 1
            stage_attempted += 1

            coord_ctx = evaluator.prepare_coord_eval_context(j)
            obj_before = float(evaluator.current_obj())
            obj_step_end = obj_before

            sj0 = float(evaluator.s[j].clamp_min(eps).item())

            best_obj_pred = obj_before
            best_sj = sj0
            best_cache: Optional[CandidateCacheGroup] = None
            coord_trace: Optional[Dict[str, Any]] = None
            coord_base = max(float(obj_before), obj_eps)
            if collect_coord_eval_traces:
                coord_trace = {
                    "sweep": int(sweep + 1),
                    "coord": int(j),
                    "attempt_index": int(attempted),
                    "obj_before": float(obj_before),
                    "rel_obj_before": 1.0,
                    "rel_obj_before_sweep": float(obj_before) / float(base),
                    "s_before": float(sj0),
                    "active_size_at_attempt": int(len(active)),
                    "in_initial_active_set": bool(j in initial_active_for_sweep),
                    "evals": [],
                }

            def _record_eval(source: str, sj_try: float, val: float) -> None:
                if coord_trace is not None:
                    coord_trace["evals"].append(
                        {
                            "source": str(source),
                            "s": float(sj_try),
                            "obj": float(val),
                            "rel_obj": float(val) / float(coord_base),
                            "rel_obj_sweep": float(val) / float(base),
                        }
                    )

            def eval_obj(sj_try: float, source: str) -> float:
                if use_exact_recompute:
                    val = float(_eval_candidate_exact_obj(j, sj_try))
                else:
                    val, _, _, _ = evaluator.eval_candidate(j, sj_try, coord_ctx=coord_ctx)
                    val = float(val)
                _record_eval(source, sj_try, val)
                return float(val)

            def try_update(sj_try: float, source: str) -> None:
                nonlocal best_obj_pred, best_cache, best_sj
                if use_exact_recompute:
                    val = eval_obj(sj_try, source)
                    if (float(val) + float(getattr(cfg, "accept_tol", 0.0))) < best_obj_pred:
                        best_obj_pred = float(val)
                        best_sj = float(sj_try)
                        best_cache = None
                else:
                    val, _, _, cache = evaluator.eval_candidate(j, sj_try, coord_ctx=coord_ctx)
                    val = float(val)
                    _record_eval(source, sj_try, val)
                    if (float(val) + float(getattr(cfg, "accept_tol", 0.0))) < best_obj_pred:
                        best_obj_pred = float(val)
                        best_cache = cache
                        best_sj = float(sj_try)

            m = float(getattr(cfg, "s_bounds_mult", 10.0))
            a = max(eps, sj0 / m)
            bnd = max(a * (1.0 + 1e-6), sj0 * m)

            try_update(a, "bracket_lo")
            try_update(bnd, "bracket_hi")

            def f_scalar(sj_float: float) -> float:
                return float(eval_obj(sj_float, "golden_iter"))

            sj_star = golden_section_search_1d(f_scalar, a, bnd, max_iter=int(getattr(cfg, "golden_max_iter", 20)))
            sj_star = max(float(sj_star), eps)
            try_update(sj_star, "golden_final")

            if int(getattr(cfg, "coord_backtrack_steps", 3)) > 0:
                log0 = math.log(max(sj0, eps))
                log1 = math.log(max(sj_star, eps))
                for kk in range(int(getattr(cfg, "coord_backtrack_steps", 3))):
                    t = 0.5 ** (kk + 1)
                    sj_try = math.exp((1.0 - t) * log0 + t * log1)
                    try_update(sj_try, f"backtrack_{kk+1}")

            if use_exact_recompute:
                if (float(best_obj_pred) + float(getattr(cfg, "accept_tol", 0.0))) < float(obj_before):
                    s_saved = evaluator.s.clone()
                    s_try = s_saved.clone()
                    s_try[j] = float(max(best_sj, eps))
                    old_eval = evaluator
                    evaluator = SCoordCandidateEvaluatorGramGroup(
                        X=X, W_fp_list=W_fp_list, s=s_try, W_hat_list=W_hat_list, cfg=cfg
                    )
                    del old_eval

                    obj_after = float(evaluator.current_obj())
                    slack = abs_slack + rel_slack * max(1.0, abs(obj_before))
                    must_be_below = obj_before - float(getattr(cfg, "accept_tol", 0.0)) + slack
                    if not (obj_after < must_be_below):
                        old_eval = evaluator
                        evaluator = SCoordCandidateEvaluatorGramGroup(
                            X=X, W_fp_list=W_fp_list, s=s_saved, W_hat_list=W_hat_list, cfg=cfg
                        )
                        del old_eval
                        obj_step_end = float(evaluator.current_obj())
                    else:
                        obj_step_end = obj_after
            elif best_cache is not None:
                s_saved = evaluator.s.clone()
                evaluator.commit(best_cache)
                obj_after = float(evaluator.current_obj())

                slack = abs_slack + rel_slack * max(1.0, abs(obj_before))
                must_be_below = obj_before - float(getattr(cfg, "accept_tol", 0.0)) + slack

                if not (obj_after < must_be_below):
                    old_eval = evaluator
                    evaluator = SCoordCandidateEvaluatorGramGroup(
                        X=X, W_fp_list=W_fp_list, s=s_saved, W_hat_list=W_hat_list, cfg=cfg
                    )
                    del old_eval
                    obj_reverted = float(evaluator.current_obj())
                    obj_step_end = obj_reverted
                else:
                    obj_step_end = obj_after

            if obj_step_end < 0.0:
                scale = max(1.0, abs(obj_step_end), abs(obj_before))
                tiny_neg_tol = 1e-8 * scale
                if obj_step_end > -tiny_neg_tol:
                    obj_step_end = 0.0
                else:
                    old_eval = evaluator
                    evaluator = SCoordCandidateEvaluatorGramGroup(
                        X=X, W_fp_list=W_fp_list, s=evaluator.s, W_hat_list=W_hat_list, cfg=cfg
                    )
                    del old_eval
                    obj_step_end = float(evaluator.current_obj())
                    if obj_step_end < -tiny_neg_tol:
                        print(
                            f"[warn] objective remained negative after evaluator rebuild: {obj_step_end:.6e} "
                            f"(coord={j}, attempt={attempted})"
                        )

            f_now = float(obj_step_end)
            if f_now < float(obj_before):
                accepted_coords_for_stage.append(int(j))
            rel_obj_curve.append(float(f_now) / base)
            if coord_trace is not None:
                coord_trace["s_best_pred"] = float(best_sj)
                coord_trace["obj_best_pred"] = float(best_obj_pred)
                coord_trace["rel_obj_best_pred"] = float(best_obj_pred) / float(coord_base)
                coord_trace["rel_obj_best_pred_sweep"] = float(best_obj_pred) / float(base)
                coord_trace["obj_after"] = float(obj_step_end)
                coord_trace["rel_obj_after"] = float(obj_step_end) / float(coord_base)
                coord_trace["rel_obj_after_sweep"] = float(obj_step_end) / float(base)
                coord_trace["accepted"] = bool(obj_step_end < obj_before)
                coord_trace["active_size_after"] = int(len(active))
                sweep_coord_eval_traces.append(coord_trace)

            if pending_batch_remaining > 0:
                pending_batch_remaining -= 1
                if pending_batch_remaining == 0:
                    if _maybe_check_append_effectiveness():
                        print(f"[stop] sweep {sweep+1}: appended batch failed to improve objective enough; stopping CD.")
                        break

            if pbar is not None:
                if len(active) > pbar.total:
                    pbar.total = len(active)
                pbar.update(1)
                postfix = {
                    "obj": f"{f_now:.3e}",
                    "rel": f"{(float(f_now)/base):.6f}",
                    "active": f"{len(active)}/{Ci}",
                }
                if pending_batch_start_obj is not None:
                    postfix["pend"] = str(pending_batch_remaining)
                pbar.set_postfix(postfix)

            if (not disable_active_set) and ((attempted % check_every) == 0):
                should_stop = _stationarity_check_and_append_batch()
                if should_stop:
                    if not sweep_hit_active_cap:
                        print(f"[stationarity] sweep {sweep+1}: outside-active stationary after {attempted} updates; stopping CD.")
                    break

        if disable_active_set and max_steps <= 0 and attempted != int(Ci):
            raise RuntimeError(
                f"Active-set ablation expected to attempt all coordinates in sweep "
                f"(attempted={attempted}, Ci={int(Ci)})."
            )

        if pbar is not None:
            pbar.close()

        _record_xinv_stage()

        sweep_obj_end = float(evaluator.current_obj())
        sweep_stats.append(
            {
                "sweep": int(sweep + 1),
                "attempted": int(attempted),
                "active_init": int(active_init),
                "active_final": int(len(active)),
                "ci": int(Ci),
            }
        )
        print(
            f"[s-sweep-end] sweep={sweep+1} obj={sweep_obj_end:.3e} "
            f"rel={float(sweep_obj_end)/base:.6f} attempted={attempted} active_final={len(active)}/{Ci}"
        )

        if plots_dir is not None:
            fname = f"{_safe_name(module_name)}__sweep_{sweep+1:02d}"
            out_base = plots_dir / fname
            save_coord_progress_plot(
                rel_obj=rel_obj_curve,
                module_name=module_name,
                sweep_idx=(sweep + 1),
                out_path_base=out_base,
            )
            print(f"[plot] saved {out_base.with_suffix('.pdf')}")
            if bool(getattr(cfg, "plot_xinv_each_sweep", False)):
                last_stage = xinv_stage_records[-1] if len(xinv_stage_records) > 0 else {}
                selected_plot_channels = list(last_stage.get("selected_channels", [])) if isinstance(last_stage, dict) else []
                updated_plot_channels = list(last_stage.get("updated_channels", [])) if isinstance(last_stage, dict) else []
                extra_plot_channels = list(last_stage.get("extra_channels", [])) if isinstance(last_stage, dict) else []
                surf_base = plots_dir / f"{fname}__x_over_s"
                save_x_over_s_surface_plot(
                    X=X,
                    s=evaluator.s,
                    module_name=module_name,
                    sweep_idx=(sweep + 1),
                    out_path_base=surf_base,
                    selected_channels=selected_plot_channels,
                    updated_channels=updated_plot_channels,
                    max_tokens_plot=int(getattr(cfg, "xinv_max_tokens_plot", 256)),
                    max_in_channels_plot=int(getattr(cfg, "xinv_max_in_channels_plot", 1024)),
                    elev=float(getattr(cfg, "xinv_elev", 25.0)),
                    azim=float(getattr(cfg, "xinv_azim", -60.0)),
                    dpi=int(getattr(cfg, "xinv_dpi", 220)),
                    p_lo=float(getattr(cfg, "xinv_p_lo", 5.0)),
                    p_hi=float(getattr(cfg, "xinv_p_hi", 99.0)),
                    alpha=float(getattr(cfg, "xinv_alpha", 0.85)),
                    cmap_name=str(getattr(cfg, "xinv_cmap", "coolwarm")),
                    shade=bool(getattr(cfg, "xinv_shade", False)),
                )
                print(
                    f"[plot] saved {surf_base.with_suffix('.png')} "
                    f"(updated={len(updated_plot_channels)}, extra={len(extra_plot_channels)})"
                )

        s = evaluator.s.clone()
        if sweep_hit_active_cap:
            stop_all_sweeps = True
            break

    # Return final s and per-branch U’s
    attempted_total = int(sum(int(x.get("attempted", 0)) for x in sweep_stats))
    active_init_total = int(sum(int(x.get("active_init", 0)) for x in sweep_stats))
    active_final_total = int(sum(int(x.get("active_final", 0)) for x in sweep_stats))
    setattr(
        cfg,
        "ablate_last_coord_stats",
        {
            "attempted_total": attempted_total,
            "active_init_total": active_init_total,
            "active_final_total": active_final_total,
            "sweeps": int(len(sweep_stats)),
            "ci": int(Ci),
            "disable_active_set": bool(disable_active_set),
            "disable_instability_screening": bool(disable_instability_screen),
            "disable_efficient_objective": bool(disable_efficient_obj),
            "active_max_size": int(max_active_size),
            "active_max_frac": float(active_max_frac),
            "stopped_on_active_cap": bool(stop_all_sweeps),
            "sweep_stats": sweep_stats,
        },
    )
    setattr(
        cfg,
        "ablate_last_coord_eval_traces",
        {
            "module_name": str(module_name),
            "coord_eval_sweeps": sweep_coord_eval_traces,
            "base_objective": float(f0) if 'f0' in locals() else float("nan"),
        },
    )
    setattr(
        cfg,
        "ablate_last_xinv_sweeps",
        {
            "module_name": str(module_name),
            "stages": xinv_stage_records,
        },
    )
    U_list = [b["U"].clone() for b in evaluator.branches]
    return evaluator.s.clone(), U_list


# ============================================================
# Group U update (per-branch; shared Gram/stats for fixed s)
# ============================================================

@torch.no_grad()
def update_U_from_current_group(
    G: torch.Tensor,
    G64: torch.Tensor,
    branch_stats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    s: torch.Tensor,
    W_hat_list: Sequence[torch.Tensor],
    cfg: SQCfg,
    u_update: str = "gptq",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, bool]:
    """
    Update U for each branch independently, accepting if total objective decreases.
    Returns:
      W_hat_new_list, U_new_list, obj_best, accepted
    """
    assert len(branch_stats) == len(W_hat_list)
    assert G.dtype == torch.float32
    assert G64.dtype == torch.float64
    eps = cfg.s_eps
    s_safe = s.clamp_min(eps)
    w_chunk_rows = _runtime_w_chunk_rows(cfg)

    # Build current U_old list and objective
    U_old_list = [
        build_U_from_s_What(
            W_hat,
            s_safe,
            cfg.w_bits,
            eps,
            w_chunk_rows=w_chunk_rows,
            w_quant_mode=str(getattr(cfg, "w_quant_mode", "symmetric")),
        )
        for W_hat in W_hat_list
    ]
    obj_old = group_obj_from_gram_stats_fp64(
        G64,
        branch_stats,
        U_old_list,
        col_chunk=int(getattr(cfg, "gu_obj_chunk_cols", 256)),
    )
    H = (2.0 * G).to(torch.float32)

    U_try_list: List[torch.Tensor] = []
    for idx, (_c64, B64) in enumerate(branch_stats):
        U_ls = ridge_ls_from_gram_fp32(G, B64.to(torch.float32), cfg)

        if u_update == "gptq":
            U_try = gptq_quantize_U_from_H(H=H, U_init=U_ls, cfg=cfg, desc=f"GPTQ(U) group[{idx}]")
        elif u_update == "ls_project":
            U_try = project_U_ls_rtn(U_ls, cfg)
        else:
            raise ValueError("u_update must be one of {'gptq','ls_project'}")

        U_try_list.append(U_try)

    obj_try = group_obj_from_gram_stats_fp64(
        G64,
        branch_stats,
        U_try_list,
        col_chunk=int(getattr(cfg, "gu_obj_chunk_cols", 256)),
    )

    if (obj_try + cfg.accept_tol) < obj_old:
        best_U_list = U_try_list
        best_obj = obj_try
        accepted = True
    else:
        best_U_list = U_old_list
        best_obj = obj_old
        accepted = False

    W_hat_new_list = [U / s_safe.unsqueeze(1) for U in best_U_list]
    return W_hat_new_list, best_U_list, float(best_obj), bool(accepted)

# quantization_utils.py
from __future__ import annotations

import torch


def qmax_signed(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def canonicalize_quant_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"symmetric", "sym"}:
        return "symmetric"
    if m in {"asymmetric", "asym", "affine"}:
        return "asymmetric"
    raise ValueError(f"Unsupported quantization mode '{mode}'.")


def quant_int_bounds(bits: int, mode: str) -> tuple[int, int]:
    m = canonicalize_quant_mode(mode)
    if m == "symmetric":
        qmax = qmax_signed(bits)
        return -qmax, qmax
    return 0, (1 << bits) - 1


def _reshape_param_for_last_dim(param: torch.Tensor, ndim: int) -> torch.Tensor:
    view_shape = [1] * ndim
    view_shape[-1] = int(param.numel())
    return param.view(*view_shape)


def _quantize_with_params_fp32(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (q - zero_point) * scale


def _ste_quantize_with_params(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    q = x / scale + zero_point
    q_ste = _ste_round_clamp(q, qmin=qmin, qmax=qmax)
    return (q_ste - zero_point) * scale


def compute_per_token_quant_params(
    x: torch.Tensor,
    bits: int,
    eps: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    m = canonicalize_quant_mode(mode)
    qmin, qmax = quant_int_bounds(bits, m)
    if m == "symmetric":
        maxabs = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = (maxabs / float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point, qmin, qmax

    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / float(qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(torch.tensor(float(qmin), device=x.device, dtype=x.dtype) - xmin / scale)
    zero_point = zero_point.clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


def compute_per_out_channel_quant_params(
    W: torch.Tensor,
    bits: int,
    eps: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    m = canonicalize_quant_mode(mode)
    qmin, qmax = quant_int_bounds(bits, m)
    if m == "symmetric":
        maxabs = W.abs().amax(dim=0, keepdim=True).clamp_min(eps)
        scale = (maxabs / float(qmax)).clamp_min(eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point, qmin, qmax

    w_min = W.amin(dim=0, keepdim=True)
    w_max = W.amax(dim=0, keepdim=True)
    scale = ((w_max - w_min) / float(qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(torch.tensor(float(qmin), device=W.device, dtype=W.dtype) - w_min / scale)
    zero_point = zero_point.clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


@torch.no_grad()
def build_Xq_from_s_streaming(
    X: torch.Tensor,
    s: torch.Tensor,
    *,
    act_bits: int,
    eps: float,
    act_chunk_cols: int = 512,
    mode: str = "symmetric",
) -> torch.Tensor:
    """
    Memory-lean activation quantization that avoids materializing X / s in full.
    """
    assert X.dtype == torch.float32 and s.dtype == torch.float32
    assert X.dim() == 2 and s.dim() == 1 and X.shape[1] == s.shape[0]

    qmin, qmax = quant_int_bounds(int(act_bits), mode)
    if qmax <= qmin:
        raise ValueError(f"Invalid integer range for act_bits={act_bits}, mode={mode}")

    N, C = X.shape
    cc = int(max(1, act_chunk_cols))
    inv_s = torch.reciprocal(s.clamp_min(float(eps)))

    quant_mode = canonicalize_quant_mode(mode)
    if quant_mode == "symmetric":
        maxabs = torch.zeros((N, 1), device=X.device, dtype=torch.float32)
        for c0 in range(0, C, cc):
            c1 = min(c0 + cc, C)
            a_chunk = X[:, c0:c1] * inv_s[c0:c1].unsqueeze(0)
            maxabs = torch.maximum(maxabs, a_chunk.abs().amax(dim=1, keepdim=True))
            del a_chunk
        scale = (maxabs.clamp_min(float(eps)) / float(qmax)).clamp_min(float(eps))
        zero_point = torch.zeros_like(scale)
    else:
        x_min = torch.full((N, 1), float("inf"), device=X.device, dtype=torch.float32)
        x_max = torch.full((N, 1), -float("inf"), device=X.device, dtype=torch.float32)
        for c0 in range(0, C, cc):
            c1 = min(c0 + cc, C)
            a_chunk = X[:, c0:c1] * inv_s[c0:c1].unsqueeze(0)
            x_min = torch.minimum(x_min, a_chunk.amin(dim=1, keepdim=True))
            x_max = torch.maximum(x_max, a_chunk.amax(dim=1, keepdim=True))
            del a_chunk
        scale = ((x_max - x_min).clamp_min(float(eps)) / float(qmax - qmin)).clamp_min(float(eps))
        zero_point = torch.round(torch.tensor(float(qmin), device=X.device, dtype=torch.float32) - x_min / scale)
        zero_point = zero_point.clamp(qmin, qmax)

    out = torch.empty_like(X)

    for c0 in range(0, C, cc):
        c1 = min(c0 + cc, C)
        a_chunk = X[:, c0:c1] * inv_s[c0:c1].unsqueeze(0)
        q = torch.round(torch.div(a_chunk, scale) + zero_point).clamp(qmin, qmax)
        out[:, c0:c1] = (q - zero_point) * scale
        del a_chunk, q

    return out


@torch.no_grad()
def quantize_per_token_maxabs_fp32(
    x: torch.Tensor,
    bits: int,
    eps: float = 1e-8,
    *,
    chunk_cols: int = 1024,
    mode: str = "symmetric",
) -> torch.Tensor:
    """
    Per-token fp32 quantization over the last dimension.
    """
    assert x.dtype == torch.float32
    scale, zero_point, qmin, qmax = compute_per_token_quant_params(x, bits, eps, mode)

    C = x.shape[-1]
    out = torch.empty_like(x)

    cc = int(max(1, chunk_cols))
    for c0 in range(0, C, cc):
        c1 = min(c0 + cc, C)
        q = torch.div(x[..., c0:c1], scale) + zero_point
        q = torch.round(q)
        q.clamp_(qmin, qmax)
        out[..., c0:c1] = (q - zero_point).mul(scale)
        del q

    return out


@torch.no_grad()
def quantize_weights_per_out_channel_fp32(
    W: torch.Tensor,
    bits: int,
    eps: float = 1e-8,
    chunk_rows: int | None = None,
    mode: str = "symmetric",
) -> torch.Tensor:
    """
    Per-output-channel fp32 quantization in fp32.
    Default chunk_rows=None processes all rows (original behavior).
    """
    assert W.dtype == torch.float32
    scale, zero_point, qmin, qmax = compute_per_out_channel_quant_params(W, bits, eps, mode)

    Ci, Co = W.shape
    out = torch.empty_like(W)

    if chunk_rows is None or int(chunk_rows) <= 0 or int(chunk_rows) >= Ci:
        q = torch.div(W, scale) + zero_point
        q = torch.round(q)
        q.clamp_(qmin, qmax)
        out.copy_((q - zero_point).mul(scale))
        return out

    rr = int(max(1, chunk_rows))
    for r0 in range(0, Ci, rr):
        r1 = min(r0 + rr, Ci)
        q = torch.div(W[r0:r1, :], scale) + zero_point
        q = torch.round(q)
        q.clamp_(qmin, qmax)
        out[r0:r1, :] = (q - zero_point).mul(scale)
        del q

    return out


@torch.no_grad()
def quantize_scalar_with_delta(x: torch.Tensor, delta: torch.Tensor, bits: int) -> torch.Tensor:
    assert x.dtype == torch.float32 and delta.dtype == torch.float32
    qm = qmax_signed(bits)
    z = torch.round(x / delta).clamp(-qm, qm)
    return z * delta


@torch.no_grad()
def quantize_with_params(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    bits: int,
    mode: str = "symmetric",
) -> torch.Tensor:
    qmin, qmax = quant_int_bounds(bits, mode)
    return _quantize_with_params_fp32(x, scale, zero_point, qmin=qmin, qmax=qmax)


def _ste_round_clamp(q: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    q_round = torch.round(q)
    q_clip = torch.clamp(q_round, qmin, qmax)
    return q + (q_clip - q).detach()


def quantize_per_token_maxabs_ste(x: torch.Tensor, bits: int, eps: float = 1e-8, mode: str = "symmetric") -> torch.Tensor:
    assert x.dtype == torch.float32
    scale, zero_point, qmin, qmax = compute_per_token_quant_params(x, bits, eps, mode)
    return _ste_quantize_with_params(x, scale, zero_point, qmin=qmin, qmax=qmax)


def quantize_weights_per_out_channel_ste(
    W: torch.Tensor,
    bits: int,
    eps: float = 1e-8,
    mode: str = "symmetric",
) -> torch.Tensor:
    assert W.dtype == torch.float32
    scale, zero_point, qmin, qmax = compute_per_out_channel_quant_params(W, bits, eps, mode)
    return _ste_quantize_with_params(W, scale, zero_point, qmin=qmin, qmax=qmax)

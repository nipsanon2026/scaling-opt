#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from inspect_coord_evals import plot_xinv_grid_from_payload


def main() -> None:
    ap = argparse.ArgumentParser("Replot saved activation-stage grid without rerunning the algorithm.")
    ap.add_argument("--data", type=str, required=True, help="Path to *__activation_sweeps_data.pt")
    ap.add_argument("--out", type=str, default="", help="Output path base without extension")
    ap.add_argument("--elev", type=float, default=None)
    ap.add_argument("--azim", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--p_lo", type=float, default=None)
    ap.add_argument("--p_hi", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--cmap", type=str, default="")
    ap.add_argument("--shade", action="store_true")
    ap.add_argument("--no_shade", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data)
    payload = torch.load(data_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type in {data_path}: {type(payload)!r}")

    grid_payload = payload.get("grid_payload", {})
    if not isinstance(grid_payload, dict) or not grid_payload:
        raise RuntimeError(f"No grid_payload found in {data_path}")

    plot_cfg: Dict[str, Any] = dict(payload.get("plot_config", {})) if isinstance(payload.get("plot_config", {}), dict) else {}
    elev = float(args.elev) if args.elev is not None else float(plot_cfg.get("elev", 25.0))
    azim = float(args.azim) if args.azim is not None else float(plot_cfg.get("azim", -60.0))
    dpi = int(args.dpi) if args.dpi is not None else int(plot_cfg.get("dpi", 220))
    p_lo = float(args.p_lo) if args.p_lo is not None else float(plot_cfg.get("p_lo", 5.0))
    p_hi = float(args.p_hi) if args.p_hi is not None else float(plot_cfg.get("p_hi", 99.0))
    alpha = float(args.alpha) if args.alpha is not None else float(plot_cfg.get("alpha", 0.85))
    cmap_name = str(args.cmap).strip() if str(args.cmap).strip() else str(plot_cfg.get("cmap_name", "coolwarm"))
    if args.shade and args.no_shade:
        raise RuntimeError("Choose at most one of --shade or --no_shade")
    if args.shade:
        shade = True
    elif args.no_shade:
        shade = False
    else:
        shade = bool(plot_cfg.get("shade", False))

    out_base = Path(args.out) if str(args.out).strip() else data_path.with_name(data_path.stem.replace("__data", "__replot"))
    plot_xinv_grid_from_payload(
        grid_payload,
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
    print(f"[save] replotted activation grid -> {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()

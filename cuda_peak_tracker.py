# cuda_peak_tracker.py
from __future__ import annotations

import torch


def _format_bytes(n: int) -> str:
    return f"{n / (1024**2):.1f} MiB"


class CUDAPeakTracker:
    def __init__(self, device: torch.device):
        self.enabled = (device.type == "cuda") and torch.cuda.is_available()
        self.device = device if self.enabled else None
        self._peak_alloc = 0
        self._peak_reserved = 0

    def reset(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_alloc = 0
        self._peak_reserved = 0

    def update(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        a = int(torch.cuda.max_memory_allocated(self.device))
        r = int(torch.cuda.max_memory_reserved(self.device))
        self._peak_alloc = max(self._peak_alloc, a)
        self._peak_reserved = max(self._peak_reserved, r)

    def snapshot_str(self) -> str:
        if not self.enabled:
            return "CPU (no CUDA)"
        self.update()
        return (
            f"peak_alloc={_format_bytes(self._peak_alloc)} "
            f"peak_reserved={_format_bytes(self._peak_reserved)}"
        )

    def snapshot(self) -> tuple[int, int]:
        if not self.enabled:
            return 0, 0
        self.update()
        return int(self._peak_alloc), int(self._peak_reserved)

    def current(self) -> tuple[int, int]:
        if not self.enabled:
            return 0, 0
        torch.cuda.synchronize(self.device)
        return (
            int(torch.cuda.memory_allocated(self.device)),
            int(torch.cuda.memory_reserved(self.device)),
        )

    @staticmethod
    def format_snapshot(peak_alloc: int, peak_reserved: int) -> str:
        return (
            f"peak_alloc={_format_bytes(int(peak_alloc))} "
            f"peak_reserved={_format_bytes(int(peak_reserved))}"
        )

    @staticmethod
    def format_current(cur_alloc: int, cur_reserved: int) -> str:
        return (
            f"cur_alloc={_format_bytes(int(cur_alloc))} "
            f"cur_reserved={_format_bytes(int(cur_reserved))}"
        )

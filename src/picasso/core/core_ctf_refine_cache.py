"""Cache of per-micrograph CTF-refinement inputs between loss evaluations.

Building a micrograph's inputs (reading the micrograph and correlation maps,
filtering, extracting particles) costs several times more than evaluating the
loss, and none of it depends on the refined parameters. The cache keeps each
micrograph's compacted inputs:

1. on its GPU while that GPU is under ``device_budget_bytes``,
2. otherwise in pinned host memory while under ``host_budget_bytes``
   (copied to the GPU for each use),
3. otherwise not at all (rebuilt for every use, as before).
"""

import os
import threading
from collections.abc import Callable
from typing import Any

import torch

GIB = 1024**3


def tensor_nbytes(obj: Any) -> int:
    """Total bytes of the tensors in a (nested) dict/list/tuple."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, dict):
        return sum(tensor_nbytes(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(tensor_nbytes(v) for v in obj)
    return 0


def move_tensors(
    obj: Any,
    device: torch.device,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> Any:
    """Copy the tensors of a (nested) dict/list/tuple to ``device``.

    Containers are copied; other values are shared.
    """
    if isinstance(obj, torch.Tensor):
        moved = obj.to(device, non_blocking=non_blocking)
        return moved.pin_memory() if pin_memory else moved
    if isinstance(obj, dict):
        return {
            k: move_tensors(v, device, pin_memory, non_blocking) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [move_tensors(v, device, pin_memory, non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_tensors(v, device, pin_memory, non_blocking) for v in obj)
    return obj


def available_host_bytes() -> int:
    """Currently available host memory (Linux); 0 if it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def resolve_budgets(
    device_list: list[torch.device],
    device_gb: float | None,
    host_gb: float | None,
    auto_fraction: float = 0.4,
) -> tuple[dict[torch.device, int], int]:
    """Cache budgets in bytes per device and for pinned host memory.

    ``None`` means automatic: ``auto_fraction`` of each GPU's total memory and
    of the currently available host memory. ``0`` disables a tier. For CPU
    devices the host budget is used as the device budget and there is no
    separate host tier.
    """
    host_budget = (
        int(auto_fraction * available_host_bytes())
        if host_gb is None
        else int(host_gb * GIB)
    )
    device_budgets: dict[torch.device, int] = {}
    for device in device_list:
        if device.type == "cuda":
            if device_gb is None:
                total = torch.cuda.get_device_properties(device).total_memory
                device_budgets[device] = int(auto_fraction * total)
            else:
                device_budgets[device] = int(device_gb * GIB)
        else:
            device_budgets[device] = host_budget
    return device_budgets, host_budget


class MicrographInputCache:
    """Keeps each micrograph's compacted inputs between loss evaluations.

    ``get`` returns a dict shared with the cache (device tier) or a fresh device
    copy (host tier); callers must not modify it in place.
    """

    def __init__(
        self,
        device_budget_bytes: dict[torch.device, int],
        host_budget_bytes: int,
    ) -> None:
        self._device_budget = dict(device_budget_bytes)
        self._host_budget = host_budget_bytes
        self._device_used = dict.fromkeys(device_budget_bytes, 0)
        self._host_used = 0
        self._entries: dict[int, tuple[str, dict[str, Any]]] = {}
        self._uncached: set[int] = set()
        self._lock = threading.Lock()
        self.builds = 0

    def get(
        self,
        index: int,
        device: torch.device,
        build: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Inputs of micrograph ``index`` on ``device``, building them if needed."""
        with self._lock:
            entry = self._entries.get(index)
        if entry is not None:
            tier, inputs = entry
            if tier == "device":
                return inputs
            return move_tensors(inputs, device, non_blocking=True)  # type: ignore[no-any-return]

        inputs = build()
        size = tensor_nbytes(inputs)
        with self._lock:
            self.builds += 1
            if index in self._uncached:
                return inputs
            if self._device_used.get(device, 0) + size <= self._device_budget.get(
                device, 0
            ):
                self._device_used[device] = self._device_used.get(device, 0) + size
                self._entries[index] = ("device", inputs)
                return inputs
            use_host = (
                device.type == "cuda" and self._host_used + size <= self._host_budget
            )
            if use_host:
                self._host_used += size
            else:
                self._uncached.add(index)
        if use_host:
            host_inputs = move_tensors(inputs, torch.device("cpu"), pin_memory=True)
            with self._lock:
                self._entries[index] = ("host", host_inputs)
        return inputs

    def summary(self) -> str:
        """One-line description of what is cached where."""
        with self._lock:
            on_device = sum(1 for tier, _ in self._entries.values() if tier == "device")
            on_host = len(self._entries) - on_device
            device_gb = sum(self._device_used.values()) / GIB
            host_gb = self._host_used / GIB
            rebuilt = len(self._uncached)
        return (
            f"Input cache: {on_device} micrograph(s) on GPU ({device_gb:.1f} GB), "
            f"{on_host} in host memory ({host_gb:.1f} GB), {rebuilt} rebuilt on "
            f"every use; {self.builds} builds so far."
        )

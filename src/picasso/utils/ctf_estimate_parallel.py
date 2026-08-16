"""Choose and run intra-device concurrency for CTF estimation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

K_MAX = 8
RAM_PER_FIT_BYTES = 1 * 1024**3
FALLBACK_IMAGE_SHAPE = (1, 4096, 4096)
DEFAULT_HOST_RAM_BYTES = 4 * 1024**3
GPU_SAFETY = 1.5


def clamp_k(k: int, n_images: int, k_max: int = K_MAX) -> int:
    """Clamp concurrency to ``[1, min(k_max, n_images)]``."""
    n_images = max(1, n_images)
    return max(1, min(k_max, n_images, max(1, k)))


def analytic_peak_bytes(
    image_shape: tuple[int, int, int],
    pixel_spacing_angstroms: float,
    target_pixel_spacing_angstroms: float,
    patch_sidelength: int,
) -> int:
    """Conservative per-fit VRAM estimate (raw + patches + rFFT + autograd)."""
    t, height, width = image_shape
    t = max(1, t)
    height = max(1, height)
    width = max(1, width)
    new_spacing = max(target_pixel_spacing_angstroms, pixel_spacing_angstroms)
    scale = pixel_spacing_angstroms / new_spacing
    prep_h = max(1, round(height * scale))
    prep_w = max(1, round(width * scale))
    if patch_sidelength < 0:
        patch = min(prep_h, prep_w)
    else:
        patch = max(1, patch_sidelength)
    step = max(1, patch // 2)
    n_gh = 1 + max(0, (prep_h - patch) // step)
    n_gw = 1 + max(0, (prep_w - patch) // step)
    n_patches = t * n_gh * n_gw
    rfft_w = patch // 2 + 1
    bytes_raw = t * height * width * 4
    bytes_prepared = t * prep_h * prep_w * 4
    bytes_patches = n_patches * patch * patch * 4
    bytes_rfft = n_patches * patch * rfft_w * 8
    bytes_ps = n_patches * patch * rfft_w * 4
    return int(bytes_raw + bytes_prepared + bytes_patches + bytes_rfft + 3 * bytes_ps)


def read_mrc_shape(path: str | Path) -> tuple[int, int, int]:
    """Return ``(t, h, w)`` from an MRC header, or a 4k still fallback."""
    try:
        import mrcfile

        with mrcfile.open(path, header_only=True, permissive=True) as mrc:
            nz = int(mrc.header.nz)
            ny = int(mrc.header.ny)
            nx = int(mrc.header.nx)
        if nz <= 1:
            return (1, max(1, ny), max(1, nx))
        return (nz, max(1, ny), max(1, nx))
    except Exception:
        return FALLBACK_IMAGE_SHAPE


def available_ram_bytes() -> int:
    """Host RAM available for CPU workers, with a 4 GB fallback."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return DEFAULT_HOST_RAM_BYTES


def resolve_k_gpu(
    *,
    forced: int | None,
    free_bytes: int,
    peak_bytes: int,
    n_images: int,
) -> int:
    """K for one CUDA device from free VRAM, or a forced value."""
    if forced is not None:
        return clamp_k(forced, n_images)
    if peak_bytes <= 0:
        return 1
    budget = int(peak_bytes * GPU_SAFETY)
    return clamp_k(free_bytes // max(budget, 1), n_images)


def resolve_k_cpu(
    *,
    forced: int | None,
    n_images: int,
    cpu_threads_per_worker: int = 1,
    n_cores: int | None = None,
    ram_bytes: int | None = None,
    ram_per_fit_bytes: int = RAM_PER_FIT_BYTES,
) -> int:
    """K for CPU from cores/RAM, or a forced value."""
    if forced is not None:
        return clamp_k(forced, n_images)
    cores = n_cores if n_cores is not None else (os.cpu_count() or 1)
    threads = max(1, cpu_threads_per_worker)
    from_cores = max(1, cores // threads)
    ram = ram_bytes if ram_bytes is not None else available_ram_bytes()
    from_ram = max(1, ram // max(ram_per_fit_bytes, 1))
    return clamp_k(min(from_cores, from_ram), n_images)


def cuda_free_bytes(device: torch.device) -> int:
    """Free bytes on a CUDA device; 0 if the query fails."""
    try:
        index = device.index if device.index is not None else 0
        free, _total = torch.cuda.mem_get_info(index)
        return int(free)
    except Exception:
        return 0


def cpu_estimate_worker(payload: dict[str, Any]) -> tuple[int, str, Any]:
    """Spawn-safe CPU worker: cap threads, reconstruct manager, estimate one image."""
    threads = int(payload["cpu_threads_per_worker"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)

    from picasso.managers.ctf_estimate_manager import CTFEstimateManager

    manager = CTFEstimateManager.model_validate(payload["manager"])
    path = payload["path"]
    result = manager.estimate_one(path, device=torch.device("cpu"))
    return payload["idx"], path, result

"""Map CTF estimation results to Leopard-EM optics and write CSV/YAML."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml
import torch
from torch_ctf_estimation.models import (
    Defocus1DResults,
    Defocus2DResults,
    LinearDefocusModel,
    OpticalParams,
    Thickness1DResults,
    Thickness2DResults,
)
from torch_ctf_estimation.utils.fitting_bounds import resolve_defocus_bounds

MICRONS_TO_ANGSTROMS = 1.0e4
BOUND_TOL = 1e-3

OPTICS_CSV_FIELDS = [
    "micrograph_path",
    "label",
    "voltage",
    "pixel_size",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
]

METRICS_CSV_FIELDS = [
    "micrograph_path",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "phase_shift",
    "ctf_B_factor",
    "cc_1d",
    "cc_2d",
    "defocus_model",
    "phase_shift_model",
    "tilt_axis_angle_deg",
    "tilt_magnitude_deg",
    "hit_defocus_bound",
    "hit_phase_bound",
    "thickness_angstroms",
    "cc_thickness_1d",
    "cc_thickness_joint",
    "defocus_spatial_min_A",
    "defocus_spatial_max_A",
    "defocus_spatial_range_A",
    "defocus_spatial_std_A",
    "defocus_1d_A",
    "defocus_2d_mean_A",
    "astig_2d_A",
    "astig_2d_angle_deg",
    "defocus_tilt_epa_A",
    "defocus_joint_A",
    "cc_tilt_epa",
]


def result_to_leopard_optics(
    result2d: Defocus2DResults,
    optical_params: OpticalParams,
    micrograph_path: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Map a 2D result to Leopard-EM OpticsGroup field names (Å, degrees)."""
    defocus_u_um = float(result2d.defocus_u) if result2d.defocus_u is not None else 0.0
    defocus_v_um = float(result2d.defocus_v) if result2d.defocus_v is not None else 0.0
    return {
        "label": label if label is not None else Path(micrograph_path).stem,
        "voltage": optical_params.voltage_kev,
        "pixel_size": optical_params.pixel_spacing_angstroms,
        "defocus_u": defocus_u_um * MICRONS_TO_ANGSTROMS,
        "defocus_v": defocus_v_um * MICRONS_TO_ANGSTROMS,
        "astigmatism_angle": float(result2d.astigmatism_angle or 0.0),
        "spherical_aberration": optical_params.spherical_aberration_mm,
        "amplitude_contrast_ratio": optical_params.amplitude_contrast_fraction,
        "phase_shift": float(result2d.phase_shift_degrees or 0.0),
        "ctf_B_factor": float(result2d.envelope_B or 0.0),
    }


def _thickness_angstroms(
    thickness_joint: Thickness1DResults | Thickness2DResults | None,
    thickness1d: Thickness1DResults | None,
) -> float | None:
    """Prefer joint-refine thickness, else the 1D grid value."""
    if isinstance(thickness_joint, Thickness2DResults):
        return thickness_joint.mean_thickness
    if isinstance(thickness_joint, Thickness1DResults):
        return thickness_joint.thickness_angstroms
    if thickness1d is not None:
        return thickness1d.thickness_angstroms
    return None


def _hit_bound(
    value: float, bounds: tuple[float, float], tol: float = BOUND_TOL
) -> bool:
    """True if value sits on either bound within ``tol``."""
    return abs(value - bounds[0]) <= tol or abs(value - bounds[1]) <= tol


def spatial_defocus_field(result2d: Defocus2DResults) -> dict[str, Any]:
    """Summarise spatial defocus variation (grid knots or linear gradient).

    Values are in Angstroms except ``defocus_grid_um`` (µm knot tensor).
    """
    empty = {
        "defocus_spatial_min_A": None,
        "defocus_spatial_max_A": None,
        "defocus_spatial_range_A": None,
        "defocus_spatial_std_A": None,
        "defocus_grid_um": None,
    }
    if result2d.defocus_model_type == "grid":
        data = result2d.defocus_model.data.detach().cpu().float()
        flat = data.reshape(-1)
        min_um = float(flat.min().item())
        max_um = float(flat.max().item())
        std_um = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
        return {
            "defocus_spatial_min_A": min_um * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_max_A": max_um * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_range_A": (max_um - min_um) * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_std_A": std_um * MICRONS_TO_ANGSTROMS,
            "defocus_grid_um": data.tolist(),
        }
    if result2d.defocus_model_type == "linear" and isinstance(
        result2d.defocus_model, LinearDefocusModel
    ):
        d0 = float(result2d.defocus_model.defocus_0)
        gm = float(result2d.defocus_model.defocus_gradient_magnitude)
        min_um = d0 - 0.5 * abs(gm)
        max_um = d0 + 0.5 * abs(gm)
        return {
            "defocus_spatial_min_A": min_um * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_max_A": max_um * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_range_A": abs(gm) * MICRONS_TO_ANGSTROMS,
            "defocus_spatial_std_A": (abs(gm) / 12.0**0.5) * MICRONS_TO_ANGSTROMS,
            "defocus_grid_um": None,
        }
    return empty


def result_to_metrics_row(
    micrograph_path: str,
    result1d: Defocus1DResults,
    result2d: Defocus2DResults,
    optical_params: OpticalParams,
    defocus_range_microns: tuple[float, float] | None,
    phase_shift_range_degrees: tuple[float, float] | None,
    thickness1d: Thickness1DResults | None,
    thickness_joint: Thickness1DResults | Thickness2DResults | None,
    stage_defocus_A: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Build one metrics-CSV row (report only; no exclude flags)."""
    optics = result_to_leopard_optics(result2d, optical_params, micrograph_path)
    defocus_bounds = resolve_defocus_bounds(defocus_range_microns)
    mean_defocus_um = (
        (float(result2d.defocus_u) + float(result2d.defocus_v)) / 2.0
        if result2d.defocus_u is not None and result2d.defocus_v is not None
        else 0.0
    )
    phase = float(result2d.phase_shift_degrees or 0.0)
    phase_bounds = phase_shift_range_degrees
    hit_phase = False
    if phase_bounds is not None:
        hit_phase = _hit_bound(phase, phase_bounds)
    cc_joint = None
    if thickness_joint is not None:
        cc_joint = thickness_joint.cross_correlation_final
    spatial = spatial_defocus_field(result2d)
    return {
        "micrograph_path": micrograph_path,
        "defocus_u": optics["defocus_u"],
        "defocus_v": optics["defocus_v"],
        "astigmatism_angle": optics["astigmatism_angle"],
        "phase_shift": optics["phase_shift"],
        "ctf_B_factor": optics["ctf_B_factor"],
        "cc_1d": result1d.cross_correlation_final,
        "cc_2d": result2d.cross_correlation_final,
        "defocus_model": result2d.defocus_model_type,
        "phase_shift_model": result2d.phase_shift_model_type,
        "tilt_axis_angle_deg": result2d.tilt_axis_angle_deg,
        "tilt_magnitude_deg": result2d.tilt_magnitude_deg,
        "hit_defocus_bound": _hit_bound(mean_defocus_um, defocus_bounds),
        "hit_phase_bound": hit_phase,
        "thickness_angstroms": _thickness_angstroms(thickness_joint, thickness1d),
        "cc_thickness_1d": (
            thickness1d.cross_correlation_final if thickness1d is not None else None
        ),
        "cc_thickness_joint": cc_joint,
        "defocus_spatial_min_A": spatial["defocus_spatial_min_A"],
        "defocus_spatial_max_A": spatial["defocus_spatial_max_A"],
        "defocus_spatial_range_A": spatial["defocus_spatial_range_A"],
        "defocus_spatial_std_A": spatial["defocus_spatial_std_A"],
        **(stage_defocus_A or {}),
    }


def write_optics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the dataset-level Leopard optics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTICS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in OPTICS_CSV_FIELDS})


def write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the per-micrograph metrics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in METRICS_CSV_FIELDS})


def _tensor_to_np(value: Any) -> Any:
    """Detach a tensor to a numpy array, or return None."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def write_estimate_debug_npz(
    path: Path,
    *,
    mean_ps: torch.Tensor,
    result1d: Defocus1DResults,
    pixel_spacing_used: float,
    thickness1d: Thickness1DResults | None = None,
    thickness_joint: Thickness1DResults | Thickness2DResults | None = None,
) -> None:
    """Save 1D spectra, thickness-grid CCs, and mean 2D power for later debug."""
    import numpy as np

    payload: dict[str, Any] = {
        "pixel_spacing_used_A": np.array(pixel_spacing_used),
        "mean_ps": _tensor_to_np(mean_ps),
        "freq_1d": _tensor_to_np(result1d.frequencies_1d),
        "power_1d": _tensor_to_np(result1d.powerspectrum_1d),
        "cc_1d": np.array(
            result1d.cross_correlation_final
            if result1d.cross_correlation_final is not None
            else np.nan
        ),
        "test_defoci_um": _tensor_to_np(result1d.test_defoci),
        "cc_defocus_grid": _tensor_to_np(result1d.cross_correlations),
        "low_frequency_fit": np.array(result1d.low_frequency_fit or np.nan),
        "high_frequency_fit": np.array(result1d.high_frequency_fit or np.nan),
    }
    if thickness1d is not None:
        payload["thickness_A"] = np.array(thickness1d.thickness_angstroms)
        payload["cc_thickness_1d"] = np.array(
            thickness1d.cross_correlation_final
            if thickness1d.cross_correlation_final is not None
            else np.nan
        )
        payload["freq_thickness_1d"] = _tensor_to_np(thickness1d.frequencies_1d)
        payload["power_thickness_1d"] = _tensor_to_np(thickness1d.powerspectrum_1d)
        payload["test_thicknesses_A"] = _tensor_to_np(thickness1d.test_thicknesses)
        payload["cc_thickness_grid"] = _tensor_to_np(thickness1d.cross_correlations)
    if isinstance(thickness_joint, Thickness1DResults):
        payload["thickness_joint_A"] = np.array(thickness_joint.thickness_angstroms)
        payload["cc_thickness_joint"] = np.array(
            thickness_joint.cross_correlation_final
            if thickness_joint.cross_correlation_final is not None
            else np.nan
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in payload.items() if v is not None})


def write_optics_yaml(
    optics: dict[str, Any],
    path: Path,
    thickness_angstroms: float | None = None,
    defocus_field: dict[str, Any] | None = None,
    stage_defocus_A: dict[str, float | None] | None = None,
) -> None:
    """Write a per-micrograph YAML with ``optics_group`` and optional extras."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"optics_group": optics}
    if thickness_angstroms is not None:
        payload["thickness_angstroms"] = thickness_angstroms
    if defocus_field is not None:
        payload["defocus_field"] = defocus_field
    if stage_defocus_A is not None:
        payload["stage_defocus_A"] = stage_defocus_A
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

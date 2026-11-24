"""Core CTF estimation chainer. Calls torch-ctf-estimation primitives only."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch_ctf_estimation.estimate_ctf_1d import (
    estimate_ctf_1d,
    estimate_thickness_1d,
    fit_background_spline_1d,
    refine_defocus_and_thickness_1d,
)
from torch_ctf_estimation.estimate_ctf_2d import (
    estimate_ctf_2d,
    estimate_defocus_2d_at_1x1,
    refine_defocus_and_thickness_2d,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_background_2d import (
    estimate_background_2d,
)
from torch_ctf_estimation.models import (
    CTFFittingParams,
    Defocus1DResults,
    Defocus2DResults,
    LaserParams,
    OpticalParams,
    Thickness1DResults,
    Thickness2DResults,
    ThicknessParams,
    linear_tilt_axis_and_magnitude_deg,
)
from torch_ctf_estimation.utils.tilt_corrected_ps import tilt_corrected_mean_ps_2d
from torch_ctf_estimation.utils.fitting_bounds import (
    resolve_defocus_bounds,
    resolve_phase_shift_fitting,
)
from torch_ctf_estimation.utils.laser_axis_mask import apply_laser_axis_mask
from torch_ctf_estimation.utils.patches import (
    compute_patch_power_spectra,
    extract_ctf_patches,
    normalised_patch_positions,
)
from torch_ctf_estimation.utils.prepare_image import prepare_image_for_ctf


class CtfEstimateCoreResult(NamedTuple):
    """Outputs of one-image CTF estimation."""

    mean_ps: torch.Tensor
    result1d: Defocus1DResults
    result2d: Defocus2DResults
    thickness1d: Thickness1DResults | None
    thickness_joint: Thickness1DResults | Thickness2DResults | None
    pixel_spacing_used: float
    stage_defocus_A: dict[str, float | None] | None = None


def _scalar(value: Any) -> float:
    """Convert a tensor or number to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def core_estimate_ctf(
    image: torch.Tensor,
    optical_params: OpticalParams,
    fitting_params: CTFFittingParams,
    device: torch.device,
    laser_params: LaserParams | None = None,
    thickness_params: ThicknessParams | None = None,
) -> CtfEstimateCoreResult:
    """Run the CTF estimation chain on one image.

    Always: prepare → patches → power spectra → 1D defocus → 2D defocus.
    If ``thickness_params`` is set: 1D thickness grid on the (optional)
    tilt-corrected equiphase spectrum using the 2D mean defocus, then optional
    refine (``"thickness"`` = thickness only, ``"1d"`` / ``"2d"`` = joint).
    """
    image, new_spacing = prepare_image_for_ctf(
        image,
        pixel_spacing_angstroms=optical_params.pixel_spacing_angstroms,
        target_pixel_spacing_angstroms=optical_params.target_pixel_spacing_angstroms,
        device=device,
    )
    t, h, w = image.shape
    patches, patch_centers, image_sidelength_for_1d, use_whole_image = (
        extract_ctf_patches(
            image,
            patch_sidelength=fitting_params.patch_sidelength,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
        )
    )
    patch_ps, mean_ps = compute_patch_power_spectra(patches)

    axis_mask: torch.Tensor | None = None
    if fitting_params.mask_laser_axis and laser_params is not None:
        patch_ps, mean_ps, axis_mask = apply_laser_axis_mask(
            patch_ps,
            mean_ps,
            laser_xy_angle_deg=laser_params.laser_xy_angle_deg,
            dual_laser=laser_params.dual_laser,
            mask_width=fitting_params.laser_axis_mask_width,
        )

    _nt, nh, nw = fitting_params.defocus_grid_resolution
    use_1d_spatial = fitting_params.use_1d_defocus_for_spatial and (
        fitting_params.defocus_model == "linear" or (nh > 1 or nw > 1)
    )
    bg_mean: Any | None = None
    if use_1d_spatial:
        bg_mean = fit_background_spline_1d(
            power_spectrum=mean_ps,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            pixel_spacing_angstroms=new_spacing,
        )

    defocus_bounds = resolve_defocus_bounds(fitting_params.defocus_range_microns)
    optimize_phase_1d, phase_shift_deg, phase_bounds = resolve_phase_shift_fitting(
        optimize_phase_shift=fitting_params.optimize_phase_shift,
        phase_shift_range_degrees=fitting_params.phase_shift_range_degrees,
        initial_phase_shift=fitting_params.initial_phase_shift,
    )
    result1d = estimate_ctf_1d(
        power_spectrum=mean_ps,
        image_sidelength=image_sidelength_for_1d,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        defocus_range_microns=defocus_bounds,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast=optical_params.amplitude_contrast_fraction,
        pixel_spacing_angstroms=new_spacing,
        optimize_envelope=fitting_params.optimize_envelope_1d,
        b_range=fitting_params.b_range_1d,
        b_step=fitting_params.b_step_1d,
        refine_steps=fitting_params.refine_steps_1d,
        background_result=bg_mean,
        optimize_phase_shift=optimize_phase_1d,
        initial_phase_shift=phase_shift_deg,
        phase_shift_range=phase_bounds,
        early_stopper=fitting_params.build_early_stopper(),
    )

    image_shape_2d = (
        (h, w)
        if use_whole_image
        else (image_sidelength_for_1d, image_sidelength_for_1d)
    )
    patch_ps_raw = patch_ps
    background_2d_power = estimate_background_2d(
        power_spectrum=mean_ps,
        image_sidelength=image_shape_2d,
    )
    patch_ps_power = patch_ps_raw - background_2d_power
    if fitting_params.use_amplitude_2d:
        patch_amp = torch.sqrt(patch_ps_raw.clamp(min=0.0))
        mean_amp = torch.sqrt(mean_ps.clamp(min=0.0))
        background_2d_amp = estimate_background_2d(
            power_spectrum=mean_amp,
            image_sidelength=image_shape_2d,
        )
        patch_ps_2d = patch_amp - background_2d_amp
    else:
        patch_ps_2d = patch_ps_power
    positions = normalised_patch_positions(patch_centers, (t, h, w))

    initial_envelope_B_2d = fitting_params.initial_envelope_B
    if initial_envelope_B_2d is None and result1d.ctf_model.envelope_B is not None:
        initial_envelope_B_2d = _scalar(result1d.ctf_model.envelope_B)
    if initial_envelope_B_2d is None:
        initial_envelope_B_2d = 0.0
    initial_defocus_2d = _scalar(result1d.ctf_model.defocus_um)
    initial_phase_shift_2d = phase_shift_deg
    if optimize_phase_1d and result1d.ctf_model.phase_shift_degrees is not None:
        initial_phase_shift_2d = _scalar(result1d.ctf_model.phase_shift_degrees)

    result2d = _run_defocus_2d(
        patch_ps=patch_ps_2d,
        positions=positions,
        fitting_params=fitting_params,
        optical_params=optical_params,
        laser_params=laser_params,
        new_spacing=new_spacing,
        image_hw=(h, w),
        image_sidelength_for_1d=image_sidelength_for_1d,
        use_1d_spatial=use_1d_spatial,
        bg_mean=bg_mean,
        defocus_bounds=defocus_bounds,
        optimize_phase_1d=optimize_phase_1d,
        phase_shift_deg=phase_shift_deg,
        phase_bounds=phase_bounds,
        initial_defocus_2d=initial_defocus_2d,
        initial_envelope_B_2d=initial_envelope_B_2d,
        initial_phase_shift_2d=initial_phase_shift_2d,
        axis_mask=axis_mask,
    )

    thickness1d: Thickness1DResults | None = None
    thickness_joint: Thickness1DResults | Thickness2DResults | None = None
    df_1d_um = _scalar(result1d.ctf_model.defocus_um)
    df_2d_mean_um = (
        float(result2d.defocus_u + result2d.defocus_v) / 2.0
        if result2d.defocus_u is not None and result2d.defocus_v is not None
        else df_1d_um
    )
    astig_2d_um = float(result2d.astigmatism or 0.0)
    astig_2d_ang = float(result2d.astigmatism_angle or 0.0)
    df_joint_um: float | None = None
    if thickness_params is not None:
        phase_for_thickness = (
            float(result2d.phase_shift_degrees)
            if result2d.phase_shift_degrees is not None
            else phase_shift_deg
        )
        thickness_ps = mean_ps
        thickness_sidelength = image_sidelength_for_1d
        thickness_bg = bg_mean
        if thickness_params.use_tilt_corrected_ps:
            result2d_tc = result2d.model_copy(
                update={"patch_power_spectra": patch_ps_raw}
            )
            optics_fit = optical_params.model_copy(
                update={"pixel_spacing_angstroms": new_spacing}
            )
            if result2d.defocus_model_type == "grid":
                d_avg = float(result2d.defocus_model.data.detach().cpu().mean().item())
            else:
                d_avg = df_2d_mean_um
            thickness_ps, _aux = tilt_corrected_mean_ps_2d(
                result1d,
                result2d_tc,
                normalised_patch_positions=positions,
                optical_params=optics_fit,
                laser_params=laser_params,
                defocus_average_um=d_avg,
            )
            thickness_sidelength = int(thickness_ps.shape[0])
            thickness_bg = None
        use_equiphase = thickness_params.use_equiphase
        if use_equiphase:
            # Rotational 1D background does not match an equiphase profile.
            thickness_bg = None
        thick_freq = (
            thickness_params.frequency_fit_range_angstroms
            if thickness_params.frequency_fit_range_angstroms is not None
            else fitting_params.frequency_fit_range_angstroms
        )
        if thick_freq != fitting_params.frequency_fit_range_angstroms:
            thickness_bg = None
        thick_defocus_bounds = (
            resolve_defocus_bounds(thickness_params.defocus_range_microns)
            if thickness_params.defocus_range_microns is not None
            else defocus_bounds
        )
        thickness1d = estimate_thickness_1d(
            power_spectrum=thickness_ps,
            image_sidelength=thickness_sidelength,
            frequency_fit_range_angstroms=thick_freq,
            defocus_um=df_2d_mean_um,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast=optical_params.amplitude_contrast_fraction,
            pixel_spacing_angstroms=new_spacing,
            phase_shift_deg=phase_for_thickness,
            thickness_range_angstroms=thickness_params.thickness_range_angstroms,
            thickness_step_angstroms=thickness_params.thickness_step_angstroms,
            background_result=thickness_bg,
            use_equiphase=use_equiphase,
            equiphase_defocus_um=df_2d_mean_um,
            equiphase_astigmatism_um=astig_2d_um,
            equiphase_astigmatism_angle_deg=astig_2d_ang,
            equiphase_phase_shift_deg=phase_for_thickness,
            laser_params=laser_params,
        )
        if thickness_params.refine_dim in ("1d", "thickness"):
            result1d_joint, thickness_joint = refine_defocus_and_thickness_1d(
                power_spectrum=thickness_ps,
                image_sidelength=thickness_sidelength,
                frequency_fit_range_angstroms=thick_freq,
                initial_defocus_um=df_2d_mean_um,
                initial_thickness_angstroms=thickness1d.thickness_angstroms,
                voltage_kev=optical_params.voltage_kev,
                spherical_aberration_mm=optical_params.spherical_aberration_mm,
                amplitude_contrast=optical_params.amplitude_contrast_fraction,
                pixel_spacing_angstroms=new_spacing,
                phase_shift_deg=phase_for_thickness,
                n_iterations=thickness_params.n_iterations,
                defocus_lr=thickness_params.defocus_lr,
                thickness_lr=thickness_params.thickness_lr,
                defocus_range_microns=thick_defocus_bounds,
                thickness_range_angstroms=thickness_params.thickness_range_angstroms,
                background_result=thickness_bg,
                laser_params=laser_params,
                early_stopper=thickness_params.build_early_stopper(),
                use_equiphase=use_equiphase,
                equiphase_defocus_um=df_2d_mean_um,
                equiphase_astigmatism_um=astig_2d_um,
                equiphase_astigmatism_angle_deg=astig_2d_ang,
                equiphase_phase_shift_deg=phase_for_thickness,
                optimize_defocus=thickness_params.refine_dim == "1d",
            )
            if thickness_params.refine_dim == "1d":
                df_joint_um = _scalar(result1d_joint.ctf_model.defocus_um)
                result2d = result2d.model_copy(
                    update={
                        "defocus_u": df_joint_um + astig_2d_um / 2.0,
                        "defocus_v": df_joint_um - astig_2d_um / 2.0,
                        "cross_correlation_final": result1d_joint.cross_correlation_final,
                    }
                )
        elif thickness_params.refine_dim == "2d":
            result2d, thickness_joint = refine_defocus_and_thickness_2d(
                patch_power_spectra=patch_ps_power,
                normalised_patch_positions=positions,
                result2d=result2d,
                initial_thickness_angstroms=thickness1d.thickness_angstroms,
                frequency_fit_range_angstroms=thick_freq,
                pixel_spacing_angstroms=new_spacing,
                voltage_kev=optical_params.voltage_kev,
                spherical_aberration_mm=optical_params.spherical_aberration_mm,
                amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
                n_iterations=thickness_params.n_iterations,
                defocus_lr=thickness_params.defocus_lr,
                thickness_lr=thickness_params.thickness_lr,
                thickness_grid_resolution=thickness_params.thickness_grid_resolution,
                defocus_bounds_microns=thick_defocus_bounds,
                laser_params=laser_params,
                axis_mask=axis_mask,
                early_stopper=thickness_params.build_early_stopper(),
            )
            if result2d.defocus_u is not None and result2d.defocus_v is not None:
                df_joint_um = float(result2d.defocus_u + result2d.defocus_v) / 2.0

    um_to_a = 1.0e4
    stage_defocus_A: dict[str, float | None] = {
        "defocus_1d_A": df_1d_um * um_to_a,
        "defocus_2d_mean_A": df_2d_mean_um * um_to_a,
        "astig_2d_A": astig_2d_um * um_to_a,
        "astig_2d_angle_deg": astig_2d_ang,
        "defocus_tilt_epa_A": None,
        "defocus_joint_A": None if df_joint_um is None else df_joint_um * um_to_a,
        "cc_tilt_epa": None,
    }

    return CtfEstimateCoreResult(
        mean_ps=mean_ps,
        result1d=result1d,
        result2d=result2d,
        thickness1d=thickness1d,
        thickness_joint=thickness_joint,
        pixel_spacing_used=new_spacing,
        stage_defocus_A=stage_defocus_A,
    )


def _run_defocus_2d(
    *,
    patch_ps: torch.Tensor,
    positions: torch.Tensor,
    fitting_params: CTFFittingParams,
    optical_params: OpticalParams,
    laser_params: LaserParams | None,
    new_spacing: float,
    image_hw: tuple[int, int],
    image_sidelength_for_1d: int,
    use_1d_spatial: bool,
    bg_mean: Any,
    defocus_bounds: tuple[float, float],
    optimize_phase_1d: bool,
    phase_shift_deg: float,
    phase_bounds: tuple[float, float],
    initial_defocus_2d: float,
    initial_envelope_B_2d: float,
    initial_phase_shift_2d: float,
    axis_mask: torch.Tensor | None,
) -> Defocus2DResults:
    """2D defocus branch (1D-spatial or full 2D fit)."""
    h, w = image_hw
    if use_1d_spatial:
        result_1x1 = estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=optimize_phase_1d,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=(
                fitting_params.phase_shift_quadratic_perpendicular_axis
            ),
            initial_phase_shift=initial_phase_shift_2d,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
            axis_mask=axis_mask,
            defocus_bounds_microns=defocus_bounds,
            phase_shift_bounds_degrees=phase_bounds,
            early_stopper=fitting_params.build_early_stopper(),
            use_amplitude=fitting_params.use_amplitude_2d,
        )
        result2d = defocus_field_from_1d_fits(
            patch_power_spectra=patch_ps,
            normalised_patch_positions=positions,
            result_1x1=result_1x1,
            defocus_model=fitting_params.defocus_model,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            initial_defocus=initial_defocus_2d,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            defocus_range_microns=defocus_bounds,
            phase_shift_range_degrees=phase_bounds,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            pixel_spacing_angstroms=new_spacing,
            optimize_envelope_1d=fitting_params.optimize_envelope_1d,
            b_range_1d=fitting_params.b_range_1d,
            b_step_1d=fitting_params.b_step_1d,
            refine_steps_1d=fitting_params.refine_steps_1d,
            background_result=bg_mean,
            device=patch_ps.device,
            optimize_phase_shift=optimize_phase_1d,
            use_equiphase_for_1d_spatial=fitting_params.use_equiphase_for_1d_spatial,
            laser_params=laser_params,
            equiphase_n_theta=fitting_params.equiphase_n_theta,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
        )
        if result2d.defocus_model_type == "linear":
            axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
                result2d, new_spacing, min(h, w)
            )
            result2d = result2d.model_copy(
                update={
                    "tilt_axis_angle_deg": axis_deg,
                    "tilt_magnitude_deg": tilt_deg,
                }
            )
        if optimize_phase_1d and result_1x1.phase_shift_degrees is not None:
            result2d = result2d.model_copy(
                update={
                    "phase_shift_degrees": result_1x1.phase_shift_degrees,
                    "phase_shift_model_type": result_1x1.phase_shift_model_type,
                    "phase_shift_model": result_1x1.phase_shift_model,
                    "phase_shift_trace": result_1x1.phase_shift_trace,
                }
            )
        elif not optimize_phase_1d:
            result2d = result2d.model_copy(
                update={"phase_shift_degrees": phase_shift_deg}
            )
        return result2d

    fix_defocus_0_val = None
    initial_astigmatism_2d = 0.0
    initial_astigmatism_angle_2d = 0.0
    initial_phase_shift_for_2d = initial_phase_shift_2d
    if (
        fitting_params.defocus_model == "linear"
        and fitting_params.linear_fix_defocus_0_from_1x1
    ):
        result_1x1 = estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=optimize_phase_1d,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=(
                fitting_params.phase_shift_quadratic_perpendicular_axis
            ),
            initial_phase_shift=initial_phase_shift_2d,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
            axis_mask=axis_mask,
            defocus_bounds_microns=defocus_bounds,
            phase_shift_bounds_degrees=phase_bounds,
            early_stopper=fitting_params.build_early_stopper(),
            use_amplitude=fitting_params.use_amplitude_2d,
        )
        fix_defocus_0_val = float(result_1x1.defocus_model.data.mean().cpu().item())
        if result_1x1.astigmatism is not None:
            initial_astigmatism_2d = result_1x1.astigmatism
        if result_1x1.astigmatism_angle is not None:
            initial_astigmatism_angle_2d = result_1x1.astigmatism_angle
        if optimize_phase_1d and result_1x1.phase_shift_degrees is not None:
            initial_phase_shift_for_2d = result_1x1.phase_shift_degrees

    result2d = estimate_ctf_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        defocus_grid_resolution=fitting_params.defocus_grid_resolution,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        initial_defocus=initial_defocus_2d,
        pixel_spacing_angstroms=new_spacing,
        debug=fitting_params.debug,
        optimize_astigmatism=fitting_params.optimize_astigmatism,
        defocus_model=fitting_params.defocus_model,
        initial_envelope_B=initial_envelope_B_2d,
        initial_astigmatism=initial_astigmatism_2d,
        initial_astigmatism_angle=initial_astigmatism_angle_2d,
        fix_defocus_0=fix_defocus_0_val,
        n_iterations=fitting_params.n_iterations_2d,
        optimize_phase_shift=optimize_phase_1d,
        phase_shift_model=fitting_params.phase_shift_model,
        phase_shift_quadratic_perpendicular_axis=(
            fitting_params.phase_shift_quadratic_perpendicular_axis
        ),
        initial_phase_shift=initial_phase_shift_for_2d,
        fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
        laser_params=laser_params,
        axis_mask=axis_mask,
        defocus_bounds_microns=defocus_bounds,
        phase_shift_bounds_degrees=phase_bounds,
        early_stopper=fitting_params.build_early_stopper(),
        use_amplitude=fitting_params.use_amplitude_2d,
    )
    if result2d.defocus_model_type == "linear":
        axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
            result2d, new_spacing, min(h, w)
        )
        result2d = result2d.model_copy(
            update={
                "tilt_axis_angle_deg": axis_deg,
                "tilt_magnitude_deg": tilt_deg,
            }
        )
    if not optimize_phase_1d:
        result2d = result2d.model_copy(update={"phase_shift_degrees": phase_shift_deg})
    return result2d

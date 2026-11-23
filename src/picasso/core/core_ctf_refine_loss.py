"""Batched, differentiable per-particle z-scores for CTF refinement.

This replaces the call into Leopard-EM's ``core_differentiable_refine`` for the
CTF refinement loss. CTF refinement scores every particle at a single, fixed
orientation, defocus offset and pixel size, so the work can be batched over
particles instead of looping over them one at a time. Batching also means each
particle gets its own CTF, which is required for per-particle and
position-dependent (quadratic) phase shifts.

The math follows Leopard-EM's ``do_batched_orientation_cross_correlate``:
extract the Fourier slice, apply the anisotropic magnification, zero the DC
term, flip contrast, apply projective filter x CTF, inverse transform, normalize
the projection, cross-correlate with the particle image and convert to a
z-score with the per-pixel correlation mean and standard deviation from the
template-matching search.
"""

from typing import Any

import roma
import torch
from leopard_em.backend.utils import EULER_ANGLE_FMT, normalize_template_projection
from leopard_em.utils.ctf_utils import (
    calculate_ctf_filter_stack_full_args,
    move_ctf_kwargs_tensors_to_device,
)
from torch_fourier_slice import extract_central_slices_rfft_3d, transform_slice_2d

_CACHE_KEY = "_picasso_loss_cache"


def _make_rfft_hermitian(spectrum: torch.Tensor, full_width: int) -> torch.Tensor:
    """Enforce Hermitian symmetry on the self-conjugate columns of an RFFT.

    In an (h, w // 2 + 1) RFFT the DC column (and the Nyquist column when w is
    even) must satisfy ``X[k] == conj(X[-k])``. Interpolated Fourier slices can
    violate this slightly near Nyquist, and then the result of ``irfftn`` is
    implementation-defined (it changed with the cuFFT batch size in testing).
    Symmetrizing these columns gives the real part of the full inverse
    transform, which is well defined and leaves consistent spectra unchanged.

    Parameters
    ----------
    spectrum : torch.Tensor
        Non-fftshifted RFFT with shape (..., h, w // 2 + 1).
    full_width : int
        Real-space width w.

    Returns
    -------
    torch.Tensor
        Spectrum with Hermitian-symmetric self-conjugate columns.
    """
    height = spectrum.shape[-2]
    mirror = (-torch.arange(height, device=spectrum.device)) % height

    def _symmetrize(column: torch.Tensor) -> torch.Tensor:
        return 0.5 * (column + column[..., mirror].conj())

    dc_column = _symmetrize(spectrum[..., :, 0]).unsqueeze(-1)
    if full_width % 2 == 0 and spectrum.shape[-1] > 1:
        nyquist_column = _symmetrize(spectrum[..., :, -1]).unsqueeze(-1)
        return torch.cat([dc_column, spectrum[..., 1:-1], nyquist_column], dim=-1)
    return torch.cat([dc_column, spectrum[..., 1:]], dim=-1)


def _is_inactive_zernike_dict(zernikes: Any) -> bool:
    """True for a Zernike dict whose coefficients are all zero and not trainable."""
    if not isinstance(zernikes, dict) or not zernikes:
        return False
    for value in zernikes.values():
        if isinstance(value, torch.Tensor):
            if value.requires_grad or bool((value != 0).any()):
                return False
        elif value != 0:
            return False
    return True


def _get_static_cache(
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Per-micrograph tensors that do not depend on the refined parameters.

    Stored in ``particle_backend_kwargs`` so repeated loss evaluations on the
    same micrograph (e.g. inside an L-BFGS closure) reuse them.
    """
    cache = particle_backend_kwargs.get(_CACHE_KEY)
    if cache is not None:
        return cache  # type: ignore[no-any-return]

    with torch.no_grad():
        particle_dft = particle_backend_kwargs["particle_stack_dft"].to(device)
        template_dft = particle_backend_kwargs["template_dft"].to(device)
        img_h = particle_dft.shape[-2]
        img_w = 2 * (particle_dft.shape[-1] - 1)
        tmpl_h = template_dft.shape[-2]
        tmpl_w = 2 * (template_dft.shape[-1] - 1)

        euler_offsets = particle_backend_kwargs["euler_angle_offsets"].to(device)
        if euler_offsets.shape[0] != 1:
            raise NotImplementedError(
                "CTF refinement loss supports exactly one Euler angle offset, got "
                f"{euler_offsets.shape[0]}."
            )
        for name in ("defocus_offsets", "pixel_size_offsets"):
            if particle_backend_kwargs[name].numel() != 1:
                raise NotImplementedError(
                    f"CTF refinement loss supports exactly one {name} value, got "
                    f"{particle_backend_kwargs[name].numel()}."
                )

        rotation = roma.euler_to_rotmat(
            EULER_ANGLE_FMT,
            particle_backend_kwargs["euler_angles"].to(device),
            degrees=True,
        ).to(torch.float32)
        offset_rotation = roma.euler_to_rotmat(
            EULER_ANGLE_FMT, euler_offsets, degrees=True
        ).to(torch.float32)
        rotation = offset_rotation @ rotation  # same composition as Leopard-EM

        dc_mask = torch.ones(
            tmpl_h, tmpl_w // 2 + 1, device=device, dtype=torch.float32
        )
        dc_mask[0, 0] = 0.0

        cache = {
            "particle_dft": particle_dft,
            "template_dft": template_dft,
            "rotation": rotation,
            "dc_mask": dc_mask,
            "img_shape": torch.tensor([img_h, img_w]),
            "tmpl_shape": torch.tensor([tmpl_h, tmpl_w]),
        }
        if img_h == tmpl_h and img_w == tmpl_w:
            # Valid cross-correlation is a single pixel (zero shift), which is
            # the real-space dot product of image and projection.
            cache["particle_real"] = torch.fft.irfftn(
                particle_dft, s=(img_h, img_w), dim=(-2, -1)
            )
    particle_backend_kwargs[_CACHE_KEY] = cache
    return cache


# pylint: disable=too-many-locals
def compute_particle_z_scores(
    particle_backend_kwargs: dict[str, Any],
    batch_size: int,
) -> torch.Tensor:
    """Differentiable z-score of each particle at its fixed pose.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Kwargs from ``setup_particle_backend_kwargs`` (Leopard-EM), with
        ``defocus_u``, ``defocus_v``, ``defocus_angle``, ``mag_matrix`` and
        ``ctf_kwargs`` possibly replaced by tensors that require grad. A
        ``ctf_kwargs["phase_shift"]`` with one value per particle is applied per
        particle.
    batch_size : int
        Number of particles processed per chunk.

    Returns
    -------
    torch.Tensor
        Shape (N,) z-scores.
    """
    kwargs = particle_backend_kwargs
    device_arg = kwargs.get("device")
    if isinstance(device_arg, list | tuple) and device_arg:
        device = torch.device(device_arg[0])
    elif isinstance(device_arg, torch.device):
        device = device_arg
    else:
        device = kwargs["particle_stack_dft"].device

    cache = _get_static_cache(kwargs, device)
    img_h, img_w = (int(v) for v in cache["img_shape"])
    tmpl_h, tmpl_w = (int(v) for v in cache["tmpl_shape"])
    crop_h = img_h - tmpl_h + 1
    crop_w = img_w - tmpl_w + 1
    num_particles = cache["particle_dft"].shape[0]

    defocus_u = torch.as_tensor(kwargs["defocus_u"], device=device).reshape(-1)
    defocus_v = torch.as_tensor(kwargs["defocus_v"], device=device).reshape(-1)
    defocus_angle = torch.as_tensor(kwargs["defocus_angle"], device=device).reshape(-1)
    defocus_u = defocus_u.expand(num_particles)
    defocus_v = defocus_v.expand(num_particles)
    defocus_angle = defocus_angle.expand(num_particles)

    projective_filters = kwargs["projective_filters"].to(device)
    corr_mean = kwargs["corr_mean"].to(device)
    corr_std = kwargs["corr_std"].to(device)
    defocus_offsets = kwargs["defocus_offsets"].to(device).reshape(1)
    pixel_size_offsets = kwargs["pixel_size_offsets"].to(device).reshape(1)
    mag_matrix = kwargs.get("mag_matrix")
    if mag_matrix is not None:
        mag_matrix = mag_matrix.to(device=device, dtype=torch.float32)

    ctf_kwargs = move_ctf_kwargs_tensors_to_device(kwargs["ctf_kwargs"], device)
    for key in ("even_zernikes", "odd_zernikes"):
        if _is_inactive_zernike_dict(ctf_kwargs.get(key)):
            # All-zero, non-trainable coefficients add nothing to the CTF but
            # still route gradients through the polar angle of the frequency grid.
            ctf_kwargs[key] = None
    phase_shift = ctf_kwargs.get("phase_shift", 0.0)
    per_particle_phase = (
        isinstance(phase_shift, torch.Tensor)
        and phase_shift.numel() == num_particles
        and num_particles > 1
    )
    if isinstance(phase_shift, torch.Tensor):
        phase_shift = phase_shift.to(device)
        if not per_particle_phase:
            if phase_shift.numel() != 1:
                raise ValueError(
                    f"phase_shift must be a scalar or have one value per particle "
                    f"({num_particles}), got shape {tuple(phase_shift.shape)}."
                )
            phase_shift = phase_shift.reshape(())

    z_scores = []
    for start in range(0, num_particles, batch_size):
        stop = min(start + batch_size, num_particles)
        count = stop - start

        chunk_ctf_kwargs = dict(ctf_kwargs)
        chunk_ctf_kwargs["phase_shift"] = (
            phase_shift.reshape(-1)[start:stop] if per_particle_phase else phase_shift
        )
        ctf = calculate_ctf_filter_stack_full_args(
            defocus_u=defocus_u[start:stop],
            defocus_v=defocus_v[start:stop],
            astigmatism_angle=defocus_angle[start:stop],
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            **chunk_ctf_kwargs,
        )[0]  # (count, h, w // 2 + 1)

        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=cache["template_dft"],
            rotation_matrices=cache["rotation"][start:stop],
        )
        if mag_matrix is not None:
            fourier_slice = transform_slice_2d(
                projection_image_dfts=fourier_slice,
                rfft_shape=(tmpl_h, tmpl_w // 2 + 1),
                stack_shape=(count,),
                transform_matrix=mag_matrix,
            )
        fourier_slice = (
            -torch.fft.ifftshift(fourier_slice, dim=(-2,)) * cache["dc_mask"]
        )
        spectrum = fourier_slice * (projective_filters[start:stop] * ctf)
        spectrum = _make_rfft_hermitian(spectrum, tmpl_w)

        projection = torch.fft.irfftn(spectrum, s=(tmpl_h, tmpl_w), dim=(-2, -1))
        projection = torch.fft.ifftshift(projection, dim=(-2, -1))
        projection = normalize_template_projection(
            projection, (tmpl_h, tmpl_w), (img_h, img_w)
        )

        if "particle_real" in cache:
            # The projection's DC is zeroed before the correlation in Leopard-EM.
            projection = projection - projection.mean(dim=(-2, -1), keepdim=True)
            cross_corr = (cache["particle_real"][start:stop] * projection).sum(
                dim=(-2, -1)
            )
            z = (cross_corr - corr_mean[start:stop].reshape(count)) / corr_std[
                start:stop
            ].reshape(count)
        else:
            projection_dft = torch.fft.rfftn(projection, s=(img_h, img_w), dim=(-2, -1))
            image_dc_mask = torch.ones_like(projection_dft.real)
            image_dc_mask[..., 0, 0] = 0.0
            projection_dft = projection_dft * image_dc_mask
            cross_corr = torch.fft.irfftn(
                cache["particle_dft"][start:stop] * projection_dft.conj(),
                s=(img_h, img_w),
                dim=(-2, -1),
            )[..., :crop_h, :crop_w]
            z = (cross_corr - corr_mean[start:stop]) / corr_std[start:stop]
            z = z.reshape(count, -1).amax(dim=-1)
        z_scores.append(z)

    return torch.cat(z_scores)

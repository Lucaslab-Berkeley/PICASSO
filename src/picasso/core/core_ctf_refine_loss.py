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


# Raw per-particle tensors of the Leopard-EM backend kwargs that the static
# inputs replace; dropped by ``compact_backend_kwargs``.
_RAW_KEYS = (
    "particle_stack_dft",
    "template_dft",
    "projective_filters",
    "corr_mean",
    "corr_std",
    "euler_angles",
)


def prepare_static_inputs(
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
    template_dft: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Tensors of one micrograph that do not depend on the refined parameters.

    Built once and stored in ``particle_backend_kwargs`` so repeated loss
    evaluations on the same micrograph reuse them.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Kwargs from Leopard-EM's ``setup_particle_backend_kwargs``.
    device : torch.device
        Device for the returned tensors.
    template_dft : torch.Tensor | None
        Shared template RFFT to use instead of the one in the kwargs (avoids a
        copy per micrograph).

    Returns
    -------
    dict[str, Any]
        Static inputs of ``compute_particle_z_scores``.
    """
    static = particle_backend_kwargs.get(_CACHE_KEY)
    if static is not None:
        return static  # type: ignore[no-any-return]

    kwargs = particle_backend_kwargs
    with torch.no_grad():
        particle_dft = kwargs["particle_stack_dft"].to(device)
        if template_dft is None:
            template_dft = kwargs["template_dft"]
        template_dft = template_dft.to(device)
        num_particles = particle_dft.shape[0]
        img_h = particle_dft.shape[-2]
        img_w = 2 * (particle_dft.shape[-1] - 1)
        tmpl_h = template_dft.shape[-2]
        tmpl_w = 2 * (template_dft.shape[-1] - 1)

        euler_offsets = kwargs["euler_angle_offsets"].to(device)
        if euler_offsets.shape[0] != 1:
            raise NotImplementedError(
                "CTF refinement loss supports exactly one Euler angle offset, got "
                f"{euler_offsets.shape[0]}."
            )
        for name in ("defocus_offsets", "pixel_size_offsets"):
            if kwargs[name].numel() != 1:
                raise NotImplementedError(
                    f"CTF refinement loss supports exactly one {name} value, got "
                    f"{kwargs[name].numel()}."
                )

        rotation = roma.euler_to_rotmat(
            EULER_ANGLE_FMT, kwargs["euler_angles"].to(device), degrees=True
        ).to(torch.float32)
        offset_rotation = roma.euler_to_rotmat(
            EULER_ANGLE_FMT, euler_offsets, degrees=True
        ).to(torch.float32)
        rotation = offset_rotation @ rotation  # same composition as Leopard-EM

        dc_mask = torch.ones(tmpl_h, tmpl_w // 2 + 1, device=device)
        dc_mask[0, 0] = 0.0

        projective_filters = kwargs["projective_filters"].to(device)
        if projective_filters.shape[0] > 1 and torch.equal(
            projective_filters, projective_filters[:1].expand_as(projective_filters)
        ):
            # With global filtering every particle shares the micrograph's filter.
            projective_filters = projective_filters[:1].clone()

        static = {
            "template_dft": template_dft,
            "rotation": rotation,
            "dc_mask": dc_mask,
            "projective_filters": projective_filters,
            "corr_mean": kwargs["corr_mean"].to(device),
            "corr_std": kwargs["corr_std"].to(device),
            "defocus_offsets": kwargs["defocus_offsets"].to(device).reshape(1),
            "pixel_size_offsets": kwargs["pixel_size_offsets"].to(device).reshape(1),
            "img_shape": (img_h, img_w),
            "tmpl_shape": (tmpl_h, tmpl_w),
            "num_particles": num_particles,
        }
        if img_h == tmpl_h and img_w == tmpl_w:
            # Valid cross-correlation is a single pixel (zero shift), which is
            # the real-space dot product of image and projection.
            static["particle_real"] = torch.fft.irfftn(
                particle_dft, s=(img_h, img_w), dim=(-2, -1)
            )
        else:
            static["particle_dft"] = particle_dft
    kwargs[_CACHE_KEY] = static
    return static


def compact_backend_kwargs(
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
    template_dft: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Prepare the static inputs and drop the raw tensors they replace.

    The returned kwargs (the same dict, modified in place) hold only what the
    loss needs, so they are suitable for caching across loss evaluations.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Kwargs from Leopard-EM's ``setup_particle_backend_kwargs``.
    device : torch.device
        Device for the static tensors.
    template_dft : torch.Tensor | None
        Shared template RFFT (see ``prepare_static_inputs``).

    Returns
    -------
    dict[str, Any]
        The compacted kwargs.
    """
    prepare_static_inputs(particle_backend_kwargs, device, template_dft)
    for key in _RAW_KEYS:
        particle_backend_kwargs.pop(key, None)
    return particle_backend_kwargs


def without_shared_template(particle_backend_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove the template RFFT from compacted kwargs (in place) before caching.

    The template RFFT is the same for every micrograph on a device, so caches
    keep one copy per device and re-attach it with ``per_call_kwargs``.
    """
    static = particle_backend_kwargs.get(_CACHE_KEY)
    if static is not None:
        static.pop("template_dft", None)
    return particle_backend_kwargs


def per_call_kwargs(
    cached: dict[str, Any],
    template_dft: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Shallow copy of cached kwargs that one loss evaluation may modify.

    Copies the top-level dict, ``ctf_kwargs`` and the static inputs (tensors are
    shared), and attaches ``template_dft`` to the static inputs if given.
    """
    kwargs = dict(cached)
    if isinstance(cached.get("ctf_kwargs"), dict):
        kwargs["ctf_kwargs"] = dict(cached["ctf_kwargs"])
    static = cached.get(_CACHE_KEY)
    if static is not None:
        static = dict(static)
        if template_dft is not None:
            static["template_dft"] = template_dft
        kwargs[_CACHE_KEY] = static
    return kwargs


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
    if _CACHE_KEY in kwargs:
        device = kwargs[_CACHE_KEY]["rotation"].device
    else:
        device_arg = kwargs.get("device")
        if isinstance(device_arg, list | tuple) and device_arg:
            device = torch.device(device_arg[0])
        elif isinstance(device_arg, torch.device):
            device = device_arg
        else:
            device = kwargs["particle_stack_dft"].device

    cache = prepare_static_inputs(kwargs, device)
    img_h, img_w = cache["img_shape"]
    tmpl_h, tmpl_w = cache["tmpl_shape"]
    crop_h = img_h - tmpl_h + 1
    crop_w = img_w - tmpl_w + 1
    num_particles = cache["num_particles"]

    defocus_u = torch.as_tensor(kwargs["defocus_u"], device=device).reshape(-1)
    defocus_v = torch.as_tensor(kwargs["defocus_v"], device=device).reshape(-1)
    defocus_angle = torch.as_tensor(kwargs["defocus_angle"], device=device).reshape(-1)
    defocus_u = defocus_u.expand(num_particles)
    defocus_v = defocus_v.expand(num_particles)
    defocus_angle = defocus_angle.expand(num_particles)

    projective_filters = cache["projective_filters"]
    shared_filter = projective_filters.shape[0] == 1
    corr_mean = cache["corr_mean"]
    corr_std = cache["corr_std"]
    defocus_offsets = cache["defocus_offsets"]
    pixel_size_offsets = cache["pixel_size_offsets"]
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
        chunk_filter = (
            projective_filters if shared_filter else projective_filters[start:stop]
        )
        spectrum = fourier_slice * (chunk_filter * ctf)
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

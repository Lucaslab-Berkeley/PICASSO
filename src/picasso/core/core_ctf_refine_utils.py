"""Utility functions shared between CTF refinement implementations."""

import itertools
import math
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.utils.backend_setup import setup_particle_backend_kwargs

from picasso.core.core_ctf_refine_loss import compute_particle_z_scores
from picasso.utils.data_io import read_mrc_to_numpy


def output_mag_matrix(mag_matrix: torch.Tensor) -> None:
    """Output magnification matrix with axis of stretching and anisotropy.

    Parameters
    ----------
    mag_matrix : torch.Tensor
        2x2 anisotropic magnification matrix.
    """
    mag_np = mag_matrix.detach().cpu().numpy()
    print("\nMagnification Matrix:")
    print(mag_np)

    eigenvalues, eigenvectors = np.linalg.eig(mag_np)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    max_idx = np.argmax(np.abs(eigenvalues))
    lambda_max = eigenvalues[max_idx]
    lambda_min = eigenvalues[1 - max_idx]

    eigenvector = eigenvectors[:, max_idx]
    norm = np.linalg.norm(eigenvector)
    if norm > 1e-10:
        eigenvector = eigenvector / norm

    angle_deg = math.degrees(math.atan2(eigenvector[1], eigenvector[0]))

    mean_mag = (lambda_max + lambda_min) / 2.0
    if abs(mean_mag) > 1e-10:
        anisotropy_pct = abs(lambda_max - lambda_min) / abs(mean_mag) * 100.0
    else:
        anisotropy_pct = 0.0

    print(f"Axis of stretching: {angle_deg:.2f} degrees")
    print(f"Magnification anisotropy: {anisotropy_pct:.2f}%")
    print()


def get_particle_positions_normalized(
    results_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Get normalized (x_norm, y_norm) positions in [0, 1] for quadratic phase shift.

    Reads the results CSV, gets micrograph_path from the first row, loads the
    micrograph MRC to obtain image dimensions, and normalizes pos_x_img and
    pos_y_img columns.

    Parameters
    ----------
    results_path : str
        Path to the results CSV file.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape (num_particles, 2) with (x_norm, y_norm) in [0, 1].
    """
    df = pd.read_csv(results_path)
    if "micrograph_path" not in df.columns:
        raise ValueError(
            f"Results CSV at {results_path} must contain 'micrograph_path' column "
            "for quadratic phase shift mode."
        )
    if "pos_x_img" not in df.columns or "pos_y_img" not in df.columns:
        raise ValueError(
            f"Results CSV at {results_path} must contain 'pos_x_img' and 'pos_y_img' "
            "columns for quadratic phase shift mode."
        )
    micrograph_path = df["micrograph_path"].iloc[0]
    micrograph = read_mrc_to_numpy(micrograph_path)
    # MRC shape: (nz, ny, nx) for 3D or (ny, nx) for 2D
    if micrograph.ndim == 3:
        height, width = int(micrograph.shape[1]), int(micrograph.shape[2])
    else:
        height, width = int(micrograph.shape[0]), int(micrograph.shape[1])
    pos_x = torch.tensor(
        df["pos_x_img"].values.astype(np.float32), device=device, dtype=torch.float32
    )
    pos_y = torch.tensor(
        df["pos_y_img"].values.astype(np.float32), device=device, dtype=torch.float32
    )
    x_norm = pos_x / width
    y_norm = pos_y / height
    return torch.stack([x_norm, y_norm], dim=-1)


def phase_shift_quadratic_at_positions(
    positions_norm_xy: torch.Tensor,
    C: torch.Tensor,
    g: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    g2: torch.Tensor | None = None,
    k2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate quadratic phase shift model at normalized positions.

    Single-axis: phase = C + g*s + k*s^2, s = x*cos(alpha) + y*sin(alpha).

    Two-axis: additionally t = -x*sin(alpha) + y*cos(alpha) and
    phase += g2*t + k2*t^2.

    x, y in [-1, 1] from positions_norm_xy in [0, 1].

    Parameters
    ----------
    positions_norm_xy : torch.Tensor
        Shape (N, 2) with (x_norm, y_norm) in [0, 1].
    C, g, k, alpha : torch.Tensor
        Scalar parameters (C in degrees, alpha in radians).
    g2, k2 : torch.Tensor, optional
        Second-axis linear and quadratic coefficients. When both are provided,
        the two-axis model is used; when both are None, single-axis only.

    Returns
    -------
    torch.Tensor
        Shape (N,) phase shift in degrees, clamped to [0, 180].
    """
    x = 2.0 * positions_norm_xy[..., 0] - 1.0
    y = 2.0 * positions_norm_xy[..., 1] - 1.0
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    s = x * ca + y * sa
    phase = C + g * s + k * (s**2)
    if g2 is not None and k2 is not None:
        t = -x * sa + y * ca
        phase = phase + g2 * t + k2 * (t**2)
    return torch.clamp(phase, min=0.0, max=180.0)


def phase_shift_quadratic_from_quad_params(
    positions_norm_xy: torch.Tensor,
    quad_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate phase shift from quad_params dict (single- or two-axis by key set)."""
    if "g1" in quad_params:
        return phase_shift_quadratic_at_positions(
            positions_norm_xy,
            quad_params["C"],
            quad_params["g1"],
            quad_params["k1"],
            quad_params["alpha"],
            g2=quad_params["g2"],
            k2=quad_params["k2"],
        )
    return phase_shift_quadratic_at_positions(
        positions_norm_xy,
        quad_params["C"],
        quad_params["g"],
        quad_params["k"],
        quad_params["alpha"],
    )


def clamp_phase_shift_quadratic_after_step(
    quad_params: dict[str, torch.Tensor],
) -> None:
    """Clamp quadratic C to [0, 180] after optimizer step."""
    with torch.no_grad():
        quad_params["C"].clamp_(min=0.0, max=180.0)


def quad_params_to_dict(
    quad_params: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Convert quadratic phase shift params to a serializable dict (C in degrees, alpha in radians)."""
    if "g1" in quad_params:
        return {
            "C": float(quad_params["C"].detach().cpu().item()),
            "g1": float(quad_params["g1"].detach().cpu().item()),
            "k1": float(quad_params["k1"].detach().cpu().item()),
            "g2": float(quad_params["g2"].detach().cpu().item()),
            "k2": float(quad_params["k2"].detach().cpu().item()),
            "alpha": float(quad_params["alpha"].detach().cpu().item()),
        }
    return {
        "C": float(quad_params["C"].detach().cpu().item()),
        "g": float(quad_params["g"].detach().cpu().item()),
        "k": float(quad_params["k"].detach().cpu().item()),
        "alpha": float(quad_params["alpha"].detach().cpu().item()),
    }


def apply_dataset_params(
    particle_backend_kwargs: dict[str, Any],
    anistropic_mag: torch.Tensor | None,
    odd_zernikes: dict[str, torch.Tensor] | None,
) -> None:
    """Put the dataset-level parameters into particle backend kwargs (in place).

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs.
    anistropic_mag : torch.Tensor | None
        Magnification matrix for the projections; None keeps the current one.
    odd_zernikes : dict[str, torch.Tensor] | None
        Beam tilt in asinh space ("asinh_Z31c", "asinh_Z31s"); None keeps the
        Zernikes from the results CSV.
    """
    if anistropic_mag is not None:
        particle_backend_kwargs["mag_matrix"] = anistropic_mag
    if odd_zernikes is not None:
        # Convert from asinh space; the scale matches setup_refinement_parameters.
        zernike_scale = 1e6
        particle_backend_kwargs["ctf_kwargs"]["odd_zernikes"] = {
            "Z31c": zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"]),
            "Z31s": zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"]),
        }


def setup_particle_backend_for_micrograph(
    results_path: str,
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    device: torch.device,
    preprocessing_filters: PreprocessingFilters,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    apply_global_filtering: bool,
    anistropic_mag: torch.Tensor | None = None,
    odd_zernikes: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Set up particle backend kwargs for a micrograph.

    Parameters
    ----------
    results_path : str
        Path to results file.
    template : torch.Tensor
        Template volume.
    template_box_size : tuple[int, int]
        Box size of template.
    device : torch.device
        Device to use.
    preprocessing_filters : PreprocessingFilters
        Filters to apply.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets.
    defocus_offsets : torch.Tensor
        Defocus offsets.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets.
    apply_global_filtering : bool
        Whether to apply global filtering.
    anistropic_mag : torch.Tensor | None, optional
        Anisotropic magnification matrix applied to the projections. If None, the
        matrix from the results CSV (if any) is used. Never applied to the CTF.
    odd_zernikes : dict[str, torch.Tensor] | None, optional
        Odd Zernike coefficients. If provided, added to CTF kwargs.

    Returns
    -------
    dict[str, Any]
        Particle backend kwargs.
    """
    # Load particle stack
    particle_stack = ParticleStack(
        df_path=results_path,
        extracted_box_size=template_box_size,
        original_template_size=template_box_size,
    )
    # Setup euler angles
    euler_angles = particle_stack.get_euler_angles(prefer_refined_angles=True).to(
        device
    )
    # Setup particle backend kwargs
    particle_backend_kwargs = setup_particle_backend_kwargs(
        particle_stack=particle_stack,
        template=template,
        preprocessing_filters=preprocessing_filters,
        euler_angles=euler_angles,
        euler_angle_offsets=euler_angle_offsets,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        apply_global_filtering=apply_global_filtering,
        device_list=[device],
    )

    # The magnification matrix distorts the projection (Fourier slice) only. The
    # CTF parameters come from CTF estimation on the recorded micrograph, whose CTF
    # rings already carry the magnification distortion (absorbed into the fitted
    # defocus/astigmatism), so stretching the CTF by the matrix as well would count
    # it twice. Leopard-EM puts the results-CSV matrix on both; drop it from the
    # CTF here. The projection keeps the CSV matrix unless a refined one is given.
    particle_backend_kwargs["ctf_kwargs"]["mag_matrix"] = None
    apply_dataset_params(particle_backend_kwargs, anistropic_mag, odd_zernikes)

    return cast("dict[str, Any]", particle_backend_kwargs)


def compute_micrograph_loss(
    particle_backend_kwargs: dict[str, Any],
    batch_size: int,
    backward_weight: float | None = None,
) -> torch.Tensor:
    """Per-particle loss (negative z-score) for one micrograph.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs.
    batch_size : int
        Number of particles processed per chunk.
    backward_weight : float | None
        If given, backpropagate ``backward_weight * loss.sum()`` chunk by chunk
        (see ``compute_particle_z_scores``) and return the loss detached.

    Returns
    -------
    torch.Tensor
        Shape (N,) negative z-scores.
    """
    weight = None if backward_weight is None else -backward_weight
    return -compute_particle_z_scores(particle_backend_kwargs, batch_size, weight)


def create_optimizer_aniso(
    anistropic_mag: torch.Tensor,
    optimizer_type: str = "adam",
    lr: float = 0.005,
    max_eval: int = 5,
) -> torch.optim.Optimizer | None:
    """Create optimizer for anisotropic magnification.

    Parameters
    ----------
    anistropic_mag : torch.Tensor
        Anisotropic magnification matrix parameter.
    optimizer_type : str, optional
        Type of optimizer to create. Either "adam" or "lbfgs". Defaults to "adam".
    lr : float, optional
        Learning rate for the optimizer. Defaults to 0.005.
    max_eval : int, optional
        Maximum number of closure evaluations for L-BFGS. Defaults to 5.

    Returns
    -------
    torch.optim.Optimizer | None
        Created optimizer, or None if not needed.
    """
    if optimizer_type == "lbfgs":
        return torch.optim.LBFGS(
            [anistropic_mag],
            lr=lr,
            line_search_fn="strong_wolfe",
            max_eval=max_eval,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam([anistropic_mag], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_optimizer_beam_tilt(
    odd_zernikes: dict[str, torch.Tensor],
    optimizer_type: str = "adam",
    lr: float = 1.0,
    max_eval: int = 5,
) -> torch.optim.Optimizer | None:
    """Create optimizer for beam tilt (odd Zernike coefficients in asinh space).

    Parameters
    ----------
    odd_zernikes : dict[str, torch.Tensor]
        Dictionary with "asinh_Z31c" and "asinh_Z31s" keys containing asinh-space parameters.
    optimizer_type : str, optional
        Type of optimizer to create. Either "adam" or "lbfgs". Defaults to "adam".
    lr : float, optional
        Learning rate for the optimizer. Defaults to 1.0.
    max_eval : int, optional
        Maximum number of closure evaluations for L-BFGS. Defaults to 5.

    Returns
    -------
    torch.optim.Optimizer | None
        Created optimizer, or None if not needed.
    """
    if optimizer_type == "lbfgs":
        return torch.optim.LBFGS(
            [odd_zernikes["asinh_Z31c"], odd_zernikes["asinh_Z31s"]],
            lr=lr,
            line_search_fn="strong_wolfe",
            max_eval=max_eval,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            [odd_zernikes["asinh_Z31c"], odd_zernikes["asinh_Z31s"]], lr=lr
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def setup_refinement_parameters(
    results_files: list[str],
    device: torch.device,
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
    initial_mag_matrix: torch.Tensor | None = None,
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
    optimizer_type: str = "adam",
    anisotropic_mag_lr: float = 0.005,
    beam_tilt_lr: float = 1.0,
    lbfgs_max_eval: int = 5,
) -> dict[str, Any]:
    """Set up refinement parameters and optimizers.

    Parameters
    ----------
    results_files : list[str]
        The list of results file paths (unused; kept for compatibility).
    device : torch.device
        The device to use for computation.
    refine_anisotropic_mag : bool
        Whether to refine the anisotropic magnitude.
    refine_beam_tilt : bool
        Whether to refine the beam tilt.
    initial_mag_matrix : torch.Tensor | None, optional
        The initial mag matrix.
    initial_odd_zernikes : dict[str, torch.Tensor] | None, optional
        The initial odd zernikes.
    optimizer_type : str, optional
        Type of optimizer to use ("adam" or "lbfgs"). Defaults to "adam".
    anisotropic_mag_lr : float, optional
        Learning rate for anisotropic magnification optimizer. Defaults to 0.005.
    beam_tilt_lr : float, optional
        Learning rate for beam tilt optimizer. Defaults to 1.0.
    lbfgs_max_eval : int, optional
        Maximum number of closure evaluations for L-BFGS optimizers. Defaults to 5.

    Returns
    -------
    dict[str, Any]
        Dictionary containing setup state including parameters and optimizers.
    """
    optimizer_aniso = None
    optimizer_beam_tilt = None

    if refine_anisotropic_mag:
        if initial_mag_matrix is not None:
            anistropic_mag = (
                initial_mag_matrix.detach()
                .clone()
                .float()
                .to(device)
                .requires_grad_(True)
            )
        else:
            anistropic_mag = torch.eye(2, 2, device=device, requires_grad=True)
        optimizer_aniso = create_optimizer_aniso(
            anistropic_mag,
            optimizer_type,
            lr=anisotropic_mag_lr,
            max_eval=lbfgs_max_eval,
        )
    else:
        if initial_mag_matrix is not None:
            anistropic_mag = initial_mag_matrix.to(device)
        else:
            anistropic_mag = torch.eye(2, 2, device=device)

    if refine_beam_tilt:
        # Scale factor for asinh transformation (values expected in range ~[-1e6, 1e6])
        zernike_scale = 1e6
        if initial_odd_zernikes is not None:
            # Convert initial values to asinh space
            z31c_init = initial_odd_zernikes["Z31c"].to(device)
            z31s_init = initial_odd_zernikes["Z31s"].to(device)
            # Transform: asinh(value / scale)
            # asinh handles both positive and negative values smoothly
            asinh_z31c = torch.asinh(z31c_init / zernike_scale)
            asinh_z31s = torch.asinh(z31s_init / zernike_scale)

            odd_zernikes = {
                "asinh_Z31c": asinh_z31c.requires_grad_(True),
                "asinh_Z31s": asinh_z31s.requires_grad_(True),
            }
        else:
            # Initialize to 0.0 in asinh space, which corresponds to actual value 0.0
            # (since asinh(0) = 0, and scale * sinh(0) = 0)
            odd_zernikes = {
                "asinh_Z31c": torch.tensor(0.0, device=device, requires_grad=True),
                "asinh_Z31s": torch.tensor(0.0, device=device, requires_grad=True),
            }
        optimizer_beam_tilt = create_optimizer_beam_tilt(
            odd_zernikes, optimizer_type, lr=beam_tilt_lr, max_eval=lbfgs_max_eval
        )
    else:
        # Scale factor for asinh transformation
        zernike_scale = 1e6
        if initial_odd_zernikes is not None:
            # For non-refined case, convert to asinh space for consistency
            z31c_init = initial_odd_zernikes["Z31c"].to(device)
            z31s_init = initial_odd_zernikes["Z31s"].to(device)
            asinh_z31c = torch.asinh(z31c_init / zernike_scale)
            asinh_z31s = torch.asinh(z31s_init / zernike_scale)
            odd_zernikes = {
                "asinh_Z31c": asinh_z31c,
                "asinh_Z31s": asinh_z31s,
            }
        else:
            odd_zernikes = {
                "asinh_Z31c": torch.tensor(0.0, device=device),
                "asinh_Z31s": torch.tensor(0.0, device=device),
            }

    pixel_size_offsets = torch.tensor([0.0], device=device)
    euler_angle_offsets = torch.zeros((1, 3), device=device)
    defocus_offsets = torch.tensor([0.0], device=device)

    return {
        "anistropic_mag": anistropic_mag,
        "odd_zernikes": odd_zernikes,
        "optimizer_aniso": optimizer_aniso,
        "optimizer_beam_tilt": optimizer_beam_tilt,
        "pixel_size_offsets": pixel_size_offsets,
        "euler_angle_offsets": euler_angle_offsets,
        "defocus_offsets": defocus_offsets,
    }


def distribute_micrographs_across_gpus(
    num_micrographs: int,
    num_gpus: int,
) -> list[list[int]]:
    """Distribute micrograph indices across GPUs.

    Parameters
    ----------
    num_micrographs : int
        Total number of micrographs.
    num_gpus : int
        Number of GPUs.

    Returns
    -------
    list[list[int]]
        List of lists, where each inner list contains micrograph indices
        assigned to that GPU.
    """
    gpu_assignments: list[list[int]] = [[] for _ in range(num_gpus)]
    for idx, mic_idx in enumerate(range(num_micrographs)):
        gpu_assignments[idx % num_gpus].append(mic_idx)
    return gpu_assignments


def balance_micrographs_across_gpus(
    weights: list[float],
    num_gpus: int,
) -> list[list[int]]:
    """Assign micrographs to GPUs so each GPU gets a similar total weight.

    Greedy longest-processing-time assignment: heaviest micrograph first, each to
    the GPU with the least weight so far. Use particle counts as weights, since
    the time per micrograph is roughly proportional to its number of particles.

    Parameters
    ----------
    weights : list[float]
        Weight (e.g. number of particles) of each micrograph.
    num_gpus : int
        Number of GPUs.

    Returns
    -------
    list[list[int]]
        Micrograph indices for each GPU, in increasing order.
    """
    loads = [0.0] * num_gpus
    assignments: list[list[int]] = [[] for _ in range(num_gpus)]
    for index in sorted(range(len(weights)), key=lambda k: (-weights[k], k)):
        gpu_id = min(range(num_gpus), key=lambda g: (loads[g], g))
        assignments[gpu_id].append(index)
        loads[gpu_id] += weights[index]
    return [sorted(indices) for indices in assignments]


def count_particles(results_path: str) -> int:
    """Number of particle rows in a results CSV (1 if it cannot be read)."""
    try:
        with open(results_path, encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 1)
    except OSError:
        return 1


def create_gpu_setup_states(
    setup_state: dict[str, Any],
    device_list: list[torch.device],
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
) -> list[dict[str, Any]]:
    """Create setup state copies for each GPU.

    Parameters
    ----------
    setup_state : dict[str, Any]
        Master setup state.
    device_list : list[torch.device]
        List of devices.
    refine_anisotropic_mag : bool
        Whether to refine anisotropic magnification.
    refine_beam_tilt : bool
        Whether to refine beam tilt.

    Returns
    -------
    list[dict[str, Any]]
        List of setup states, one per GPU.
    """
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]

    setup_states = []
    for device in device_list:
        if refine_anisotropic_mag:
            aniso_mag_gpu = anistropic_mag.to(device).detach().requires_grad_(True)
        else:
            aniso_mag_gpu = anistropic_mag.to(device)
        if refine_beam_tilt:
            asinh_z31c_gpu = (
                odd_zernikes["asinh_Z31c"].to(device).detach().requires_grad_(True)
            )
            asinh_z31s_gpu = (
                odd_zernikes["asinh_Z31s"].to(device).detach().requires_grad_(True)
            )
        else:
            asinh_z31c_gpu = odd_zernikes["asinh_Z31c"].to(device)
            asinh_z31s_gpu = odd_zernikes["asinh_Z31s"].to(device)
        state_copy = {
            "anistropic_mag": aniso_mag_gpu,
            "odd_zernikes": {
                "asinh_Z31c": asinh_z31c_gpu,
                "asinh_Z31s": asinh_z31s_gpu,
            },
            "pixel_size_offsets": setup_state["pixel_size_offsets"].to(device),
            "euler_angle_offsets": setup_state["euler_angle_offsets"].to(device),
            "defocus_offsets": setup_state["defocus_offsets"].to(device),
            "optimizer_aniso": None,
            "optimizer_beam_tilt": None,
        }
        setup_states.append(state_copy)
    return setup_states


def update_gpu_setup_states(
    setup_states: list[dict[str, Any]],
    setup_state: dict[str, Any],
    device_list: list[torch.device],
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
) -> None:
    """Update GPU-local setup states with master parameter values.

    Parameters
    ----------
    setup_states : list[dict[str, Any]]
        List of GPU-local setup states to update.
    setup_state : dict[str, Any]
        Master setup state with updated parameters.
    device_list : list[torch.device]
        List of devices.
    refine_anisotropic_mag : bool
        Whether to refine anisotropic magnification.
    refine_beam_tilt : bool
        Whether to refine beam tilt.
    """
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]

    for gpu_id, device in enumerate(device_list):
        if refine_anisotropic_mag:
            setup_states[gpu_id]["anistropic_mag"].data.copy_(
                anistropic_mag.data.to(device)
            )
        if refine_beam_tilt:
            setup_states[gpu_id]["odd_zernikes"]["asinh_Z31c"].data.copy_(
                odd_zernikes["asinh_Z31c"].data.to(device)
            )
            setup_states[gpu_id]["odd_zernikes"]["asinh_Z31s"].data.copy_(
                odd_zernikes["asinh_Z31s"].data.to(device)
            )


def check_convergence(
    loss_history: list[float],
    convergence_threshold: float,
    num_convergence_iterations: int = 5,
    label: str = "dataset loss",
) -> tuple[bool, list[float]]:
    """Check whether the loss changed by at most a threshold n times in a row.

    Parameters
    ----------
    loss_history : list[float]
        History of loss values, oldest first.
    convergence_threshold : float
        Maximum absolute change between consecutive losses.
    num_convergence_iterations : int, optional
        Number n of consecutive changes that must all be within the threshold.
        Values <= 0 disable the check. Defaults to 5.
    label : str, optional
        Name of the loss for the log message. Defaults to "dataset loss".

    Returns
    -------
    tuple[bool, list[float]]
        Tuple containing:
        - converged: bool indicating if convergence was detected
        - updated_loss_history: list with at most the last n + 1 values
    """
    if num_convergence_iterations <= 0:
        return False, loss_history
    min_history_len = num_convergence_iterations + 1
    if len(loss_history) < min_history_len:
        return False, loss_history
    recent = loss_history[-min_history_len:]
    changes = [abs(after - before) for before, after in itertools.pairwise(recent)]
    if all(change <= convergence_threshold for change in changes):
        print(
            f"Converged: absolute change in {label} ({changes[-1]:.6f}) <= "
            f"threshold ({convergence_threshold:.6f}) for "
            f"{num_convergence_iterations} consecutive iterations. Stopping early."
        )
        return True, recent
    return False, recent


def create_final_results_dict(
    setup_state: dict[str, Any],
    astigmatism_angle_all: torch.Tensor,
    astigmatism_magnitude_all: torch.Tensor,
    dataset_losses: list[tuple[int, float]],
    micrograph_losses: dict[int, list[tuple[int, float]]],
    defocus_all: dict[int, torch.Tensor] | None = None,
    phase_shift_all: dict[int, torch.Tensor] | None = None,
    phase_shift_quadratic_params_all: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Create final results dictionary.

    Parameters
    ----------
    setup_state : dict[str, Any]
        Setup state containing parameters.
    astigmatism_angle_all : torch.Tensor
        All astigmatism angles.
    astigmatism_magnitude_all : torch.Tensor
        All astigmatism magnitudes.
    dataset_losses : list[tuple[int, float]]
        Dataset-level loss history.
    micrograph_losses : dict[int, list[tuple[int, float]]]
        Micrograph-level loss history.
    defocus_all : dict[int, torch.Tensor] | None, optional
        Per-micrograph defocus tensors. If None, defocus was not refined.
    phase_shift_all : dict[int, torch.Tensor] | None, optional
        Per-micrograph phase_shift tensors. If None, phase_shift was not refined.
    phase_shift_quadratic_params_all : dict[int, dict[str, float]] | None, optional
        Per-micrograph quadratic params (C, g, k, alpha). If None, quadratic phase was not used.

    Returns
    -------
    dict[str, Any]
        Final results dictionary.
    """
    # Convert from asinh space to actual values for final results
    odd_zernikes_asinh = setup_state["odd_zernikes"]
    zernike_scale = 1e6
    result = {
        "anistropic_mag": setup_state["anistropic_mag"].detach(),
        "odd_zernikes": {
            "Z31c": (
                zernike_scale * torch.sinh(odd_zernikes_asinh["asinh_Z31c"])
            ).detach(),
            "Z31s": (
                zernike_scale * torch.sinh(odd_zernikes_asinh["asinh_Z31s"])
            ).detach(),
        },
        "astigmatism_angle": astigmatism_angle_all.detach(),
        "astigmatism_magnitude": astigmatism_magnitude_all.detach(),
        "dataset_losses": dataset_losses,
        "micrograph_losses": micrograph_losses,
    }
    # Only include defocus_all if it was refined (not None)
    if defocus_all is not None:
        result["defocus_all"] = {k: v.detach() for k, v in defocus_all.items()}
    # Only include phase_shift_all if it was refined (not None)
    if phase_shift_all is not None:
        result["phase_shift_all"] = {k: v.detach() for k, v in phase_shift_all.items()}
    if phase_shift_quadratic_params_all is not None:
        result["phase_shift_quadratic_params_all"] = phase_shift_quadratic_params_all
    return result


def accumulate_gradients_from_gpu_worker(
    grads: dict[str, torch.Tensor],
    anistropic_mag: torch.Tensor,
    odd_zernikes: dict[str, torch.Tensor],
    primary_device: torch.device,
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
) -> None:
    """Accumulate gradients from GPU worker to master parameters.

    Parameters
    ----------
    grads : dict[str, torch.Tensor]
        Dictionary of gradients collected from GPU worker.
    anistropic_mag : torch.Tensor
        Master anisotropic magnification parameter.
    odd_zernikes : dict[str, torch.Tensor]
        Master odd Zernike coefficients.
    primary_device : torch.device
        Primary device for master parameters.
    refine_anisotropic_mag : bool
        Whether to refine anisotropic magnification.
    refine_beam_tilt : bool
        Whether to refine beam tilt.
    """
    if refine_anisotropic_mag and "anistropic_mag" in grads:
        if anistropic_mag.grad is None:
            anistropic_mag.grad = grads["anistropic_mag"].to(primary_device)
        else:
            anistropic_mag.grad.add_(grads["anistropic_mag"].to(primary_device))

    if refine_beam_tilt:
        if "asinh_Z31c" in grads:
            if odd_zernikes["asinh_Z31c"].grad is None:
                odd_zernikes["asinh_Z31c"].grad = grads["asinh_Z31c"].to(primary_device)
            else:
                odd_zernikes["asinh_Z31c"].grad.add_(
                    grads["asinh_Z31c"].to(primary_device)
                )
        if "asinh_Z31s" in grads:
            if odd_zernikes["asinh_Z31s"].grad is None:
                odd_zernikes["asinh_Z31s"].grad = grads["asinh_Z31s"].to(primary_device)
            else:
                odd_zernikes["asinh_Z31s"].grad.add_(
                    grads["asinh_Z31s"].to(primary_device)
                )

"""Utility functions shared between CTF refinement implementations."""

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

# Disable Dynamo before importing leopard_em so its compiled code is not used
# (avoids "FX to symbolically trace a dynamo-optimized function" in refinement).
if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
    torch._dynamo.disable()

from leopard_em.backend.core_differentiable_refine import core_differentiable_refine
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.utils.backend_setup import setup_particle_backend_kwargs

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


def initialize_astigmatism_parameters(
    i: int,
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
    initial_astigmatism_angle: torch.Tensor | None,
    initial_astigmatism_magnitude: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Initialize astigmatism parameters for a micrograph.

    Parameters
    ----------
    i : int
        Micrograph index.
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs containing defocus information.
    device : torch.device
        Device to use for tensors.
    initial_astigmatism_angle : torch.Tensor | None
        Initial astigmatism angle tensor.
    initial_astigmatism_magnitude : torch.Tensor | None
        Initial astigmatism magnitude tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Tuple containing:
        - astigmatism_angle: torch.Tensor with requires_grad=True
        - astigmatism_magnitude: torch.Tensor with requires_grad=True
        - initial_defocus_mean: torch.Tensor | None (mean defocus for broadcasting)
    """
    # Get initial values from provided initial tensors or particle_backend_kwargs
    if initial_astigmatism_angle is not None:
        initial_angle = initial_astigmatism_angle[i].to(device)
    else:
        defocus_angle_val = particle_backend_kwargs["defocus_angle"]
        # Extract first element if tensor (same for all particles)
        if (
            isinstance(defocus_angle_val, torch.Tensor)
            and defocus_angle_val.numel() > 1
        ):
            defocus_angle_val = defocus_angle_val[0].item()
        initial_angle = torch.tensor(defocus_angle_val, device=device)

    # Get defocus values - keep as tensors for per-particle defocus
    defocus_u_val = particle_backend_kwargs["defocus_u"]
    defocus_v_val = particle_backend_kwargs["defocus_v"]

    defocus_u_shape = (
        defocus_u_val.shape if isinstance(defocus_u_val, torch.Tensor) else "N/A"
    )
    defocus_v_shape = (
        defocus_v_val.shape if isinstance(defocus_v_val, torch.Tensor) else "N/A"
    )
    print(
        f"[SHAPE DEBUG] defocus_u_val type: {type(defocus_u_val)}, "
        f"shape: {defocus_u_shape}"
    )
    print(
        f"[SHAPE DEBUG] defocus_v_val type: {type(defocus_v_val)}, "
        f"shape: {defocus_v_shape}"
    )

    # Convert to tensors if needed, but keep full shape
    if not isinstance(defocus_u_val, torch.Tensor):
        defocus_u_val = torch.tensor(defocus_u_val, device=device)
    else:
        defocus_u_val = defocus_u_val.to(device)

    if not isinstance(defocus_v_val, torch.Tensor):
        defocus_v_val = torch.tensor(defocus_v_val, device=device)
    else:
        defocus_v_val = defocus_v_val.to(device)

    print(
        f"[SHAPE DEBUG] After conversion - "
        f"defocus_u_val shape: {defocus_u_val.shape}, "
        f"defocus_v_val shape: {defocus_v_val.shape}"
    )

    if initial_astigmatism_magnitude is not None:
        initial_magnitude = initial_astigmatism_magnitude[i].to(device)
        # Compute defocus_mean from full tensors (per particle)
        initial_defocus_mean = (defocus_u_val + defocus_v_val) / 2.0
        print(f"[SHAPE DEBUG] initial_defocus_mean shape: {initial_defocus_mean.shape}")
    else:
        # Compute initial magnitude from mean difference across particles
        # (astigmatism magnitude is same for all particles in micrograph)
        initial_magnitude_scalar = (defocus_u_val - defocus_v_val).mean().item()
        initial_magnitude = torch.tensor(initial_magnitude_scalar, device=device)
        # Compute defocus_mean from full tensors (per particle)
        initial_defocus_mean = (defocus_u_val + defocus_v_val) / 2.0
        print(f"[SHAPE DEBUG] initial_magnitude (scalar): {initial_magnitude}")
        print(f"[SHAPE DEBUG] initial_defocus_mean shape: {initial_defocus_mean.shape}")

    # Create tensors with requires_grad, handling both tensor and scalar inputs
    if isinstance(initial_angle, torch.Tensor):
        astigmatism_angle = initial_angle.detach().clone().requires_grad_(True)
    else:
        astigmatism_angle = torch.tensor(
            initial_angle, device=device, requires_grad=True
        )
    if isinstance(initial_magnitude, torch.Tensor):
        astigmatism_magnitude = initial_magnitude.detach().clone().requires_grad_(True)
    else:
        astigmatism_magnitude = torch.tensor(
            initial_magnitude, device=device, requires_grad=True
        )

    return astigmatism_angle, astigmatism_magnitude, initial_defocus_mean


def get_non_refined_astigmatism_values(
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get astigmatism values when not refining astigmatism.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs containing defocus information.
    device : torch.device
        Device to use for tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing (astigmatism_angle, astigmatism_magnitude).
    """
    defocus_angle_val = particle_backend_kwargs["defocus_angle"]
    # Extract first element if tensor (same for all particles)
    if isinstance(defocus_angle_val, torch.Tensor) and defocus_angle_val.numel() > 1:
        defocus_angle_val = defocus_angle_val[0].item()
    astigmatism_angle = torch.tensor(defocus_angle_val, device=device)
    defocus_u_val = particle_backend_kwargs["defocus_u"]
    defocus_v_val = particle_backend_kwargs["defocus_v"]
    # Extract first element if tensor (difference is same for all particles)
    if isinstance(defocus_u_val, torch.Tensor) and defocus_u_val.numel() > 1:
        defocus_u_val = defocus_u_val[0].item()
    if isinstance(defocus_v_val, torch.Tensor) and defocus_v_val.numel() > 1:
        defocus_v_val = defocus_v_val[0].item()
    defocus_u = torch.tensor(defocus_u_val, device=device)
    defocus_v = torch.tensor(defocus_v_val, device=device)
    astigmatism_magnitude = defocus_u - defocus_v
    return astigmatism_angle, astigmatism_magnitude


def initialize_defocus_parameters(
    results_path: str,
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Initialize defocus parameters for a micrograph.

    Parameters
    ----------
    results_path : str
        Path to results CSV file.
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs containing defocus information.
        Used to determine the number of particles.
    device : torch.device
        Device to use for tensors.

    Returns
    -------
    torch.Tensor
        Per-particle defocus tensor initialized to zeros with requires_grad=True.
        Note: The defocus values from CSV are already included in defocus_u/defocus_v
        from particle_backend_kwargs, so we initialize to zero and add our refinement.
    """
    # Get number of particles from defocus_u in particle_backend_kwargs
    defocus_u = particle_backend_kwargs["defocus_u"]
    if isinstance(defocus_u, torch.Tensor):
        num_particles = defocus_u.shape[0]
    else:
        # If it's a scalar or list, we need to check the CSV
        df = pd.read_csv(results_path)
        num_particles = len(df)

    # Initialize to zeros - the CSV defocus values are already in defocus_u/defocus_v
    defocus_tensor = torch.zeros(
        num_particles, dtype=torch.float32, device=device, requires_grad=True
    )

    return defocus_tensor


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
    quad_params: dict[str, torch.nn.Parameter],
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
    quad_params: dict[str, torch.nn.Parameter],
) -> None:
    """Clamp quadratic C to [0, 180] after optimizer step."""
    with torch.no_grad():
        quad_params["C"].clamp_(min=0.0, max=180.0)


def quad_params_to_dict(
    quad_params: dict[str, torch.nn.Parameter],
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


def initialize_phase_shift_parameters(
    i: int,
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
    phase_shift_mode: str,
    initial_phase_shift: torch.Tensor | None,
    initial_phase_shift_quadratic_C: float | None = None,
    initial_phase_shift_quadratic_g: float = 0.0,
    initial_phase_shift_quadratic_k: float = 0.0,
    initial_phase_shift_quadratic_alpha: float = 0.0,
    phase_shift_quadratic_two_axes: bool = False,
    initial_phase_shift_quadratic_g2: float = 0.0,
    initial_phase_shift_quadratic_k2: float = 0.0,
) -> torch.Tensor | dict[str, torch.nn.Parameter]:
    """Initialize phase_shift parameters for a micrograph.

    Parameters
    ----------
    i : int
        Micrograph index.
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs containing CTF information.
        Used to determine the number of particles and extract initial phase_shift.
    device : torch.device
        Device to use for tensors.
    phase_shift_mode : str
        Mode for phase_shift refinement: 'per_micrograph', 'per_particle', or 'quadratic'.
    initial_phase_shift : torch.Tensor | None
        Initial phase_shift tensor. If provided, should have shape:
        - (num_micrographs,) for per_micrograph mode
        - (num_micrographs, num_particles) for per_particle mode
    initial_phase_shift_quadratic_C : float or None, optional
        Initial C (degrees) for quadratic mode. If None, use mean phase shift
        from the results file for this micrograph. Default 50.0 when not None.
    initial_phase_shift_quadratic_g : float, optional
        Initial g for quadratic mode. Default 0.0.
    initial_phase_shift_quadratic_k : float, optional
        Initial k for quadratic mode. Default 0.0.
    initial_phase_shift_quadratic_alpha : float, optional
        Initial alpha (radians) for quadratic mode. Default 0.0.
    phase_shift_quadratic_two_axes : bool, optional
        If True, use six-parameter model with g1, k1, g2, k2. Default False.
    initial_phase_shift_quadratic_g2 : float, optional
        Initial g2 (t axis) for two-axis quadratic. Default 0.0.
    initial_phase_shift_quadratic_k2 : float, optional
        Initial k2 (t axis) for two-axis quadratic. Default 0.0.

    Returns
    -------
    torch.Tensor or dict[str, torch.nn.Parameter]
        For per_micrograph/per_particle: phase shift tensor with requires_grad=True.
        For quadratic: dict with keys "C", "g", "k", "alpha" or two-axis
        "C", "g1", "k1", "g2", "k2", "alpha" of nn.Parameter.
    """
    if phase_shift_mode == "quadratic":
        # If C not given, use mean phase shift from results for this micrograph
        if initial_phase_shift_quadratic_C is None:
            phase_shift_val = particle_backend_kwargs.get("ctf_kwargs", {}).get(
                "phase_shift", 0.0
            )
            if initial_phase_shift is not None:
                val = initial_phase_shift[i]
                if isinstance(val, torch.Tensor):
                    initial_C = val.mean().item()
                else:
                    initial_C = float(val)
            elif isinstance(phase_shift_val, torch.Tensor):
                initial_C = phase_shift_val.mean().item()
            else:
                initial_C = float(phase_shift_val)
        else:
            initial_C = initial_phase_shift_quadratic_C

        if phase_shift_quadratic_two_axes:
            return {
                "C": torch.nn.Parameter(
                    torch.tensor(initial_C, device=device, dtype=torch.float32)
                ),
                "g1": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift_quadratic_g,
                        device=device,
                        dtype=torch.float32,
                    )
                ),
                "k1": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift_quadratic_k,
                        device=device,
                        dtype=torch.float32,
                    )
                ),
                "g2": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift_quadratic_g2,
                        device=device,
                        dtype=torch.float32,
                    )
                ),
                "k2": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift_quadratic_k2,
                        device=device,
                        dtype=torch.float32,
                    )
                ),
                "alpha": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift_quadratic_alpha,
                        device=device,
                        dtype=torch.float32,
                    )
                ),
            }

        quad_params = {
            "C": torch.nn.Parameter(
                torch.tensor(initial_C, device=device, dtype=torch.float32)
            ),
            "g": torch.nn.Parameter(
                torch.tensor(
                    initial_phase_shift_quadratic_g,
                    device=device,
                    dtype=torch.float32,
                )
            ),
            "k": torch.nn.Parameter(
                torch.tensor(
                    initial_phase_shift_quadratic_k,
                    device=device,
                    dtype=torch.float32,
                )
            ),
            "alpha": torch.nn.Parameter(
                torch.tensor(
                    initial_phase_shift_quadratic_alpha,
                    device=device,
                    dtype=torch.float32,
                )
            ),
        }
        return quad_params

    # Get number of particles from defocus_u in particle_backend_kwargs
    defocus_u = particle_backend_kwargs["defocus_u"]
    if isinstance(defocus_u, torch.Tensor):
        num_particles = defocus_u.shape[0]
    else:
        # If it's a scalar, assume single particle (shouldn't happen in practice)
        num_particles = 1

    # Get initial value from provided initial_phase_shift or particle_backend_kwargs
    if initial_phase_shift is not None:
        if phase_shift_mode == "per_micrograph":
            # Expect shape (num_micrographs,)
            initial_value = initial_phase_shift[i].to(device)
        else:  # per_particle
            # Expect shape (num_micrographs, num_particles) or dict with micrograph indices
            if isinstance(initial_phase_shift, dict):
                initial_value = initial_phase_shift[i].to(device)
            else:
                initial_value = initial_phase_shift[i].to(device)
    else:
        # Extract from particle_backend_kwargs
        phase_shift_val = particle_backend_kwargs.get("ctf_kwargs", {}).get(
            "phase_shift", 0.0
        )

        # Convert to tensor if needed
        if not isinstance(phase_shift_val, torch.Tensor):
            if phase_shift_mode == "per_micrograph":
                initial_value = torch.tensor(phase_shift_val, device=device)
            else:  # per_particle
                # Create tensor for all particles with same initial value
                initial_value = torch.full(
                    (num_particles,), phase_shift_val, device=device
                )
        else:
            phase_shift_val = phase_shift_val.to(device)
            if phase_shift_mode == "per_micrograph":
                # Take mean if it's per-particle tensor
                if phase_shift_val.numel() > 1:
                    initial_value = phase_shift_val.mean()
                else:
                    initial_value = phase_shift_val.squeeze()
            else:  # per_particle
                # Ensure it matches num_particles
                if phase_shift_val.numel() == 1:
                    # Broadcast scalar to all particles
                    initial_value = phase_shift_val.expand(num_particles).clone()
                else:
                    initial_value = phase_shift_val

    # Create tensor with requires_grad
    if isinstance(initial_value, torch.Tensor):
        phase_shift_tensor = initial_value.detach().clone().requires_grad_(True)
    else:
        phase_shift_tensor = torch.tensor(
            initial_value, device=device, requires_grad=True
        )

    return phase_shift_tensor


def get_non_refined_phase_shift_values(
    particle_backend_kwargs: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Get phase_shift values when not refining phase_shift.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs containing CTF information.
    device : torch.device
        Device to use for tensors.

    Returns
    -------
    torch.Tensor
        Phase shift tensor without gradients.
    """
    phase_shift_val = particle_backend_kwargs.get("ctf_kwargs", {}).get(
        "phase_shift", 0.0
    )

    if isinstance(phase_shift_val, torch.Tensor):
        return phase_shift_val.to(device)
    else:
        return torch.tensor(phase_shift_val, device=device)


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
        Anisotropic magnification matrix. If provided, used as mag_matrix.
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

    # Change params for optimizable parameters
    if anistropic_mag is not None:
        particle_backend_kwargs["mag_matrix"] = anistropic_mag
    if odd_zernikes is not None:
        # Convert from asinh space to actual values using sinh()
        # Scale factor matches the one used in setup_refinement_parameters
        zernike_scale = 1e6
        particle_backend_kwargs["ctf_kwargs"]["odd_zernikes"] = {
            "Z31c": zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"]),
            "Z31s": zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"]),
        }

    return cast("dict[str, Any]", particle_backend_kwargs)


def compute_micrograph_loss(
    particle_backend_kwargs: dict[str, Any],
    batch_size: int,
) -> torch.Tensor:
    """Compute micrograph loss.

    Parameters
    ----------
    particle_backend_kwargs : dict[str, Any]
        Particle backend kwargs.
    batch_size : int
        Batch size for computation.

    Returns
    -------
    torch.Tensor
        Micrograph loss tensor.
    """
    result = core_differentiable_refine(
        batch_size=batch_size,
        num_cuda_streams=1,
        **particle_backend_kwargs,
    )
    refined_scaled_mip = result["refined_z_score"]
    micrograph_loss = -refined_scaled_mip
    return micrograph_loss


def get_loss_scalar(loss: torch.Tensor) -> float:
    """Extract scalar value from loss tensor.

    Parameters
    ----------
    loss : torch.Tensor
        Loss tensor.

    Returns
    -------
    float
        Scalar loss value.
    """
    if loss.numel() > 1:
        loss_val: float = loss.mean().item()
    elif hasattr(loss, "item"):
        loss_val = loss.item()
    else:
        loss_val = float(loss)
    return loss_val


def get_loss_scalar_tensor(loss: torch.Tensor) -> torch.Tensor:
    """Get scalar tensor from loss (for backward pass).

    Parameters
    ----------
    loss : torch.Tensor
        Loss tensor.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    if loss.numel() > 1:
        return loss.mean()
    return loss


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
    initial_astigmatism_angle: torch.Tensor | None = None,
    initial_astigmatism_magnitude: torch.Tensor | None = None,
    optimizer_type: str = "adam",
    anisotropic_mag_lr: float = 0.005,
    beam_tilt_lr: float = 1.0,
    lbfgs_max_eval: int = 5,
) -> dict[str, Any]:
    """Set up refinement parameters and optimizers.

    Parameters
    ----------
    results_files : list[str]
        The list of results file paths.
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
    initial_astigmatism_angle : torch.Tensor | None, optional
        The initial astigmatism angle.
    initial_astigmatism_magnitude : torch.Tensor | None, optional
        The initial astigmatism magnitude.
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
            anistropic_mag = initial_mag_matrix.to(device).requires_grad_(True)
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

    if initial_astigmatism_angle is not None:
        if initial_astigmatism_angle.shape != (len(results_files),):
            raise ValueError(
                f"initial_astigmatism_angle must have shape ({len(results_files)},), "
                f"got {initial_astigmatism_angle.shape}"
            )
        astigmatism_angle_all = initial_astigmatism_angle.to(device)
    else:
        astigmatism_angle_all = torch.zeros(len(results_files), device=device)

    if initial_astigmatism_magnitude is not None:
        if initial_astigmatism_magnitude.shape != (len(results_files),):
            raise ValueError(
                f"initial_astigmatism_magnitude must have shape "
                f"({len(results_files)},), got "
                f"{initial_astigmatism_magnitude.shape}"
            )
        astigmatism_magnitude_all = initial_astigmatism_magnitude.to(device)
    else:
        astigmatism_magnitude_all = torch.zeros(len(results_files), device=device)

    pixel_size_offsets = torch.tensor([0.0], device=device)
    euler_angle_offsets = torch.zeros((1, 3), device=device)
    defocus_offsets = torch.tensor([0.0], device=device)

    return {
        "anistropic_mag": anistropic_mag,
        "odd_zernikes": odd_zernikes,
        "astigmatism_angle_all": astigmatism_angle_all,
        "astigmatism_magnitude_all": astigmatism_magnitude_all,
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
) -> tuple[bool, list[float]]:
    """Check if loss has converged based on threshold.

    Parameters
    ----------
    loss_history : list[float]
        History of loss values.
    convergence_threshold : float
        Threshold for convergence.
    num_convergence_iterations : int, optional
        Number of consecutive iterations to check for convergence. Defaults to 5.

    Returns
    -------
    tuple[bool, list[float]]
        Tuple containing:
        - converged: bool indicating if convergence was detected
        - updated_loss_history: list with at most last
          (num_convergence_iterations + 1) values
    """
    min_history_len = num_convergence_iterations + 1
    if len(loss_history) >= min_history_len:
        converged = True
        start_idx = len(loss_history) - num_convergence_iterations
        for j in range(start_idx, len(loss_history) - 1):
            abs_change = abs(loss_history[j + 1] - loss_history[j])
            if abs_change > convergence_threshold:
                converged = False
                break

        if converged:
            last_change = abs(loss_history[-1] - loss_history[-2])
            print(
                f"Converged: absolute change in dataset loss "
                f"({last_change:.6f}) <= threshold "
                f"({convergence_threshold:.6f}) for "
                f"{num_convergence_iterations} consecutive "
                f"iterations. Stopping early."
            )
            return True, loss_history[-min_history_len:]

        return False, loss_history[
            -min_history_len:
        ]  # Keep only last (num_convergence_iterations + 1) values
    return False, loss_history


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


def collect_gradients_from_micrograph_loss(
    micrograph_loss: torch.Tensor,
    local_setup: dict[str, Any],
    num_micrographs: int,
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
) -> dict[str, torch.Tensor]:
    """Collect gradients from micrograph loss for dataset-level optimization.

    Parameters
    ----------
    micrograph_loss : torch.Tensor
        Loss tensor from micrograph processing.
    local_setup : dict[str, Any]
        Local setup state with parameters.
    num_micrographs : int
        Total number of micrographs.
    refine_anisotropic_mag : bool
        Whether to refine anisotropic magnification.
    refine_beam_tilt : bool
        Whether to refine beam tilt.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary of collected gradients.
    """
    grads = {}
    if refine_anisotropic_mag or refine_beam_tilt:
        # Check if computation graph is still intact
        if micrograph_loss.requires_grad:
            # Compute gradients for dataset-level parameters using the final loss
            micrograph_loss_scalar = get_loss_scalar_tensor(micrograph_loss)
            scaled_loss = micrograph_loss_scalar / num_micrographs

            # Zero any existing gradients first
            if refine_anisotropic_mag and local_setup["anistropic_mag"].requires_grad:
                if local_setup["anistropic_mag"].grad is not None:
                    local_setup["anistropic_mag"].grad.zero_()
            if refine_beam_tilt:
                asinh_z31c = local_setup["odd_zernikes"]["asinh_Z31c"]
                if asinh_z31c.requires_grad and asinh_z31c.grad is not None:
                    asinh_z31c.grad.zero_()
                asinh_z31s = local_setup["odd_zernikes"]["asinh_Z31s"]
                if asinh_z31s.requires_grad and asinh_z31s.grad is not None:
                    asinh_z31s.grad.zero_()

            # Compute gradients
            scaled_loss.backward(retain_graph=False)

            # Collect gradients
            if refine_anisotropic_mag and local_setup["anistropic_mag"].requires_grad:
                aniso_grad = local_setup["anistropic_mag"].grad
                if aniso_grad is not None:
                    grads["anistropic_mag"] = aniso_grad.detach().clone().cpu()
                    local_setup["anistropic_mag"].grad = None

            if refine_beam_tilt:
                asinh_z31c = local_setup["odd_zernikes"]["asinh_Z31c"]
                if asinh_z31c.requires_grad and asinh_z31c.grad is not None:
                    grads["asinh_Z31c"] = asinh_z31c.grad.detach().clone().cpu()
                    asinh_z31c.grad = None
                asinh_z31s = local_setup["odd_zernikes"]["asinh_Z31s"]
                if asinh_z31s.requires_grad and asinh_z31s.grad is not None:
                    grads["asinh_Z31s"] = asinh_z31s.grad.detach().clone().cpu()
                    asinh_z31s.grad = None

    return grads

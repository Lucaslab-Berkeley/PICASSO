"""Utility functions shared between CTF refinement implementations."""

import math
from typing import Any, cast

import numpy as np
import torch
from leopard_em.backend.core_differentiable_refine import core_differentiable_refine
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.pydantic_models.data_structures import ParticleStack
from leopard_em.utils.backend_setup import setup_particle_backend_kwargs


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
        Anisotropic magnification matrix. If provided, used as transform_matrix.
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
        device_list=[str(device)],
    )

    # Change params for optimizable parameters
    if anistropic_mag is not None:
        particle_backend_kwargs["transform_matrix"] = anistropic_mag
    if odd_zernikes is not None:
        # Convert from log space to actual values using exp()
        particle_backend_kwargs["ctf_kwargs"]["odd_zernikes"] = {
            "Z31c": torch.exp(odd_zernikes["log_Z31c"]),
            "Z31s": torch.exp(odd_zernikes["log_Z31s"]),
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
    """Create optimizer for beam tilt (odd Zernike coefficients in log space).

    Parameters
    ----------
    odd_zernikes : dict[str, torch.Tensor]
        Dictionary with "log_Z31c" and "log_Z31s" keys containing log-space parameters.
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
            [odd_zernikes["log_Z31c"], odd_zernikes["log_Z31s"]],
            lr=lr,
            line_search_fn="strong_wolfe",
            max_eval=max_eval,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            [odd_zernikes["log_Z31c"], odd_zernikes["log_Z31s"]], lr=lr
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
        if initial_odd_zernikes is not None:
            # Convert initial values to log space
            # Use torch.clamp to avoid log(0) or log(negative)
            z31c_init = initial_odd_zernikes["Z31c"].to(device)
            z31s_init = initial_odd_zernikes["Z31s"].to(device)
            # Clamp to positive values before taking log
            z31c_init = torch.clamp(z31c_init, min=1e-10)
            z31s_init = torch.clamp(z31s_init, min=1e-10)
            odd_zernikes = {
                "log_Z31c": torch.log(z31c_init).requires_grad_(True),
                "log_Z31s": torch.log(z31s_init).requires_grad_(True),
            }
        else:
            # Initialize to log(1.0) = 0.0, which corresponds to exp(0) = 1.0
            # This is a reasonable starting point that will scale up during optimization
            odd_zernikes = {
                "log_Z31c": torch.tensor(0.0, device=device, requires_grad=True),
                "log_Z31s": torch.tensor(0.0, device=device, requires_grad=True),
            }
        optimizer_beam_tilt = create_optimizer_beam_tilt(
            odd_zernikes, optimizer_type, lr=beam_tilt_lr, max_eval=lbfgs_max_eval
        )
    else:
        if initial_odd_zernikes is not None:
            # For non-refined case, still convert to log space for consistency
            z31c_init = initial_odd_zernikes["Z31c"].to(device)
            z31s_init = initial_odd_zernikes["Z31s"].to(device)
            z31c_init = torch.clamp(z31c_init, min=1e-10)
            z31s_init = torch.clamp(z31s_init, min=1e-10)
            odd_zernikes = {
                "log_Z31c": torch.log(z31c_init),
                "log_Z31s": torch.log(z31s_init),
            }
        else:
            odd_zernikes = {
                "log_Z31c": torch.tensor(0.0, device=device),
                "log_Z31s": torch.tensor(0.0, device=device),
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
            log_z31c_gpu = (
                odd_zernikes["log_Z31c"].to(device).detach().requires_grad_(True)
            )
            log_z31s_gpu = (
                odd_zernikes["log_Z31s"].to(device).detach().requires_grad_(True)
            )
        else:
            log_z31c_gpu = odd_zernikes["log_Z31c"].to(device)
            log_z31s_gpu = odd_zernikes["log_Z31s"].to(device)
        state_copy = {
            "anistropic_mag": aniso_mag_gpu,
            "odd_zernikes": {
                "log_Z31c": log_z31c_gpu,
                "log_Z31s": log_z31s_gpu,
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
            setup_states[gpu_id]["odd_zernikes"]["log_Z31c"].data.copy_(
                odd_zernikes["log_Z31c"].data.to(device)
            )
            setup_states[gpu_id]["odd_zernikes"]["log_Z31s"].data.copy_(
                odd_zernikes["log_Z31s"].data.to(device)
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

    Returns
    -------
    dict[str, Any]
        Final results dictionary.
    """
    # Convert from log space to actual values for final results
    odd_zernikes_log = setup_state["odd_zernikes"]
    return {
        "anistropic_mag": setup_state["anistropic_mag"].detach(),
        "odd_zernikes": {
            "Z31c": torch.exp(odd_zernikes_log["log_Z31c"]).detach(),
            "Z31s": torch.exp(odd_zernikes_log["log_Z31s"]).detach(),
        },
        "astigmatism_angle": astigmatism_angle_all.detach(),
        "astigmatism_magnitude": astigmatism_magnitude_all.detach(),
        "dataset_losses": dataset_losses,
        "micrograph_losses": micrograph_losses,
    }


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
        if "log_Z31c" in grads:
            if odd_zernikes["log_Z31c"].grad is None:
                odd_zernikes["log_Z31c"].grad = grads["log_Z31c"].to(primary_device)
            else:
                odd_zernikes["log_Z31c"].grad.add_(grads["log_Z31c"].to(primary_device))
        if "log_Z31s" in grads:
            if odd_zernikes["log_Z31s"].grad is None:
                odd_zernikes["log_Z31s"].grad = grads["log_Z31s"].to(primary_device)
            else:
                odd_zernikes["log_Z31s"].grad.add_(grads["log_Z31s"].to(primary_device))


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
                log_z31c = local_setup["odd_zernikes"]["log_Z31c"]
                if log_z31c.requires_grad and log_z31c.grad is not None:
                    log_z31c.grad.zero_()
                log_z31s = local_setup["odd_zernikes"]["log_Z31s"]
                if log_z31s.requires_grad and log_z31s.grad is not None:
                    log_z31s.grad.zero_()

            # Compute gradients
            scaled_loss.backward(retain_graph=False)

            # Collect gradients
            if refine_anisotropic_mag and local_setup["anistropic_mag"].requires_grad:
                aniso_grad = local_setup["anistropic_mag"].grad
                if aniso_grad is not None:
                    grads["anistropic_mag"] = aniso_grad.detach().clone().cpu()
                    local_setup["anistropic_mag"].grad = None

            if refine_beam_tilt:
                log_z31c = local_setup["odd_zernikes"]["log_Z31c"]
                if log_z31c.requires_grad and log_z31c.grad is not None:
                    grads["log_Z31c"] = log_z31c.grad.detach().clone().cpu()
                    log_z31c.grad = None
                log_z31s = local_setup["odd_zernikes"]["log_Z31s"]
                if log_z31s.requires_grad and log_z31s.grad is not None:
                    grads["log_Z31s"] = log_z31s.grad.detach().clone().cpu()
                    log_z31s.grad = None

    return grads

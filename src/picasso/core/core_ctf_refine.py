"""Core functions for CTF refinement."""

import math
from typing import Any

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
    # Convert to numpy for eigenvalue calculation
    mag_np = mag_matrix.detach().cpu().numpy()

    print("\nMagnification Matrix:")
    print(mag_np)

    # Calculate eigenvalues and eigenvectors using numpy
    eigenvalues, eigenvectors = np.linalg.eig(mag_np)

    # Get real parts (in case of numerical issues)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Find the index of the largest eigenvalue
    max_idx = np.argmax(np.abs(eigenvalues))
    lambda_max = eigenvalues[max_idx]
    lambda_min = eigenvalues[1 - max_idx]  # The other eigenvalue

    # Get the eigenvector corresponding to the largest eigenvalue
    eigenvector = eigenvectors[:, max_idx]

    # Normalize eigenvector
    norm = np.linalg.norm(eigenvector)
    if norm > 1e-10:
        eigenvector = eigenvector / norm

    # Calculate angle in degrees (axis of stretching)
    angle_deg = math.degrees(math.atan2(eigenvector[1], eigenvector[0]))

    # Calculate anisotropy percentage
    # Anisotropy = (max - min) / mean * 100
    mean_mag = (lambda_max + lambda_min) / 2.0
    if abs(mean_mag) > 1e-10:
        anisotropy_pct = abs(lambda_max - lambda_min) / abs(mean_mag) * 100.0
    else:
        anisotropy_pct = 0.0

    print(f"Axis of stretching: {angle_deg:.2f} degrees")
    print(f"Magnification anisotropy: {anisotropy_pct:.2f}%")
    print()


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def core_refine_ctf(
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    results_files: list[str],
    device: str,
    refine_anisotropic_mag: bool,
    refine_astigmatism: bool,
    refine_beam_tilt: bool,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    batch_size: int,
    per_micrograph_iterations: int,
    per_dataset_iterations: int,
    initial_mag_matrix: torch.Tensor | None = None,
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
    initial_astigmatism_angle: torch.Tensor | None = None,
    initial_astigmatism_magnitude: torch.Tensor | None = None,
    convergence_threshold: float = 0.0005,
) -> dict[str, Any]:
    """
    Refine the CTF parameters for the micrographs.

    Parameters
    ----------
    template : torch.Tensor
        The template volume.
    template_box_size : tuple[int, int]
        The box size of the template.
    results_files : list[str]
        The list of results file paths.
    device : str
        The device to use for computation.
    refine_anisotropic_mag : bool
        Whether to refine the anisotropic magnitude.
    refine_astigmatism : bool
        Whether to refine the astigmatism.
    refine_beam_tilt : bool
        Whether to refine the beam tilt.
    apply_global_filtering : bool
        Whether to apply global filtering to the micrographs.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    batch_size : int
        The batch size to use for computation.
    per_micrograph_iterations : int
        The number of iterations to run per micrograph.
    per_dataset_iterations : int
        The number of iterations to run per dataset.
    initial_mag_matrix: torch.Tensor | None = None,
        The initial mag matrix.
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
        The initial odd zernikes.
    initial_astigmatism_angle: torch.Tensor | None = None,
        The initial astigmatism angle.
    initial_astigmatism_magnitude: torch.Tensor | None = None,
        The initial astigmatism magnitude.
    convergence_threshold: float = 0.0005,
        Convergence threshold. If loss doesn't improve by more than this
        amount for 5 iterations, stop early.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the refined CTF parameters:
        - "anistropic_mag": torch.Tensor
        - "odd_zernikes": dict[str, torch.Tensor]
        - "astigmatism_angle": torch.Tensor
        - "astigmatism_magnitude": torch.Tensor
    """
    torch.set_grad_enabled(True)
    device = torch.device(device)

    # Setup refinement parameters and optimizers
    setup_state = _setup_refinement(
        results_files=results_files,
        device=device,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_beam_tilt=refine_beam_tilt,
        initial_mag_matrix=initial_mag_matrix,
        initial_odd_zernikes=initial_odd_zernikes,
        initial_astigmatism_angle=initial_astigmatism_angle,
        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
    )

    # Run the dataset loop
    return _dataset_loop(
        results_files=results_files,
        template=template,
        template_box_size=template_box_size,
        device=device,
        batch_size=batch_size,
        per_micrograph_iterations=per_micrograph_iterations,
        per_dataset_iterations=per_dataset_iterations,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_astigmatism=refine_astigmatism,
        refine_beam_tilt=refine_beam_tilt,
        apply_global_filtering=apply_global_filtering,
        preprocessing_filters=preprocessing_filters,
        convergence_threshold=convergence_threshold,
        initial_astigmatism_angle=initial_astigmatism_angle,
        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
        setup_state=setup_state,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def _setup_refinement(
    results_files: list[str],
    device: torch.device,
    refine_anisotropic_mag: bool,
    refine_beam_tilt: bool,
    initial_mag_matrix: torch.Tensor | None = None,
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
    initial_astigmatism_angle: torch.Tensor | None = None,
    initial_astigmatism_magnitude: torch.Tensor | None = None,
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
        optimizer_aniso = torch.optim.Adam([anistropic_mag], lr=0.005)
    else:
        if initial_mag_matrix is not None:
            anistropic_mag = initial_mag_matrix.to(device)
        else:
            anistropic_mag = torch.eye(2, 2, device=device)

    if refine_beam_tilt:
        if initial_odd_zernikes is not None:
            odd_zernikes = {
                "Z31c": initial_odd_zernikes["Z31c"].to(device).requires_grad_(True),
                "Z31s": initial_odd_zernikes["Z31s"].to(device).requires_grad_(True),
            }
        else:
            odd_zernikes = {
                "Z31c": torch.tensor(0.0, device=device, requires_grad=True),
                "Z31s": torch.tensor(0.0, device=device, requires_grad=True),
            }
        optimizer_beam_tilt = torch.optim.Adam(
            [odd_zernikes["Z31c"], odd_zernikes["Z31s"]], lr=1.0
        )
    else:
        if initial_odd_zernikes is not None:
            odd_zernikes = {
                "Z31c": initial_odd_zernikes["Z31c"].to(device),
                "Z31s": initial_odd_zernikes["Z31s"].to(device),
            }
        else:
            odd_zernikes = {
                "Z31c": torch.tensor(0.0, device=device),
                "Z31s": torch.tensor(0.0, device=device),
            }

    # Validate initial astigmatism parameters if provided
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def _micrograph_loop(
    i: int,
    results_path: str,
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
    per_micrograph_iterations: int,
    outer_iteration: int,
    refine_anisotropic_mag: bool,
    refine_astigmatism: bool,
    refine_beam_tilt: bool,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    convergence_threshold: float,
    num_results_files: int,
    initial_astigmatism_angle: torch.Tensor | None,
    initial_astigmatism_magnitude: torch.Tensor | None,
    setup_state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Process a single micrograph through inner iterations.

    Parameters
    ----------
    i : int
        Micrograph index.
    results_path : str
        Path to results file for this micrograph.
    template : torch.Tensor
        The template volume.
    template_box_size : tuple[int, int]
        The box size of the template.
    device : torch.device
        The device to use for computation.
    batch_size : int
        The batch size to use for computation.
    per_micrograph_iterations : int
        The number of iterations to run per micrograph.
    outer_iteration : int
        Current outer iteration number.
    refine_anisotropic_mag : bool
        Whether to refine the anisotropic magnitude.
    refine_astigmatism : bool
        Whether to refine the astigmatism.
    refine_beam_tilt : bool
        Whether to refine the beam tilt.
    apply_global_filtering : bool
        Whether to apply global filtering to the micrographs.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    convergence_threshold : float
        Convergence threshold for early stopping.
    num_results_files : int
        Total number of micrographs.
    initial_astigmatism_angle : torch.Tensor | None
        Initial astigmatism angle tensor.
    initial_astigmatism_magnitude : torch.Tensor | None
        Initial astigmatism magnitude tensor.
    setup_state : dict[str, Any]
        Setup state from _setup_refinement.

    Returns
    -------
    tuple[torch.Tensor, dict[str, Any]]
        Micrograph loss and updated astigmatism values.
    """
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]
    pixel_size_offsets = setup_state["pixel_size_offsets"]
    euler_angle_offsets = setup_state["euler_angle_offsets"]
    defocus_offsets = setup_state["defocus_offsets"]

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

    # Initialize astigmatism parameters for this micrograph
    optimizer_astig = None
    if refine_astigmatism:
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
            print(
                f"[SHAPE DEBUG] initial_defocus_mean shape: "
                f"{initial_defocus_mean.shape}"
            )
        else:
            # Compute initial magnitude from mean difference across particles
            # (astigmatism magnitude is same for all particles in micrograph)
            initial_magnitude_scalar = (defocus_u_val - defocus_v_val).mean().item()
            initial_magnitude = torch.tensor(initial_magnitude_scalar, device=device)
            # Compute defocus_mean from full tensors (per particle)
            initial_defocus_mean = (defocus_u_val + defocus_v_val) / 2.0
            print(f"[SHAPE DEBUG] initial_magnitude (scalar): {initial_magnitude}")
            print(
                f"[SHAPE DEBUG] initial_defocus_mean shape: "
                f"{initial_defocus_mean.shape}"
            )

        # Create tensors with requires_grad, handling both tensor and scalar inputs
        if isinstance(initial_angle, torch.Tensor):
            astigmatism_angle = initial_angle.detach().clone().requires_grad_(True)
        else:
            astigmatism_angle = torch.tensor(
                initial_angle, device=device, requires_grad=True
            )
        if isinstance(initial_magnitude, torch.Tensor):
            astigmatism_magnitude = (
                initial_magnitude.detach().clone().requires_grad_(True)
            )
        else:
            astigmatism_magnitude = torch.tensor(
                initial_magnitude, device=device, requires_grad=True
            )

        optimizer_astig = torch.optim.Adam(
            [
                {"params": [astigmatism_angle], "lr": 1.0},
                {"params": [astigmatism_magnitude], "lr": 5.0},
            ]
        )
    else:
        defocus_angle_val = particle_backend_kwargs["defocus_angle"]
        # Extract first element if tensor (same for all particles)
        if (
            isinstance(defocus_angle_val, torch.Tensor)
            and defocus_angle_val.numel() > 1
        ):
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
        initial_defocus_mean = None

    # Change params for optimizable parameters
    if refine_anisotropic_mag:
        particle_backend_kwargs["transform_matrix"] = anistropic_mag
    if refine_beam_tilt:
        particle_backend_kwargs["ctf_kwargs"]["odd_zernikes"] = odd_zernikes

    # Only run inner loop (per-micrograph optimizer) on iteration 0
    # On later iterations, just compute loss once for dataset optimizers
    inner_iterations = per_micrograph_iterations if outer_iteration == 0 else 1
    # Track micrograph loss history for inner loop convergence
    micrograph_loss_history: list[float] = []
    for inner_iter in range(inner_iterations):
        # Update defocus_u and defocus_v from astigmatism parameters if optimizing
        if refine_astigmatism:
            # Broadcast astigmatism_magnitude (scalar) to match initial_defocus_mean
            # shape
            defocus_u = initial_defocus_mean + astigmatism_magnitude / 2.0
            defocus_v = initial_defocus_mean - astigmatism_magnitude / 2.0
            # Broadcast astigmatism_angle to match defocus_u/defocus_v shape
            # (one value per particle)
            num_particles = defocus_u.shape[0]
            astigmatism_angle_broadcast = astigmatism_angle.expand(num_particles)
            # Pass tensors directly to preserve gradients
            particle_backend_kwargs["defocus_u"] = defocus_u
            particle_backend_kwargs["defocus_v"] = defocus_v
            particle_backend_kwargs["defocus_angle"] = astigmatism_angle_broadcast

        result = core_differentiable_refine(
            batch_size=batch_size,
            num_cuda_streams=1,
            **particle_backend_kwargs,
        )
        refined_scaled_mip = result["refined_z_score"]
        micrograph_loss = -refined_scaled_mip

        # Print micrograph loss
        if micrograph_loss.numel() > 1:
            loss_val = micrograph_loss.mean().item()
        else:
            if hasattr(micrograph_loss, "item"):
                loss_val = micrograph_loss.item()
            else:
                loss_val = float(micrograph_loss)
        print(
            f"  Micrograph {i + 1}/{num_results_files}, "
            f"inner iter {inner_iter + 1}/{inner_iterations}, "
            f"loss: {loss_val:.6f}"
        )

        # Check for convergence based on micrograph loss (end of inner loop)
        if refine_astigmatism and outer_iteration == 0:
            micrograph_loss_history.append(loss_val)
            if len(micrograph_loss_history) >= 6:
                # Check if absolute change has been less than threshold
                # for the last 5 consecutive iterations
                converged = True
                hist_len = len(micrograph_loss_history)
                for j in range(hist_len - 5, hist_len - 1):
                    abs_change = abs(
                        micrograph_loss_history[j + 1] - micrograph_loss_history[j]
                    )
                    if abs_change > convergence_threshold:
                        converged = False
                        break

                if converged:
                    last_change = abs(
                        micrograph_loss_history[-1] - micrograph_loss_history[-2]
                    )
                    print(
                        f"  Micrograph {i + 1} converged: absolute change "
                        f"in micrograph loss ({last_change:.6f}) <= "
                        f"threshold ({convergence_threshold:.6f}) for 5 "
                        f"consecutive iterations. Stopping inner loop early."
                    )
                    break
                # Keep only last 6 values for efficiency
                micrograph_loss_history = micrograph_loss_history[-6:]

        # Optimize astigmatism parameters (per micrograph, inner loop)
        # Only optimize on iteration 0
        if refine_astigmatism and outer_iteration == 0 and optimizer_astig is not None:
            # Ensure micrograph_loss is a scalar for backward()
            if micrograph_loss.numel() > 1:
                micrograph_loss_scalar = micrograph_loss.mean()
            else:
                micrograph_loss_scalar = micrograph_loss
            optimizer_astig.zero_grad()
            # Retain graph if we need gradients for dataset-level optimizers
            retain_graph = refine_anisotropic_mag or refine_beam_tilt
            micrograph_loss_scalar.backward(retain_graph=retain_graph)
            optimizer_astig.step()
            # Print astigmatism parameters after optimization
            if hasattr(astigmatism_angle, "item"):
                angle_val = astigmatism_angle.item()
            else:
                angle_val = float(astigmatism_angle)
            if hasattr(astigmatism_magnitude, "item"):
                mag_val = astigmatism_magnitude.item()
            else:
                mag_val = float(astigmatism_magnitude)
            print(
                f"    Micrograph {i + 1} astigmatism: "
                f"angle={angle_val:.2f}°, "
                f"magnitude={mag_val:.2f} Å"
            )

    # Store final astigmatism values
    astigmatism_result = {}
    if refine_astigmatism:
        astigmatism_result["angle"] = astigmatism_angle.detach()
        astigmatism_result["magnitude"] = astigmatism_magnitude.detach()
    else:
        astigmatism_result["angle"] = astigmatism_angle
        astigmatism_result["magnitude"] = astigmatism_magnitude

    return micrograph_loss, astigmatism_result


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def _dataset_loop(
    results_files: list[str],
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
    per_micrograph_iterations: int,
    per_dataset_iterations: int,
    refine_anisotropic_mag: bool,
    refine_astigmatism: bool,
    refine_beam_tilt: bool,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    convergence_threshold: float,
    initial_astigmatism_angle: torch.Tensor | None,
    initial_astigmatism_magnitude: torch.Tensor | None,
    setup_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the outer dataset loop for dataset-level optimization.

    Parameters
    ----------
    results_files : list[str]
        The list of results file paths.
    template : torch.Tensor
        The template volume.
    template_box_size : tuple[int, int]
        The box size of the template.
    device : torch.device
        The device to use for computation.
    batch_size : int
        The batch size to use for computation.
    per_micrograph_iterations : int
        The number of iterations to run per micrograph.
    per_dataset_iterations : int
        The number of iterations to run per dataset.
    refine_anisotropic_mag : bool
        Whether to refine the anisotropic magnitude.
    refine_astigmatism : bool
        Whether to refine the astigmatism.
    refine_beam_tilt : bool
        Whether to refine the beam tilt.
    apply_global_filtering : bool
        Whether to apply global filtering to the micrographs.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    convergence_threshold : float
        Convergence threshold for early stopping.
    initial_astigmatism_angle : torch.Tensor | None
        Initial astigmatism angle tensor.
    initial_astigmatism_magnitude : torch.Tensor | None
        Initial astigmatism magnitude tensor.
    setup_state : dict[str, Any]
        Setup state from _setup_refinement.

    Returns
    -------
    dict[str, Any]
        Refined CTF parameters and loss tracking data.
    """
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]
    astigmatism_angle_all = setup_state["astigmatism_angle_all"]
    astigmatism_magnitude_all = setup_state["astigmatism_magnitude_all"]
    optimizer_aniso = setup_state["optimizer_aniso"]
    optimizer_beam_tilt = setup_state["optimizer_beam_tilt"]

    # Track loss history for convergence checking
    loss_history: list[float] = []
    # Track losses for CSV output
    dataset_losses: list[tuple[int, float]] = []
    micrograph_losses: dict[int, list[tuple[int, float]]] = {}

    for outer_iteration in range(per_dataset_iterations):
        dataset_loss: torch.Tensor | None = None
        # Loop through results files
        for i, results_path in enumerate(results_files):
            micrograph_loss, astigmatism_result = _micrograph_loop(
                i=i,
                results_path=results_path,
                template=template,
                template_box_size=template_box_size,
                device=device,
                batch_size=batch_size,
                per_micrograph_iterations=per_micrograph_iterations,
                outer_iteration=outer_iteration,
                refine_anisotropic_mag=refine_anisotropic_mag,
                refine_astigmatism=refine_astigmatism,
                refine_beam_tilt=refine_beam_tilt,
                apply_global_filtering=apply_global_filtering,
                preprocessing_filters=preprocessing_filters,
                convergence_threshold=convergence_threshold,
                num_results_files=len(results_files),
                initial_astigmatism_angle=initial_astigmatism_angle,
                initial_astigmatism_magnitude=initial_astigmatism_magnitude,
                setup_state=setup_state,
            )

            # Store final astigmatism values
            astigmatism_angle_all[i] = astigmatism_result["angle"]
            astigmatism_magnitude_all[i] = astigmatism_result["magnitude"]

            # Track micrograph loss for CSV output
            if i not in micrograph_losses:
                micrograph_losses[i] = []
            if micrograph_loss.numel() > 1:
                loss_val = micrograph_loss.mean().item()
            else:
                if hasattr(micrograph_loss, "item"):
                    loss_val = micrograph_loss.item()
                else:
                    loss_val = float(micrograph_loss)
            micrograph_losses[i].append((outer_iteration, loss_val))

            # Accumulate loss for dataset-level optimizers
            # Need to keep gradients for dataset-level parameters
            if refine_anisotropic_mag or refine_beam_tilt:
                # Ensure micrograph_loss is a scalar
                if micrograph_loss.numel() > 1:
                    micrograph_loss = micrograph_loss.mean()
                if dataset_loss is None:
                    dataset_loss = micrograph_loss
                else:
                    dataset_loss = dataset_loss + micrograph_loss
                # Print micrograph loss
                print(f"  Micrograph {i + 1}/{len(results_files)} loss: {loss_val:.6f}")

        # Optimize dataset-level parameters (outer loop)
        if (
            refine_anisotropic_mag
            and dataset_loss is not None
            and optimizer_aniso is not None
        ):
            optimizer_aniso.zero_grad()
            dataset_loss.backward()
            optimizer_aniso.step()

        if (
            refine_beam_tilt
            and dataset_loss is not None
            and optimizer_beam_tilt is not None
        ):
            optimizer_beam_tilt.zero_grad()
            dataset_loss.backward()
            optimizer_beam_tilt.step()
            # Print Zernike coefficients after optimization
            if hasattr(odd_zernikes["Z31c"], "item"):
                z31c_val = odd_zernikes["Z31c"].item()
            else:
                z31c_val = float(odd_zernikes["Z31c"])
            if hasattr(odd_zernikes["Z31s"], "item"):
                z31s_val = odd_zernikes["Z31s"].item()
            else:
                z31s_val = float(odd_zernikes["Z31s"])
            print(f"Zernike coefficients: Z31c={z31c_val:.6f}, Z31s={z31s_val:.6f}")

        # Print dataset-level loss after each outer iteration
        if dataset_loss is not None:
            if hasattr(dataset_loss, "item"):
                loss_val = dataset_loss.item()
            else:
                loss_val = float(dataset_loss)
            print(
                f"Dataset iteration {outer_iteration + 1}/"
                f"{per_dataset_iterations} loss: {loss_val:.6f}"
            )
            # Track dataset loss for CSV output
            dataset_losses.append((outer_iteration, loss_val))

            # Check for convergence based on dataset loss (end of outer loop)
            loss_history.append(loss_val)
            if len(loss_history) >= 6:  # Need at least 6 to check last 5 changes
                # Check if absolute change has been less than threshold
                # for the last 5 consecutive iterations
                converged = True
                for j in range(len(loss_history) - 5, len(loss_history) - 1):
                    abs_change = abs(loss_history[j + 1] - loss_history[j])
                    if abs_change > convergence_threshold:
                        converged = False
                        break

                if converged:
                    last_change = abs(loss_history[-1] - loss_history[-2])
                    print(
                        f"Converged: absolute change in dataset loss "
                        f"({last_change:.6f}) <= threshold "
                        f"({convergence_threshold:.6f}) for 5 consecutive "
                        f"iterations. Stopping early."
                    )
                    break
                # Keep only last 6 values for efficiency
                loss_history = loss_history[-6:]

        # Output magnification matrix information
        if refine_anisotropic_mag:
            output_mag_matrix(anistropic_mag)

    return {
        "anistropic_mag": anistropic_mag.detach(),
        "odd_zernikes": {
            "Z31c": odd_zernikes["Z31c"].detach(),
            "Z31s": odd_zernikes["Z31s"].detach(),
        },
        "astigmatism_angle": astigmatism_angle_all.detach(),
        "astigmatism_magnitude": astigmatism_magnitude_all.detach(),
        "dataset_losses": dataset_losses,
        "micrograph_losses": micrograph_losses,
    }

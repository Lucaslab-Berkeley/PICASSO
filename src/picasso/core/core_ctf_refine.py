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
    # Initialize parameters with gradients for optimization
    # beam tilt x and y (Zernike polynomials)
    # astigmatism optimized per micrograph so later
    # anisotropic mag
    device = torch.device(device)
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
            [odd_zernikes["Z31c"], odd_zernikes["Z31s"]], lr=0.01
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

    for outer_iteration in range(per_dataset_iterations):
        dataset_loss: torch.Tensor | None = None
        # Loop through results files
        for i, results_path in enumerate(results_files):
            # load particle_stack
            particle_stack = ParticleStack(
                df_path=results_path,
                extracted_box_size=template_box_size,
                original_template_size=template_box_size,
            )
            # setup euler angles
            euler_angles = particle_stack.get_euler_angles(
                prefer_refined_angles=True
            ).to(device)
            # setup particle backend kwargs
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

            # Initialize astigmatism parameters for this micrograph
            optimizer_astig = None
            if refine_astigmatism:
                # Get initial values from provided initial tensors or
                # particle_backend_kwargs
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

                if initial_astigmatism_magnitude is not None:
                    initial_magnitude = initial_astigmatism_magnitude[i].to(device)
                    # Need to compute defocus_mean from initial values
                    defocus_u_val = particle_backend_kwargs["defocus_u"]
                    defocus_v_val = particle_backend_kwargs["defocus_v"]
                    # Extract first element if tensor
                    # (difference is same for all particles)
                    if (
                        isinstance(defocus_u_val, torch.Tensor)
                        and defocus_u_val.numel() > 1
                    ):
                        defocus_u_val = defocus_u_val[0].item()
                    if (
                        isinstance(defocus_v_val, torch.Tensor)
                        and defocus_v_val.numel() > 1
                    ):
                        defocus_v_val = defocus_v_val[0].item()
                    initial_defocus_u = torch.tensor(defocus_u_val, device=device)
                    initial_defocus_v = torch.tensor(defocus_v_val, device=device)
                    initial_defocus_mean = (initial_defocus_u + initial_defocus_v) / 2.0
                else:
                    defocus_u_val = particle_backend_kwargs["defocus_u"]
                    defocus_v_val = particle_backend_kwargs["defocus_v"]
                    # Extract first element if tensor
                    # (difference is same for all particles)
                    if (
                        isinstance(defocus_u_val, torch.Tensor)
                        and defocus_u_val.numel() > 1
                    ):
                        defocus_u_val = defocus_u_val[0].item()
                    if (
                        isinstance(defocus_v_val, torch.Tensor)
                        and defocus_v_val.numel() > 1
                    ):
                        defocus_v_val = defocus_v_val[0].item()
                    initial_defocus_u = torch.tensor(defocus_u_val, device=device)
                    initial_defocus_v = torch.tensor(defocus_v_val, device=device)
                    initial_magnitude = initial_defocus_u - initial_defocus_v
                    initial_defocus_mean = (initial_defocus_u + initial_defocus_v) / 2.0

                astigmatism_angle = torch.tensor(
                    initial_angle, device=device, requires_grad=True
                )
                astigmatism_magnitude = torch.tensor(
                    initial_magnitude, device=device, requires_grad=True
                )

                optimizer_astig = torch.optim.Adam(
                    [astigmatism_angle, astigmatism_magnitude], lr=0.5
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
                # Extract first element if tensor
                # (difference is same for all particles)
                if (
                    isinstance(defocus_u_val, torch.Tensor)
                    and defocus_u_val.numel() > 1
                ):
                    defocus_u_val = defocus_u_val[0].item()
                if (
                    isinstance(defocus_v_val, torch.Tensor)
                    and defocus_v_val.numel() > 1
                ):
                    defocus_v_val = defocus_v_val[0].item()
                defocus_u = torch.tensor(defocus_u_val, device=device)
                defocus_v = torch.tensor(defocus_v_val, device=device)
                astigmatism_magnitude = defocus_u - defocus_v

            # change params for optimizable parameters
            if refine_anisotropic_mag:
                particle_backend_kwargs["transform_matrix"] = anistropic_mag
            if refine_beam_tilt:
                particle_backend_kwargs["odd_zernikes"] = odd_zernikes

            # Only run inner loop (per-micrograph optimizer) on iteration 0
            # On later iterations, just compute loss once for dataset optimizers
            inner_iterations = per_micrograph_iterations if outer_iteration == 0 else 1
            for _ in range(inner_iterations):
                # Update defocus_u and defocus_v from astigmatism parameters
                # if optimizing
                if refine_astigmatism:
                    defocus_u = initial_defocus_mean + astigmatism_magnitude / 2.0
                    defocus_v = initial_defocus_mean - astigmatism_magnitude / 2.0
                    # Pass tensors directly to preserve gradients
                    # The backend should handle tensor inputs
                    particle_backend_kwargs["defocus_u"] = defocus_u
                    particle_backend_kwargs["defocus_v"] = defocus_v
                    particle_backend_kwargs["defocus_angle"] = astigmatism_angle

                result = core_differentiable_refine(
                    batch_size=batch_size,
                    num_cuda_streams=1,
                    **particle_backend_kwargs,
                )
                refined_scaled_mip = result["refined_z_score"]
                micrograph_loss = -refined_scaled_mip

                # Optimize astigmatism parameters (per micrograph, inner loop)
                # Only optimize on iteration 0
                if (
                    refine_astigmatism
                    and outer_iteration == 0
                    and optimizer_astig is not None
                ):
                    optimizer_astig.zero_grad()
                    # Retain graph if we need gradients for dataset-level optimizers
                    retain_graph = refine_anisotropic_mag or refine_beam_tilt
                    micrograph_loss.backward(retain_graph=retain_graph)
                    optimizer_astig.step()

            # Store final astigmatism values
            if refine_astigmatism:
                astigmatism_angle_all[i] = astigmatism_angle.detach()
                astigmatism_magnitude_all[i] = astigmatism_magnitude.detach()
            else:
                astigmatism_angle_all[i] = astigmatism_angle
                astigmatism_magnitude_all[i] = astigmatism_magnitude

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
                if hasattr(micrograph_loss, "item"):
                    loss_val = micrograph_loss.item()
                else:
                    loss_val = float(micrograph_loss)
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
    }

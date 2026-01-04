"""Core functions for CTF refinement with multi-GPU support."""
# pylint: disable=too-many-lines

import math
from queue import Queue
from threading import Thread
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def core_refine_ctf(
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    results_files: list[str],
    device_list: list[torch.device],
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
    device_list : list[torch.device]
        List of device objects (e.g., [torch.device("cuda:0"), torch.device("cuda:1")]).
        Results files are distributed across these GPUs.
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

    # Use first device as primary device for shared parameters
    primary_device = device_list[0]

    # Setup refinement parameters and optimizers
    setup_state = _setup_refinement(
        results_files=results_files,
        device=primary_device,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_beam_tilt=refine_beam_tilt,
        initial_mag_matrix=initial_mag_matrix,
        initial_odd_zernikes=initial_odd_zernikes,
        initial_astigmatism_angle=initial_astigmatism_angle,
        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
    )

    # Run the dataset loop with multiple devices
    return _dataset_loop(
        results_files=results_files,
        template=template,
        template_box_size=template_box_size,
        device_list=device_list,
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
def _dataset_loop(
    results_files: list[str],
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    device_list: list[torch.device],
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
    device_list : list[torch.device]
        The devices to use for computation.
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
    primary_device = device_list[0]
    num_gpus = len(device_list)

    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]
    astigmatism_angle_all = setup_state["astigmatism_angle_all"]
    astigmatism_magnitude_all = setup_state["astigmatism_magnitude_all"]
    optimizer_aniso = setup_state["optimizer_aniso"]
    optimizer_beam_tilt = setup_state["optimizer_beam_tilt"]

    loss_history: list[float] = []
    dataset_losses: list[tuple[int, float]] = []
    micrograph_losses: dict[int, list[tuple[int, float]]] = {}
    num_micrographs = len(results_files)

    # Distribute micrographs across GPUs
    gpu_assignments: list[list[int]] = [[] for _ in range(num_gpus)]
    for idx, mic_idx in enumerate(range(num_micrographs)):
        gpu_assignments[idx % num_gpus].append(mic_idx)

    print(f"\nUsing {num_gpus} GPUs:")
    for gpu_id, indices in enumerate(gpu_assignments):
        print(f"  GPU {gpu_id} ({device_list[gpu_id]}): {len(indices)} micrographs")

    # Create replicas of template on each GPU
    templates = [template.to(device) for device in device_list]

    # Create setup_state copies for each GPU
    setup_states = []
    for device in device_list:
        if refine_anisotropic_mag:
            aniso_mag_gpu = anistropic_mag.to(device).detach().requires_grad_(True)
        else:
            aniso_mag_gpu = anistropic_mag.to(device)
        if refine_beam_tilt:
            z31c_gpu = odd_zernikes["Z31c"].to(device).detach().requires_grad_(True)
            z31s_gpu = odd_zernikes["Z31s"].to(device).detach().requires_grad_(True)
        else:
            z31c_gpu = odd_zernikes["Z31c"].to(device)
            z31s_gpu = odd_zernikes["Z31s"].to(device)
        state_copy = {
            "anistropic_mag": aniso_mag_gpu,
            "odd_zernikes": {
                "Z31c": z31c_gpu,
                "Z31s": z31s_gpu,
            },
            "pixel_size_offsets": setup_state["pixel_size_offsets"].to(device),
            "euler_angle_offsets": setup_state["euler_angle_offsets"].to(device),
            "defocus_offsets": setup_state["defocus_offsets"].to(device),
            "optimizer_aniso": None,
            "optimizer_beam_tilt": None,
        }
        setup_states.append(state_copy)

    for outer_iteration in range(per_dataset_iterations):
        dataset_loss_value: float = 0.0

        if refine_anisotropic_mag and optimizer_aniso is not None:
            optimizer_aniso.zero_grad()
        if refine_beam_tilt and optimizer_beam_tilt is not None:
            optimizer_beam_tilt.zero_grad()

        if num_gpus > 1:
            # Use threading with a work queue for each GPU
            result_queue: Queue[dict[str, Any]] = Queue()
            work_queues: list[Queue[int | None]] = [Queue() for _ in range(num_gpus)]

            # Distribute work to queues
            for gpu_id, indices in enumerate(gpu_assignments):
                for idx in indices:
                    work_queues[gpu_id].put(idx)
                # Add sentinel to signal end
                work_queues[gpu_id].put(None)

            # Worker function that processes from its queue
            # pylint: disable=too-many-locals,too-many-branches,too-many-statements
            def gpu_worker(
                gpu_id: int,
                work_queue: Queue[int | None],
                iter_val: int = outer_iteration,
                queue: Queue[dict[str, Any]] = result_queue,
            ) -> None:
                device = device_list[gpu_id]
                local_template = templates[gpu_id]
                local_setup = setup_states[gpu_id]

                while True:
                    i = work_queue.get()
                    if i is None:  # Sentinel value
                        break

                    results_path = results_files[i]

                    micrograph_loss, astigmatism_result = _micrograph_loop(
                        i=i,
                        results_path=results_path,
                        template=local_template,
                        template_box_size=template_box_size,
                        device=device,
                        batch_size=batch_size,
                        per_micrograph_iterations=per_micrograph_iterations,
                        outer_iteration=iter_val,
                        refine_anisotropic_mag=refine_anisotropic_mag,
                        refine_astigmatism=refine_astigmatism,
                        refine_beam_tilt=refine_beam_tilt,
                        apply_global_filtering=apply_global_filtering,
                        preprocessing_filters=preprocessing_filters,
                        convergence_threshold=convergence_threshold,
                        num_results_files=num_micrographs,
                        initial_astigmatism_angle=initial_astigmatism_angle,
                        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
                        setup_state=local_setup,
                    )

                    if micrograph_loss.numel() > 1:
                        loss_val = micrograph_loss.mean().item()
                    elif hasattr(micrograph_loss, "item"):
                        loss_val = micrograph_loss.item()
                    else:
                        loss_val = float(micrograph_loss)

                    print(
                        f"  GPU {gpu_id}: Micrograph {i + 1}/{num_micrographs} "
                        f"loss: {loss_val:.6f}"
                    )

                    # Collect gradients from GPU-local parameters for
                    # dataset-level optimization. _micrograph_loop may have
                    # already computed gradients if retain_graph=True was used.
                    # We need to compute them here to ensure they're available
                    # for the final loss value.
                    grads = {}
                    if refine_anisotropic_mag or refine_beam_tilt:
                        # Check if computation graph is still intact
                        if micrograph_loss.requires_grad:
                            # Compute gradients for dataset-level parameters
                            # using the final loss
                            if micrograph_loss.numel() > 1:
                                micrograph_loss_scalar = micrograph_loss.mean()
                            else:
                                micrograph_loss_scalar = micrograph_loss
                            scaled_loss = micrograph_loss_scalar / num_micrographs

                            # Zero any existing gradients first
                            if (
                                refine_anisotropic_mag
                                and local_setup["anistropic_mag"].requires_grad
                            ):
                                if local_setup["anistropic_mag"].grad is not None:
                                    local_setup["anistropic_mag"].grad.zero_()
                            if refine_beam_tilt:
                                z31c = local_setup["odd_zernikes"]["Z31c"]
                                if z31c.requires_grad and z31c.grad is not None:
                                    z31c.grad.zero_()
                                z31s = local_setup["odd_zernikes"]["Z31s"]
                                if z31s.requires_grad and z31s.grad is not None:
                                    z31s.grad.zero_()

                            # Compute gradients
                            scaled_loss.backward(retain_graph=False)

                            # Collect gradients
                            if (
                                refine_anisotropic_mag
                                and local_setup["anistropic_mag"].requires_grad
                            ):
                                aniso_grad = local_setup["anistropic_mag"].grad
                                if aniso_grad is not None:
                                    grads["anistropic_mag"] = (
                                        aniso_grad.detach().clone().cpu()
                                    )
                                    local_setup["anistropic_mag"].grad = None

                            if refine_beam_tilt:
                                z31c = local_setup["odd_zernikes"]["Z31c"]
                                if z31c.requires_grad and z31c.grad is not None:
                                    grads["Z31c"] = z31c.grad.detach().clone().cpu()
                                    z31c.grad = None
                                z31s = local_setup["odd_zernikes"]["Z31s"]
                                if z31s.requires_grad and z31s.grad is not None:
                                    grads["Z31s"] = z31s.grad.detach().clone().cpu()
                                    z31s.grad = None

                    queue.put(
                        {
                            "index": i,
                            "loss": loss_val,
                            "astigmatism_angle": astigmatism_result["angle"],
                            "astigmatism_magnitude": astigmatism_result["magnitude"],
                            "grads": grads,
                        }
                    )

                    # Free memory
                    del micrograph_loss
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # Start all worker threads
            threads = []
            for gpu_id in range(num_gpus):
                thread = Thread(target=gpu_worker, args=(gpu_id, work_queues[gpu_id]))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Collect all results
            all_results = []
            while not result_queue.empty():
                all_results.append(result_queue.get())

            # Sort by index to maintain order
            all_results.sort(key=lambda x: x["index"])

            # Process results and accumulate gradients
            for result in all_results:
                i = result["index"]
                loss_val = result["loss"]

                astigmatism_angle_all[i] = result["astigmatism_angle"].to(
                    primary_device
                )
                astigmatism_magnitude_all[i] = result["astigmatism_magnitude"].to(
                    primary_device
                )

                if i not in micrograph_losses:
                    micrograph_losses[i] = []
                micrograph_losses[i].append((outer_iteration, loss_val))
                dataset_loss_value += loss_val

                # Accumulate gradients from GPU-local parameters to master parameters
                grads = result.get("grads", {})
                if refine_anisotropic_mag and "anistropic_mag" in grads:
                    if anistropic_mag.grad is None:
                        anistropic_mag.grad = grads["anistropic_mag"].to(primary_device)
                    else:
                        anistropic_mag.grad.add_(
                            grads["anistropic_mag"].to(primary_device)
                        )

                if refine_beam_tilt:
                    if "Z31c" in grads:
                        if odd_zernikes["Z31c"].grad is None:
                            odd_zernikes["Z31c"].grad = grads["Z31c"].to(primary_device)
                        else:
                            odd_zernikes["Z31c"].grad.add_(
                                grads["Z31c"].to(primary_device)
                            )
                    if "Z31s" in grads:
                        if odd_zernikes["Z31s"].grad is None:
                            odd_zernikes["Z31s"].grad = grads["Z31s"].to(primary_device)
                        else:
                            odd_zernikes["Z31s"].grad.add_(
                                grads["Z31s"].to(primary_device)
                            )

        else:
            # Single GPU path
            for i, results_path in enumerate(results_files):
                micrograph_loss, astigmatism_result = _micrograph_loop(
                    i=i,
                    results_path=results_path,
                    template=template,
                    template_box_size=template_box_size,
                    device=primary_device,
                    batch_size=batch_size,
                    per_micrograph_iterations=per_micrograph_iterations,
                    outer_iteration=outer_iteration,
                    refine_anisotropic_mag=refine_anisotropic_mag,
                    refine_astigmatism=refine_astigmatism,
                    refine_beam_tilt=refine_beam_tilt,
                    apply_global_filtering=apply_global_filtering,
                    preprocessing_filters=preprocessing_filters,
                    convergence_threshold=convergence_threshold,
                    num_results_files=num_micrographs,
                    initial_astigmatism_angle=initial_astigmatism_angle,
                    initial_astigmatism_magnitude=initial_astigmatism_magnitude,
                    setup_state=setup_state,
                )

                astigmatism_angle_all[i] = astigmatism_result["angle"]
                astigmatism_magnitude_all[i] = astigmatism_result["magnitude"]

                if i not in micrograph_losses:
                    micrograph_losses[i] = []

                if micrograph_loss.numel() > 1:
                    loss_val = micrograph_loss.mean().item()
                elif hasattr(micrograph_loss, "item"):
                    loss_val = micrograph_loss.item()
                else:
                    loss_val = float(micrograph_loss)
                micrograph_losses[i].append((outer_iteration, loss_val))
                dataset_loss_value += loss_val

                if refine_anisotropic_mag or refine_beam_tilt:
                    if micrograph_loss.numel() > 1:
                        micrograph_loss_scalar = micrograph_loss.mean()
                    else:
                        micrograph_loss_scalar = micrograph_loss
                    scaled_loss = micrograph_loss_scalar / num_micrographs
                    scaled_loss.backward()

                print(f"  Micrograph {i + 1}/{num_micrographs} loss: {loss_val:.6f}")

                if primary_device.type == "cuda":
                    torch.cuda.empty_cache()

        # Optimize dataset-level parameters
        if refine_anisotropic_mag and optimizer_aniso is not None:
            optimizer_aniso.step()
            # Print mag matrix after each update
            print("\nMagnification matrix after update:")
            print(anistropic_mag.detach().cpu().numpy())
        if refine_beam_tilt and optimizer_beam_tilt is not None:
            optimizer_beam_tilt.step()

        # Update GPU-local copies with new master parameter values
        if num_gpus > 1:
            for gpu_id, device in enumerate(device_list):
                if refine_anisotropic_mag:
                    setup_states[gpu_id]["anistropic_mag"].data.copy_(
                        anistropic_mag.data.to(device)
                    )
                if refine_beam_tilt:
                    setup_states[gpu_id]["odd_zernikes"]["Z31c"].data.copy_(
                        odd_zernikes["Z31c"].data.to(device)
                    )
                    setup_states[gpu_id]["odd_zernikes"]["Z31s"].data.copy_(
                        odd_zernikes["Z31s"].data.to(device)
                    )

        # Print statistics
        z31c = odd_zernikes["Z31c"]
        z31c_val = z31c.item() if hasattr(z31c, "item") else float(z31c)
        z31s = odd_zernikes["Z31s"]
        z31s_val = z31s.item() if hasattr(z31s, "item") else float(z31s)
        print(f"Zernike coefficients: Z31c={z31c_val:.6f}, Z31s={z31s_val:.6f}")

        avg_dataset_loss = dataset_loss_value / num_micrographs
        print(
            f"Dataset iteration {outer_iteration + 1}/"
            f"{per_dataset_iterations} loss: {avg_dataset_loss:.6f}\n"
        )

        dataset_losses.append((outer_iteration, avg_dataset_loss))

        # Check convergence
        loss_history.append(avg_dataset_loss)
        if len(loss_history) >= 6:
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

            loss_history = loss_history[-6:]

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


# Note: Include your original _micrograph_loop function here


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

    # Explicitly free particle stack and clear cache to reduce memory usage
    del particle_stack, euler_angles, particle_backend_kwargs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return micrograph_loss, astigmatism_result

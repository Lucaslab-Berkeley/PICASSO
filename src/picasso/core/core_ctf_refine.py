"""Core functions for CTF refinement with multi-GPU support."""
# pylint: disable=too-many-lines

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import torch
from leopard_em.pydantic_models.config import PreprocessingFilters

from picasso.core.core_ctf_refine_utils import (
    accumulate_gradients_from_gpu_worker,
    check_convergence,
    collect_gradients_from_micrograph_loss,
    compute_micrograph_loss,
    create_final_results_dict,
    create_gpu_setup_states,
    distribute_micrographs_across_gpus,
    get_loss_scalar,
    get_loss_scalar_tensor,
    get_non_refined_astigmatism_values,
    initialize_astigmatism_parameters,
    initialize_defocus_parameters,
    output_mag_matrix,
    setup_particle_backend_for_micrograph,
    setup_refinement_parameters,
    update_gpu_setup_states,
)


def _log_message(message: str, log_file: Any = None) -> None:
    """Write message to both console and log file if provided.

    Parameters
    ----------
    message : str
        Message to log.
    log_file : Any, optional
        File handle to write to. If None, only prints to console.
    """
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()  # Ensure it's written immediately


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def core_refine_ctf(
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    results_files: list[str],
    device_list: list[torch.device],
    refine_anisotropic_mag: bool,
    refine_astigmatism: bool,
    refine_beam_tilt: bool,
    refine_defocus: bool,
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
    num_convergence_iterations: int = 5,
    lbfgs_max_eval: int = 5,  # pylint: disable=unused-argument
    anisotropic_mag_lr: float = 0.005,
    beam_tilt_lr: float = 1.0,
    astigmatism_angle_lr: float = 1.0,
    astigmatism_magnitude_lr: float = 5.0,
    defocus_lr: float = 1.0,
    output_directory: str | Path | None = None,
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
        amount for num_convergence_iterations iterations, stop early.
    num_convergence_iterations: int = 5,
        Number of consecutive iterations to check for convergence.
    lbfgs_max_eval: int = 5,
        Maximum number of closure evaluations for L-BFGS optimizers.
        Not used for Adam optimizer, but included for signature compatibility.
    anisotropic_mag_lr: float = 0.005,
        Learning rate for anisotropic magnification optimizer.
    beam_tilt_lr: float = 1.0,
        Learning rate for beam tilt optimizer.
    astigmatism_angle_lr: float = 1.0,
        Learning rate for astigmatism angle optimizer.
    astigmatism_magnitude_lr: float = 5.0,
        Learning rate for astigmatism magnitude optimizer.
    defocus_lr: float = 1.0,
        Learning rate for defocus optimizer.
    output_directory: str | Path | None, optional
        Directory where log file will be written. If None, no log file is created.

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
    setup_state = setup_refinement_parameters(
        results_files=results_files,
        device=primary_device,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_beam_tilt=refine_beam_tilt,
        initial_mag_matrix=initial_mag_matrix,
        initial_odd_zernikes=initial_odd_zernikes,
        initial_astigmatism_angle=initial_astigmatism_angle,
        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
        optimizer_type="adam",
        anisotropic_mag_lr=anisotropic_mag_lr,
        beam_tilt_lr=beam_tilt_lr,
        lbfgs_max_eval=lbfgs_max_eval,  # Not used for Adam, but required
    )

    # Setup log file if output directory is provided
    log_file = None
    log_file_path = None
    if output_directory is not None:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = output_dir / "ctf_refinement.log"
        log_file = open(log_file_path, "w", encoding="utf-8")
        log_file.write("CTF Refinement Log\n")
        log_file.write("=" * 50 + "\n\n")
        log_file.flush()

    try:
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
            refine_defocus=refine_defocus,
            apply_global_filtering=apply_global_filtering,
            preprocessing_filters=preprocessing_filters,
            convergence_threshold=convergence_threshold,
            num_convergence_iterations=num_convergence_iterations,
            astigmatism_angle_lr=astigmatism_angle_lr,
            astigmatism_magnitude_lr=astigmatism_magnitude_lr,
            defocus_lr=defocus_lr,
            initial_astigmatism_angle=initial_astigmatism_angle,
            initial_astigmatism_magnitude=initial_astigmatism_magnitude,
            setup_state=setup_state,
            log_file=log_file,
        )
    finally:
        if log_file is not None:
            log_file.close()
            if log_file_path is not None:
                print(f"CTF refinement log saved to: {log_file_path}")


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
    refine_defocus: bool,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    convergence_threshold: float,
    num_convergence_iterations: int,
    astigmatism_angle_lr: float,
    astigmatism_magnitude_lr: float,
    defocus_lr: float,
    initial_astigmatism_angle: torch.Tensor | None,
    initial_astigmatism_magnitude: torch.Tensor | None,
    setup_state: dict[str, Any],
    log_file: Any = None,
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
    num_convergence_iterations : int
        Number of consecutive iterations to check for convergence.
    astigmatism_angle_lr : float
        Learning rate for astigmatism angle optimizer.
    astigmatism_magnitude_lr : float
        Learning rate for astigmatism magnitude optimizer.
    defocus_lr : float
        Learning rate for defocus optimizer.
    initial_astigmatism_angle : torch.Tensor | None
        Initial astigmatism angle tensor.
    initial_astigmatism_magnitude : torch.Tensor | None
        Initial astigmatism magnitude tensor.
    setup_state : dict[str, Any]
        Setup state from _setup_refinement.
    log_file : Any, optional
        File handle for logging. If provided, debug messages will be written
        to this file in addition to console output.

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

    # Store defocus results per micrograph
    defocus_all: dict[int, torch.Tensor] = {}

    loss_history: list[float] = []
    dataset_losses: list[tuple[int, float]] = []
    micrograph_losses: dict[int, list[tuple[int, float]]] = {}
    num_micrographs = len(results_files)

    # Distribute micrographs across GPUs
    gpu_assignments = distribute_micrographs_across_gpus(num_micrographs, num_gpus)

    _log_message(f"\nUsing {num_gpus} GPUs:", log_file)
    for gpu_id, indices in enumerate(gpu_assignments):
        _log_message(
            f"  GPU {gpu_id} ({device_list[gpu_id]}): {len(indices)} micrographs",
            log_file,
        )

    # Create replicas of template on each GPU
    templates = [template.to(device) for device in device_list]

    # Create setup_state copies for each GPU
    setup_states = create_gpu_setup_states(
        setup_state=setup_state,
        device_list=device_list,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_beam_tilt=refine_beam_tilt,
    )

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

                    micrograph_loss, astigmatism_result, defocus_result = _micrograph_loop(
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
                        refine_defocus=refine_defocus,
                        apply_global_filtering=apply_global_filtering,
                        preprocessing_filters=preprocessing_filters,
                        convergence_threshold=convergence_threshold,
                        num_convergence_iterations=num_convergence_iterations,
                        astigmatism_angle_lr=astigmatism_angle_lr,
                        astigmatism_magnitude_lr=astigmatism_magnitude_lr,
                        defocus_lr=defocus_lr,
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

                    _log_message(
                        f"  GPU {gpu_id}: Micrograph {i + 1}/{num_micrographs} "
                        f"loss: {loss_val:.6f}",
                        log_file,
                    )

                    # Collect gradients from GPU-local parameters for
                    # dataset-level optimization. _micrograph_loop may have
                    # already computed gradients if retain_graph=True was used.
                    # We need to compute them here to ensure they're available
                    # for the final loss value.
                    grads = collect_gradients_from_micrograph_loss(
                        micrograph_loss=micrograph_loss,
                        local_setup=local_setup,
                        num_micrographs=num_micrographs,
                        refine_anisotropic_mag=refine_anisotropic_mag,
                        refine_beam_tilt=refine_beam_tilt,
                    )

                    queue.put(
                        {
                            "index": i,
                            "loss": loss_val,
                            "astigmatism_angle": astigmatism_result["angle"],
                            "astigmatism_magnitude": astigmatism_result["magnitude"],
                            "defocus": defocus_result,
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
                if refine_defocus and result.get("defocus") is not None:
                    defocus_all[i] = result["defocus"].to(primary_device)

                if i not in micrograph_losses:
                    micrograph_losses[i] = []
                micrograph_losses[i].append((outer_iteration, loss_val))
                dataset_loss_value += loss_val

                # Accumulate gradients from GPU-local parameters to master parameters
                grads = result.get("grads", {})
                accumulate_gradients_from_gpu_worker(
                    grads=grads,
                    anistropic_mag=anistropic_mag,
                    odd_zernikes=odd_zernikes,
                    primary_device=primary_device,
                    refine_anisotropic_mag=refine_anisotropic_mag,
                    refine_beam_tilt=refine_beam_tilt,
                )

        else:
            # Single GPU path
            for i, results_path in enumerate(results_files):
                micrograph_loss, astigmatism_result, defocus_result = _micrograph_loop(
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
                    refine_defocus=refine_defocus,
                    apply_global_filtering=apply_global_filtering,
                    preprocessing_filters=preprocessing_filters,
                    convergence_threshold=convergence_threshold,
                    num_convergence_iterations=num_convergence_iterations,
                    astigmatism_angle_lr=astigmatism_angle_lr,
                    astigmatism_magnitude_lr=astigmatism_magnitude_lr,
                    defocus_lr=defocus_lr,
                    num_results_files=num_micrographs,
                    initial_astigmatism_angle=initial_astigmatism_angle,
                    initial_astigmatism_magnitude=initial_astigmatism_magnitude,
                    setup_state=setup_state,
                )

                astigmatism_angle_all[i] = astigmatism_result["angle"]
                astigmatism_magnitude_all[i] = astigmatism_result["magnitude"]
                if refine_defocus and defocus_result is not None:
                    defocus_all[i] = defocus_result

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

                _log_message(
                    f"  Micrograph {i + 1}/{num_micrographs} loss: {loss_val:.6f}",
                    log_file,
                )

                if primary_device.type == "cuda":
                    torch.cuda.empty_cache()

        # Optimize dataset-level parameters
        if refine_anisotropic_mag and optimizer_aniso is not None:
            optimizer_aniso.step()
            # Print mag matrix after each update
            _log_message("\nMagnification matrix after update:", log_file)
            _log_message(
                str(anistropic_mag.detach().cpu().numpy()), log_file
            )
        if refine_beam_tilt and optimizer_beam_tilt is not None:
            # Store values before optimization for comparison
            zernike_scale = 1e6
            asinh_z31c_before = odd_zernikes["asinh_Z31c"].item()
            asinh_z31s_before = odd_zernikes["asinh_Z31s"].item()
            z31c_before = (
                zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"])
            ).item()
            z31s_before = (
                zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"])
            ).item()

            optimizer_beam_tilt.step()

            # Log values after optimization
            asinh_z31c_after = odd_zernikes["asinh_Z31c"].item()
            asinh_z31s_after = odd_zernikes["asinh_Z31s"].item()
            z31c_after = (
                zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"])
            ).item()
            z31s_after = (
                zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"])
            ).item()

            _log_message(
                f"  Beam tilt optimization (asinh-space): "
                f"asinh_Z31c: {asinh_z31c_before:.6f} -> {asinh_z31c_after:.6f}, "
                f"asinh_Z31s: {asinh_z31s_before:.6f} -> {asinh_z31s_after:.6f}",
                log_file,
            )
            _log_message(
                f"  Beam tilt optimization (actual): "
                f"Z31c: {z31c_before:.6f} -> {z31c_after:.6f}, "
                f"Z31s: {z31s_before:.6f} -> {z31s_after:.6f}",
                log_file,
            )

        # Update GPU-local copies with new master parameter values
        if num_gpus > 1:
            update_gpu_setup_states(
                setup_states=setup_states,
                setup_state=setup_state,
                device_list=device_list,
                refine_anisotropic_mag=refine_anisotropic_mag,
                refine_beam_tilt=refine_beam_tilt,
            )

        # Print statistics (show both asinh-space and actual values)
        zernike_scale = 1e6
        asinh_z31c = odd_zernikes["asinh_Z31c"]
        asinh_z31s = odd_zernikes["asinh_Z31s"]
        asinh_z31c_val = (
            asinh_z31c.item() if hasattr(asinh_z31c, "item") else float(asinh_z31c)
        )
        asinh_z31s_val = (
            asinh_z31s.item() if hasattr(asinh_z31s, "item") else float(asinh_z31s)
        )
        z31c_actual = zernike_scale * torch.sinh(asinh_z31c)
        z31s_actual = zernike_scale * torch.sinh(asinh_z31s)
        z31c_val = (
            z31c_actual.item() if hasattr(z31c_actual, "item") else float(z31c_actual)
        )
        z31s_val = (
            z31s_actual.item() if hasattr(z31s_actual, "item") else float(z31s_actual)
        )
        _log_message(
            f"Zernike coefficients (asinh-space): "
            f"asinh_Z31c={asinh_z31c_val:.6f}, "
            f"asinh_Z31s={asinh_z31s_val:.6f}",
            log_file,
        )
        _log_message(
            f"Zernike coefficients (actual): Z31c={z31c_val:.6f}, Z31s={z31s_val:.6f}",
            log_file,
        )

        avg_dataset_loss = dataset_loss_value / num_micrographs
        _log_message(
            f"Dataset iteration {outer_iteration + 1}/"
            f"{per_dataset_iterations} loss: {avg_dataset_loss:.6f}\n",
            log_file,
        )

        dataset_losses.append((outer_iteration, avg_dataset_loss))

        # Check convergence
        loss_history.append(avg_dataset_loss)
        converged, loss_history = check_convergence(
            loss_history, convergence_threshold, num_convergence_iterations
        )
        if converged:
            break

    if refine_anisotropic_mag:
        output_mag_matrix(anistropic_mag)

    return create_final_results_dict(
        setup_state=setup_state,
        astigmatism_angle_all=astigmatism_angle_all,
        astigmatism_magnitude_all=astigmatism_magnitude_all,
        dataset_losses=dataset_losses,
        micrograph_losses=micrograph_losses,
        defocus_all=defocus_all if refine_defocus else None,
    )


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
    refine_defocus: bool,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    convergence_threshold: float,
    num_convergence_iterations: int,
    astigmatism_angle_lr: float,
    astigmatism_magnitude_lr: float,
    defocus_lr: float,
    num_results_files: int,
    initial_astigmatism_angle: torch.Tensor | None,
    initial_astigmatism_magnitude: torch.Tensor | None,
    setup_state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
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
    refine_defocus : bool
        Whether to refine per-particle defocus.
    apply_global_filtering : bool
        Whether to apply global filtering to the micrographs.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    convergence_threshold : float
        Convergence threshold for early stopping.
    num_convergence_iterations : int
        Number of consecutive iterations to check for convergence.
    astigmatism_angle_lr : float
        Learning rate for astigmatism angle optimizer.
    astigmatism_magnitude_lr : float
        Learning rate for astigmatism magnitude optimizer.
    defocus_lr : float
        Learning rate for defocus optimizer.
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
    tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]
        Micrograph loss, updated astigmatism values, and defocus tensor (if refined).
    """
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]
    pixel_size_offsets = setup_state["pixel_size_offsets"]
    euler_angle_offsets = setup_state["euler_angle_offsets"]
    defocus_offsets = setup_state["defocus_offsets"]

    # Setup particle backend kwargs
    particle_backend_kwargs = setup_particle_backend_for_micrograph(
        results_path=results_path,
        template=template,
        template_box_size=template_box_size,
        device=device,
        preprocessing_filters=preprocessing_filters,
        euler_angle_offsets=euler_angle_offsets,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        apply_global_filtering=apply_global_filtering,
        anistropic_mag=anistropic_mag if refine_anisotropic_mag else None,
        odd_zernikes=odd_zernikes if refine_beam_tilt else None,
    )

    # Initialize astigmatism parameters for this micrograph
    optimizer_astig = None
    if refine_astigmatism:
        (
            astigmatism_angle,
            astigmatism_magnitude,
            initial_defocus_mean,
        ) = initialize_astigmatism_parameters(
            i=i,
            particle_backend_kwargs=particle_backend_kwargs,
            device=device,
            initial_astigmatism_angle=initial_astigmatism_angle,
            initial_astigmatism_magnitude=initial_astigmatism_magnitude,
        )

        optimizer_astig = torch.optim.Adam(
            [
                {"params": [astigmatism_angle], "lr": astigmatism_angle_lr},
                {"params": [astigmatism_magnitude], "lr": astigmatism_magnitude_lr},
            ]
        )
    else:
        astigmatism_angle, astigmatism_magnitude = get_non_refined_astigmatism_values(
            particle_backend_kwargs=particle_backend_kwargs,
            device=device,
        )
        initial_defocus_mean = None

    # Initialize defocus parameters for this micrograph
    optimizer_defocus = None
    defocus_tensor = None
    initial_defocus_u = None
    initial_defocus_v = None
    if refine_defocus:
        defocus_tensor = initialize_defocus_parameters(
            results_path=results_path,
            particle_backend_kwargs=particle_backend_kwargs,
            device=device,
        )
        # Get initial defocus_u and defocus_v to add defocus_tensor to them
        initial_defocus_u = particle_backend_kwargs["defocus_u"]
        initial_defocus_v = particle_backend_kwargs["defocus_v"]
        # Convert to tensors if needed
        if not isinstance(initial_defocus_u, torch.Tensor):
            initial_defocus_u = torch.tensor(initial_defocus_u, device=device)
        else:
            initial_defocus_u = initial_defocus_u.to(device)
        if not isinstance(initial_defocus_v, torch.Tensor):
            initial_defocus_v = torch.tensor(initial_defocus_v, device=device)
        else:
            initial_defocus_v = initial_defocus_v.to(device)
        
        optimizer_defocus = torch.optim.Adam([defocus_tensor], lr=defocus_lr)

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

        # Update defocus_u and defocus_v from defocus parameters if optimizing
        if refine_defocus and defocus_tensor is not None:
            # Add defocus_tensor to initial defocus values
            particle_backend_kwargs["defocus_u"] = initial_defocus_u + defocus_tensor
            particle_backend_kwargs["defocus_v"] = initial_defocus_v + defocus_tensor

        micrograph_loss = compute_micrograph_loss(
            particle_backend_kwargs=particle_backend_kwargs,
            batch_size=batch_size,
        )

        # Print micrograph loss
        loss_val = get_loss_scalar(micrograph_loss)
        print(
            f"  Micrograph {i + 1}/{num_results_files}, "
            f"inner iter {inner_iter + 1}/{inner_iterations}, "
            f"loss: {loss_val:.6f}"
        )

        # Check for convergence based on micrograph loss (end of inner loop)
        if (refine_astigmatism or refine_defocus) and outer_iteration == 0:
            micrograph_loss_history.append(loss_val)
            min_history_len = num_convergence_iterations + 1
            if len(micrograph_loss_history) >= min_history_len:
                # Check if absolute change has been less than threshold
                # for the last num_convergence_iterations consecutive iterations
                converged = True
                hist_len = len(micrograph_loss_history)
                start_idx = hist_len - num_convergence_iterations
                for j in range(start_idx, hist_len - 1):
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
                        f"threshold ({convergence_threshold:.6f}) for "
                        f"{num_convergence_iterations} "
                        f"consecutive iterations. Stopping inner loop early."
                    )
                    break
                # Keep only last (num_convergence_iterations + 1) values for efficiency
                micrograph_loss_history = micrograph_loss_history[-min_history_len:]

        # Optimize astigmatism parameters (per micrograph, inner loop)
        # Only optimize on iteration 0
        if refine_astigmatism and outer_iteration == 0 and optimizer_astig is not None:
            # Ensure micrograph_loss is a scalar for backward()
            micrograph_loss_scalar = get_loss_scalar_tensor(micrograph_loss)
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

        # Optimize defocus parameters (per micrograph, inner loop)
        # Only optimize on iteration 0
        if refine_defocus and outer_iteration == 0 and optimizer_defocus is not None:
            # Ensure micrograph_loss is a scalar for backward()
            micrograph_loss_scalar = get_loss_scalar_tensor(micrograph_loss)
            optimizer_defocus.zero_grad()
            # Retain graph if we need gradients for dataset-level optimizers
            retain_graph = refine_anisotropic_mag or refine_beam_tilt
            micrograph_loss_scalar.backward(retain_graph=retain_graph)
            optimizer_defocus.step()
            # Print defocus statistics after optimization
            if defocus_tensor is not None:
                defocus_mean = defocus_tensor.mean().item()
                defocus_std = defocus_tensor.std().item()
                print(
                    f"    Micrograph {i + 1} defocus: "
                    f"mean={defocus_mean:.2f} Å, "
                    f"std={defocus_std:.2f} Å"
                )

    # Store final astigmatism values
    astigmatism_result = {}
    if refine_astigmatism:
        astigmatism_result["angle"] = astigmatism_angle.detach()
        astigmatism_result["magnitude"] = astigmatism_magnitude.detach()
    else:
        astigmatism_result["angle"] = astigmatism_angle
        astigmatism_result["magnitude"] = astigmatism_magnitude

    # Store final defocus values
    defocus_result = None
    if refine_defocus and defocus_tensor is not None:
        defocus_result = defocus_tensor.detach()

    # Explicitly free particle backend kwargs and clear cache to reduce memory usage
    del particle_backend_kwargs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return micrograph_loss, astigmatism_result, defocus_result

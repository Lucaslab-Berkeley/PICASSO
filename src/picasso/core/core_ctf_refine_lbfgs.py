"""Core functions for CTF refinement with multi-GPU support using L-BFGS optimizer."""
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
    lbfgs_max_eval: int = 5,
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
        optimizer_type="lbfgs",
        anisotropic_mag_lr=anisotropic_mag_lr,
        beam_tilt_lr=beam_tilt_lr,
        lbfgs_max_eval=lbfgs_max_eval,
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
            lbfgs_max_eval=lbfgs_max_eval,
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
    lbfgs_max_eval: int,
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
    lbfgs_max_eval : int
        Maximum number of closure evaluations for L-BFGS optimizers.
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
                        lbfgs_max_eval=lbfgs_max_eval,
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
                    lbfgs_max_eval=lbfgs_max_eval,
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

                # For L-BFGS, don't accumulate gradients here - closure will recompute
                # This prevents OOM from accumulating computation graphs
                # L-BFGS optimizers will recompute everything in the closure
                if refine_anisotropic_mag or refine_beam_tilt:
                    # Check if we're using L-BFGS (closure will handle gradients)
                    using_lbfgs = (
                        refine_anisotropic_mag
                        and optimizer_aniso is not None
                        and isinstance(optimizer_aniso, torch.optim.LBFGS)
                    ) or (
                        refine_beam_tilt
                        and optimizer_beam_tilt is not None
                        and isinstance(optimizer_beam_tilt, torch.optim.LBFGS)
                    )
                    if not using_lbfgs:
                        # Only accumulate gradients if not using L-BFGS
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

        # Optimize dataset-level parameters with L-BFGS closure
        # Clear any accumulated gradients before closure (closure will recompute)
        if refine_anisotropic_mag and optimizer_aniso is not None:
            # Clear gradients that were accumulated in the loop (closure will recompute)
            if anistropic_mag.grad is not None:
                anistropic_mag.grad = None

            # Track closure calls for debugging
            closure_call_count = [0]  # Use list to allow modification in closure

            # Define closure for anisotropic magnification optimizer
            # The closure recomputes loss for L-BFGS line search
            # Process each micrograph separately to avoid deep computation graphs
            # NOTE: L-BFGS calls this closure multiple times during line search
            # closure_call_count is redefined each iteration, so safe
            def closure_aniso() -> torch.Tensor:
                closure_call_count[0] += 1
                if closure_call_count[0] == 1:
                    print(
                        f"\nL-BFGS closure call #{closure_call_count[0]} "
                        f"(computing initial loss and gradients)..."
                    )
                else:
                    print(
                        f"\nL-BFGS closure call #{closure_call_count[0]} "
                        f"(line search evaluation)..."
                    )

                optimizer_aniso.zero_grad()
                # Accumulate scalar loss (detached) for return value
                total_loss_scalar = 0.0

                for i, results_path in enumerate(results_files):
                    # Clear cache before each micrograph to prevent OOM
                    if primary_device.type == "cuda":
                        torch.cuda.empty_cache()

                    micrograph_loss, _, _ = _micrograph_loop(
                        i=i,
                        results_path=results_path,
                        template=template,
                        template_box_size=template_box_size,
                        device=primary_device,
                        batch_size=batch_size,
                        per_micrograph_iterations=1,  # Single forward pass
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
                        lbfgs_max_eval=lbfgs_max_eval,
                        num_results_files=num_micrographs,
                        initial_astigmatism_angle=astigmatism_angle_all,
                        initial_astigmatism_magnitude=astigmatism_magnitude_all,
                        setup_state=setup_state,
                    )

                    # Compute scalar loss for accumulation
                    if micrograph_loss.numel() > 1:
                        micrograph_loss_scalar = micrograph_loss.mean()
                    else:
                        micrograph_loss_scalar = micrograph_loss

                    # Accumulate scalar loss (detached) for return value
                    total_loss_scalar += micrograph_loss_scalar.item() / num_micrographs

                    # Compute gradients for this micrograph and accumulate
                    scaled_loss = micrograph_loss_scalar / num_micrographs
                    
                    # Debug: check computation graph before backward
                    if closure_call_count[0] == 1 and i == 0:
                        asinh_z31c = odd_zernikes["asinh_Z31c"]
                        asinh_z31s = odd_zernikes["asinh_Z31s"]
                        _log_message(
                            f"  Debug: Before backward - asinh_Z31c.requires_grad={asinh_z31c.requires_grad}, "
                            f"asinh_Z31s.requires_grad={asinh_z31s.requires_grad}",
                            log_file,
                        )
                        _log_message(
                            f"  Debug: Before backward - scaled_loss.requires_grad={scaled_loss.requires_grad}",
                            log_file,
                        )
                    
                    scaled_loss.backward()
                    
                    # Debug: check gradients after backward
                    if closure_call_count[0] == 1 and i == 0:
                        asinh_z31c = odd_zernikes["asinh_Z31c"]
                        asinh_z31s = odd_zernikes["asinh_Z31s"]
                        grad_z31c = asinh_z31c.grad
                        grad_z31s = asinh_z31s.grad
                        _log_message(
                            f"  Debug: After backward (micrograph {i+1}) - "
                            f"asinh_Z31c.grad={'None' if grad_z31c is None else f'{grad_z31c.item():.6e}'}, "
                            f"asinh_Z31s.grad={'None' if grad_z31s is None else f'{grad_z31s.item():.6e}'}",
                            log_file,
                        )

                    # Clear computation graph after each micrograph to prevent OOM
                    del micrograph_loss, scaled_loss, micrograph_loss_scalar
                    if primary_device.type == "cuda":
                        torch.cuda.empty_cache()

                # Return a tensor with the accumulated loss (for L-BFGS)
                # This doesn't need gradients since we already computed them
                loss_tensor = torch.tensor(
                    total_loss_scalar, device=primary_device, requires_grad=False
                )
                _log_message(
                    f"  Closure #{closure_call_count[0]} complete: "
                    f"total loss = {total_loss_scalar:.6f}",
                    log_file,
                )
                return loss_tensor

            optimizer_aniso.step(closure_aniso)
            _log_message(
                f"\nL-BFGS optimization complete after "
                f"{closure_call_count[0]} closure calls",
                log_file,
            )
            # Print mag matrix after each update
            _log_message("\nMagnification matrix after update:", log_file)
            _log_message(
                str(anistropic_mag.detach().cpu().numpy()), log_file
            )

        if refine_beam_tilt and optimizer_beam_tilt is not None:
            # Clear gradients that were accumulated in the loop (closure will recompute)
            if odd_zernikes["asinh_Z31c"].grad is not None:
                odd_zernikes["asinh_Z31c"].grad = None
            if odd_zernikes["asinh_Z31s"].grad is not None:
                odd_zernikes["asinh_Z31s"].grad = None

            # Track closure calls for debugging
            closure_call_count = [0]  # Use list to allow modification in closure

            # Define closure for beam tilt optimizer
            # The closure recomputes loss for L-BFGS line search
            # Process each micrograph separately to avoid deep computation graphs
            # NOTE: L-BFGS calls this closure multiple times during line search
            # closure_call_count is redefined each iteration, so safe
            def closure_beam_tilt() -> torch.Tensor:
                closure_call_count[0] += 1
                if closure_call_count[0] == 1:
                    print(
                        f"\nL-BFGS closure call #{closure_call_count[0]} "
                        f"(computing initial loss and gradients)..."
                    )
                else:
                    print(
                        f"\nL-BFGS closure call #{closure_call_count[0]} "
                        f"(line search evaluation)..."
                    )

                optimizer_beam_tilt.zero_grad()
                # Accumulate scalar loss (detached) for return value
                total_loss_scalar = 0.0

                for i, results_path in enumerate(results_files):
                    # Clear cache before each micrograph to prevent OOM
                    if primary_device.type == "cuda":
                        torch.cuda.empty_cache()

                    micrograph_loss, _, _ = _micrograph_loop(
                        i=i,
                        results_path=results_path,
                        template=template,
                        template_box_size=template_box_size,
                        device=primary_device,
                        batch_size=batch_size,
                        per_micrograph_iterations=1,  # Single forward pass
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
                        lbfgs_max_eval=lbfgs_max_eval,
                        num_results_files=num_micrographs,
                        initial_astigmatism_angle=astigmatism_angle_all,
                        initial_astigmatism_magnitude=astigmatism_magnitude_all,
                        setup_state=setup_state,
                    )

                    # Compute scalar loss for accumulation
                    if micrograph_loss.numel() > 1:
                        micrograph_loss_scalar = micrograph_loss.mean()
                    else:
                        micrograph_loss_scalar = micrograph_loss

                    # Accumulate scalar loss (detached) for return value
                    total_loss_scalar += micrograph_loss_scalar.item() / num_micrographs

                    # Check if loss requires grad and depends on Zernike parameters
                    if closure_call_count[0] == 1 and i == 0:
                        # Debug: check if computation graph is connected
                        zernike_scale = 1e6
                        asinh_z31c = odd_zernikes["asinh_Z31c"]
                        asinh_z31s = odd_zernikes["asinh_Z31s"]
                        z31c_actual = zernike_scale * torch.sinh(asinh_z31c)
                        z31s_actual = zernike_scale * torch.sinh(asinh_z31s)
                        _log_message(
                            f"  Debug: asinh_Z31c requires_grad={asinh_z31c.requires_grad}, "
                            f"value={asinh_z31c.item():.6f}",
                            log_file,
                        )
                        _log_message(
                            f"  Debug: sinh(asinh_Z31c) * scale requires_grad={z31c_actual.requires_grad}, "
                            f"value={z31c_actual.item():.6f}",
                            log_file,
                        )
                        _log_message(
                            f"  Debug: micrograph_loss requires_grad={micrograph_loss_scalar.requires_grad}",
                            log_file,
                        )

                    # Compute gradients for this micrograph and accumulate
                    scaled_loss = micrograph_loss_scalar / num_micrographs
                    scaled_loss.backward()
                    
                    # Debug: check gradients after backward
                    if closure_call_count[0] == 1 and i == 0:
                        asinh_z31c = odd_zernikes["asinh_Z31c"]
                        asinh_z31s = odd_zernikes["asinh_Z31s"]
                        grad_z31c = asinh_z31c.grad
                        grad_z31s = asinh_z31s.grad
                        _log_message(
                            f"  Debug: After backward, asinh_Z31c.grad is "
                            f"{'None' if grad_z31c is None else f'{grad_z31c.item():.6f}'}",
                            log_file,
                        )
                        _log_message(
                            f"  Debug: After backward, asinh_Z31s.grad is "
                            f"{'None' if grad_z31s is None else f'{grad_z31s.item():.6f}'}",
                            log_file,
                        )

                    # Clear computation graph after each micrograph to prevent OOM
                    del micrograph_loss, scaled_loss, micrograph_loss_scalar
                    if primary_device.type == "cuda":
                        torch.cuda.empty_cache()

                # Debug: check accumulated gradients after all micrographs
                if closure_call_count[0] == 1:
                    asinh_z31c = odd_zernikes["asinh_Z31c"]
                    asinh_z31s = odd_zernikes["asinh_Z31s"]
                    grad_z31c = asinh_z31c.grad
                    grad_z31s = asinh_z31s.grad
                    _log_message(
                        f"  Debug: After all micrographs - "
                        f"asinh_Z31c.grad={'None' if grad_z31c is None else f'{grad_z31c.item():.6e}'}, "
                        f"asinh_Z31s.grad={'None' if grad_z31s is None else f'{grad_z31s.item():.6e}'}",
                        log_file,
                    )

                # Return a tensor with the accumulated loss (for L-BFGS)
                # This doesn't need gradients since we already computed them
                loss_tensor = torch.tensor(
                    total_loss_scalar, device=primary_device, requires_grad=False
                )
                _log_message(
                    f"  Closure #{closure_call_count[0]} complete: "
                    f"total loss = {total_loss_scalar:.6f}",
                    log_file,
                )
                return loss_tensor

            # Store values before optimization for comparison
            zernike_scale = 1e6
            asinh_z31c_before = odd_zernikes["asinh_Z31c"].item()
            asinh_z31s_before = odd_zernikes["asinh_Z31s"].item()
            z31c_before = (zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"])).item()
            z31s_before = (zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"])).item()

            optimizer_beam_tilt.step(closure_beam_tilt)
            _log_message(
                f"\nL-BFGS optimization complete after "
                f"{closure_call_count[0]} closure calls",
                log_file,
            )
            # Debug: log gradient and value information
            asinh_z31c_after = odd_zernikes["asinh_Z31c"].item()
            asinh_z31s_after = odd_zernikes["asinh_Z31s"].item()
            z31c_after = (zernike_scale * torch.sinh(odd_zernikes["asinh_Z31c"])).item()
            z31s_after = (zernike_scale * torch.sinh(odd_zernikes["asinh_Z31s"])).item()

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

            # Check if gradients were computed (they may be cleared after step)
            if odd_zernikes["asinh_Z31c"].grad is not None:
                _log_message(
                    f"  asinh_Z31c gradient after step: "
                    f"{odd_zernikes['asinh_Z31c'].grad.item():.6f}",
                    log_file,
                )
            if odd_zernikes["asinh_Z31s"].grad is not None:
                _log_message(
                    f"  asinh_Z31s gradient after step: "
                    f"{odd_zernikes['asinh_Z31s'].grad.item():.6f}",
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
        asinh_z31c_val = asinh_z31c.item() if hasattr(asinh_z31c, "item") else float(asinh_z31c)
        asinh_z31s_val = asinh_z31s.item() if hasattr(asinh_z31s, "item") else float(asinh_z31s)
        z31c_actual = zernike_scale * torch.sinh(asinh_z31c)
        z31s_actual = zernike_scale * torch.sinh(asinh_z31s)
        z31c_val = z31c_actual.item() if hasattr(z31c_actual, "item") else float(z31c_actual)
        z31s_val = z31s_actual.item() if hasattr(z31s_actual, "item") else float(z31s_actual)
        _log_message(
            f"Zernike coefficients (asinh-space): asinh_Z31c={asinh_z31c_val:.6f}, "
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
    lbfgs_max_eval: int,
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
    lbfgs_max_eval : int
        Maximum number of closure evaluations for L-BFGS optimizers.
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

        # Scale magnitude parameter so both parameters are on same scale for LBFGS
        # LBFGS doesn't support per-parameter options, so we scale the magnitude
        # by dividing by the learning rate ratio, then use the lower learning rate
        # This makes the scaled parameter's gradient larger, compensating for the
        # lower learning rate
        lr_ratio = astigmatism_magnitude_lr / astigmatism_angle_lr
        # Create a leaf tensor for optimization by detaching and cloning
        astigmatism_magnitude_scaled = (
            (astigmatism_magnitude / lr_ratio).detach().clone().requires_grad_(True)
        )

        optimizer_astig = torch.optim.LBFGS(
            [astigmatism_angle, astigmatism_magnitude_scaled],
            lr=astigmatism_angle_lr,
            line_search_fn="strong_wolfe",
        )
    else:
        astigmatism_angle, astigmatism_magnitude = get_non_refined_astigmatism_values(
            particle_backend_kwargs=particle_backend_kwargs,
            device=device,
        )
        initial_defocus_mean = None
        lr_ratio = 1.0  # Not used when not refining, but needed for consistency
        astigmatism_magnitude_scaled = astigmatism_magnitude  # For consistency

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
        # Detach and clone to ensure they're constants in the computation graph
        initial_defocus_u = particle_backend_kwargs["defocus_u"]
        initial_defocus_v = particle_backend_kwargs["defocus_v"]
        # Convert to tensors if needed and detach to make them constants
        # Ensure they have the same shape as defocus_tensor (per-particle)
        if not isinstance(initial_defocus_u, torch.Tensor):
            initial_defocus_u = torch.tensor(initial_defocus_u, device=device)
        else:
            initial_defocus_u = initial_defocus_u.to(device).detach().clone()
        if not isinstance(initial_defocus_v, torch.Tensor):
            initial_defocus_v = torch.tensor(initial_defocus_v, device=device)
        else:
            initial_defocus_v = initial_defocus_v.to(device).detach().clone()
        
        # Ensure shapes match defocus_tensor (broadcast if needed)
        if initial_defocus_u.numel() == 1:
            # If scalar, expand to match defocus_tensor shape
            initial_defocus_u = initial_defocus_u.expand_as(defocus_tensor)
        elif initial_defocus_u.shape[0] != defocus_tensor.shape[0]:
            # If shape doesn't match, this is an error
            raise ValueError(
                f"initial_defocus_u shape {initial_defocus_u.shape} doesn't match "
                f"defocus_tensor shape {defocus_tensor.shape}"
            )
        if initial_defocus_v.numel() == 1:
            # If scalar, expand to match defocus_tensor shape
            initial_defocus_v = initial_defocus_v.expand_as(defocus_tensor)
        elif initial_defocus_v.shape[0] != defocus_tensor.shape[0]:
            # If shape doesn't match, this is an error
            raise ValueError(
                f"initial_defocus_v shape {initial_defocus_v.shape} doesn't match "
                f"defocus_tensor shape {defocus_tensor.shape}"
            )
        
        optimizer_defocus = torch.optim.LBFGS(
            [defocus_tensor],
            lr=1000.0 * defocus_lr,
            line_search_fn="strong_wolfe",
            #max_eval=lbfgs_max_eval,
        )

    # Only run inner loop (per-micrograph optimizer) on iteration 0
    # On later iterations, just compute loss once for dataset optimizers
    inner_iterations = per_micrograph_iterations if outer_iteration == 0 else 1
    # Track micrograph loss history for inner loop convergence
    micrograph_loss_history: list[float] = []
    for inner_iter in range(inner_iterations):
        # Update defocus_u and defocus_v from astigmatism parameters if optimizing
        if refine_astigmatism:
            # Unscale magnitude to get original value for calculations
            astigmatism_magnitude_unscaled = astigmatism_magnitude_scaled * lr_ratio
            # Broadcast astigmatism_magnitude (scalar) to match initial_defocus_mean
            # shape
            defocus_u = initial_defocus_mean + astigmatism_magnitude_unscaled / 2.0
            defocus_v = initial_defocus_mean - astigmatism_magnitude_unscaled / 2.0
            # Broadcast astigmatism_angle to match defocus_u/defocus_v shape
            # (one value per particle)
            num_particles = defocus_u.shape[0]
            astigmatism_angle_broadcast = astigmatism_angle.expand(num_particles)
            # Pass tensors directly to preserve gradients
            particle_backend_kwargs["defocus_u"] = defocus_u
            particle_backend_kwargs["defocus_v"] = defocus_v
            particle_backend_kwargs["defocus_angle"] = astigmatism_angle_broadcast

        # Update defocus_u and defocus_v from defocus parameters if optimizing
        # If astigmatism is also refined, add defocus_tensor to the astigmatism-based values
        if refine_defocus and defocus_tensor is not None:
            if refine_astigmatism:
                # Add defocus_tensor to the astigmatism-based defocus values
                particle_backend_kwargs["defocus_u"] = (
                    particle_backend_kwargs["defocus_u"] + defocus_tensor
                )
                particle_backend_kwargs["defocus_v"] = (
                    particle_backend_kwargs["defocus_v"] + defocus_tensor
                )
            else:
                # Add defocus_tensor to initial defocus values
                particle_backend_kwargs["defocus_u"] = (
                    initial_defocus_u + defocus_tensor
                )
                particle_backend_kwargs["defocus_v"] = (
                    initial_defocus_v + defocus_tensor
                )

        micrograph_loss = compute_micrograph_loss(
            particle_backend_kwargs=particle_backend_kwargs,
            batch_size=batch_size,
        )

        # Print micrograph loss (note: log_file not available in _micrograph_loop)
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
            # Debug: Print initial values
            angle_before = (
                astigmatism_angle.item()
                if hasattr(astigmatism_angle, "item")
                else float(astigmatism_angle)
            )
            mag_before = (
                (astigmatism_magnitude_scaled * lr_ratio).item()
                if hasattr(astigmatism_magnitude_scaled, "item")
                else float(astigmatism_magnitude_scaled * lr_ratio)
            )
            print(
                f"[DEBUG] Micrograph {i + 1} ASTIG LBFGS: "
                f"Before optimization - angle={angle_before:.4f}°, "
                f"magnitude={mag_before:.4f} Å"
            )

            # Define closure for astigmatism optimizer
            # Capture variables from outer scope
            # pylint: disable=cell-var-from-loop,undefined-variable
            closure_call_count_astig = [0]  # Use list to allow modification

            def closure_astig() -> torch.Tensor:
                closure_call_count_astig[0] += 1
                optimizer_astig.zero_grad()
                # Recompute forward pass and loss
                if refine_astigmatism:
                    # Unscale magnitude to get original value for calculations
                    astigmatism_magnitude_unscaled = (
                        astigmatism_magnitude_scaled * lr_ratio
                    )
                    defocus_u = (
                        initial_defocus_mean + astigmatism_magnitude_unscaled / 2.0
                    )
                    defocus_v = (
                        initial_defocus_mean - astigmatism_magnitude_unscaled / 2.0
                    )
                    num_particles = defocus_u.shape[0]
                    astigmatism_angle_broadcast = astigmatism_angle.expand(
                        num_particles
                    )
                    # particle_backend_kwargs is defined in outer scope
                    particle_backend_kwargs["defocus_u"] = defocus_u  # noqa: F821
                    particle_backend_kwargs["defocus_v"] = defocus_v  # noqa: F821
                    particle_backend_kwargs["defocus_angle"] = (  # noqa: F821
                        astigmatism_angle_broadcast
                    )

                loss = compute_micrograph_loss(
                    particle_backend_kwargs=particle_backend_kwargs,  # noqa: F821
                    batch_size=batch_size,
                )

                # Ensure loss is a scalar for backward()
                loss_scalar = get_loss_scalar_tensor(loss)

                # Retain graph if we need gradients for dataset-level optimizers
                retain_graph = refine_anisotropic_mag or refine_beam_tilt
                loss_scalar.backward(retain_graph=retain_graph)

                # Debug: Print closure call info
                loss_val = (
                    loss_scalar.item()
                    if hasattr(loss_scalar, "item")
                    else float(loss_scalar)
                )
                angle_val = (
                    astigmatism_angle.item()
                    if hasattr(astigmatism_angle, "item")
                    else float(astigmatism_angle)
                )
                mag_val = (
                    (astigmatism_magnitude_scaled * lr_ratio).item()
                    if hasattr(astigmatism_magnitude_scaled, "item")
                    else float(astigmatism_magnitude_scaled * lr_ratio)
                )
                angle_grad = (
                    astigmatism_angle.grad.item()
                    if astigmatism_angle.grad is not None
                    else None
                )
                mag_grad = (
                    astigmatism_magnitude_scaled.grad.item()
                    if astigmatism_magnitude_scaled.grad is not None
                    else None
                )
                print(
                    f"[DEBUG] ASTIG closure call #{closure_call_count_astig[0]}: "
                    f"loss={loss_val:.6f}, angle={angle_val:.4f}°, "
                    f"mag={mag_val:.4f} Å, angle_grad={angle_grad}, "
                    f"mag_grad={mag_grad}"
                )

                return loss_scalar

            optimizer_astig.step(closure_astig)

            # Debug: Print final values
            angle_after = (
                astigmatism_angle.item()
                if hasattr(astigmatism_angle, "item")
                else float(astigmatism_angle)
            )
            mag_after = (
                (astigmatism_magnitude_scaled * lr_ratio).item()
                if hasattr(astigmatism_magnitude_scaled, "item")
                else float(astigmatism_magnitude_scaled * lr_ratio)
            )
            print(
                f"[DEBUG] Micrograph {i + 1} ASTIG LBFGS: "
                f"After optimization - angle={angle_after:.4f}° "
                f"(Δ={angle_after-angle_before:.6f}), "
                f"magnitude={mag_after:.4f} Å (Δ={mag_after-mag_before:.6f}), "
                f"closure_calls={closure_call_count_astig[0]}"
            )
            # Print astigmatism parameters after optimization
            if hasattr(astigmatism_angle, "item"):
                angle_val = astigmatism_angle.item()
            else:
                angle_val = float(astigmatism_angle)
            # Unscale magnitude for display
            astigmatism_magnitude_unscaled = astigmatism_magnitude_scaled * lr_ratio
            if hasattr(astigmatism_magnitude_unscaled, "item"):
                mag_val = astigmatism_magnitude_unscaled.item()
            else:
                mag_val = float(astigmatism_magnitude_unscaled)
            print(
                f"    Micrograph {i + 1} astigmatism: "
                f"angle={angle_val:.2f}°, "
                f"magnitude={mag_val:.2f} Å"
            )

        # Optimize defocus parameters (per micrograph, inner loop)
        # Only optimize on iteration 0
        if refine_defocus and outer_iteration == 0 and optimizer_defocus is not None:
            # Debug: Print initial values
            defocus_before = defocus_tensor.detach().clone() if defocus_tensor is not None else None
            defocus_mean_before = defocus_before.mean().item() if defocus_before is not None else None
            defocus_std_before = defocus_before.std().item() if defocus_before is not None else None
            defocus_min_before = defocus_before.min().item() if defocus_before is not None else None
            defocus_max_before = defocus_before.max().item() if defocus_before is not None else None
            print(f"[DEBUG] Micrograph {i + 1} DEFOCUS LBFGS: Before optimization - mean={defocus_mean_before:.4f} Å, std={defocus_std_before:.4f} Å, range=[{defocus_min_before:.4f}, {defocus_max_before:.4f}] Å")
            print(f"[DEBUG] DEFOCUS optimizer: {type(optimizer_defocus)}, lr={optimizer_defocus.param_groups[0]['lr']}, max_eval={optimizer_defocus.param_groups[0].get('max_eval', 'N/A')}")
            
            # Define closure for defocus optimizer
            # pylint: disable=cell-var-from-loop,undefined-variable
            closure_call_count_defocus = [0]  # Use list to allow modification in closure
            def closure_defocus() -> torch.Tensor:
                closure_call_count_defocus[0] += 1
                optimizer_defocus.zero_grad()
                # Recompute forward pass and loss
                if refine_defocus and defocus_tensor is not None:
                    if refine_astigmatism:
                        # Recompute astigmatism-based defocus first
                        astigmatism_magnitude_unscaled = (
                            astigmatism_magnitude_scaled * lr_ratio
                        )
                        defocus_u_astig = (
                            initial_defocus_mean + astigmatism_magnitude_unscaled / 2.0
                        )
                        defocus_v_astig = (
                            initial_defocus_mean - astigmatism_magnitude_unscaled / 2.0
                        )
                        num_particles = defocus_u_astig.shape[0]
                        astigmatism_angle_broadcast = astigmatism_angle.expand(
                            num_particles
                        )
                        # Add defocus_tensor to astigmatism-based values
                        particle_backend_kwargs["defocus_u"] = (  # noqa: F821
                            defocus_u_astig + defocus_tensor
                        )
                        particle_backend_kwargs["defocus_v"] = (  # noqa: F821
                            defocus_v_astig + defocus_tensor
                        )
                        particle_backend_kwargs["defocus_angle"] = (  # noqa: F821
                            astigmatism_angle_broadcast
                        )
                    else:
                        # Add defocus_tensor to initial defocus values
                        particle_backend_kwargs["defocus_u"] = (  # noqa: F821
                            initial_defocus_u + defocus_tensor
                        )
                        particle_backend_kwargs["defocus_v"] = (  # noqa: F821
                            initial_defocus_v + defocus_tensor
                        )

                loss = compute_micrograph_loss(
                    particle_backend_kwargs=particle_backend_kwargs,  # noqa: F821
                    batch_size=batch_size,
                )

                # Ensure loss is a scalar for backward()
                loss_scalar = get_loss_scalar_tensor(loss)

                # Retain graph if we need gradients for dataset-level optimizers
                retain_graph = refine_anisotropic_mag or refine_beam_tilt
                loss_scalar.backward(retain_graph=retain_graph)
                
                # Debug: Print closure call info
                loss_val = loss_scalar.item() if hasattr(loss_scalar, "item") else float(loss_scalar)
                defocus_mean = defocus_tensor.mean().item() if defocus_tensor is not None else None
                defocus_std = defocus_tensor.std().item() if defocus_tensor is not None else None
                defocus_grad_mean = defocus_tensor.grad.mean().item() if defocus_tensor is not None and defocus_tensor.grad is not None else None
                defocus_grad_std = defocus_tensor.grad.std().item() if defocus_tensor is not None and defocus_tensor.grad is not None else None
                defocus_grad_norm = defocus_tensor.grad.norm().item() if defocus_tensor is not None and defocus_tensor.grad is not None else None
                print(f"[DEBUG] DEFOCUS closure call #{closure_call_count_defocus[0]}: loss={loss_val:.6f}, defocus_mean={defocus_mean:.4f} Å, defocus_std={defocus_std:.4f} Å, grad_mean={defocus_grad_mean}, grad_std={defocus_grad_std}, grad_norm={defocus_grad_norm}")
                
                return loss_scalar

            optimizer_defocus.step(closure_defocus)
            
            # Debug: Print final values
            defocus_after = defocus_tensor.detach().clone() if defocus_tensor is not None else None
            defocus_mean_after = defocus_after.mean().item() if defocus_after is not None else None
            defocus_std_after = defocus_after.std().item() if defocus_after is not None else None
            defocus_min_after = defocus_after.min().item() if defocus_after is not None else None
            defocus_max_after = defocus_after.max().item() if defocus_after is not None else None
            defocus_change = defocus_after - defocus_before if defocus_after is not None and defocus_before is not None else None
            defocus_change_mean = defocus_change.mean().item() if defocus_change is not None else None
            defocus_change_std = defocus_change.std().item() if defocus_change is not None else None
            print(f"[DEBUG] Micrograph {i + 1} DEFOCUS LBFGS: After optimization - mean={defocus_mean_after:.4f} Å (Δ={defocus_change_mean:.6f}), std={defocus_std_after:.4f} Å, range=[{defocus_min_after:.4f}, {defocus_max_after:.4f}] Å, closure_calls={closure_call_count_defocus[0]}")
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
        # Unscale magnitude before returning
        astigmatism_result["magnitude"] = (
            astigmatism_magnitude_scaled * lr_ratio
        ).detach()
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

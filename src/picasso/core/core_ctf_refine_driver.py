"""Shared driver for PICASSO CTF refinement (L-BFGS and Adam).

Structure of a run:

1. An initial loss-only pass records, for every micrograph, the starting loss and
   the starting per-micrograph parameters (astigmatism, defocus offsets, phase
   shift), taken from the explicit initial values or from the results CSV.
2. Outer (dataset) iteration 0 optimizes the per-micrograph parameters. A
   micrograph's new values are kept only if its loss improved.
3. Every outer iteration then takes a step on the dataset-level parameters
   (anisotropic magnification, beam tilt) with the per-micrograph parameters
   fixed at their accepted values, and re-evaluates the dataset loss.
4. If the dataset-level step did not improve on the loss reached after step 2,
   the dataset-level parameters are reverted.

All loss evaluations read the per-micrograph parameters from a single
``MicrographState`` per micrograph, so every pass sees the same, accepted values.
"""

# pylint: disable=too-many-lines,too-many-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-positional-arguments

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from leopard_em.pydantic_models.config import PreprocessingFilters
from leopard_em.utils.image_processing import volume_to_rfft_fourier_slice

from picasso.core.core_ctf_refine_cache import MicrographInputCache, resolve_budgets
from picasso.core.core_ctf_refine_loss import (
    compact_backend_kwargs,
    per_call_kwargs,
    without_shared_template,
)
from picasso.core.core_ctf_refine_utils import (
    accumulate_gradients_from_gpu_worker,
    apply_dataset_params,
    balance_micrographs_across_gpus,
    check_convergence,
    clamp_phase_shift_quadratic_after_step,
    compute_micrograph_loss,
    count_particles,
    create_final_results_dict,
    create_gpu_setup_states,
    get_particle_positions_normalized,
    output_mag_matrix,
    phase_shift_quadratic_from_quad_params,
    quad_params_to_dict,
    setup_particle_backend_for_micrograph,
    setup_refinement_parameters,
    update_gpu_setup_states,
)

ZERNIKE_SCALE = 1e6
T = TypeVar("T")


@dataclass(frozen=True)
class RefineOptions:
    """Static options for one CTF refinement run."""

    template_box_size: tuple[int, int]
    batch_size: int
    optimizer_type: str
    refine_anisotropic_mag: bool
    refine_astigmatism: bool
    refine_beam_tilt: bool
    refine_defocus: bool
    refine_phase_shift: bool
    phase_shift_mode: str
    apply_global_filtering: bool
    preprocessing_filters: PreprocessingFilters
    per_micrograph_iterations: int
    per_dataset_iterations: int
    convergence_threshold: float
    num_convergence_iterations: int
    lbfgs_max_eval: int
    astigmatism_angle_lr: float
    astigmatism_magnitude_lr: float
    defocus_lr: float
    phase_shift_lr: float
    initial_phase_shift_quadratic_C: float | None
    initial_phase_shift_quadratic_g: float
    initial_phase_shift_quadratic_k: float
    initial_phase_shift_quadratic_alpha: float
    phase_shift_quadratic_two_axes: bool
    initial_phase_shift_quadratic_g2: float
    initial_phase_shift_quadratic_k2: float
    setup_cache_gpu_gb: float | None = None
    setup_cache_host_gb: float | None = None

    @property
    def refine_per_micrograph(self) -> bool:
        """Whether any per-micrograph parameter is refined."""
        return self.refine_astigmatism or self.refine_defocus or self.refine_phase_shift

    @property
    def refine_dataset_level(self) -> bool:
        """Whether any dataset-level parameter is refined."""
        return self.refine_anisotropic_mag or self.refine_beam_tilt


@dataclass
class MicrographState:
    """Current (accepted) per-micrograph parameters and their loss.

    ``None`` means "use the value from the results CSV".
    """

    loss: float = float("inf")
    astigmatism_angle: torch.Tensor | None = None  # degrees, scalar
    astigmatism_magnitude: torch.Tensor | None = None  # defocus_u - defocus_v, A
    defocus: torch.Tensor | None = None  # per-particle offsets (N,), A
    phase_shift: torch.Tensor | None = None  # degrees, scalar or (N,)
    quad_params: dict[str, float] | None = None


def _raise_if_nonfinite(
    loss: torch.Tensor | float,
    params: list[torch.Tensor],
    context: str,
) -> None:
    """Stop with a clear error instead of letting an optimizer step on NaN/inf.

    L-BFGS does not fail on non-finite values; it silently stalls or produces
    garbage parameters.
    """
    loss_ok = bool(torch.isfinite(torch.as_tensor(loss)).all())
    bad = [
        index
        for index, param in enumerate(params)
        if param.grad is not None and not bool(torch.isfinite(param.grad).all())
    ]
    if loss_ok and not bad:
        return
    problem = "loss" if not loss_ok else f"gradient of parameter(s) {bad}"
    raise FloatingPointError(f"Non-finite {problem} in {context}.")


def _log_message(message: str, log_file: Any = None) -> None:
    """Write message to the console and, if given, to the log file."""
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def _as_scalar_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Return ``value`` as a detached float32 0-d tensor on ``device``."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() != 1:
        tensor = tensor.mean()
    return tensor.detach().reshape(()).to(device).clone()


def _initial_quad_values(
    opts: RefineOptions,
    state: MicrographState,
    csv_phase_shift: Any,
) -> dict[str, float]:
    """Starting quadratic phase-shift parameters for one micrograph."""
    if state.quad_params is not None:
        return dict(state.quad_params)
    if opts.initial_phase_shift_quadratic_C is not None:
        c_value = float(opts.initial_phase_shift_quadratic_C)
    elif state.phase_shift is not None:
        c_value = float(torch.as_tensor(state.phase_shift).float().mean())
    else:
        c_value = float(torch.as_tensor(csv_phase_shift).float().mean())
    if opts.phase_shift_quadratic_two_axes:
        return {
            "C": c_value,
            "g1": opts.initial_phase_shift_quadratic_g,
            "k1": opts.initial_phase_shift_quadratic_k,
            "g2": opts.initial_phase_shift_quadratic_g2,
            "k2": opts.initial_phase_shift_quadratic_k2,
            "alpha": opts.initial_phase_shift_quadratic_alpha,
        }
    return {
        "C": c_value,
        "g": opts.initial_phase_shift_quadratic_g,
        "k": opts.initial_phase_shift_quadratic_k,
        "alpha": opts.initial_phase_shift_quadratic_alpha,
    }


def _make_optimizer(
    opts: RefineOptions,
    param_groups: list[dict[str, Any]],
    lbfgs_lr: float,
) -> torch.optim.Optimizer:
    """L-BFGS over all groups at ``lbfgs_lr``, or Adam with per-group lr."""
    if opts.optimizer_type == "lbfgs":
        params = [p for group in param_groups for p in group["params"]]
        return torch.optim.LBFGS(
            params,
            lr=lbfgs_lr,
            line_search_fn="strong_wolfe",
            max_eval=opts.lbfgs_max_eval,
        )
    return torch.optim.Adam(param_groups)


def _build_micrograph_inputs(
    results_path: str,
    template: torch.Tensor,
    template_dft: torch.Tensor,
    device: torch.device,
    opts: RefineOptions,
    dataset_params: dict[str, Any],
) -> dict[str, Any]:
    """Loss inputs of one micrograph that do not depend on refined parameters.

    Runs Leopard-EM's setup (reads the micrograph and correlation maps, filters,
    extracts particles), then keeps only what the loss needs. The template RFFT is
    shared per device and left out (re-attached by ``per_call_kwargs``).
    """
    with torch.no_grad():
        pbk = setup_particle_backend_for_micrograph(
            results_path=results_path,
            template=template,
            template_box_size=opts.template_box_size,
            device=device,
            preprocessing_filters=opts.preprocessing_filters,
            euler_angle_offsets=dataset_params["euler_angle_offsets"],
            defocus_offsets=dataset_params["defocus_offsets"],
            pixel_size_offsets=dataset_params["pixel_size_offsets"],
            apply_global_filtering=opts.apply_global_filtering,
        )
        if "particle_stack_dft" in pbk:
            compact_backend_kwargs(pbk, device, template_dft=template_dft)
            without_shared_template(pbk)
    return pbk


def _micrograph_loop(
    i: int,
    results_path: str,
    get_inputs: Callable[[], dict[str, Any]],
    device: torch.device,
    opts: RefineOptions,
    state: MicrographState,
    dataset_params: dict[str, Any],
    optimize: bool,
    grad_weight: float | None,
    num_micrographs: int,
) -> tuple[torch.Tensor, MicrographState]:
    """Evaluate (and optionally optimize) one micrograph.

    Parameters
    ----------
    i : int
        Micrograph index.
    results_path : str
        Path to the results CSV of this micrograph.
    get_inputs : Callable[[], dict[str, Any]]
        Returns this micrograph's loss inputs (a copy this call may modify),
        e.g. from the input cache.
    device : torch.device
        Device for this micrograph.
    opts : RefineOptions
        Run options.
    state : MicrographState
        Starting per-micrograph parameters.
    dataset_params : dict[str, Any]
        Dataset-level parameters on ``device`` (mag matrix, Zernikes, offsets).
    optimize : bool
        Run the per-micrograph optimizers before the final evaluation.
    grad_weight : float | None
        If given, the final evaluation also accumulates the gradient of
        ``grad_weight * loss`` into the dataset-level parameters.
    num_micrographs : int
        Total number of micrographs, for log messages.

    Returns
    -------
    tuple[torch.Tensor, MicrographState]
        Scalar loss (mean of -z over particles, detached) at the returned
        parameters, and the returned per-micrograph state (``loss`` filled in).
    """
    pbk = get_inputs()
    # Grad enabled so the beam-tilt Zernikes derived from the (possibly trainable)
    # asinh parameters keep their autograd graph.
    with torch.enable_grad():
        apply_dataset_params(
            pbk,
            dataset_params["anistropic_mag"] if opts.refine_anisotropic_mag else None,
            dataset_params["odd_zernikes"] if opts.refine_beam_tilt else None,
        )

    csv_du = torch.as_tensor(pbk["defocus_u"]).float().reshape(-1).to(device)
    csv_dv = torch.as_tensor(pbk["defocus_v"]).float().reshape(-1).to(device)
    num_particles = csv_du.shape[0]
    csv_angle = (
        torch.as_tensor(pbk["defocus_angle"]).float().reshape(-1).to(device)
    ).expand(num_particles)
    csv_phase_shift = pbk["ctf_kwargs"].get("phase_shift", 0.0)

    trainable = optimize
    groups: dict[str, list[dict[str, Any]]] = {}
    lbfgs_lrs: dict[str, float] = {}

    # --- astigmatism (per micrograph) ------------------------------------------
    magnitude_scale = 1.0
    if opts.refine_astigmatism:
        start_angle = (
            state.astigmatism_angle
            if state.astigmatism_angle is not None
            else csv_angle[0]
        )
        start_magnitude = (
            state.astigmatism_magnitude
            if state.astigmatism_magnitude is not None
            else (csv_du - csv_dv).mean()
        )
        if opts.optimizer_type == "lbfgs":
            # L-BFGS has a single lr; rescale the magnitude so the configured
            # angle/magnitude lr ratio still applies.
            magnitude_scale = opts.astigmatism_magnitude_lr / opts.astigmatism_angle_lr
        astig_angle = _as_scalar_tensor(start_angle, device).requires_grad_(trainable)
        astig_mag_scaled = (
            _as_scalar_tensor(start_magnitude, device) / magnitude_scale
        ).requires_grad_(trainable)
        defocus_mean = (csv_du + csv_dv) / 2.0
        groups["astigmatism"] = [
            {"params": [astig_angle], "lr": opts.astigmatism_angle_lr},
            {"params": [astig_mag_scaled], "lr": opts.astigmatism_magnitude_lr},
        ]
        lbfgs_lrs["astigmatism"] = opts.astigmatism_angle_lr

    # --- per-particle defocus offsets ------------------------------------------
    if opts.refine_defocus:
        if state.defocus is not None:
            if state.defocus.numel() != num_particles:
                raise ValueError(
                    f"Micrograph {i}: stored defocus has {state.defocus.numel()} "
                    f"values but the results file has {num_particles} particles."
                )
            start_defocus = state.defocus.reshape(-1).float().to(device)
        else:
            start_defocus = torch.zeros(num_particles, device=device)
        defocus_offsets = start_defocus.detach().clone().requires_grad_(trainable)
        groups["defocus"] = [{"params": [defocus_offsets], "lr": opts.defocus_lr}]
        lbfgs_lrs["defocus"] = 1000.0 * opts.defocus_lr

    # --- phase shift --------------------------------------------------------------
    quad_params: dict[str, torch.Tensor] | None = None
    positions: torch.Tensor | None = None
    if opts.refine_phase_shift:
        if opts.phase_shift_mode == "quadratic":
            start_quad = _initial_quad_values(opts, state, csv_phase_shift)
            quad_params = {
                key: torch.tensor(value, device=device, dtype=torch.float32)
                .detach()
                .requires_grad_(trainable)
                for key, value in start_quad.items()
            }
            positions = get_particle_positions_normalized(results_path, device)
            if positions.shape[0] != num_particles:
                raise ValueError(
                    f"Micrograph {i}: {positions.shape[0]} particle positions but "
                    f"{num_particles} particles in the particle stack."
                )
            groups["phase_shift"] = [
                {"params": list(quad_params.values()), "lr": opts.phase_shift_lr}
            ]
        else:
            start_phase = (
                state.phase_shift if state.phase_shift is not None else csv_phase_shift
            )
            start_phase = torch.as_tensor(start_phase, dtype=torch.float32).to(device)
            if opts.phase_shift_mode == "per_micrograph":
                start_phase = start_phase.mean().reshape(())
            elif start_phase.numel() == 1:
                start_phase = start_phase.reshape(()).expand(num_particles)
            elif start_phase.numel() != num_particles:
                raise ValueError(
                    f"Micrograph {i}: initial per-particle phase shift has "
                    f"{start_phase.numel()} values for {num_particles} particles."
                )
            phase_shift = start_phase.detach().clone().requires_grad_(trainable)
            groups["phase_shift"] = [
                {"params": [phase_shift], "lr": opts.phase_shift_lr}
            ]
        lbfgs_lrs["phase_shift"] = opts.phase_shift_lr

    def apply_parameters() -> None:
        if opts.refine_astigmatism:
            magnitude = astig_mag_scaled * magnitude_scale
            defocus_u = defocus_mean + magnitude / 2.0
            defocus_v = defocus_mean - magnitude / 2.0
            angle = astig_angle.expand(num_particles)
        else:
            defocus_u, defocus_v, angle = csv_du, csv_dv, csv_angle
        if opts.refine_defocus:
            defocus_u = defocus_u + defocus_offsets
            defocus_v = defocus_v + defocus_offsets
        pbk["defocus_u"] = defocus_u
        pbk["defocus_v"] = defocus_v
        pbk["defocus_angle"] = angle
        if opts.refine_phase_shift:
            if quad_params is not None and positions is not None:
                pbk["ctf_kwargs"]["phase_shift"] = (
                    phase_shift_quadratic_from_quad_params(positions, quad_params)
                )
            else:
                pbk["ctf_kwargs"]["phase_shift"] = phase_shift

    def evaluate(backward_weight: float | None = None) -> torch.Tensor:
        """Mean loss over particles.

        With ``backward_weight``, also accumulate the gradient of
        ``backward_weight * mean loss``, chunk by chunk, and return it detached.
        """
        apply_parameters()
        if backward_weight is None:
            return compute_micrograph_loss(pbk, opts.batch_size).mean()
        return compute_micrograph_loss(
            pbk, opts.batch_size, backward_weight=backward_weight / num_particles
        ).mean()

    all_params = [p for group in groups.values() for pg in group for p in pg["params"]]

    def clear_grads() -> None:
        for param in all_params:
            param.grad = None

    def after_step() -> None:
        if quad_params is not None:
            clamp_phase_shift_quadratic_after_step(quad_params)

    if optimize and groups:
        optimizers = {
            name: _make_optimizer(opts, group, lbfgs_lrs[name])
            for name, group in groups.items()
        }
        history: list[float] = []
        with torch.enable_grad():
            for inner_iter in range(opts.per_micrograph_iterations):
                if opts.optimizer_type == "lbfgs":
                    # Block-coordinate L-BFGS: one optimizer per parameter group.
                    start_loss: float | None = None
                    for optimizer in optimizers.values():

                        def closure() -> torch.Tensor:
                            clear_grads()
                            loss = evaluate(1.0)
                            _raise_if_nonfinite(loss, all_params, f"micrograph {i + 1}")
                            return loss

                        step_loss = float(optimizer.step(closure))  # type: ignore[arg-type]
                        if start_loss is None:
                            start_loss = step_loss
                        after_step()
                    assert start_loss is not None
                else:
                    # Adam: one forward/backward, then step every group.
                    clear_grads()
                    loss = evaluate(1.0)
                    _raise_if_nonfinite(loss, all_params, f"micrograph {i + 1}")
                    start_loss = float(loss)
                    for optimizer in optimizers.values():
                        optimizer.step()
                    after_step()

                print(
                    f"  Micrograph {i + 1}/{num_micrographs}, inner iter "
                    f"{inner_iter + 1}/{opts.per_micrograph_iterations}, "
                    f"loss before step: {start_loss:.6f}"
                )
                history.append(start_loss)
                converged, history = check_convergence(
                    history,
                    opts.convergence_threshold,
                    opts.num_convergence_iterations,
                    label=f"micrograph {i + 1} loss",
                )
                if converged:
                    break
        clear_grads()
        for param in all_params:
            param.requires_grad_(False)

    # Final evaluation at the returned parameters.
    if grad_weight is not None:
        with torch.enable_grad():
            loss = evaluate(grad_weight)
    else:
        with torch.no_grad():
            loss = evaluate()

    with torch.no_grad():
        if opts.refine_astigmatism:
            out_angle = astig_angle.detach().clone()
            out_magnitude = (astig_mag_scaled * magnitude_scale).detach().clone()
        else:
            out_angle = csv_angle[0].detach().clone()
            out_magnitude = (csv_du - csv_dv).mean().detach().clone()
        out_defocus = defocus_offsets.detach().clone() if opts.refine_defocus else None
        out_phase: torch.Tensor | None = None
        out_quad: dict[str, float] | None = None
        if opts.refine_phase_shift:
            if quad_params is not None and positions is not None:
                out_phase = phase_shift_quadratic_from_quad_params(
                    positions, quad_params
                ).detach()
                out_quad = quad_params_to_dict(quad_params)
            else:
                out_phase = phase_shift.detach().clone()

    new_state = MicrographState(
        loss=float(loss.detach()),
        astigmatism_angle=out_angle,
        astigmatism_magnitude=out_magnitude,
        defocus=out_defocus,
        phase_shift=out_phase,
        quad_params=out_quad,
    )
    if optimize and groups:
        _print_micrograph_summary(i, num_micrographs, opts, new_state)

    return loss.detach(), new_state


def _print_micrograph_summary(
    i: int, num_micrographs: int, opts: RefineOptions, state: MicrographState
) -> None:
    """Print the optimized per-micrograph parameters."""
    parts = [f"loss={state.loss:.6f}"]
    if opts.refine_astigmatism and state.astigmatism_angle is not None:
        parts.append(
            f"astig angle={float(state.astigmatism_angle):.2f} deg, "
            f"magnitude={float(state.astigmatism_magnitude):.2f} A"  # type: ignore[arg-type]
        )
    if opts.refine_defocus and state.defocus is not None:
        std = float(state.defocus.std()) if state.defocus.numel() > 1 else 0.0
        parts.append(
            f"defocus offset mean={float(state.defocus.mean()):.2f} A, std={std:.2f} A"
        )
    if opts.refine_phase_shift and state.phase_shift is not None:
        if state.quad_params is not None:
            quad = ", ".join(f"{k}={v:.4f}" for k, v in state.quad_params.items())
            parts.append(f"phase quadratic: {quad}")
        else:
            parts.append(f"phase mean={float(state.phase_shift.mean()):.3f} deg")
    print(f"    Micrograph {i + 1}/{num_micrographs} optimized: " + "; ".join(parts))


def _run_over_micrographs(
    job: Callable[[int, int], T],
    gpu_assignments: list[list[int]],
) -> dict[int, T]:
    """Run ``job(micrograph_index, gpu_id)`` for every micrograph.

    With several GPUs, each GPU's micrographs run in their own thread. An
    exception in any worker is re-raised here instead of silently dropping
    that micrograph.
    """
    results: dict[int, T] = {}
    if len(gpu_assignments) == 1:
        for index in gpu_assignments[0]:
            results[index] = job(index, 0)
        return results

    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        for index in gpu_assignments[gpu_id]:
            try:
                result = job(index, gpu_id)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                with lock:
                    errors.append((index, exc))
                return
            with lock:
                results[index] = result

    threads = [
        threading.Thread(target=worker, args=(gpu_id,))
        for gpu_id in range(len(gpu_assignments))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        index, exc = min(errors, key=lambda item: item[0])
        raise RuntimeError(f"CTF refinement failed on micrograph {index}") from exc
    return results


def _collect_local_grads(
    local_params: dict[str, Any],
    primary_device: torch.device,
    opts: RefineOptions,
) -> dict[str, torch.Tensor]:
    """Move gradients of one GPU's dataset-level copies to the primary device."""
    grads: dict[str, torch.Tensor] = {}
    named: list[tuple[str, torch.Tensor]] = []
    if opts.refine_anisotropic_mag:
        named.append(("anistropic_mag", local_params["anistropic_mag"]))
    if opts.refine_beam_tilt:
        named.append(("asinh_Z31c", local_params["odd_zernikes"]["asinh_Z31c"]))
        named.append(("asinh_Z31s", local_params["odd_zernikes"]["asinh_Z31s"]))
    for name, tensor in named:
        if tensor.grad is not None:
            grads[name] = tensor.grad.detach().to(primary_device)
            tensor.grad = None
    return grads


def run_ctf_refinement(
    template: torch.Tensor,
    results_files: list[str],
    device_list: list[torch.device],
    opts: RefineOptions,
    initial_mag_matrix: torch.Tensor | None = None,
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
    initial_astigmatism_angle: torch.Tensor | None = None,
    initial_astigmatism_magnitude: torch.Tensor | None = None,
    initial_phase_shift: Any = None,
    anisotropic_mag_lr: float = 0.005,
    beam_tilt_lr: float = 1.0,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Refine CTF parameters for a set of micrographs.

    See ``picasso.core.core_ctf_refine_lbfgs.core_refine_ctf`` for the meaning of
    the arguments and the returned dictionary.
    """
    torch.set_grad_enabled(True)
    primary_device = device_list[0]
    num_micrographs = len(results_files)
    num_gpus = len(device_list)

    for name, value in (
        ("initial_astigmatism_angle", initial_astigmatism_angle),
        ("initial_astigmatism_magnitude", initial_astigmatism_magnitude),
    ):
        if value is not None and tuple(value.shape) != (num_micrographs,):
            raise ValueError(
                f"{name} must have shape ({num_micrographs},), got {tuple(value.shape)}"
            )

    setup_state = setup_refinement_parameters(
        results_files=results_files,
        device=primary_device,
        refine_anisotropic_mag=opts.refine_anisotropic_mag,
        refine_beam_tilt=opts.refine_beam_tilt,
        initial_mag_matrix=initial_mag_matrix,
        initial_odd_zernikes=initial_odd_zernikes,
        optimizer_type=opts.optimizer_type,
        anisotropic_mag_lr=anisotropic_mag_lr,
        beam_tilt_lr=beam_tilt_lr,
        lbfgs_max_eval=opts.lbfgs_max_eval,
    )
    anistropic_mag = setup_state["anistropic_mag"]
    odd_zernikes = setup_state["odd_zernikes"]
    dataset_optimizers = [
        opt
        for opt in (setup_state["optimizer_aniso"], setup_state["optimizer_beam_tilt"])
        if opt is not None
    ]

    log_file = None
    log_file_path = None
    if output_directory is not None:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = output_dir / "ctf_refinement.log"
        log_file = open(log_file_path, "w", encoding="utf-8")
        log_file.write("CTF Refinement Log\n" + "=" * 50 + "\n\n")
        log_file.flush()

    try:
        gpu_assignments = balance_micrographs_across_gpus(
            [count_particles(path) for path in results_files], num_gpus
        )
        _log_message(f"\nUsing {num_gpus} device(s):", log_file)
        for gpu_id, indices in enumerate(gpu_assignments):
            particles = sum(count_particles(results_files[i]) for i in indices)
            _log_message(
                f"  {device_list[gpu_id]}: {len(indices)} micrographs, "
                f"~{particles} particles",
                log_file,
            )

        templates = [template.to(device) for device in device_list]
        with torch.no_grad():
            template_dfts = [volume_to_rfft_fourier_slice(t) for t in templates]
        device_budgets, host_budget = resolve_budgets(
            device_list, opts.setup_cache_gpu_gb, opts.setup_cache_host_gb
        )
        input_cache = MicrographInputCache(device_budgets, host_budget)

        def inputs_getter(i: int, gpu_id: int) -> Callable[[], dict[str, Any]]:
            device = device_list[gpu_id]

            def build() -> dict[str, Any]:
                return _build_micrograph_inputs(
                    results_path=results_files[i],
                    template=templates[gpu_id],
                    template_dft=template_dfts[gpu_id],
                    device=device,
                    opts=opts,
                    dataset_params=local_params[gpu_id],
                )

            def get_inputs() -> dict[str, Any]:
                return per_call_kwargs(
                    input_cache.get(i, device, build), template_dfts[gpu_id]
                )

            return get_inputs

        if num_gpus == 1:
            local_params = [setup_state]
        else:
            local_params = create_gpu_setup_states(
                setup_state=setup_state,
                device_list=device_list,
                refine_anisotropic_mag=opts.refine_anisotropic_mag,
                refine_beam_tilt=opts.refine_beam_tilt,
            )

        def sync_local_params() -> None:
            if num_gpus > 1:
                update_gpu_setup_states(
                    setup_states=local_params,
                    setup_state=setup_state,
                    device_list=device_list,
                    refine_anisotropic_mag=opts.refine_anisotropic_mag,
                    refine_beam_tilt=opts.refine_beam_tilt,
                )

        def make_job(
            states: dict[int, MicrographState], optimize: bool
        ) -> Callable[[int, int], MicrographState]:
            def job(i: int, gpu_id: int) -> MicrographState:
                _, new_state = _micrograph_loop(
                    i=i,
                    results_path=results_files[i],
                    get_inputs=inputs_getter(i, gpu_id),
                    device=device_list[gpu_id],
                    opts=opts,
                    state=states[i],
                    dataset_params=local_params[gpu_id],
                    optimize=optimize,
                    grad_weight=None,
                    num_micrographs=num_micrographs,
                )
                return new_state

            return job

        def dataset_gradient_pass(
            states: dict[int, MicrographState],
        ) -> tuple[float, dict[int, MicrographState]]:
            """Loss of every micrograph and d(mean loss)/d(dataset params).

            Gradients are accumulated into the master parameters' ``.grad``.
            """
            sync_local_params()

            def job(
                i: int, gpu_id: int
            ) -> tuple[MicrographState, dict[str, torch.Tensor]]:
                _, new_state = _micrograph_loop(
                    i=i,
                    results_path=results_files[i],
                    get_inputs=inputs_getter(i, gpu_id),
                    device=device_list[gpu_id],
                    opts=opts,
                    state=states[i],
                    dataset_params=local_params[gpu_id],
                    optimize=False,
                    grad_weight=1.0 / num_micrographs,
                    num_micrographs=num_micrographs,
                )
                grads = (
                    _collect_local_grads(local_params[gpu_id], primary_device, opts)
                    if num_gpus > 1
                    else {}
                )
                return new_state, grads

            results = _run_over_micrographs(job, gpu_assignments)
            for _, grads in results.values():
                accumulate_gradients_from_gpu_worker(
                    grads=grads,
                    anistropic_mag=anistropic_mag,
                    odd_zernikes=odd_zernikes,
                    primary_device=primary_device,
                    refine_anisotropic_mag=opts.refine_anisotropic_mag,
                    refine_beam_tilt=opts.refine_beam_tilt,
                )
            new_states = {i: state for i, (state, _) in results.items()}
            avg_loss = sum(s.loss for s in new_states.values()) / num_micrographs
            _raise_if_nonfinite(
                avg_loss,
                dataset_param_list,
                "the dataset-level gradient (anisotropic magnification / beam tilt)",
            )
            return avg_loss, new_states

        def zero_dataset_grads() -> None:
            for optimizer in dataset_optimizers:
                optimizer.zero_grad(set_to_none=True)

        # The last evaluation (loss, per-micrograph states and gradients), keyed by
        # the dataset-level parameter values and a version of the per-micrograph
        # values. Each L-BFGS step starts by re-evaluating where the previous step
        # ended, and the loss after a step is usually the last closure's.
        dataset_param_list = [anistropic_mag, *odd_zernikes.values()]
        last_evaluation: dict[str, Any] = {}
        states_version = [0]

        def evaluation_key() -> tuple[Any, ...]:
            return (
                states_version[0],
                tuple(p.detach().cpu().numpy().tobytes() for p in dataset_param_list),
            )

        def evaluate_dataset(
            states: dict[int, MicrographState],
        ) -> tuple[float, dict[int, MicrographState], bool]:
            """Dataset loss and gradients at the current dataset-level parameters.

            Sets ``.grad`` of the dataset-level parameters. Returns the average
            loss, the evaluated states and whether a previous result was reused.
            """
            key = evaluation_key()
            if last_evaluation.get("key") == key:
                for param, grad in zip(
                    dataset_param_list, last_evaluation["grads"], strict=True
                ):
                    param.grad = None if grad is None else grad.clone()
                return last_evaluation["avg"], last_evaluation["states"], True
            zero_dataset_grads()
            avg, new_states = dataset_gradient_pass(states)
            last_evaluation.update(
                key=key,
                avg=avg,
                states=new_states,
                grads=[
                    None if p.grad is None else p.grad.detach().clone()
                    for p in dataset_param_list
                ],
            )
            return avg, new_states, False

        def lbfgs_dataset_step(
            optimizer: torch.optim.Optimizer, states: dict[int, MicrographState]
        ) -> None:
            """One L-BFGS step; each closure call is a loss + gradient pass."""
            evaluations = [0]

            def closure() -> torch.Tensor:
                evaluations[0] += 1
                avg, _, reused = evaluate_dataset(states)
                _log_message(
                    f"  L-BFGS closure #{evaluations[0]}: avg loss {avg:.6f}"
                    + (" (reused)" if reused else ""),
                    log_file,
                )
                return torch.tensor(avg, device=primary_device)

            optimizer.step(closure)  # type: ignore[arg-type]

        # ---- 1. initial loss and starting per-micrograph values -------------------
        _log_message("\nComputing initial loss before optimization...", log_file)
        start_states: dict[int, MicrographState] = {}
        for i in range(num_micrographs):
            start = MicrographState()
            if initial_astigmatism_angle is not None:
                start.astigmatism_angle = initial_astigmatism_angle[i]
            if initial_astigmatism_magnitude is not None:
                start.astigmatism_magnitude = initial_astigmatism_magnitude[i]
            if initial_phase_shift is not None:
                start.phase_shift = initial_phase_shift[i]
            start_states[i] = start
        sync_local_params()
        if opts.refine_dataset_level and not opts.refine_per_micrograph:
            # Also gives the gradients the first dataset-level step starts from.
            _, states, _ = evaluate_dataset(start_states)
        else:
            states = _run_over_micrographs(
                make_job(start_states, optimize=False), gpu_assignments
            )
        initial_avg_loss = sum(s.loss for s in states.values()) / num_micrographs
        _log_message(
            f"Initial avg loss (before any optimization): {initial_avg_loss:.6f}",
            log_file,
        )
        _log_message(input_cache.summary() + "\n", log_file)

        global_initial_mag = anistropic_mag.detach().clone()
        global_initial_zernikes = {
            k: v.detach().clone() for k, v in odd_zernikes.items()
        }
        reference_states = dict(states)
        reference_avg_loss = initial_avg_loss

        dataset_losses: list[tuple[int, float]] = []
        micrograph_losses: dict[int, list[tuple[int, float]]] = {
            i: [] for i in range(num_micrographs)
        }
        loss_history: list[float] = []

        for outer_iteration in range(opts.per_dataset_iterations):
            # ---- 2. per-micrograph parameters (first outer iteration only) -------
            if outer_iteration == 0 and opts.refine_per_micrograph:
                sync_local_params()
                candidates = _run_over_micrographs(
                    make_job(states, optimize=True), gpu_assignments
                )
                for i in range(num_micrographs):
                    candidate = candidates[i]
                    if candidate.loss < states[i].loss:
                        states[i] = candidate
                    else:
                        _log_message(
                            f"  Micrograph {i + 1}: per-micrograph refinement did not "
                            f"improve the loss ({candidate.loss:.6f} >= "
                            f"{states[i].loss:.6f}); keeping starting values.",
                            log_file,
                        )
                states_version[0] += 1
                reference_states = dict(states)
                reference_avg_loss = (
                    sum(s.loss for s in states.values()) / num_micrographs
                )
                _log_message(
                    f"Avg loss after per-micrograph refinement: "
                    f"{reference_avg_loss:.6f}",
                    log_file,
                )

            # ---- 3. dataset-level parameters --------------------------------------
            if opts.refine_dataset_level:
                if opts.optimizer_type == "lbfgs":
                    for optimizer in dataset_optimizers:
                        lbfgs_dataset_step(optimizer, states)
                else:
                    evaluate_dataset(states)
                    for optimizer in dataset_optimizers:
                        optimizer.step()
                # Loss at the new parameters: usually the last L-BFGS closure's; for
                # Adam this pass also gives the next step's gradients.
                _, states, _ = evaluate_dataset(states)
                zero_dataset_grads()
                sync_local_params()
                _log_dataset_params(opts, anistropic_mag, odd_zernikes, log_file)

            avg_loss = sum(s.loss for s in states.values()) / num_micrographs
            for i in range(num_micrographs):
                micrograph_losses[i].append((outer_iteration, states[i].loss))
            dataset_losses.append((outer_iteration, avg_loss))
            _log_message(
                f"Dataset iteration {outer_iteration + 1}/"
                f"{opts.per_dataset_iterations} avg loss: {avg_loss:.6f}\n",
                log_file,
            )

            if not opts.refine_dataset_level:
                # Per-micrograph parameters are only optimized in iteration 0;
                # further iterations would only re-evaluate the same loss.
                break
            loss_history.append(avg_loss)
            converged, loss_history = check_convergence(
                loss_history,
                opts.convergence_threshold,
                opts.num_convergence_iterations,
            )
            if converged:
                break

        # ---- 4. revert dataset-level parameters if they did not help --------------
        if opts.refine_dataset_level:
            final_avg_loss = (
                dataset_losses[-1][1] if dataset_losses else reference_avg_loss
            )
            if final_avg_loss >= reference_avg_loss:
                _log_message(
                    f"\nFinal check: dataset-level refinement did not improve the loss "
                    f"(before: {reference_avg_loss:.6f}, after: {final_avg_loss:.6f}). "
                    f"Reverting anisotropic magnification and beam tilt.",
                    log_file,
                )
                with torch.no_grad():
                    if opts.refine_anisotropic_mag:
                        anistropic_mag.copy_(global_initial_mag)
                    if opts.refine_beam_tilt:
                        for key, value in odd_zernikes.items():
                            value.copy_(global_initial_zernikes[key])
                sync_local_params()
                states = reference_states
            else:
                _log_message(
                    f"\nFinal check: dataset-level refinement improved the loss "
                    f"(before: {reference_avg_loss:.6f}, after: {final_avg_loss:.6f}).",
                    log_file,
                )

        _log_message(input_cache.summary(), log_file)
        if opts.refine_anisotropic_mag:
            output_mag_matrix(anistropic_mag)

        ordered = [states[i] for i in range(num_micrographs)]
        astigmatism_angle_all = torch.stack(
            [s.astigmatism_angle.to(primary_device) for s in ordered]  # type: ignore[union-attr]
        )
        astigmatism_magnitude_all = torch.stack(
            [s.astigmatism_magnitude.to(primary_device) for s in ordered]  # type: ignore[union-attr]
        )
        defocus_all = (
            {i: s.defocus.to(primary_device) for i, s in enumerate(ordered)}  # type: ignore[union-attr]
            if opts.refine_defocus
            else None
        )
        phase_shift_all = (
            {i: s.phase_shift.to(primary_device) for i, s in enumerate(ordered)}  # type: ignore[union-attr]
            if opts.refine_phase_shift
            else None
        )
        quad_all = (
            {i: dict(s.quad_params or {}) for i, s in enumerate(ordered)}
            if opts.refine_phase_shift and opts.phase_shift_mode == "quadratic"
            else None
        )
        return create_final_results_dict(
            setup_state=setup_state,
            astigmatism_angle_all=astigmatism_angle_all,
            astigmatism_magnitude_all=astigmatism_magnitude_all,
            dataset_losses=dataset_losses,
            micrograph_losses=micrograph_losses,
            defocus_all=defocus_all,
            phase_shift_all=phase_shift_all,
            phase_shift_quadratic_params_all=quad_all,
        )
    finally:
        if log_file is not None:
            log_file.close()
            print(f"CTF refinement log saved to: {log_file_path}")


def _log_dataset_params(
    opts: RefineOptions,
    anistropic_mag: torch.Tensor,
    odd_zernikes: dict[str, torch.Tensor],
    log_file: Any,
) -> None:
    """Log the current dataset-level parameters."""
    if opts.refine_anisotropic_mag:
        _log_message(
            "Magnification matrix: "
            + str(anistropic_mag.detach().cpu().numpy().round(6).tolist()),
            log_file,
        )
    if opts.refine_beam_tilt:
        z31c = float(ZERNIKE_SCALE * torch.sinh(odd_zernikes["asinh_Z31c"].detach()))
        z31s = float(ZERNIKE_SCALE * torch.sinh(odd_zernikes["asinh_Z31s"].detach()))
        _log_message(f"Beam tilt Zernikes: Z31c={z31c:.6f}, Z31s={z31s:.6f}", log_file)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def refine_ctf(
    optimizer_type: str,
    template: torch.Tensor,
    template_box_size: tuple[int, int],
    results_files: list[str],
    device_list: list[torch.device],
    refine_anisotropic_mag: bool,
    refine_astigmatism: bool,
    refine_beam_tilt: bool,
    refine_defocus: bool,
    refine_phase_shift: bool,
    phase_shift_mode: str,
    apply_global_filtering: bool,
    preprocessing_filters: PreprocessingFilters,
    batch_size: int,
    per_micrograph_iterations: int,
    per_dataset_iterations: int,
    initial_mag_matrix: torch.Tensor | None = None,
    initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
    initial_astigmatism_angle: torch.Tensor | None = None,
    initial_astigmatism_magnitude: torch.Tensor | None = None,
    initial_phase_shift: Any = None,
    initial_phase_shift_quadratic_C: float | None = None,
    initial_phase_shift_quadratic_g: float = 0.0,
    initial_phase_shift_quadratic_k: float = 0.0,
    initial_phase_shift_quadratic_alpha: float = 0.0,
    phase_shift_quadratic_two_axes: bool = False,
    initial_phase_shift_quadratic_g2: float = 0.0,
    initial_phase_shift_quadratic_k2: float = 0.0,
    convergence_threshold: float = 0.0005,
    num_convergence_iterations: int = 5,
    lbfgs_max_eval: int = 5,
    anisotropic_mag_lr: float = 0.005,
    beam_tilt_lr: float = 1.0,
    astigmatism_angle_lr: float = 1.0,
    astigmatism_magnitude_lr: float = 5.0,
    defocus_lr: float = 1.0,
    phase_shift_lr: float = 1.0,
    output_directory: str | Path | None = None,
    setup_cache_gpu_gb: float | None = None,
    setup_cache_host_gb: float | None = None,
) -> dict[str, Any]:
    """Refine the CTF parameters for the micrographs.

    Parameters
    ----------
    optimizer_type : str
        "lbfgs" or "adam"; used for both per-micrograph and dataset-level
        parameters.
    template : torch.Tensor
        The template volume.
    template_box_size : tuple[int, int]
        The box size of the template.
    results_files : list[str]
        The list of results file paths (one per micrograph).
    device_list : list[torch.device]
        Devices to use; micrographs are distributed across them.
    refine_anisotropic_mag, refine_astigmatism, refine_beam_tilt : bool
        Which parameters to refine.
    refine_defocus, refine_phase_shift : bool
        Which parameters to refine.
    phase_shift_mode : str
        'per_micrograph', 'per_particle' or 'quadratic'.
    apply_global_filtering : bool
        Whether to filter whole micrographs before particle extraction.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    batch_size : int
        Number of particles per chunk in the loss computation.
    per_micrograph_iterations : int
        Maximum optimizer iterations per micrograph (first dataset iteration).
    per_dataset_iterations : int
        Maximum dataset-level iterations.
    initial_mag_matrix, initial_odd_zernikes : optional
        Starting dataset-level parameters.
    initial_astigmatism_angle, initial_astigmatism_magnitude : torch.Tensor, optional
        Starting astigmatism, shape (num_micrographs,). Defaults to the CSV values.
    initial_phase_shift : optional
        Starting phase shift per micrograph, indexable by micrograph index
        (tensor or dict). Defaults to the CSV value.
    initial_phase_shift_quadratic_C, initial_phase_shift_quadratic_g : optional
        Starting quadratic phase-shift parameters (C in degrees).
    initial_phase_shift_quadratic_k, initial_phase_shift_quadratic_alpha : float
        Starting quadratic phase-shift parameters (alpha in radians).
    phase_shift_quadratic_two_axes : bool
        Use the two-axis quadratic model.
    initial_phase_shift_quadratic_g2, initial_phase_shift_quadratic_k2 : float
        Starting second-axis quadratic parameters.
    convergence_threshold : float
        Stop when the loss changes by at most this much for
        ``num_convergence_iterations`` consecutive iterations.
    num_convergence_iterations : int
        See ``convergence_threshold``. Values <= 0 disable early stopping.
    lbfgs_max_eval : int
        ``max_eval`` for the L-BFGS optimizers.
    anisotropic_mag_lr, beam_tilt_lr : float
        Learning rates of the dataset-level optimizers.
    astigmatism_angle_lr, astigmatism_magnitude_lr : float
        Learning rates for astigmatism.
    defocus_lr, phase_shift_lr : float
        Learning rates for defocus and phase shift.
    output_directory : str | Path | None
        Directory for ``ctf_refinement.log``. No log file if None.
    setup_cache_gpu_gb : float | None
        Per-GPU memory (GB) for keeping micrograph inputs between passes. None:
        40% of each GPU's memory. 0: do not keep inputs on the GPU.
    setup_cache_host_gb : float | None
        Pinned host memory (GB) for micrograph inputs that do not fit on the GPU.
        None: 40% of the available host memory. 0: no host cache.

    Returns
    -------
    dict[str, Any]
        "anistropic_mag", "odd_zernikes", "astigmatism_angle",
        "astigmatism_magnitude", "dataset_losses", "micrograph_losses", and when
        refined "defocus_all", "phase_shift_all" and
        "phase_shift_quadratic_params_all".
    """
    if optimizer_type not in ("lbfgs", "adam"):
        raise ValueError(
            f"optimizer_type must be 'lbfgs' or 'adam', got {optimizer_type}"
        )
    opts = RefineOptions(
        template_box_size=template_box_size,
        batch_size=batch_size,
        optimizer_type=optimizer_type,
        refine_anisotropic_mag=refine_anisotropic_mag,
        refine_astigmatism=refine_astigmatism,
        refine_beam_tilt=refine_beam_tilt,
        refine_defocus=refine_defocus,
        refine_phase_shift=refine_phase_shift,
        phase_shift_mode=phase_shift_mode,
        apply_global_filtering=apply_global_filtering,
        preprocessing_filters=preprocessing_filters,
        per_micrograph_iterations=per_micrograph_iterations,
        per_dataset_iterations=per_dataset_iterations,
        convergence_threshold=convergence_threshold,
        num_convergence_iterations=num_convergence_iterations,
        lbfgs_max_eval=lbfgs_max_eval,
        astigmatism_angle_lr=astigmatism_angle_lr,
        astigmatism_magnitude_lr=astigmatism_magnitude_lr,
        defocus_lr=defocus_lr,
        phase_shift_lr=phase_shift_lr,
        initial_phase_shift_quadratic_C=initial_phase_shift_quadratic_C,
        initial_phase_shift_quadratic_g=initial_phase_shift_quadratic_g,
        initial_phase_shift_quadratic_k=initial_phase_shift_quadratic_k,
        initial_phase_shift_quadratic_alpha=initial_phase_shift_quadratic_alpha,
        phase_shift_quadratic_two_axes=phase_shift_quadratic_two_axes,
        initial_phase_shift_quadratic_g2=initial_phase_shift_quadratic_g2,
        initial_phase_shift_quadratic_k2=initial_phase_shift_quadratic_k2,
        setup_cache_gpu_gb=setup_cache_gpu_gb,
        setup_cache_host_gb=setup_cache_host_gb,
    )
    return run_ctf_refinement(
        template=template,
        results_files=results_files,
        device_list=device_list,
        opts=opts,
        initial_mag_matrix=initial_mag_matrix,
        initial_odd_zernikes=initial_odd_zernikes,
        initial_astigmatism_angle=initial_astigmatism_angle,
        initial_astigmatism_magnitude=initial_astigmatism_magnitude,
        initial_phase_shift=initial_phase_shift,
        anisotropic_mag_lr=anisotropic_mag_lr,
        beam_tilt_lr=beam_tilt_lr,
        output_directory=output_directory,
    )

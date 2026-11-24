"""Tests for the CTF refinement driver, loss and CSV writer.

Driver tests replace the Leopard-EM setup and the image-based loss with a smooth
synthetic loss so the control flow can be checked on the CPU.
"""

import json
import sys
from typing import Any

import pandas as pd
import pytest
import torch

import picasso.core.core_ctf_refine_driver as driver
from picasso.config import CtfRefineConfig, InputConfig, OutputConfig
from picasso.core.core_ctf_refine_loss import (
    _make_rfft_hermitian,
    compute_particle_z_scores,
)
from picasso.core.core_ctf_refine_utils import check_convergence
from picasso.managers.ctf_refine_manager import CTFRefineManager

# ---------------------------------------------------------------------------
# Synthetic loss used in place of the image-based z-score
# ---------------------------------------------------------------------------

N_PARTICLES = 8
CSV_DU, CSV_DV, CSV_ANGLE = 12100.0, 11700.0, 30.0  # magnitude 400 A in the CSV
TRUE_ANGLE, TRUE_MAG = 35.0, 450.0
TRUE_DEFOCUS = torch.linspace(-60.0, 60.0, N_PARTICLES)
TRUE_PHASE = 20.0


class SyntheticProblem:
    """Fake setup + loss with a known optimum."""

    def __init__(self, target_mag: torch.Tensor) -> None:
        self.target_mag = target_mag

    def setup(self, **kwargs: Any) -> dict[str, Any]:
        pbk: dict[str, Any] = {
            "defocus_u": torch.full((N_PARTICLES,), CSV_DU),
            "defocus_v": torch.full((N_PARTICLES,), CSV_DV),
            "defocus_angle": torch.full((N_PARTICLES,), CSV_ANGLE),
            "ctf_kwargs": {"phase_shift": 0.0},
            "mag_matrix": torch.eye(2),
        }
        if kwargs.get("anistropic_mag") is not None:
            pbk["mag_matrix"] = kwargs["anistropic_mag"]
        return pbk

    def loss(
        self,
        pbk: dict[str, Any],
        batch_size: int,
        backward_weight: float | None = None,
    ) -> torch.Tensor:
        loss = self._loss(pbk)
        if backward_weight is not None:
            (backward_weight * loss.sum()).backward()
            return loss.detach()
        return loss

    def _loss(self, pbk: dict[str, Any]) -> torch.Tensor:
        mean_df = (pbk["defocus_u"] + pbk["defocus_v"]) / 2
        magnitude = pbk["defocus_u"] - pbk["defocus_v"]
        true_mean = (CSV_DU + CSV_DV) / 2 + TRUE_DEFOCUS
        phase = torch.as_tensor(pbk["ctf_kwargs"]["phase_shift"])
        return (
            ((mean_df - true_mean) / 100.0) ** 2
            + ((magnitude - TRUE_MAG) / 100.0) ** 2
            # Astigmatism angles are periodic in 180 degrees, as in a real CTF.
            + (
                (torch.remainder(pbk["defocus_angle"] - TRUE_ANGLE + 90, 180) - 90)
                / 10.0
            )
            ** 2
            + ((phase - TRUE_PHASE) / 10.0) ** 2
            + 1e4 * ((pbk["mag_matrix"] - self.target_mag) ** 2).sum()
            - 10.0
        )


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch):
    """Install a synthetic problem; returns a function to set the target mag."""
    problem = SyntheticProblem(torch.eye(2))
    monkeypatch.setattr(driver, "setup_particle_backend_for_micrograph", problem.setup)
    monkeypatch.setattr(driver, "compute_micrograph_loss", problem.loss)
    return problem


def run(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "optimizer_type": "lbfgs",
        "template": torch.zeros(4, 4, 4),
        "template_box_size": (4, 4),
        "results_files": ["m0.csv", "m1.csv"],
        "device_list": [torch.device("cpu")],
        "refine_anisotropic_mag": False,
        "refine_astigmatism": False,
        "refine_beam_tilt": False,
        "refine_defocus": False,
        "refine_phase_shift": False,
        "phase_shift_mode": "per_micrograph",
        "apply_global_filtering": True,
        "preprocessing_filters": None,
        "batch_size": 8,
        "per_micrograph_iterations": 5,
        "per_dataset_iterations": 3,
        "astigmatism_angle_lr": 1.0,
        "astigmatism_magnitude_lr": 1.0,
    }
    kwargs.update(overrides)
    return driver.refine_ctf(**kwargs)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_later_dataset_iterations_keep_per_micrograph_results(synthetic):
    """Dataset-level iterations must not reset refined astigmatism/defocus."""
    synthetic.target_mag = torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    result = run(
        refine_anisotropic_mag=True, refine_astigmatism=True, refine_defocus=True
    )
    torch.testing.assert_close(
        result["astigmatism_magnitude"], torch.full((2,), TRUE_MAG), atol=1.0, rtol=0
    )
    torch.testing.assert_close(
        result["astigmatism_angle"], torch.full((2,), TRUE_ANGLE), atol=0.1, rtol=0
    )
    for offsets in result["defocus_all"].values():
        torch.testing.assert_close(offsets, TRUE_DEFOCUS, atol=1.0, rtol=0)
    # The mag matrix moved toward the target and the dataset loss went down.
    start_error = (torch.eye(2) - synthetic.target_mag).norm()
    assert (result["anistropic_mag"] - synthetic.target_mag).norm() < start_error
    losses = [loss for _, loss in result["dataset_losses"]]
    assert losses == sorted(losses, reverse=True)


def test_unimproved_micrographs_keep_csv_astigmatism(synthetic):
    """If nothing improves, reported astigmatism is the CSV value, not zero."""
    result = run(refine_anisotropic_mag=True)  # identity mag is already optimal
    torch.testing.assert_close(result["astigmatism_angle"], torch.full((2,), CSV_ANGLE))
    torch.testing.assert_close(
        result["astigmatism_magnitude"], torch.full((2,), CSV_DU - CSV_DV)
    )


def test_per_micrograph_optimization_runs_once_per_micrograph(synthetic, monkeypatch):
    """Dataset-level closures only evaluate; they never re-optimize micrographs."""
    synthetic.target_mag = torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    calls = {"optimize": 0}
    original = driver._micrograph_loop

    def counting_loop(**kwargs: Any):
        calls["optimize"] += int(kwargs["optimize"])
        return original(**kwargs)

    monkeypatch.setattr(driver, "_micrograph_loop", counting_loop)
    run(refine_anisotropic_mag=True, refine_astigmatism=True)
    assert calls["optimize"] == 2


@pytest.mark.parametrize("optimizer_type", ["lbfgs", "adam"])
def test_several_per_micrograph_groups(synthetic, optimizer_type):
    """Astigmatism + defocus + phase together (Adam used to crash here)."""
    result = run(
        optimizer_type=optimizer_type,
        refine_astigmatism=True,
        refine_defocus=True,
        refine_phase_shift=True,
        per_micrograph_iterations=200 if optimizer_type == "adam" else 5,
        astigmatism_angle_lr=0.5,
        astigmatism_magnitude_lr=0.5,
        defocus_lr=0.5,
        phase_shift_lr=0.5,
        num_convergence_iterations=0,
    )
    losses = [loss for _, loss in result["micrograph_losses"][0]]
    assert losses[-1] < -9.0  # starts near -5.4; optimum is -10
    # The magnitude must receive a gradient even with defocus refined too.
    assert abs(float(result["astigmatism_magnitude"][0]) - (CSV_DU - CSV_DV)) > 1.0


def test_per_particle_phase_shift(synthetic):
    result = run(
        refine_phase_shift=True,
        phase_shift_mode="per_particle",
        phase_shift_lr=1.0,
    )
    for phase in result["phase_shift_all"].values():
        assert phase.shape == (N_PARTICLES,)
        torch.testing.assert_close(
            phase, torch.full((N_PARTICLES,), TRUE_PHASE), atol=0.1, rtol=0
        )


def test_worker_errors_are_raised():
    def job(index: int, gpu_id: int) -> int:
        if index == 3:
            raise ValueError("boom")
        return index

    with pytest.raises(RuntimeError, match="micrograph 3"):
        driver._run_over_micrographs(job, [[0, 2], [1, 3]])
    assert driver._run_over_micrographs(job, [[0, 2], [1]]) == {0: 0, 1: 1, 2: 2}


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


def test_check_convergence_counts_all_changes():
    # Last 3 changes: 1.0, 1e-4, 1e-4 -> not converged with n=3.
    assert not check_convergence([0.0, 1.0, 1.0001, 1.0002], 1e-3, 3)[0]
    assert check_convergence([1.0, 1.0001, 1.0002, 1.0003], 1e-3, 3)[0]


def test_check_convergence_single_iteration_needs_small_change():
    assert not check_convergence([0.0, -5.0], 1e-3, 1)[0]
    assert check_convergence([0.0, 1e-4], 1e-3, 1)[0]
    assert not check_convergence([0.0, 0.0, 0.0], 1e-3, 0)[0]


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def test_csv_writer_only_touches_refined_columns(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    original_mag = "[1.01, 0.0, 0.0, 0.99]"
    original_zernikes = json.dumps({"Z31c": 5.0, "Z31s": -3.0})
    pd.DataFrame(
        {
            "defocus_u": [12100.0, 12100.0],
            "defocus_v": [11700.0, 11700.0],
            "astigmatism_angle": [30.0, 30.0],
            "refined_relative_defocus": [10.0, -10.0],
            "mag_matrix": [original_mag, original_mag],
            "odd_zernikes": [original_zernikes, original_zernikes],
        }
    ).to_csv(results_dir / "a_results.csv", index=False)

    manager = CTFRefineManager.model_construct(
        input_config=InputConfig.model_construct(
            template_path="unused.mrc",
            template_box_size=(4, 4),
            results_directory=str(results_dir),
            results_pattern="*_results.csv",
            batch_size=8,
        ),
        output_config=OutputConfig(output_directory=str(tmp_path / "out")),
        computational_config=None,
        ctf_refine_config=CtfRefineConfig(refine_defocus=True),
        preprocessing_filters=None,
    )
    refined = {
        "anistropic_mag": torch.eye(2),
        "odd_zernikes": {"Z31c": torch.tensor(0.0), "Z31s": torch.tensor(0.0)},
        "astigmatism_angle": torch.tensor([0.0]),
        "astigmatism_magnitude": torch.tensor([0.0]),
        "defocus_all": {0: torch.tensor([1.0, 2.0])},
    }
    manager.updated_results_csvs(refined)
    out = pd.read_csv(tmp_path / "out" / "a_results_ctf-refine.csv")
    assert list(out["mag_matrix"]) == [original_mag] * 2
    assert list(out["odd_zernikes"]) == [original_zernikes] * 2
    assert list(out["defocus_u"]) == [12100.0] * 2
    assert list(out["defocus_v"]) == [11700.0] * 2
    assert list(out["astigmatism_angle"]) == [30.0] * 2
    assert list(out["refined_relative_defocus"]) == [11.0, -8.0]


# ---------------------------------------------------------------------------
# Batched loss
# ---------------------------------------------------------------------------


def _band_limited_volume(size: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    volume = torch.randn(size, size, size, generator=gen)
    freq = torch.fft.fftfreq(size)
    radius = torch.sqrt(
        freq[:, None, None] ** 2
        + freq[None, :, None] ** 2
        + torch.fft.rfftfreq(size)[None, None, :] ** 2
    )
    spectrum = torch.fft.rfftn(volume) * (radius < 0.4)
    return torch.fft.irfftn(spectrum, s=volume.shape)


def _synthetic_backend_kwargs(
    size: int, num_particles: int, device: torch.device, seed: int = 0
) -> dict[str, Any]:
    from leopard_em.utils.image_processing import volume_to_rfft_fourier_slice

    gen = torch.Generator().manual_seed(seed)
    images = torch.randn(num_particles, size, size, generator=gen)
    particle_dft = torch.fft.rfftn(images, dim=(-2, -1))
    particle_dft[..., 0, 0] = 0
    ky = torch.fft.fftfreq(size)[:, None]
    kx = torch.fft.rfftfreq(size)[None, :]
    radius = torch.sqrt(ky**2 + kx**2)
    scales = 20 + 10 * torch.rand(num_particles, 1, 1, generator=gen)
    return {
        "particle_stack_dft": particle_dft.to(device),
        "template_dft": volume_to_rfft_fourier_slice(_band_limited_volume(size)).to(
            device
        ),
        "euler_angles": (
            torch.rand(num_particles, 3, generator=gen)
            * torch.tensor([360.0, 180.0, 360.0])
        ).to(device),
        "euler_angle_offsets": torch.zeros(1, 3, device=device),
        "defocus_offsets": torch.zeros(1, device=device),
        "pixel_size_offsets": torch.zeros(1, device=device),
        "defocus_u": (12000 + 50 * torch.randn(num_particles, generator=gen)).to(
            device
        ),
        "defocus_v": (11600 + 50 * torch.randn(num_particles, generator=gen)).to(
            device
        ),
        "defocus_angle": torch.full((num_particles,), 30.0, device=device),
        "corr_mean": (0.01 * torch.randn(num_particles, 1, 1, generator=gen)).to(
            device
        ),
        "corr_std": (1.0 + torch.rand(num_particles, 1, 1, generator=gen)).to(device),
        "ctf_kwargs": {
            "voltage": 300.0,
            "spherical_aberration": 2.7,
            "amplitude_contrast_ratio": 0.07,
            "ctf_B_factor": 0.0,
            "phase_shift": 0.0,
            "pixel_size": 1.055,
            "template_shape": (size, size),
            "even_zernikes": None,
            "odd_zernikes": None,
            "mag_matrix": None,
        },
        "projective_filters": (1.0 + 0.5 * torch.cos(radius[None] * scales)).to(device),
        "device": [device],
        "mag_matrix": None,
    }


def _subset(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    per_particle = {
        "particle_stack_dft",
        "euler_angles",
        "defocus_u",
        "defocus_v",
        "defocus_angle",
        "corr_mean",
        "corr_std",
        "projective_filters",
    }
    out = {
        key: (value[index : index + 1] if key in per_particle else value)
        for key, value in kwargs.items()
        if not key.startswith("_picasso")  # drop the full-stack loss cache
    }
    out["ctf_kwargs"] = dict(kwargs["ctf_kwargs"])
    return out


def test_per_particle_phase_shift_is_applied_per_particle():
    kwargs = _synthetic_backend_kwargs(32, 5, torch.device("cpu"))
    phases = torch.tensor([0.0, 20.0, 45.0, 70.0, 90.0])
    kwargs["ctf_kwargs"]["phase_shift"] = phases
    batched = compute_particle_z_scores(kwargs, batch_size=2)
    for j in range(5):
        single = _subset(kwargs, j)
        single["ctf_kwargs"]["phase_shift"] = phases[j]
        expected = compute_particle_z_scores(single, batch_size=1)
        torch.testing.assert_close(batched[j : j + 1], expected, rtol=1e-4, atol=1e-4)


def test_loss_does_not_depend_on_batch_size():
    kwargs = _synthetic_backend_kwargs(32, 7, torch.device("cpu"))
    a = compute_particle_z_scores(dict(kwargs), batch_size=7)
    b = compute_particle_z_scores(dict(kwargs), batch_size=3)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_make_rfft_hermitian_is_real_part_of_inverse():
    gen = torch.Generator().manual_seed(1)
    height, width = 8, 8
    spectrum = torch.randn(height, width // 2 + 1, dtype=torch.complex64, generator=gen)
    fixed = _make_rfft_hermitian(spectrum, width)
    # Unchanged when already consistent:
    consistent = torch.fft.rfftn(torch.randn(height, width, generator=gen))
    torch.testing.assert_close(_make_rfft_hermitian(consistent, width), consistent)
    # Same as taking the real part of the full complex inverse transform:
    full = torch.zeros(height, width, dtype=torch.complex64)
    full[:, : width // 2 + 1] = spectrum
    for kx in range(1, width // 2):
        full[:, width - kx] = spectrum[(-torch.arange(height)) % height, kx].conj()
    expected = torch.fft.ifftn(full).real
    torch.testing.assert_close(torch.fft.irfftn(fixed, s=(height, width)), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Leopard-EM needs CUDA")
# Leopard-EM's torch.compile'd projection normalization warns during tracing on
# recent PyTorch (.grad of a non-leaf tensor); with warnings as errors, dynamo
# turns that into a failure. The warning is in the reference implementation.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_loss_matches_leopard_em():
    from leopard_em.backend.core_differentiable_refine import (
        core_differentiable_refine,
    )

    torch.set_grad_enabled(True)
    device = torch.device("cuda:0")
    kwargs = _synthetic_backend_kwargs(64, 20, device)
    mag = torch.tensor([[1.002, 0.001], [0.0005, 0.998]], device=device)

    def evaluate(use_leopard: bool):
        params = {
            "du": kwargs["defocus_u"].clone().requires_grad_(True),
            "angle": torch.tensor(30.0, device=device, requires_grad=True),
            "mag": mag.clone().requires_grad_(True),
        }
        local = dict(kwargs)
        local["defocus_u"] = params["du"]
        local["defocus_angle"] = params["angle"].expand(20)
        local["mag_matrix"] = params["mag"]
        local["ctf_kwargs"] = dict(kwargs["ctf_kwargs"], mag_matrix=params["mag"])
        if use_leopard:
            z = core_differentiable_refine(batch_size=8, num_cuda_streams=1, **local)[
                "refined_z_score"
            ]
        else:
            z = compute_particle_z_scores(local, batch_size=8)
        z.mean().backward()
        return z.detach(), {k: v.grad.detach() for k, v in params.items()}

    z_ref, g_ref = evaluate(True)
    z_new, g_new = evaluate(False)
    torch.testing.assert_close(z_new, z_ref, rtol=1e-4, atol=1e-3)
    for key, grad in g_ref.items():
        torch.testing.assert_close(g_new[key], grad, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# Non-finite gradients (Zernike terms + trainable magnification matrix)
# ---------------------------------------------------------------------------


def _mag_gradient(zernikes: dict[str, Any]) -> torch.Tensor:
    torch.set_grad_enabled(True)
    kwargs = _synthetic_backend_kwargs(32, 4, torch.device("cpu"))
    mag = torch.tensor([[1.002, 0.001], [0.0005, 0.998]], requires_grad=True)
    kwargs["mag_matrix"] = mag
    kwargs["ctf_kwargs"] = dict(kwargs["ctf_kwargs"], mag_matrix=mag, **zernikes)
    compute_particle_z_scores(kwargs, batch_size=4).mean().backward()
    assert mag.grad is not None
    return mag.grad


def test_mag_gradient_finite_with_zero_zernike_dicts():
    """Leopard-EM results CSVs carry all-zero Zernike dicts."""
    zero = torch.tensor(0.0)
    grad = _mag_gradient(
        {
            "even_zernikes": {"Z44c": zero, "Z44s": zero, "Z60": zero},
            "odd_zernikes": {"Z31c": zero, "Z31s": zero, "Z33c": zero, "Z33s": zero},
        }
    )
    assert torch.isfinite(grad).all()


def test_mag_gradient_finite_with_active_beam_tilt():
    """Needs torch-grid-utils with the atan2-at-origin fix on older PyTorch."""
    coeff = torch.tensor(50.0, requires_grad=True)
    grad = _mag_gradient({"odd_zernikes": {"Z31c": coeff, "Z31s": torch.tensor(-20.0)}})
    assert torch.isfinite(grad).all()


def test_inactive_zernike_dict_detection():
    from picasso.core.core_ctf_refine_loss import _is_inactive_zernike_dict

    assert _is_inactive_zernike_dict({"Z31c": torch.tensor(0.0), "Z31s": 0.0})
    assert not _is_inactive_zernike_dict({"Z31c": torch.tensor(1.0)})
    assert not _is_inactive_zernike_dict(
        {"Z31c": torch.tensor(0.0, requires_grad=True)}
    )
    assert not _is_inactive_zernike_dict(None)
    assert not _is_inactive_zernike_dict({})


def test_non_finite_loss_stops_refinement(synthetic, monkeypatch):
    def nan_loss(
        pbk: dict[str, Any], batch_size: int, backward_weight: float | None = None
    ) -> torch.Tensor:
        loss = synthetic._loss(pbk) * torch.tensor(float("nan"))
        if backward_weight is not None:
            (backward_weight * loss.sum()).backward()
            return loss.detach()
        return loss

    monkeypatch.setattr(driver, "compute_micrograph_loss", nan_loss)
    with pytest.raises(FloatingPointError, match="Non-finite loss in micrograph 1"):
        run(refine_astigmatism=True)


# ---------------------------------------------------------------------------
# Magnification matrix goes to the projection only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refined", [True, False])
def test_mag_matrix_is_applied_to_projection_only(monkeypatch, refined):
    """The CTF is never stretched: CTF parameters come from the recorded image."""
    import picasso.core.core_ctf_refine_utils as utils

    csv_matrix = torch.tensor([[1.01, 0.0], [0.0, 0.99]])

    class FakeParticleStack:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_euler_angles(self, prefer_refined_angles: bool) -> torch.Tensor:
            return torch.zeros(3, 3)

    def fake_setup(**kwargs: Any) -> dict[str, Any]:
        # Leopard-EM puts the results-CSV matrix on both projection and CTF.
        return {"mag_matrix": csv_matrix, "ctf_kwargs": {"mag_matrix": csv_matrix}}

    monkeypatch.setattr(utils, "ParticleStack", FakeParticleStack)
    monkeypatch.setattr(utils, "setup_particle_backend_kwargs", fake_setup)
    refined_matrix = torch.tensor([[0.998, 0.001], [0.001, 0.997]])
    pbk = utils.setup_particle_backend_for_micrograph(
        results_path="unused.csv",
        template=torch.zeros(4, 4, 4),
        template_box_size=(4, 4),
        device=torch.device("cpu"),
        preprocessing_filters=None,
        euler_angle_offsets=torch.zeros(1, 3),
        defocus_offsets=torch.zeros(1),
        pixel_size_offsets=torch.zeros(1),
        apply_global_filtering=True,
        anistropic_mag=refined_matrix if refined else None,
    )
    assert pbk["ctf_kwargs"]["mag_matrix"] is None
    assert pbk["mag_matrix"] is (refined_matrix if refined else csv_matrix)


# ---------------------------------------------------------------------------
# Input cache
# ---------------------------------------------------------------------------


def test_inputs_built_once_per_micrograph(synthetic, monkeypatch):
    calls = {"setup": 0}
    original = synthetic.setup

    def counting_setup(**kwargs: Any) -> dict[str, Any]:
        calls["setup"] += 1
        return original(**kwargs)

    monkeypatch.setattr(driver, "setup_particle_backend_for_micrograph", counting_setup)
    synthetic.target_mag = torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    flags = {"refine_anisotropic_mag": True, "refine_astigmatism": True}

    cached = run(**flags)
    assert calls["setup"] == 2  # one build per micrograph for the whole run

    calls["setup"] = 0
    uncached = run(**flags, setup_cache_gpu_gb=0, setup_cache_host_gb=0)
    assert calls["setup"] > 10  # rebuilt for every evaluation
    torch.testing.assert_close(cached["anistropic_mag"], uncached["anistropic_mag"])
    torch.testing.assert_close(
        cached["astigmatism_magnitude"], uncached["astigmatism_magnitude"]
    )


def test_input_cache_respects_budget():
    from picasso.core.core_ctf_refine_cache import MicrographInputCache

    cpu = torch.device("cpu")
    entry_bytes = 4 * 100
    cache = MicrographInputCache({cpu: entry_bytes}, host_budget_bytes=0)
    builds = []

    def build(index: int):
        def _build() -> dict[str, Any]:
            builds.append(index)
            return {"x": torch.full((100,), float(index))}

        return _build

    first = cache.get(0, cpu, build(0))
    assert cache.get(0, cpu, build(0)) is first  # cached on the device tier
    cache.get(1, cpu, build(1))
    cache.get(1, cpu, build(1))  # over budget: rebuilt
    assert builds == [0, 1, 1]
    assert "1 micrograph(s) on GPU" in cache.summary()
    assert "1 rebuilt on every use" in cache.summary()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_input_cache_host_tier():
    from picasso.core.core_ctf_refine_cache import MicrographInputCache

    gpu = torch.device("cuda:0")
    cache = MicrographInputCache({gpu: 0}, host_budget_bytes=10**6)
    builds = []

    def build() -> dict[str, Any]:
        builds.append(1)
        return {"x": torch.arange(10.0, device=gpu), "meta": (3, 4)}

    cache.get(0, gpu, build)
    again = cache.get(0, gpu, build)
    assert builds == [1]
    assert again["x"].device == gpu and again["meta"] == (3, 4)
    torch.testing.assert_close(again["x"].cpu(), torch.arange(10.0))
    assert "1 in host memory" in cache.summary()


@pytest.mark.parametrize("shared_filter", [False, True])
def test_compacted_inputs_give_same_z_scores(shared_filter):
    from picasso.core.core_ctf_refine_loss import (
        compact_backend_kwargs,
        per_call_kwargs,
        without_shared_template,
    )

    kwargs = _synthetic_backend_kwargs(32, 6, torch.device("cpu"))
    if shared_filter:
        kwargs["projective_filters"] = (
            kwargs["projective_filters"][:1].expand(6, -1, -1).clone()
        )
    template_dft = kwargs["template_dft"]
    reference = compute_particle_z_scores(dict(kwargs), batch_size=4)

    cached = compact_backend_kwargs(dict(kwargs), torch.device("cpu"), template_dft)
    without_shared_template(cached)
    assert "particle_stack_dft" not in cached
    static = cached["_picasso_loss_cache"]
    assert "template_dft" not in static
    assert static["projective_filters"].shape[0] == (1 if shared_filter else 6)
    for _ in range(2):  # repeated evaluations reuse the same cached inputs
        z = compute_particle_z_scores(
            per_call_kwargs(cached, template_dft), batch_size=4
        )
        torch.testing.assert_close(z, reference)


# ---------------------------------------------------------------------------
# Chunked backward, reused evaluations, GPU balancing
# ---------------------------------------------------------------------------


def test_chunked_backward_matches_full_backward():
    torch.set_grad_enabled(True)
    kwargs = _synthetic_backend_kwargs(32, 7, torch.device("cpu"))

    def grads(chunked: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        du = kwargs["defocus_u"].clone().requires_grad_(True)
        mag = torch.tensor([[1.002, 0.001], [0.0005, 0.998]], requires_grad=True)
        local = dict(kwargs, defocus_u=du, mag_matrix=mag)
        if chunked:
            z = compute_particle_z_scores(
                local, batch_size=2, backward_weight=0.5, backward_group=3
            )
            assert not z.requires_grad
        else:
            z = compute_particle_z_scores(local, batch_size=2)
            (0.5 * z.sum()).backward()
        return z.detach(), [du.grad, mag.grad]

    z_full, g_full = grads(chunked=False)
    z_chunk, g_chunk = grads(chunked=True)
    torch.testing.assert_close(z_chunk, z_full)
    for a, b in zip(g_chunk, g_full, strict=True):
        # Summing gradients group by group changes float32 rounding slightly.
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_lbfgs_step_start_is_not_recomputed(synthetic, monkeypatch, capsys):
    """Each L-BFGS step's first closure and the post-step loss reuse results."""
    synthetic.target_mag = torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    calls = {"grad": 0}
    original = driver._micrograph_loop

    def counting_loop(**kwargs: Any):
        calls["grad"] += int(kwargs["grad_weight"] is not None)
        return original(**kwargs)

    monkeypatch.setattr(driver, "_micrograph_loop", counting_loop)
    capsys.readouterr()
    result = run(refine_anisotropic_mag=True, per_dataset_iterations=3)
    out = capsys.readouterr().out
    iterations = len(result["dataset_losses"])
    closures = out.count("L-BFGS closure #")
    reused = out.count("(reused)")
    assert reused >= iterations  # every step starts where the previous one ended
    gradient_passes = calls["grad"] / 2  # two micrographs
    # Initial pass + the closures that were actually computed, plus at most one
    # post-step evaluation per iteration that the last closure did not cover.
    computed = 1 + closures - reused
    assert computed <= gradient_passes <= computed + iterations


def test_balance_micrographs_across_gpus():
    from picasso.core.core_ctf_refine_utils import balance_micrographs_across_gpus

    assignments = balance_micrographs_across_gpus([100, 100, 100, 100], 3)
    assert sorted(len(a) for a in assignments) == [1, 1, 2]
    assignments = balance_micrographs_across_gpus([400, 50, 50, 50, 50, 50], 2)
    assert assignments == [[0], [1, 2, 3, 4, 5]]  # 400 vs 250 particles
    assert balance_micrographs_across_gpus([5, 5], 1) == [[0, 1]]


# ---------------------------------------------------------------------------
# Cartesian astigmatism; magnification matrix from the results CSVs
# ---------------------------------------------------------------------------


def _set_synthetic_astigmatism(monkeypatch, du, dv, angle, true_mag, true_angle):
    module = sys.modules[__name__]
    for name, value in {
        "CSV_DU": du,
        "CSV_DV": dv,
        "CSV_ANGLE": angle,
        "TRUE_MAG": true_mag,
        "TRUE_ANGLE": true_angle,
    }.items():
        monkeypatch.setattr(module, name, value)


def test_astigmatism_angle_reported_near_start(synthetic, monkeypatch):
    """170 deg -> 175 deg must not come back as -5 deg (same astigmatism)."""
    _set_synthetic_astigmatism(monkeypatch, 12100.0, 11700.0, 170.0, 450.0, 175.0)
    result = run(refine_astigmatism=True, per_micrograph_iterations=10)
    torch.testing.assert_close(
        result["astigmatism_angle"], torch.full((2,), 175.0), atol=0.1, rtol=0
    )
    torch.testing.assert_close(
        result["astigmatism_magnitude"], torch.full((2,), 450.0), atol=1.0, rtol=0
    )


def test_astigmatism_refines_from_zero(synthetic, monkeypatch):
    """A micrograph with defocus_u == defocus_v can still gain astigmatism."""
    _set_synthetic_astigmatism(monkeypatch, 11900.0, 11900.0, 0.0, 300.0, 20.0)
    result = run(refine_astigmatism=True, per_micrograph_iterations=10)
    torch.testing.assert_close(
        result["astigmatism_magnitude"], torch.full((2,), 300.0), atol=1.0, rtol=0
    )
    torch.testing.assert_close(
        result["astigmatism_angle"], torch.full((2,), 20.0), atol=0.1, rtol=0
    )


def _write_results_csv(path, mag_matrix: str | None) -> str:
    frame = {"defocus_u": [1.0, 1.0]}
    if mag_matrix is not None:
        frame["mag_matrix"] = [mag_matrix, mag_matrix]
    pd.DataFrame(frame).to_csv(path, index=False)
    return str(path)


def test_read_results_mag_matrix(tmp_path):
    from picasso.core.core_ctf_refine_utils import (
        common_results_mag_matrix,
        read_results_mag_matrix,
    )

    a = _write_results_csv(tmp_path / "a.csv", "[1.01, 0.002, 0.001, 0.99]")
    b = _write_results_csv(tmp_path / "b.csv", "[1.01, 0.002, 0.001, 0.99]")
    c = _write_results_csv(tmp_path / "c.csv", "[1.0, 0.0, 0.0, 1.0]")
    empty = _write_results_csv(tmp_path / "d.csv", "")
    missing = _write_results_csv(tmp_path / "e.csv", None)

    expected = torch.tensor([[1.01, 0.002], [0.001, 0.99]])
    torch.testing.assert_close(read_results_mag_matrix(a), expected)
    assert read_results_mag_matrix(empty) is None
    assert read_results_mag_matrix(missing) is None
    assert read_results_mag_matrix(str(tmp_path / "nope.csv")) is None

    matrix, consistent = common_results_mag_matrix([a, b])
    assert consistent
    torch.testing.assert_close(matrix, expected)
    assert common_results_mag_matrix([a, c]) == (None, False)
    assert common_results_mag_matrix([c, empty, missing]) == (None, True)


@pytest.mark.parametrize("refine", [False, True])
def test_mag_matrix_starts_from_results_csv(synthetic, tmp_path, refine):
    csv_matrix = "[1.01, 0.0, 0.0, 0.99]"
    files = [
        _write_results_csv(tmp_path / f"m{k}_results.csv", csv_matrix) for k in (0, 1)
    ]
    synthetic.target_mag = torch.tensor([[1.01, 0.0], [0.0, 0.99]])
    result = run(results_files=files, refine_anisotropic_mag=refine)
    # Reported (not refined) or kept (already optimal) rather than identity.
    torch.testing.assert_close(
        result["anistropic_mag"], synthetic.target_mag, atol=1e-6, rtol=0
    )


# ---------------------------------------------------------------------------
# Joint vs block-coordinate per-micrograph L-BFGS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint", [True, False])
def test_per_micrograph_lbfgs_modes_reach_optimum(synthetic, joint):
    result = run(
        refine_astigmatism=True,
        refine_defocus=True,
        refine_phase_shift=True,
        per_micrograph_iterations=10,
        astigmatism_magnitude_lr=1.0,
        defocus_lr=0.05,
        phase_shift_lr=1.0,
        joint_per_micrograph_lbfgs=joint,
    )
    torch.testing.assert_close(
        result["astigmatism_magnitude"], torch.full((2,), TRUE_MAG), atol=1.0, rtol=0
    )
    torch.testing.assert_close(
        result["astigmatism_angle"], torch.full((2,), TRUE_ANGLE), atol=0.1, rtol=0
    )
    for offsets in result["defocus_all"].values():
        torch.testing.assert_close(offsets, TRUE_DEFOCUS, atol=1.0, rtol=0)
    for phase in result["phase_shift_all"].values():
        torch.testing.assert_close(phase, torch.tensor(TRUE_PHASE), atol=0.1, rtol=0)


def test_joint_lbfgs_uses_one_optimizer(synthetic, monkeypatch):
    created = []
    original = torch.optim.LBFGS

    class CountingLBFGS(original):  # type: ignore[misc,valid-type]
        def __init__(self, params: Any, **kwargs: Any) -> None:
            created.append(kwargs.get("lr"))
            super().__init__(params, **kwargs)

    monkeypatch.setattr(torch.optim, "LBFGS", CountingLBFGS)
    run(
        refine_astigmatism=True,
        refine_defocus=True,
        per_micrograph_iterations=2,
        joint_per_micrograph_lbfgs=True,
    )
    assert created == [1.0, 1.0]  # one optimizer per micrograph
    created.clear()
    run(
        refine_astigmatism=True,
        refine_defocus=True,
        per_micrograph_iterations=2,
        joint_per_micrograph_lbfgs=False,
    )
    assert len(created) == 4  # two groups per micrograph

"""Tests for CTF estimation configs, mapping, writers, and manager wiring."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml
from torch_ctf_estimation.models import (
    CTF,
    CTFFittingParams,
    Defocus1DResults,
    Defocus2DResults,
    LinearDefocusModel,
    OpticalParams,
    Thickness1DResults,
    ThicknessParams,
)

from picasso.config import (
    ComputationalConfig,
    CtfEstimateInputConfig,
    CtfEstimateOutputConfig,
)
from picasso.core.core_ctf_estimate import CtfEstimateCoreResult
from picasso.managers.ctf_estimate_manager import CTFEstimateManager
from picasso.utils.ctf_estimate_io import (
    result_to_leopard_optics,
    result_to_metrics_row,
    write_metrics_csv,
    write_optics_csv,
    write_optics_yaml,
)
from picasso.utils.ctf_estimate_parallel import (
    analytic_peak_bytes,
    clamp_k,
    resolve_k_cpu,
    resolve_k_gpu,
)

# Implemented OpticsGroup fields (Leopard-EM extra="forbid"; no thickness).
LEOPARD_OPTICS_FIELDS = {
    "label",
    "voltage",
    "pixel_size",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "programs/ctf_estimation/ctf_estimate_example_config.yaml"
LASER_EXAMPLE_YAML = (
    REPO_ROOT / "programs/ctf_estimation/ctf_estimate_laser_example_config.yaml"
)


def _touch_mrc(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def _optical_params() -> OpticalParams:
    return OpticalParams(
        pixel_spacing_angstroms=1.06,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.07,
        target_pixel_spacing_angstroms=3.0,
    )


def _fake_result1d() -> Defocus1DResults:
    return Defocus1DResults(
        frequencies_1d=torch.linspace(0, 0.5, 16),
        ctf_model=CTF(
            defocus_um=torch.tensor(1.5),
            voltage_kev=torch.tensor(300.0),
            spherical_aberration_mm=torch.tensor(2.7),
            amplitude_contrast_fraction=torch.tensor(0.07),
            phase_shift_degrees=torch.tensor(0.0),
        ),
        cross_correlation_final=0.7,
    )


def _fake_result2d(
    *, defocus_u: float = 1.6, defocus_v: float = 1.4
) -> Defocus2DResults:
    return Defocus2DResults(
        defocus_model=LinearDefocusModel(
            defocus_0=1.5,
            defocus_gradient_magnitude=0.0,
            defocus_gradient_angle=0.0,
        ),
        defocus_model_type="linear",
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        astigmatism=0.2,
        astigmatism_angle=25.0,
        envelope_B=40.0,
        phase_shift_degrees=12.0,
        phase_shift_model_type="grid",
        cross_correlation_final=0.8,
        tilt_axis_angle_deg=10.0,
        tilt_magnitude_deg=2.0,
    )


def _fake_thickness1d() -> Thickness1DResults:
    return Thickness1DResults(
        thickness_angstroms=1200.0,
        frequencies_1d=torch.linspace(0, 0.5, 16),
        cross_correlation_final=0.6,
    )


def _fake_core_result() -> CtfEstimateCoreResult:
    return CtfEstimateCoreResult(
        mean_ps=torch.ones(8, 5),
        result1d=_fake_result1d(),
        result2d=_fake_result2d(),
        thickness1d=_fake_thickness1d(),
        thickness_joint=_fake_thickness1d(),
        pixel_spacing_used=3.0,
    )


class TestCtfEstimateInputConfig:
    """Image discovery from a single path and/or a directory glob."""

    def test_requires_at_least_one_source(self):
        with pytest.raises(ValueError, match="image_path and/or images_directory"):
            CtfEstimateInputConfig()

    def test_discover_single_path(self, tmp_path: Path):
        image = _touch_mrc(tmp_path / "solo.mrc")
        config = CtfEstimateInputConfig(image_path=str(image))
        assert config.discover_images() == [image]

    def test_discover_directory_glob(self, tmp_path: Path):
        first = _touch_mrc(tmp_path / "a.mrc")
        second = _touch_mrc(tmp_path / "b.mrc")
        _touch_mrc(tmp_path / "skip.txt")
        config = CtfEstimateInputConfig(images_directory=str(tmp_path))
        assert config.discover_images() == [first, second]

    def test_discover_both_deduplicates(self, tmp_path: Path):
        shared = _touch_mrc(tmp_path / "shared.mrc")
        extra = _touch_mrc(tmp_path / "extra.mrc")
        config = CtfEstimateInputConfig(
            image_path=str(shared),
            images_directory=str(tmp_path),
        )
        discovered = config.discover_images()
        assert discovered[0] == shared
        assert extra in discovered
        assert len(discovered) == 2

    def test_missing_image_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="image_path does not exist"):
            CtfEstimateInputConfig(image_path=str(tmp_path / "missing.mrc"))


class TestCtfEstimateOutputConfig:
    """Output directory and plot flags."""

    def test_defaults(self, tmp_path: Path):
        config = CtfEstimateOutputConfig(output_directory=str(tmp_path / "out"))
        assert config.output_plots is False
        assert config.save_debug_arrays is False
        assert config.optics_csv_filename == "ctf_estimate_optics.csv"
        assert config.metrics_csv_filename == "ctf_estimate_metrics.csv"


class TestCtfEstimateMapping:
    """µm → Å conversion and Leopard OpticsGroup field names."""

    def test_microns_to_angstroms_and_original_pixel_size(self):
        optics = result_to_leopard_optics(
            _fake_result2d(),
            _optical_params(),
            "/data/micro.mrc",
            label="micro",
        )
        assert optics["defocus_u"] == pytest.approx(16000.0)
        assert optics["defocus_v"] == pytest.approx(14000.0)
        assert optics["pixel_size"] == pytest.approx(1.06)
        assert optics["pixel_size"] != pytest.approx(3.0)
        assert optics["phase_shift"] == pytest.approx(12.0)
        assert optics["ctf_B_factor"] == pytest.approx(40.0)
        assert optics["astigmatism_angle"] == pytest.approx(25.0)
        assert set(optics) == LEOPARD_OPTICS_FIELDS

    def test_metrics_bound_hits_and_thickness(self):
        row = result_to_metrics_row(
            "/data/micro.mrc",
            _fake_result1d(),
            _fake_result2d(defocus_u=5.0, defocus_v=5.0),
            _optical_params(),
            (0.0, 5.0),
            (0.0, 12.0),
            _fake_thickness1d(),
            _fake_thickness1d(),
        )
        assert row["hit_defocus_bound"] is True
        assert row["hit_phase_bound"] is True
        assert row["thickness_angstroms"] == pytest.approx(1200.0)
        assert row["cc_thickness_1d"] == pytest.approx(0.6)
        assert row["cc_thickness_joint"] == pytest.approx(0.6)
        assert row["defocus_spatial_range_A"] == pytest.approx(0.0)
        assert "exclude" not in row

    def test_spatial_defocus_field_grid_range(self):
        from torch_cubic_spline_grids import CubicCatmullRomGrid3d

        from picasso.utils.ctf_estimate_io import spatial_defocus_field

        knots = torch.tensor([[[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]])
        result = Defocus2DResults(
            defocus_model_type="grid",
            defocus_model=CubicCatmullRomGrid3d.from_grid_data(knots),
            defocus_u=1.4,
            defocus_v=1.4,
        )
        field = spatial_defocus_field(result)
        assert field["defocus_spatial_min_A"] == pytest.approx(10000.0)
        assert field["defocus_spatial_max_A"] == pytest.approx(18000.0)
        assert field["defocus_spatial_range_A"] == pytest.approx(8000.0)
        assert field["defocus_grid_um"] is not None


class TestConcurrencyHelpers:
    """K selection from VRAM / cores / RAM."""

    def test_clamp_k(self):
        assert clamp_k(0, 10) == 1
        assert clamp_k(100, 3) == 3
        assert clamp_k(100, 20) == 8
        assert clamp_k(2, 1) == 1

    def test_resolve_k_gpu_forced_and_auto(self):
        assert resolve_k_gpu(forced=2, free_bytes=0, peak_bytes=1, n_images=4) == 2
        peak = 100
        free = int(peak * 1.5 * 4)
        k = resolve_k_gpu(forced=None, free_bytes=free, peak_bytes=peak, n_images=10)
        assert k == 4

    def test_resolve_k_cpu_forced_and_auto(self):
        assert resolve_k_cpu(forced=3, n_images=10) == 3
        assert (
            resolve_k_cpu(
                forced=None,
                n_images=10,
                cpu_threads_per_worker=1,
                n_cores=8,
                ram_bytes=8 * 1024**3,
            )
            == 8
        )
        assert (
            resolve_k_cpu(
                forced=None,
                n_images=10,
                cpu_threads_per_worker=2,
                n_cores=8,
                ram_bytes=3 * 1024**3,
            )
            == 3
        )

    def test_analytic_peak_bytes_positive(self):
        peak = analytic_peak_bytes((1, 4096, 4096), 1.06, 3.0, 256)
        assert peak > 64 * 1024 * 1024

    def test_reject_mixed_cpu_gpu(self):
        with pytest.raises(ValueError, match="cannot mix CPU and GPU"):
            ComputationalConfig(gpu_ids=[0, "cpu"])


class TestCtfEstimateWriters:
    """CSV and per-micrograph YAML writers."""

    def test_optics_yaml_thickness_is_sibling(self, tmp_path: Path):
        optics = result_to_leopard_optics(
            _fake_result2d(),
            _optical_params(),
            "/data/micro.mrc",
            label="micro",
        )
        path = tmp_path / "micro_optics.yaml"
        write_optics_yaml(optics, path, thickness_angstroms=1200.0)
        payload = yaml.safe_load(path.read_text())
        assert "optics_group" in payload
        assert payload["thickness_angstroms"] == pytest.approx(1200.0)
        assert "thickness_angstroms" not in payload["optics_group"]
        assert set(payload["optics_group"]) == LEOPARD_OPTICS_FIELDS

    def test_optics_yaml_omits_thickness_when_unset(self, tmp_path: Path):
        optics = result_to_leopard_optics(
            _fake_result2d(),
            _optical_params(),
            "/data/micro.mrc",
        )
        path = tmp_path / "micro_optics.yaml"
        write_optics_yaml(optics, path, thickness_angstroms=None)
        payload = yaml.safe_load(path.read_text())
        assert "thickness_angstroms" not in payload

    def test_csv_writers(self, tmp_path: Path):
        optics = result_to_leopard_optics(
            _fake_result2d(),
            _optical_params(),
            "/data/micro.mrc",
            label="micro",
        )
        optics_path = tmp_path / "optics.csv"
        write_optics_csv(
            [{"micrograph_path": "/data/micro.mrc", **optics}],
            optics_path,
        )
        with optics_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["label"] == "micro"
        assert float(rows[0]["defocus_u"]) == pytest.approx(16000.0)

        metrics = result_to_metrics_row(
            "/data/micro.mrc",
            _fake_result1d(),
            _fake_result2d(),
            _optical_params(),
            (0.0, 5.0),
            None,
            _fake_thickness1d(),
            None,
        )
        metrics_path = tmp_path / "metrics.csv"
        write_metrics_csv([metrics], metrics_path)
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        assert "exclude" not in metric_rows[0]
        assert metric_rows[0]["thickness_angstroms"] == "1200.0"


class TestCTFEstimateManager:
    """Manager construction and export without running a GPU fit."""

    def _write_manager_yaml(
        self,
        tmp_path: Path,
        image: Path | None = None,
        images_directory: Path | None = None,
    ) -> Path:
        output_dir = tmp_path / "output"
        if images_directory is not None:
            input_config: dict = {"images_directory": str(images_directory)}
        else:
            input_config = {"image_path": str(image)}
        payload = {
            "input_config": input_config,
            "output_config": {
                "output_directory": str(output_dir),
                "allow_file_overwrite": True,
                "output_plots": False,
            },
            "computational_config": {
                "gpu_ids": "cpu",
                "images_per_device": 1,
            },
            "optical_config": {
                "pixel_spacing_angstroms": 1.06,
                "voltage_kev": 300.0,
            },
            "fitting_config": {
                "defocus_grid_resolution": [1, 1, 1],
                "frequency_fit_range_angstroms": [30.0, 5.0],
                "early_stopping": True,
                "early_stopping_patience": 4,
            },
            "thickness_config": {
                "refine_dim": "2d",
                "thickness_range_angstroms": [300.0, 4000.0],
                "frequency_fit_range_angstroms": [20.0, 4.0],
                "defocus_range_microns": [0.5, 3.0],
                "early_stopping": False,
            },
        }
        yaml_path = tmp_path / "estimate.yaml"
        yaml_path.write_text(yaml.safe_dump(payload))
        return yaml_path

    def test_from_temp_yaml(self, tmp_path: Path):
        image = _touch_mrc(tmp_path / "micro.mrc")
        yaml_path = self._write_manager_yaml(tmp_path, image)
        manager = CTFEstimateManager.from_yaml(yaml_path)
        assert manager.input_config.discover_images() == [image]
        assert isinstance(manager.optical_config, OpticalParams)
        assert isinstance(manager.fitting_config, CTFFittingParams)
        assert isinstance(manager.thickness_config, ThicknessParams)
        assert manager.thickness_config.refine_dim == "2d"
        assert manager.fitting_config.early_stopping is True
        assert manager.fitting_config.early_stopping_patience == 4
        assert manager.fitting_config.build_early_stopper() is not None
        assert manager.thickness_config.early_stopping is False
        assert manager.thickness_config.frequency_fit_range_angstroms == (20.0, 4.0)
        assert manager.thickness_config.defocus_range_microns == (0.5, 3.0)
        assert manager.laser_config is None

    def test_estimate_and_export_with_mocked_core(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        image = _touch_mrc(tmp_path / "micro.mrc")
        yaml_path = self._write_manager_yaml(tmp_path, image)
        manager = CTFEstimateManager.from_yaml(yaml_path)
        fake = _fake_core_result()

        def _fake_estimate(_self, _path, device=None):
            return fake

        monkeypatch.setattr(CTFEstimateManager, "estimate_one", _fake_estimate)

        results = manager.estimate_and_export()
        assert len(results) == 1
        output_dir = Path(manager.output_config.output_directory)
        assert (output_dir / "ctf_estimate_optics.csv").is_file()
        assert (output_dir / "ctf_estimate_metrics.csv").is_file()
        yaml_path = output_dir / "micro_optics.yaml"
        payload = yaml.safe_load(yaml_path.read_text())
        assert payload["thickness_angstroms"] == pytest.approx(1200.0)
        assert set(payload["optics_group"]) == LEOPARD_OPTICS_FIELDS

    def test_estimate_and_export_dispatches_across_devices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        first = _touch_mrc(tmp_path / "a.mrc")
        second = _touch_mrc(tmp_path / "b.mrc")
        yaml_path = self._write_manager_yaml(tmp_path, images_directory=tmp_path)
        manager = CTFEstimateManager.from_yaml(yaml_path)
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        monkeypatch.setattr(
            type(manager.computational_config),
            "gpu_devices",
            property(lambda _self: devices),
        )
        fake = _fake_core_result()
        calls: list[tuple[str, str]] = []

        def _fake_estimate(_self, path, device=None):
            calls.append((Path(path).name, str(device)))
            return fake

        monkeypatch.setattr(CTFEstimateManager, "estimate_one", _fake_estimate)
        manager.computational_config.images_per_device = 1

        results = manager.estimate_and_export()
        assert len(results) == 2
        assert {device for _name, device in calls} == {"cuda:0", "cuda:1"}
        assert [name for name, _device in sorted(calls)] == ["a.mrc", "b.mrc"]
        output_dir = Path(manager.output_config.output_directory)
        with (output_dir / "ctf_estimate_optics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert [Path(row["micrograph_path"]).name for row in rows] == [
            first.name,
            second.name,
        ]
        assert (output_dir / "a_optics.yaml").is_file()
        assert (output_dir / "b_optics.yaml").is_file()

    def test_single_gpu_thread_pool_forced_k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _touch_mrc(tmp_path / "a.mrc")
        _touch_mrc(tmp_path / "b.mrc")
        yaml_path = self._write_manager_yaml(tmp_path, images_directory=tmp_path)
        manager = CTFEstimateManager.from_yaml(yaml_path)
        manager.computational_config.images_per_device = 2
        monkeypatch.setattr(
            type(manager.computational_config),
            "gpu_devices",
            property(lambda _self: [torch.device("cuda:0")]),
        )
        fake = _fake_core_result()
        calls: list[str] = []

        def _fake_estimate(_self, path, device=None):
            calls.append(str(device))
            return fake

        monkeypatch.setattr(CTFEstimateManager, "estimate_one", _fake_estimate)
        results = manager.estimate_and_export()
        assert len(results) == 2
        assert calls == ["cuda:0", "cuda:0"]

    def test_cpu_process_pool_mocked_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _touch_mrc(tmp_path / "a.mrc")
        _touch_mrc(tmp_path / "b.mrc")
        yaml_path = self._write_manager_yaml(tmp_path, images_directory=tmp_path)
        manager = CTFEstimateManager.from_yaml(yaml_path)
        manager.computational_config.images_per_device = 2
        fake = _fake_core_result()

        def _fake_estimate(_self, path, device=None):
            return fake

        monkeypatch.setattr(CTFEstimateManager, "estimate_one", _fake_estimate)

        class _InlinePool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, payload):
                from concurrent.futures import Future

                future = Future()
                future.set_result(fn(payload))
                return future

        monkeypatch.setattr(
            "picasso.managers.ctf_estimate_manager.ProcessPoolExecutor",
            _InlinePool,
        )
        results = manager.estimate_and_export()
        assert len(results) == 2
        output_dir = Path(manager.output_config.output_directory)
        with (output_dir / "ctf_estimate_optics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert [Path(row["micrograph_path"]).name for row in rows] == [
            "a.mrc",
            "b.mrc",
        ]

    def test_example_yaml_loads_after_rewriting_paths(self, tmp_path: Path):
        image = _touch_mrc(tmp_path / "micro.mrc")
        data = yaml.safe_load(EXAMPLE_YAML.read_text())
        data["input_config"]["image_path"] = str(image)
        data["input_config"]["images_directory"] = None
        data["output_config"]["output_directory"] = str(tmp_path / "out")
        data["computational_config"]["gpu_ids"] = "cpu"
        yaml_path = tmp_path / "example.yaml"
        yaml_path.write_text(yaml.safe_dump(data))
        manager = CTFEstimateManager.from_yaml(yaml_path)
        assert manager.thickness_config is not None
        assert manager.thickness_config.refine_dim == "thickness"
        assert manager.laser_config is None

    def test_laser_example_yaml_loads_after_rewriting_paths(self, tmp_path: Path):
        image = _touch_mrc(tmp_path / "micro.mrc")
        data = yaml.safe_load(LASER_EXAMPLE_YAML.read_text())
        data["input_config"]["image_path"] = str(image)
        data["input_config"]["images_directory"] = None
        data["output_config"]["output_directory"] = str(tmp_path / "out")
        data["computational_config"]["gpu_ids"] = "cpu"
        yaml_path = tmp_path / "laser.yaml"
        yaml_path.write_text(yaml.safe_dump(data))
        manager = CTFEstimateManager.from_yaml(yaml_path)
        assert manager.laser_config is not None
        assert manager.laser_config.model_laser is True
        assert manager.fitting_config.mask_laser_axis is True
        assert manager.thickness_config is not None


def test_core_does_not_import_convenience_estimate_ctf():
    """PICASSO core must chain primitives, not estimate_ctf."""
    source = (REPO_ROOT / "src/picasso/core/core_ctf_estimate.py").read_text(
        encoding="utf-8"
    )
    assert "from torch_ctf_estimation.estimate_ctf import" not in source
    assert "from torch_ctf_estimation import estimate_ctf" not in source

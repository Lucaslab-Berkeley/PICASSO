"""Manager for CTF estimation over one image or a directory of micrographs."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from threading import Lock, Thread
from typing import ClassVar

import torch
import tqdm
from pydantic import ConfigDict
from torch_ctf_estimation.models import (
    CTFFittingParams,
    LaserParams,
    OpticalParams,
    ThicknessParams,
)

from picasso.config import (
    ComputationalConfig,
    CtfEstimateInputConfig,
    CtfEstimateOutputConfig,
)
from picasso.core.core_ctf_estimate import CtfEstimateCoreResult, core_estimate_ctf
from picasso.utils.ctf_estimate_io import (
    result_to_leopard_optics,
    result_to_metrics_row,
    spatial_defocus_field,
    write_estimate_debug_npz,
    write_metrics_csv,
    write_optics_csv,
    write_optics_yaml,
)
from picasso.utils.ctf_estimate_parallel import (
    analytic_peak_bytes,
    cpu_estimate_worker,
    cuda_free_bytes,
    read_mrc_shape,
    resolve_k_cpu,
    resolve_k_gpu,
)
from picasso.utils.custom_types import BaseModelPICASSO
from picasso.utils.data_io import read_mrc_to_tensor


def _round_robin_indices(n_items: int, n_devices: int) -> list[list[int]]:
    """Assign item indices to devices (same algorithm as refine)."""
    assignments: list[list[int]] = [[] for _ in range(n_devices)]
    for idx in range(n_items):
        assignments[idx % n_devices].append(idx)
    return assignments


class CTFEstimateManager(BaseModelPICASSO):
    """YAML-facing manager for the CTF estimation program."""

    model_config: ClassVar = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    input_config: CtfEstimateInputConfig
    output_config: CtfEstimateOutputConfig
    computational_config: ComputationalConfig
    optical_config: OpticalParams
    fitting_config: CTFFittingParams
    laser_config: LaserParams | None = None
    thickness_config: ThicknessParams | None = None

    def estimate_one(
        self,
        image_path: str | Path,
        device: torch.device | None = None,
    ) -> CtfEstimateCoreResult:
        """Load one MRC and run the core estimation chain.

        Parameters
        ----------
        image_path : str | Path
            Path to the micrograph MRC.
        device : torch.device | None
            Device for this image. If None, uses the first configured device.
        """
        if device is None:
            device = self.computational_config.gpu_device
        image = read_mrc_to_tensor(image_path)
        return core_estimate_ctf(
            image=image,
            optical_params=self.optical_config,
            fitting_params=self.fitting_config,
            device=device,
            laser_params=self.laser_config,
            thickness_params=self.thickness_config,
        )

    def worker_counts(
        self,
        n_images: int,
        image_shape: tuple[int, int, int] | None = None,
    ) -> list[tuple[torch.device, int]]:
        """Return ``(device, K)`` for each configured device."""
        devices = self.computational_config.gpu_devices
        if image_shape is None:
            image_shape = (1, 4096, 4096)
        return [
            (device, self._k_for_device(device, n_images, image_shape))
            for device in devices
        ]

    def estimate_and_export(self) -> list[CtfEstimateCoreResult]:
        """Discover images, estimate (optionally concurrent), and write outputs.

        Per-micrograph YAML/npz/plots are written as each image finishes. Dataset
        CSVs are rewritten after each completion so they stay current.
        """
        image_paths = self.input_config.discover_images()
        output_dir = Path(self.output_config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        devices = self.computational_config.gpu_devices
        image_shape = read_mrc_shape(image_paths[0])
        n_images = len(image_paths)

        global_pbar = tqdm.tqdm(
            total=n_images,
            desc="CTF estimate",
            dynamic_ncols=True,
            smoothing=0.1,
            unit="mgraph",
            leave=True,
            position=0,
        )
        device_pbars: dict[int, tqdm.tqdm] = {}
        if not all(device.type == "cpu" for device in devices):
            assignments = _round_robin_indices(n_images, len(devices))
            for gpu_id, device in enumerate(devices):
                device_pbars[gpu_id] = tqdm.tqdm(
                    total=len(assignments[gpu_id]),
                    desc=f"device - {device.type} {device.index}",
                    dynamic_ncols=True,
                    smoothing=0.1,
                    unit="mgraph",
                    position=gpu_id + 1,
                    leave=True,
                )

        lock = Lock()
        optics_by_idx: dict[int, dict] = {}
        metrics_by_idx: dict[int, dict] = {}
        result_by_idx: dict[int, CtfEstimateCoreResult] = {}

        def on_complete(
            idx: int,
            image_path: Path,
            core_result: CtfEstimateCoreResult,
            gpu_id: int | None = None,
        ) -> None:
            with lock:
                optics_row, metrics_row = self._export_one(image_path, core_result)
                optics_by_idx[idx] = optics_row
                metrics_by_idx[idx] = metrics_row
                result_by_idx[idx] = core_result
                ordered = sorted(optics_by_idx)
                write_optics_csv(
                    [optics_by_idx[i] for i in ordered],
                    output_dir / self.output_config.optics_csv_filename,
                )
                write_metrics_csv(
                    [metrics_by_idx[i] for i in ordered],
                    output_dir / self.output_config.metrics_csv_filename,
                )
                global_pbar.update(1)
                global_pbar.set_postfix_str(image_path.name, refresh=False)
                if gpu_id is not None and gpu_id in device_pbars:
                    device_pbars[gpu_id].update(1)

        try:
            if all(device.type == "cpu" for device in devices):
                self._estimate_on_cpu(image_paths, image_shape, on_complete)
            else:
                self._estimate_on_devices(
                    image_paths, devices, image_shape, on_complete
                )
        finally:
            global_pbar.close()
            for pbar in device_pbars.values():
                pbar.close()

        return [result_by_idx[i] for i in range(n_images) if i in result_by_idx]

    def _k_for_device(
        self,
        device: torch.device,
        n_images: int,
        image_shape: tuple[int, int, int],
    ) -> int:
        """Resolve K for one device from config and memory/cores."""
        forced = self.computational_config.images_per_device
        if device.type == "cpu":
            return resolve_k_cpu(
                forced=forced,
                n_images=n_images,
                cpu_threads_per_worker=self.computational_config.cpu_threads_per_worker,
            )
        peak = analytic_peak_bytes(
            image_shape,
            self.optical_config.pixel_spacing_angstroms,
            self.optical_config.target_pixel_spacing_angstroms,
            self.fitting_config.patch_sidelength,
        )
        return resolve_k_gpu(
            forced=forced,
            free_bytes=cuda_free_bytes(device),
            peak_bytes=peak,
            n_images=n_images,
        )

    def _estimate_on_cpu(
        self,
        image_paths: list[Path],
        image_shape: tuple[int, int, int],
        on_complete: Callable[
            [int, Path, CtfEstimateCoreResult, int | None], None
        ],
    ) -> None:
        """Serial if K==1, otherwise a spawn process pool."""
        k = self._k_for_device(torch.device("cpu"), len(image_paths), image_shape)
        if k == 1:
            for idx, path in enumerate(image_paths):
                result = self.estimate_one(path, device=torch.device("cpu"))
                on_complete(idx, path, result, None)
            return

        payloads = [
            {
                "idx": idx,
                "path": str(path),
                "manager": self.model_dump(),
                "cpu_threads_per_worker": (
                    self.computational_config.cpu_threads_per_worker
                ),
            }
            for idx, path in enumerate(image_paths)
        ]
        with ProcessPoolExecutor(
            max_workers=k, mp_context=get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(cpu_estimate_worker, payload) for payload in payloads
            ]
            for future in as_completed(futures):
                idx, path_str, result = future.result()
                on_complete(idx, Path(path_str), result, None)

    def _estimate_on_devices(
        self,
        image_paths: list[Path],
        devices: list[torch.device],
        image_shape: tuple[int, int, int],
        on_complete: Callable[
            [int, Path, CtfEstimateCoreResult, int | None], None
        ],
    ) -> None:
        """One supervisor per GPU; each GPU runs a thread pool of size K."""
        assignments = _round_robin_indices(len(image_paths), len(devices))

        def run_device(gpu_id: int) -> None:
            device = devices[gpu_id]
            indices = assignments[gpu_id]
            k = self._k_for_device(device, len(indices), image_shape)
            if k == 1:
                for idx in indices:
                    path = image_paths[idx]
                    result = self.estimate_one(path, device)
                    on_complete(idx, path, result, gpu_id)
            else:
                with ThreadPoolExecutor(max_workers=k) as pool:
                    futures = {
                        pool.submit(self.estimate_one, image_paths[idx], device): idx
                        for idx in indices
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        on_complete(idx, image_paths[idx], future.result(), gpu_id)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if len(devices) == 1:
            run_device(0)
            return

        thread_errors: list[BaseException] = []

        def worker(gpu_id: int) -> None:
            try:
                run_device(gpu_id)
            except BaseException as exc:  # noqa: BLE001
                thread_errors.append(exc)

        threads = [
            Thread(target=worker, args=(gpu_id,)) for gpu_id in range(len(devices))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if thread_errors:
            raise thread_errors[0]

    def _export_one(
        self,
        image_path: Path,
        core_result: CtfEstimateCoreResult,
    ) -> tuple[dict, dict]:
        """Write per-micrograph YAML/npz/plots; return CSV rows."""
        output_dir = Path(self.output_config.output_directory)
        path_str = str(image_path)
        label = image_path.stem
        optics = result_to_leopard_optics(
            core_result.result2d,
            self.optical_config,
            path_str,
            label=label,
        )
        optics_row = {"micrograph_path": path_str, **optics}
        metrics_row = result_to_metrics_row(
            path_str,
            core_result.result1d,
            core_result.result2d,
            self.optical_config,
            self.fitting_config.defocus_range_microns,
            self.fitting_config.phase_shift_range_degrees,
            core_result.thickness1d,
            core_result.thickness_joint,
            stage_defocus_A=core_result.stage_defocus_A,
        )
        write_optics_yaml(
            optics,
            output_dir / f"{label}_optics.yaml",
            thickness_angstroms=metrics_row["thickness_angstroms"],
            defocus_field=spatial_defocus_field(core_result.result2d),
            stage_defocus_A=core_result.stage_defocus_A,
        )
        if self.output_config.save_debug_arrays:
            write_estimate_debug_npz(
                output_dir / f"{label}_debug.npz",
                mean_ps=core_result.mean_ps,
                result1d=core_result.result1d,
                pixel_spacing_used=core_result.pixel_spacing_used,
                thickness1d=core_result.thickness1d,
                thickness_joint=core_result.thickness_joint,
            )
        if self.output_config.output_plots:
            self._write_plots(core_result, output_dir / f"{label}_ctf_plots")
        return optics_row, metrics_row

    def _write_plots(self, core_result: CtfEstimateCoreResult, stem: Path) -> None:
        """Save 1D/2D diagnostic plots when matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            from torch_ctf_estimation.utils.plotting import (
                plot_1d_spectrum,
                plot_2d_spectrum,
                plot_2d_spectrum_images,
            )
        except ImportError:
            print(
                "warning: output_plots is true but matplotlib is unavailable; "
                "skipping diagnostic PNGs."
            )
            return
        plot_1d_spectrum(core_result.result1d)
        plt.savefig(f"{stem}_1d.png")
        plt.close()
        plot_2d_spectrum(core_result.result2d, core_result.result1d)
        plt.savefig(f"{stem}_2d.png")
        plt.close()
        thickness1d = core_result.thickness1d
        if (
            thickness1d is not None
            and thickness1d.test_thicknesses is not None
            and thickness1d.cross_correlations is not None
        ):
            t_grid = thickness1d.test_thicknesses.detach().cpu().numpy()
            cc_grid = thickness1d.cross_correlations.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.plot(t_grid, cc_grid)
            ax.axvline(thickness1d.thickness_angstroms, color="C1", linestyle="--")
            ax.set_xlabel("thickness (Å)")
            ax.set_ylabel("ZNCC")
            ax.set_title("1D thickness grid search")
            fig.savefig(f"{stem}_thickness_cc.png")
            plt.close(fig)
        if self.fitting_config.debug:
            plot_2d_spectrum_images(core_result.result2d)
            plt.savefig(f"{stem}_2d_images.png")
            plt.close()

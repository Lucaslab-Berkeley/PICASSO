"""Manager for refining CTF parameters for 2D in situ data."""

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from leopard_em.pydantic_models.config import PreprocessingFilters

from picasso.config import (
    ComputationalConfig,
    CtfRefineConfig,
    InputConfig,
    OutputConfig,
)
from picasso.core import core_refine_ctf, core_refine_ctf_lbfgs
from picasso.utils.custom_types import BaseModelPICASSO
from picasso.utils.data_io import (
    get_results_files,
    load_template_tensor,
    read_refined_ctf_params,
    write_loss_csvs,
    write_refined_ctf_params,
)


class CTFRefineManager(BaseModelPICASSO):
    """Model holding parameters necessary for running the CTF refinement program.

    Attributes
    ----------
    input_config : InputConfig
        Configuration for the input files.
    output_config : OutputConfig
        Configuration for the output files.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    ctf_refine_config : CtfRefineConfig
        Configuration for the CTF refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.

    Methods
    -------
    setup_backend_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    refine_ctf(self) -> None
        Run the CTF refinement program.
    """

    input_config: InputConfig
    output_config: OutputConfig
    computational_config: ComputationalConfig
    ctf_refine_config: CtfRefineConfig
    preprocessing_filters: PreprocessingFilters

    def setup_backend_kwargs(
        self,
    ) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function."""
        # Ensure the template is loaded in as a Tensor object
        template = load_template_tensor(
            template_volume=None,
            template_volume_path=self.input_config.template_path,
        )

        results_files = get_results_files(
            results_directory=self.input_config.results_directory,
            results_pattern=self.input_config.results_pattern,
        )

        # Get list of devices from config
        gpu_devices = self.computational_config.gpu_devices

        backend_kwargs = {
            "template": template,
            "template_box_size": self.input_config.template_box_size,
            "results_files": results_files,
            "device_list": gpu_devices,
            "refine_anisotropic_mag": self.ctf_refine_config.refine_anisotropic_mag,
            "refine_astigmatism": self.ctf_refine_config.refine_astigmatism,
            "refine_beam_tilt": self.ctf_refine_config.refine_beam_tilt,
            "refine_defocus": self.ctf_refine_config.refine_defocus,
            "refine_phase_shift": self.ctf_refine_config.refine_phase_shift,
            "phase_shift_mode": self.ctf_refine_config.phase_shift_mode,
            "apply_global_filtering": self.ctf_refine_config.apply_global_filtering,
            "preprocessing_filters": self.preprocessing_filters,
            "batch_size": self.input_config.batch_size,
            "per_micrograph_iterations": (
                self.ctf_refine_config.per_micrograph_iterations
            ),
            "per_dataset_iterations": (self.ctf_refine_config.per_dataset_iterations),
            "convergence_threshold": self.ctf_refine_config.convergence_threshold,
            "num_convergence_iterations": (
                self.ctf_refine_config.num_convergence_iterations
            ),
            "lbfgs_max_eval": self.ctf_refine_config.lbfgs_max_eval,
            "anisotropic_mag_lr": self.ctf_refine_config.anisotropic_mag_lr,
            "beam_tilt_lr": self.ctf_refine_config.beam_tilt_lr,
            "astigmatism_angle_lr": self.ctf_refine_config.astigmatism_angle_lr,
            "astigmatism_magnitude_lr": (
                self.ctf_refine_config.astigmatism_magnitude_lr
            ),
            "defocus_lr": self.ctf_refine_config.defocus_lr,
            "phase_shift_lr": self.ctf_refine_config.phase_shift_lr,
            "initial_phase_shift_quadratic_C": (
                self.ctf_refine_config.phase_shift_quadratic_C
            ),
            "initial_phase_shift_quadratic_g": (
                self.ctf_refine_config.phase_shift_quadratic_g
            ),
            "initial_phase_shift_quadratic_k": (
                self.ctf_refine_config.phase_shift_quadratic_k
            ),
            "initial_phase_shift_quadratic_alpha": (
                self.ctf_refine_config.phase_shift_quadratic_alpha
            ),
        }

        return backend_kwargs

    def write_refined_ctf_params(
        self,
        refined_ctf_params: dict[str, Any],
        filename: str = "refined_ctf_params.pkl",
    ) -> None:
        """Write refined CTF parameters to a file.

        Parameters
        ----------
        refined_ctf_params : dict[str, Any]
            Dictionary containing the refined CTF parameters.
        filename : str, optional
            Name of the output file. Defaults to "refined_ctf_params.pkl".

        Raises
        ------
        ValueError
            If output_directory is None or if file exists and overwrite is not
            allowed.
        """
        if self.output_config.output_directory is None:
            raise ValueError("output_directory must be set to write results.")

        output_path = Path(self.output_config.output_directory) / filename

        write_refined_ctf_params(
            refined_ctf_params=refined_ctf_params,
            output_path=output_path,
            allow_file_overwrite=self.output_config.allow_file_overwrite,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def refine_ctf(
        self,
        initial_mag_matrix: torch.Tensor | None = None,
        initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
        initial_astigmatism_angle: torch.Tensor | None = None,
        initial_astigmatism_magnitude: torch.Tensor | None = None,
        initial_phase_shift: torch.Tensor | None = None,
        initial_params_pkl_path: str | os.PathLike | Path | None = None,
    ) -> dict[str, Any]:
        """Refine the CTF parameters for the micrographs.

        Parameters
        ----------
        initial_mag_matrix : torch.Tensor | None, optional
            Initial anisotropic magnification matrix. Defaults to None.
        initial_odd_zernikes : dict[str, torch.Tensor] | None, optional
            Initial odd Zernike coefficients. Defaults to None.
        initial_astigmatism_angle : torch.Tensor | None, optional
            Initial astigmatism angles for each micrograph. Defaults to None.
        initial_astigmatism_magnitude : torch.Tensor | None, optional
            Initial astigmatism magnitudes for each micrograph. Defaults to None.
        initial_phase_shift : torch.Tensor | None, optional
            Initial phase shift values. Defaults to None.
        initial_params_pkl_path : str | os.PathLike | Path | None, optional
            Path to a pickle file containing initial CTF parameters.
            If provided and explicit parameters are also given, explicit
            parameters take precedence with a warning.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the refined CTF parameters:
            - "anistropic_mag": torch.Tensor
            - "odd_zernikes": dict[str, torch.Tensor]
            - "astigmatism_angle": torch.Tensor
            - "astigmatism_magnitude": torch.Tensor
            - "phase_shift_all": dict[int, torch.Tensor] (if refine_phase_shift is True)
        """
        backend_kwargs = self.setup_backend_kwargs()

        # Load initial parameters from pkl file if provided
        pkl_params: dict[str, Any] | None = None
        if initial_params_pkl_path is not None:
            pkl_params = read_refined_ctf_params(initial_params_pkl_path)

        # Check for conflicts and use explicit parameters over pkl
        has_explicit_params = any(
            [
                initial_mag_matrix is not None,
                initial_odd_zernikes is not None,
                initial_astigmatism_angle is not None,
                initial_astigmatism_magnitude is not None,
                initial_phase_shift is not None,
            ]
        )

        if pkl_params is not None and has_explicit_params:
            warnings.warn(
                "Both pkl file and explicit initial parameters provided. "
                "Using explicitly provided parameters over pkl file values.",
                UserWarning,
                stacklevel=2,
            )

        # Use explicit parameters if provided, otherwise use pkl file values
        final_initial_mag_matrix = initial_mag_matrix
        if final_initial_mag_matrix is None and pkl_params is not None:
            final_initial_mag_matrix = pkl_params.get("anistropic_mag")

        final_initial_odd_zernikes = initial_odd_zernikes
        if final_initial_odd_zernikes is None and pkl_params is not None:
            final_initial_odd_zernikes = pkl_params.get("odd_zernikes")

        final_initial_astigmatism_angle = initial_astigmatism_angle
        if final_initial_astigmatism_angle is None and pkl_params is not None:
            final_initial_astigmatism_angle = pkl_params.get("astigmatism_angle")

        final_initial_astigmatism_magnitude = initial_astigmatism_magnitude
        if final_initial_astigmatism_magnitude is None and pkl_params is not None:
            final_initial_astigmatism_magnitude = pkl_params.get(
                "astigmatism_magnitude"
            )

        final_initial_phase_shift = initial_phase_shift
        if final_initial_phase_shift is None and pkl_params is not None:
            final_initial_phase_shift = pkl_params.get("phase_shift_all")

        # Choose core function based on optimizer type from config
        if self.ctf_refine_config.optimizer == "lbfgs":
            core_function = core_refine_ctf_lbfgs
        else:
            core_function = core_refine_ctf

        refined_ctf_params = core_function(
            **backend_kwargs,
            initial_mag_matrix=final_initial_mag_matrix,
            initial_odd_zernikes=final_initial_odd_zernikes,
            initial_astigmatism_angle=final_initial_astigmatism_angle,
            initial_astigmatism_magnitude=final_initial_astigmatism_magnitude,
            initial_phase_shift=final_initial_phase_shift,
            output_directory=self.output_config.output_directory,
        )

        # Write loss CSV files if requested
        if (
            self.output_config.output_losses
            and self.output_config.output_directory is not None
        ):
            write_loss_csvs(
                refined_ctf_params=refined_ctf_params,
                output_directory=self.output_config.output_directory,
            )

        return refined_ctf_params

    def updated_results_csvs(
        self,
        refined_ctf_params: dict[str, Any],
    ) -> None:
        """Update results CSV files with refined CTF parameters.

        This function creates new CSV files with updated mag_matrix, odd_zernikes,
        and defocus_u/defocus_v columns based on refined values from
        refined_ctf_params. Original files are not modified.

        Parameters
        ----------
        refined_ctf_params : dict[str, Any]
            Dictionary containing the refined CTF parameters from refine_ctf().

        Raises
        ------
        ValueError
            If output_directory is None.

        Notes
        -----
        - New files are written to output_directory with `_ctf-refine.csv` suffix
          (e.g., `results.csv` -> `results_ctf-refine.csv`)
        - mag_matrix is converted from [[1,2],[3,4]] format to [1,2,3,4] format
        - odd_zernikes is converted from dict to list format [val1, val2, ...]
        - For astigmatism, defocus_u and defocus_v are updated while keeping
          the mean the same (mean = (defocus_u + defocus_v) / 2)
        - Columns are created if they don't exist
        """
        if self.output_config.output_directory is None:
            raise ValueError("output_directory must be set to write results.")

        # Get results files
        results_files = get_results_files(
            results_directory=self.input_config.results_directory,
            results_pattern=self.input_config.results_pattern,
        )

        # Create output directory if it doesn't exist
        output_dir = Path(self.output_config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each results file
        for micrograph_idx, results_file in enumerate(results_files):
            # Read CSV file
            df = pd.read_csv(results_file)

            # Update mag_matrix
            if "anistropic_mag" in refined_ctf_params:
                mag_matrix = refined_ctf_params["anistropic_mag"]
                # Convert 2x2 matrix to flat list [a, b, c, d]
                if isinstance(mag_matrix, torch.Tensor):
                    mag_np = mag_matrix.cpu().numpy()
                else:
                    mag_np = mag_matrix
                # Flatten: [[a, b], [c, d]] -> [a, b, c, d]
                mag_flat = mag_np.flatten().tolist()
                mag_str = f"[{', '.join(map(str, mag_flat))}]"

                # Create column if it doesn't exist
                if "mag_matrix" not in df.columns:
                    df["mag_matrix"] = ""

                # Update all rows with the same mag_matrix value
                df["mag_matrix"] = mag_str

            # Update odd_zernikes
            if "odd_zernikes" in refined_ctf_params:
                odd_zernikes = refined_ctf_params["odd_zernikes"]
                # Convert dict to string format like {"Z31c": 0.0, "Z31s": 0.0}
                zernike_dict = {}
                for key, val in odd_zernikes.items():
                    if isinstance(val, torch.Tensor):
                        zernike_dict[key] = val.cpu().item()
                    else:
                        zernike_dict[key] = float(val)
                # Format as JSON string with double quotes
                zernike_str = json.dumps(zernike_dict)

                # Create column if it doesn't exist
                if "odd_zernikes" not in df.columns:
                    df["odd_zernikes"] = ""

                # Update all rows with the same odd_zernikes value
                df["odd_zernikes"] = zernike_str

            # Update defocus_u and defocus_v from astigmatism
            if (
                "astigmatism_angle" in refined_ctf_params
                and "astigmatism_magnitude" in refined_ctf_params
            ):
                astigmatism_angle = refined_ctf_params["astigmatism_angle"]
                astigmatism_magnitude = refined_ctf_params["astigmatism_magnitude"]

                # Get astigmatism values for this micrograph
                if isinstance(astigmatism_angle, torch.Tensor):
                    if astigmatism_angle.numel() > 1:
                        astig_angle = astigmatism_angle[micrograph_idx].cpu().item()
                    else:
                        astig_angle = astigmatism_angle.cpu().item()
                else:
                    astig_angle = float(astigmatism_angle)

                if isinstance(astigmatism_magnitude, torch.Tensor):
                    if astigmatism_magnitude.numel() > 1:
                        astig_mag = astigmatism_magnitude[micrograph_idx].cpu().item()
                    else:
                        astig_mag = astigmatism_magnitude.cpu().item()
                else:
                    astig_mag = float(astigmatism_magnitude)

                # Create columns if they don't exist
                if "defocus_u" not in df.columns:
                    df["defocus_u"] = 0.0
                if "defocus_v" not in df.columns:
                    df["defocus_v"] = 0.0

                # Calculate mean defocus from existing values (keep mean same)
                if len(df) > 0:
                    defocus_u_vals = pd.to_numeric(
                        df["defocus_u"], errors="coerce"
                    ).fillna(0.0)
                    defocus_v_vals = pd.to_numeric(
                        df["defocus_v"], errors="coerce"
                    ).fillna(0.0)
                    defocus_mean = (defocus_u_vals + defocus_v_vals) / 2.0

                    # Update defocus_u and defocus_v based on astigmatism
                    # defocus_u = mean + magnitude / 2
                    # defocus_v = mean - magnitude / 2
                    df["defocus_u"] = defocus_mean + astig_mag / 2.0
                    df["defocus_v"] = defocus_mean - astig_mag / 2.0

                # Update astigmatism_angle column
                if "astigmatism_angle" not in df.columns:
                    df["astigmatism_angle"] = ""
                df["astigmatism_angle"] = astig_angle

            # Update defocus values (refined_relative_defocus or relative_defocus)
            # Add the refined defocus_tensor to the existing column values
            if "defocus_all" in refined_ctf_params:
                defocus_all = refined_ctf_params["defocus_all"]
                if micrograph_idx in defocus_all:
                    defocus_tensor = defocus_all[micrograph_idx]
                    if isinstance(defocus_tensor, torch.Tensor):
                        defocus_values = defocus_tensor.cpu().numpy()
                    else:
                        defocus_values = np.array(defocus_tensor)

                    # Try to add to refined_relative_defocus first,
                    # then relative_defocus
                    if "refined_relative_defocus" in df.columns:
                        # Add refined defocus to existing values
                        existing_values = pd.to_numeric(
                            df["refined_relative_defocus"], errors="coerce"
                        ).fillna(0.0)
                        df["refined_relative_defocus"] = (
                            existing_values.values + defocus_values
                        )
                    elif "relative_defocus" in df.columns:
                        # Add refined defocus to existing values
                        existing_values = pd.to_numeric(
                            df["relative_defocus"], errors="coerce"
                        ).fillna(0.0)
                        df["relative_defocus"] = existing_values.values + defocus_values
                    else:
                        # Create refined_relative_defocus column if neither exists
                        df["refined_relative_defocus"] = defocus_values

            # Update phase_shift values
            if "phase_shift_all" in refined_ctf_params:
                phase_shift_all = refined_ctf_params["phase_shift_all"]
                if micrograph_idx in phase_shift_all:
                    phase_shift_tensor = phase_shift_all[micrograph_idx]
                    if isinstance(phase_shift_tensor, torch.Tensor):
                        phase_shift_values = phase_shift_tensor.cpu().numpy()
                    else:
                        phase_shift_values = np.array(phase_shift_tensor)

                    # Check if per-micrograph (scalar) or per-particle (array)
                    if phase_shift_values.ndim == 0 or len(phase_shift_values) == 1:
                        # Per-micrograph: single value for all particles
                        phase_shift_val = float(phase_shift_values)
                        if "phase_shift" not in df.columns:
                            df["phase_shift"] = ""
                        df["phase_shift"] = phase_shift_val
                    else:
                        # Per-particle: array of values
                        if "phase_shift" not in df.columns:
                            df["phase_shift"] = 0.0
                        df["phase_shift"] = phase_shift_values

            # Create new filename: replace .csv with _ctf-refine.csv
            results_path = Path(results_file)
            new_filename = results_path.stem + "_ctf-refine.csv"
            output_path = output_dir / new_filename

            # Write updated CSV to new file
            df.to_csv(output_path, index=False)

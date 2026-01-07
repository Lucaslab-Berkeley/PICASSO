"""Manager for refining CTF parameters for 2D in situ data."""

import os
import warnings
from pathlib import Path
from typing import Any

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
            "apply_global_filtering": self.ctf_refine_config.apply_global_filtering,
            "preprocessing_filters": self.preprocessing_filters,
            "batch_size": self.input_config.batch_size,
            "per_micrograph_iterations": (
                self.ctf_refine_config.per_micrograph_iterations
            ),
            "per_dataset_iterations": (self.ctf_refine_config.per_dataset_iterations),
            "convergence_threshold": self.ctf_refine_config.convergence_threshold,
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

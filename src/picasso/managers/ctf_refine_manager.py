"""Manager for refining CTF parameters for 2D in situ data."""

import os
import pickle
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
from picasso.core import core_refine_ctf
from picasso.utils.custom_types import BaseModelPICASSO
from picasso.utils.data_io import get_results_files, load_template_tensor


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

        backend_kwargs = {
            "template": template,
            "template_box_size": self.input_config.template_box_size,
            "results_files": results_files,
            "device": self.computational_config.gpu_id,
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

        # Check if file exists and overwrite is not allowed
        if not self.output_config.allow_file_overwrite and output_path.exists():
            raise ValueError(
                f"File '{output_path}' already exists, but "
                "'allow_file_overwrite' is False. Set "
                "'allow_file_overwrite' to True to permit overwriting."
            )

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check write permissions
        if not os.access(output_path.parent, os.W_OK):
            raise ValueError(
                f"Directory '{output_path.parent}' does not permit writing."
            )

        # Write the parameters using pickle
        with open(output_path, "wb") as f:
            pickle.dump(refined_ctf_params, f)

        # Also write a human-readable text file
        txt_output_path = output_path.with_suffix(".txt")
        with open(txt_output_path, "w") as f:
            f.write("Refined CTF Parameters\n")
            f.write("=" * 50 + "\n\n")

            if "anistropic_mag" in refined_ctf_params:
                f.write("Anisotropic Magnification Matrix:\n")
                mag_matrix = refined_ctf_params["anistropic_mag"]
                if isinstance(mag_matrix, torch.Tensor):
                    mag_matrix_np = mag_matrix.cpu().numpy()
                    f.write(f"{mag_matrix_np}\n\n")
                else:
                    f.write(f"{mag_matrix}\n\n")

            if "odd_zernikes" in refined_ctf_params:
                f.write("Odd Zernike Coefficients:\n")
                zernikes = refined_ctf_params["odd_zernikes"]
                for key, value in zernikes.items():
                    if isinstance(value, torch.Tensor):
                        val = value.cpu().item()
                    else:
                        val = value
                    f.write(f"  {key}: {val:.6f}\n")
                f.write("\n")

            if "astigmatism_angle" in refined_ctf_params:
                f.write("Astigmatism Angles (per micrograph):\n")
                angles = refined_ctf_params["astigmatism_angle"]
                if isinstance(angles, torch.Tensor):
                    angles_np = angles.cpu().numpy()
                    for i, angle in enumerate(angles_np):
                        f.write(f"  Micrograph {i + 1}: {angle:.6f}\n")
                else:
                    f.write(f"{angles}\n")
                f.write("\n")

            if "astigmatism_magnitude" in refined_ctf_params:
                f.write("Astigmatism Magnitudes (per micrograph):\n")
                magnitudes = refined_ctf_params["astigmatism_magnitude"]
                if isinstance(magnitudes, torch.Tensor):
                    magnitudes_np = magnitudes.cpu().numpy()
                    for i, mag in enumerate(magnitudes_np):
                        f.write(f"  Micrograph {i + 1}: {mag:.6f}\n")
                else:
                    f.write(f"{magnitudes}\n")

    def refine_ctf(
        self,
        initial_mag_matrix: torch.Tensor | None = None,
        initial_odd_zernikes: dict[str, torch.Tensor] | None = None,
        initial_astigmatism_angle: torch.Tensor | None = None,
        initial_astigmatism_magnitude: torch.Tensor | None = None,
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

        refined_ctf_params = core_refine_ctf(
            **backend_kwargs,
            initial_mag_matrix=initial_mag_matrix,
            initial_odd_zernikes=initial_odd_zernikes,
            initial_astigmatism_angle=initial_astigmatism_angle,
            initial_astigmatism_magnitude=initial_astigmatism_magnitude,
        )

        return refined_ctf_params

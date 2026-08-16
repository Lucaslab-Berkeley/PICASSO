"""Output configuration for CTF estimation in PICASSO."""

from pydantic import model_validator
from typing_extensions import Self

from picasso.config.output_config import check_file_path_and_permissions
from picasso.utils.custom_types import BaseModelPICASSO


class CtfEstimateOutputConfig(BaseModelPICASSO):
    """Output directory and optional plots for CTF estimation."""

    output_directory: str
    allow_file_overwrite: bool = True
    output_plots: bool = False
    save_debug_arrays: bool = False
    optics_csv_filename: str = "ctf_estimate_optics.csv"
    metrics_csv_filename: str = "ctf_estimate_metrics.csv"

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Ensure the output directory is writable."""
        check_file_path_and_permissions(
            self.output_directory, self.allow_file_overwrite
        )
        return self

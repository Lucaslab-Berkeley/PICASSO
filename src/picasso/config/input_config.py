"""Configuration for output files for PICASSO."""

import os
from pathlib import Path

from pydantic import model_validator
from typing_extensions import Self

from picasso.utils.custom_types import BaseModelPICASSO


class InputConfig(BaseModelPICASSO):
    """Configuration for input files for PICASSO."""

    template_path: str
    template_box_size: tuple[int, int]
    micrograph_directory: str
    micrograph_pattern: str = "*.mrc"
    results_directory: str
    results_pattern: str = "*_results.csv"
    batch_size: int = 32

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Validate input paths for existence."""
        if self.template_path is not None and not os.path.exists(self.template_path):
            raise ValueError(f"Template path '{self.template_path}' does not exist.")
        if self.micrograph_directory is not None and not os.path.exists(
            self.micrograph_directory
        ):
            raise ValueError(
                f"Micrograph directory '{self.micrograph_directory}' does not exist."
            )
        if self.results_directory is not None and not os.path.exists(
            self.results_directory
        ):
            raise ValueError(
                f"Results directory '{self.results_directory}' does not exist."
            )

        # Validate that files matching the pattern exist
        if (
            self.micrograph_directory is not None
            and self.micrograph_pattern is not None
        ):
            pattern_path = Path(self.micrograph_directory) / self.micrograph_pattern
            matching_files = list(pattern_path.parent.glob(pattern_path.name))
            if not matching_files:
                raise ValueError(
                    f"No files matching pattern '{self.micrograph_pattern}' found in "
                    f"micrograph directory '{self.micrograph_directory}'."
                )

        if self.results_directory is not None and self.results_pattern is not None:
            pattern_path = Path(self.results_directory) / self.results_pattern
            matching_files = list(pattern_path.parent.glob(pattern_path.name))
            if not matching_files:
                raise ValueError(
                    f"No files matching pattern '{self.results_pattern}' found in "
                    f"results directory '{self.results_directory}'."
                )

        return self

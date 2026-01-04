"""Configuration for output files for PICASSO."""

import os

from pydantic import model_validator
from typing_extensions import Self

from picasso.utils.custom_types import BaseModelPICASSO


def check_file_path_and_permissions(path: str | None, allow_overwrite: bool) -> None:
    """Ensures path is writable and it does not exist, if `allow_overwrite` is False."""
    # If path is None, skip validation (no output file specified)
    if path is None:
        return

    # 1. Create path to file, if it does not exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2. Check write permissions
    if directory and not os.access(directory, os.W_OK):
        raise ValueError(
            f"Directory '{directory}' does not permit writing."
            f"Will be unable to write results to '{path}'."
        )

    # 3. Check if file exists
    if not allow_overwrite and os.path.exists(path):
        raise ValueError(
            f"File '{path}' already exists, but 'allow_file_overwrite' "
            "is False. Set 'allow_file_overwrite' to True to permit. "
            "overwriting.\n"
            "WARNING: Overwriting will delete the existing file(s)!"
        )


class OutputConfig(BaseModelPICASSO):
    """Configuration for output files for PICASSO."""

    allow_file_overwrite: bool = True
    output_directory: str | None = None
    output_losses: bool = False

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Validate output paths for write permissions and overwriting.

        Note: This method runs after instantiation, so attributes are already
        set. We can safely access them with `self`.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If the output paths are not writable or do not permit overwriting.
        """
        # 1. Check write permissions and overwriting for each path
        paths = [
            self.output_directory,
        ]
        for path in paths:
            check_file_path_and_permissions(path, self.allow_file_overwrite)

        return self

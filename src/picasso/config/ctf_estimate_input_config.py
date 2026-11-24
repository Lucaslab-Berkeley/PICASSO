"""Input configuration for CTF estimation in PICASSO."""

from pathlib import Path

from pydantic import model_validator
from typing_extensions import Self

from picasso.utils.custom_types import BaseModelPICASSO


class CtfEstimateInputConfig(BaseModelPICASSO):
    """Image path(s) for CTF estimation.

    Provide ``image_path`` and/or ``images_directory`` (with a glob pattern).
    At least one is required.
    """

    image_path: str | None = None
    images_directory: str | None = None
    images_pattern: str = "*.mrc"

    @model_validator(mode="after")  # type: ignore
    def validate_sources(self) -> Self:
        """Require at least one image source and that paths exist."""
        if self.image_path is None and self.images_directory is None:
            raise ValueError(
                "Provide image_path and/or images_directory for CTF estimation."
            )
        if self.image_path is not None and not Path(self.image_path).is_file():
            raise ValueError(f"image_path does not exist: {self.image_path}")
        if (
            self.images_directory is not None
            and not Path(self.images_directory).is_dir()
        ):
            raise ValueError(
                f"images_directory does not exist: {self.images_directory}"
            )
        return self

    def discover_images(self) -> list[Path]:
        """Return unique image paths from the single file and/or directory glob."""
        paths: list[Path] = []
        if self.image_path is not None:
            paths.append(Path(self.image_path))
        if self.images_directory is not None:
            directory = Path(self.images_directory)
            paths.extend(sorted(directory.glob(self.images_pattern)))
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
        if not unique:
            raise ValueError(
                "No images found. Check image_path / images_directory / "
                f"images_pattern='{self.images_pattern}'."
            )
        return unique

"""Tests for configuration classes."""

import tempfile
from pathlib import Path

import pytest

from picasso.config import CtfRefineConfig, OutputConfig


class TestCtfRefineConfig:
    """Tests for CtfRefineConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = CtfRefineConfig()
        assert config.refine_anisotropic_mag is False
        assert config.refine_astigmatism is False
        assert config.refine_beam_tilt is False
        assert config.apply_global_filtering is True
        assert config.per_micrograph_iterations == 10
        assert config.per_dataset_iterations == 10
        assert config.convergence_threshold == 0.0005

    def test_custom_values(self):
        """Test setting custom values."""
        config = CtfRefineConfig(
            refine_anisotropic_mag=True,
            refine_astigmatism=True,
            per_micrograph_iterations=20,
            convergence_threshold=0.001,
        )
        assert config.refine_anisotropic_mag is True
        assert config.refine_astigmatism is True
        assert config.refine_beam_tilt is False
        assert config.per_micrograph_iterations == 20
        assert config.convergence_threshold == 0.001


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = OutputConfig()
        assert config.allow_file_overwrite is True
        assert config.output_directory is None
        assert config.output_losses is False

    def test_custom_values(self):
        """Test setting custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a subdirectory to avoid conflicts
            subdir = Path(tmpdir) / "output"
            config = OutputConfig(
                allow_file_overwrite=True,
                output_directory=str(subdir),
                output_losses=True,
            )
            assert config.allow_file_overwrite is True
            assert config.output_directory == str(subdir)
            assert config.output_losses is True

    def test_validate_paths_creates_directory(self):
        """Test that validation creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a file path, not a directory path, to test directory creation
            new_file = Path(tmpdir) / "new_output_dir" / "output.txt"
            _ = OutputConfig(output_directory=str(new_file))
            # The parent directory should be created
            assert new_file.parent.exists()
            assert new_file.parent.is_dir()

    def test_validate_paths_no_overwrite_existing_file(self):
        """Test that validation raises error if file exists and overwrite is False."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_path = tmpfile.name

        try:
            with pytest.raises(ValueError, match="already exists"):
                OutputConfig(
                    output_directory=tmpfile_path,
                    allow_file_overwrite=False,
                )
        finally:
            Path(tmpfile_path).unlink()

    def test_validate_paths_allows_overwrite(self):
        """Test that validation allows overwrite when flag is True."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile_path = tmpfile.name

        try:
            # Should not raise an error
            config = OutputConfig(
                output_directory=tmpfile_path,
                allow_file_overwrite=True,
            )
            assert config.output_directory == tmpfile_path
        finally:
            Path(tmpfile_path).unlink()

    def test_validate_paths_none_directory(self):
        """Test that validation passes when directory is None."""
        config = OutputConfig(output_directory=None)
        assert config.output_directory is None

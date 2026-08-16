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

    def test_phase_shift_quadratic_two_axes_requires_quadratic_mode(self):
        """Two-axis quadratic model is only valid with phase_shift_mode quadratic."""
        with pytest.raises(ValueError, match="phase_shift_quadratic_two_axes"):
            CtfRefineConfig(
                phase_shift_mode="per_micrograph",
                phase_shift_quadratic_two_axes=True,
            )

    def test_phase_shift_quadratic_two_axes_with_quadratic_mode(self):
        """Two-axis flag is allowed when phase_shift_mode is quadratic."""
        config = CtfRefineConfig(
            phase_shift_mode="quadratic",
            phase_shift_quadratic_two_axes=True,
            phase_shift_quadratic_g2=0.1,
            phase_shift_quadratic_k2=-0.05,
        )
        assert config.phase_shift_quadratic_two_axes is True
        assert config.phase_shift_quadratic_g2 == 0.1
        assert config.phase_shift_quadratic_k2 == -0.05


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

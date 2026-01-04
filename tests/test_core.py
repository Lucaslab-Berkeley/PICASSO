"""Tests for core functions."""

import numpy as np
import torch

from picasso.core.core_ctf_refine import output_mag_matrix


class TestOutputMagMatrix:
    """Tests for output_mag_matrix function."""

    def test_output_mag_matrix_identity(self, capsys):
        """Test output_mag_matrix with identity matrix."""
        mag_matrix = torch.eye(2)
        output_mag_matrix(mag_matrix)

        captured = capsys.readouterr()
        assert "Magnification Matrix:" in captured.out
        assert "Axis of stretching:" in captured.out
        assert "Magnification anisotropy:" in captured.out

    def test_output_mag_matrix_anisotropic(self, capsys):
        """Test output_mag_matrix with anisotropic matrix."""
        # Create a simple anisotropic matrix (scaling in x direction)
        mag_matrix = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        output_mag_matrix(mag_matrix)

        captured = capsys.readouterr()
        assert "Magnification Matrix:" in captured.out
        assert "Axis of stretching:" in captured.out
        assert "Magnification anisotropy:" in captured.out
        # Anisotropy should be > 0 for this matrix
        assert "anisotropy:" in captured.out

    def test_output_mag_matrix_rotated(self, capsys):
        """Test output_mag_matrix with rotated anisotropic matrix."""
        # Create a rotated scaling matrix
        angle = np.pi / 4  # 45 degrees
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Use float32 to match torch default
        rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        scale = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mag_matrix = rotation @ scale @ rotation.T

        output_mag_matrix(mag_matrix)

        captured = capsys.readouterr()
        assert "Magnification Matrix:" in captured.out
        assert "Axis of stretching:" in captured.out
        assert "Magnification anisotropy:" in captured.out

    def test_output_mag_matrix_detached(self, capsys):
        """Test that function works with detached tensors."""
        mag_matrix = torch.eye(2, requires_grad=True)
        detached = mag_matrix.detach()
        output_mag_matrix(detached)

        captured = capsys.readouterr()
        assert "Magnification Matrix:" in captured.out

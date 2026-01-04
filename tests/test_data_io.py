"""Tests for data I/O functions."""

import pickle
import tempfile
from pathlib import Path

import pytest
import torch

from picasso.utils.data_io import (
    read_refined_ctf_params,
    write_loss_csvs,
    write_refined_ctf_params,
)


class TestWriteRefinedCtfParams:
    """Tests for write_refined_ctf_params."""

    def test_write_pkl_and_txt(self):
        """Test writing both pickle and text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_params.pkl"
            params = {
                "anistropic_mag": torch.eye(2),
                "odd_zernikes": {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)},
                "astigmatism_angle": torch.tensor([1.0, 2.0]),
                "astigmatism_magnitude": torch.tensor([3.0, 4.0]),
            }

            write_refined_ctf_params(
                refined_ctf_params=params,
                output_path=output_path,
                allow_file_overwrite=True,
            )

            # Check that both files were created
            assert output_path.exists()
            txt_path = output_path.with_suffix(".txt")
            assert txt_path.exists()

            # Check that pickle file can be read
            with open(output_path, "rb") as f:
                loaded_params = pickle.load(f)
            assert "anistropic_mag" in loaded_params
            assert "odd_zernikes" in loaded_params

            # Check that text file has content
            txt_content = txt_path.read_text(encoding="utf-8")
            assert "Refined CTF Parameters" in txt_content
            assert "Anisotropic Magnification Matrix" in txt_content
            assert "Odd Zernike Coefficients" in txt_content

    def test_no_overwrite_when_disabled(self):
        """Test that function raises error when overwrite is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_params.pkl"
            output_path.touch()  # Create empty file

            params = {"anistropic_mag": torch.eye(2)}

            with pytest.raises(ValueError, match="already exists"):
                write_refined_ctf_params(
                    refined_ctf_params=params,
                    output_path=output_path,
                    allow_file_overwrite=False,
                )


class TestReadRefinedCtfParams:
    """Tests for read_refined_ctf_params."""

    def test_read_pkl_file(self):
        """Test reading a pickle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_params.pkl"
            params = {
                "anistropic_mag": torch.eye(2),
                "odd_zernikes": {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)},
            }

            # Write the file
            with open(pkl_path, "wb") as f:
                pickle.dump(params, f)

            # Read it back
            loaded_params = read_refined_ctf_params(pkl_path)
            assert "anistropic_mag" in loaded_params
            assert "odd_zernikes" in loaded_params
            assert torch.allclose(loaded_params["anistropic_mag"], torch.eye(2))

    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "nonexistent.pkl"
            with pytest.raises(FileNotFoundError):
                read_refined_ctf_params(pkl_path)


class TestWriteLossCsvs:
    """Tests for write_loss_csvs."""

    def test_write_dataset_losses(self):
        """Test writing dataset losses to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "dataset_losses": [(0, 1.5), (1, 1.3), (2, 1.2)],
            }

            write_loss_csvs(
                refined_ctf_params=params,
                output_directory=tmpdir,
            )

            csv_path = Path(tmpdir) / "dataset_losses.csv"
            assert csv_path.exists()

            # Check CSV content
            content = csv_path.read_text(encoding="utf-8")
            assert "outer_iteration,loss" in content
            assert "0,1.5" in content
            assert "1,1.3" in content

    def test_write_micrograph_losses(self):
        """Test writing micrograph losses to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "micrograph_losses": {
                    0: [(0, 1.5), (1, 1.3)],
                    1: [(0, 2.0), (1, 1.8)],
                },
            }

            write_loss_csvs(
                refined_ctf_params=params,
                output_directory=tmpdir,
            )

            # Check that both micrograph files were created
            csv1_path = Path(tmpdir) / "micrograph_losses_1.csv"
            csv2_path = Path(tmpdir) / "micrograph_losses_2.csv"
            assert csv1_path.exists()
            assert csv2_path.exists()

            # Check content
            content1 = csv1_path.read_text(encoding="utf-8")
            assert "outer_iteration,loss" in content1
            assert "0,1.5" in content1

    def test_write_both_losses(self):
        """Test writing both dataset and micrograph losses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "dataset_losses": [(0, 1.5), (1, 1.3)],
                "micrograph_losses": {
                    0: [(0, 1.5)],
                },
            }

            write_loss_csvs(
                refined_ctf_params=params,
                output_directory=tmpdir,
            )

            dataset_csv = Path(tmpdir) / "dataset_losses.csv"
            micrograph_csv = Path(tmpdir) / "micrograph_losses_1.csv"
            assert dataset_csv.exists()
            assert micrograph_csv.exists()

    def test_write_creates_directory(self):
        """Test that function creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_output_dir"
            params = {"dataset_losses": [(0, 1.5)]}

            write_loss_csvs(
                refined_ctf_params=params,
                output_directory=new_dir,
            )

            assert new_dir.exists()
            assert (new_dir / "dataset_losses.csv").exists()

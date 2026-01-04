"""Utility functions dealing with basic data I/O operations."""

import csv
import os
import pickle
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch


def read_mrc_to_numpy(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    """Reads an MRC file and returns the data as a numpy array.

    Parameters
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        The MRC data as a numpy array, copied.
    """
    with mrcfile.open(mrc_path) as mrc:
        return mrc.data.copy()


def read_mrc_to_tensor(mrc_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads an MRC file and returns the data as a torch tensor.

    Parameters
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC data as a tensor, copied and converted to float32 if needed.
    """
    tensor = torch.tensor(read_mrc_to_numpy(mrc_path))
    # Convert float16 to float32 for FFT compatibility
    if tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)
    return tensor


def load_mrc_image(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an two-dimensional MRC image into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC image as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not two-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 2D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 2:
        raise ValueError(f"MRC file is not two-dimensional. Got shape: {tensor.shape}")

    return tensor


def load_mrc_volume(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an three-dimensional MRC volume into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC volume as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not three-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 3D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 3:
        raise ValueError(
            f"MRC file is not three-dimensional. Got shape: {tensor.shape}"
        )

    return tensor


def load_template_tensor(
    template_volume: torch.Tensor | Any | None = None,
    template_volume_path: str | os.PathLike | Path | None = None,
) -> torch.Tensor:
    """Load and convert template volume to a torch.Tensor.

    This function ensures that the template volume is a torch.Tensor.
    If template_volume is None, it loads the volume from template_volume_path.
    If template_volume is not a torch.Tensor, it converts it to one.

    Parameters
    ----------
    template_volume : Optional[Union[torch.Tensor, Any]], optional
        The template volume object, by default None
    template_volume_path : Optional[Union[str, os.PathLike, Path]], optional
        Path to the template volume file, by default None

    Returns
    -------
    torch.Tensor
        The template volume as a torch.Tensor

    Raises
    ------
    ValueError
        If both template_volume and template_volume_path are None
    """
    if template_volume is None:
        if template_volume_path is None:
            raise ValueError("template_volume or template_volume_path must be provided")
        template_volume = load_mrc_volume(template_volume_path)

    if not isinstance(template_volume, torch.Tensor):
        template = torch.from_numpy(template_volume)
    else:
        template = template_volume

    # Convert float16 to float32 for FFT compatibility
    if template.dtype == torch.float16:
        template = template.to(torch.float32)

    return template


def get_results_files(
    results_directory: str | os.PathLike | Path,
    results_pattern: str,
) -> list[str]:
    """Get list of results files matching the pattern.

    Parameters
    ----------
    results_directory : str | os.PathLike | Path
        Directory containing results files.
    results_pattern : str
        Glob pattern for results files (e.g., "*_results.csv").

    Returns
    -------
    list[str]
        List of paths to results files matching the pattern, sorted.
    """
    results_dir = Path(results_directory)

    # Get all results files matching the pattern
    results_files = sorted(results_dir.glob(results_pattern))

    return [str(f) for f in results_files]


def read_refined_ctf_params(
    pkl_path: str | os.PathLike | Path,
) -> dict[str, Any]:
    """Read refined CTF parameters from a pickle file.

    Parameters
    ----------
    pkl_path : str | os.PathLike | Path
        Path to the pickle file containing refined CTF parameters.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the refined CTF parameters:
        - "anistropic_mag": torch.Tensor
        - "odd_zernikes": dict[str, torch.Tensor]
        - "astigmatism_angle": torch.Tensor
        - "astigmatism_magnitude": torch.Tensor

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        refined_ctf_params: dict[str, Any] = pickle.load(f)

    return refined_ctf_params


# pylint: disable=too-many-locals,too-many-branches
def write_refined_ctf_params(
    refined_ctf_params: dict[str, Any],
    output_path: str | os.PathLike | Path,
    allow_file_overwrite: bool = True,
) -> None:
    """Write refined CTF parameters to pickle and text files.

    Parameters
    ----------
    refined_ctf_params : dict[str, Any]
        Dictionary containing the refined CTF parameters.
    output_path : str | os.PathLike | Path
        Path where the output files will be written.
    allow_file_overwrite : bool, default True
        Whether to allow overwriting existing files.

    Raises
    ------
    ValueError
        If file exists and overwrite is not allowed, or if directory
        does not permit writing.
    """
    output_path = Path(output_path)

    # Check if file exists and overwrite is not allowed
    if not allow_file_overwrite and output_path.exists():
        raise ValueError(
            f"File '{output_path}' already exists, but "
            "'allow_file_overwrite' is False. Set "
            "'allow_file_overwrite' to True to permit overwriting."
        )

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check write permissions
    if not os.access(output_path.parent, os.W_OK):
        raise ValueError(f"Directory '{output_path.parent}' does not permit writing.")

    # Write the parameters using pickle
    with open(output_path, "wb") as f:
        pickle.dump(refined_ctf_params, f)

    # Also write a human-readable text file
    txt_output_path = output_path.with_suffix(".txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
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


def write_loss_csvs(
    refined_ctf_params: dict[str, Any],
    output_directory: str | os.PathLike | Path,
) -> None:
    """Write loss values to CSV files.

    Parameters
    ----------
    refined_ctf_params : dict[str, Any]
        Dictionary containing refined CTF parameters including losses.
    output_directory : str | os.PathLike | Path
        Directory where CSV files will be written.

    Raises
    ------
    ValueError
        If directory does not permit writing.
    """
    output_dir = Path(output_directory)
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check write permissions
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Directory '{output_dir}' does not permit writing.")

    # Write dataset losses
    if "dataset_losses" in refined_ctf_params:
        dataset_losses = refined_ctf_params["dataset_losses"]
        dataset_csv_path = output_dir / "dataset_losses.csv"
        with open(dataset_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["outer_iteration", "loss"])
            for outer_iter, loss_val in dataset_losses:
                writer.writerow([outer_iter, loss_val])

    # Write micrograph losses
    if "micrograph_losses" in refined_ctf_params:
        micrograph_losses = refined_ctf_params["micrograph_losses"]
        for micrograph_num, losses in micrograph_losses.items():
            micrograph_csv_path = (
                output_dir / f"micrograph_losses_{micrograph_num + 1}.csv"
            )
            with open(micrograph_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["outer_iteration", "loss"])
                for outer_iter, loss_val in losses:
                    writer.writerow([outer_iter, loss_val])

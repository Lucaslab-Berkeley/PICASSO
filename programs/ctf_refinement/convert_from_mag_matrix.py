"""Convenience script to convert anisotropic mag matrix to pixel size measurements.

This script converts a 2x2 anisotropic magnification matrix (and mean pixel size)
into measured pixel sizes (major axis, minor axis, and angle). The mag matrix
transforms an image of mean pixel size to the measured anisotropic pixel size.
"""

import argparse
import math

import numpy as np
import torch


def convert_from_mag_matrix(
    mag_matrix: torch.Tensor,
    mean_px: float,
) -> tuple[float, float, float, float]:
    """Convert anisotropic magnification matrix to pixel size measurements.

    Parameters
    ----------
    mag_matrix : torch.Tensor
        2x2 anisotropic magnification matrix that transforms an image of
        mean pixel size to the measured anisotropic pixel size.
    mean_px : float
        Mean pixel size used to construct the mag matrix.

    Returns
    -------
    px_major : float
        Measured pixel size on major axis (lowest mag/highest pixel size).
    px_minor : float
        Measured pixel size on minor axis.
    angle : float
        Angle of the major axis in radians (minor is perpendicular to it).
    anisotropy_pct : float
        Anisotropy percentage.
    """
    # Convert to numpy for eigenvalue decomposition
    if isinstance(mag_matrix, torch.Tensor):
        mag_np = mag_matrix.detach().cpu().numpy()
    else:
        mag_np = mag_matrix

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(mag_np)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Find the major and minor axes
    # The smaller eigenvalue corresponds to the major axis (larger pixel size)
    # The larger eigenvalue corresponds to the minor axis (smaller pixel size)
    # From convert_to_mag_matrix: scale = mean_px / px
    # So px_major > px_minor implies scale_major < scale_minor
    min_idx = np.argmin(eigenvalues)
    max_idx = 1 - min_idx

    lambda_min = eigenvalues[min_idx]  # Smaller eigenvalue -> major axis
    lambda_max = eigenvalues[max_idx]   # Larger eigenvalue -> minor axis

    # The smaller eigenvalue is the scale factor for the major axis
    scale_major = lambda_min
    scale_minor = lambda_max
    eigenvector_major = eigenvectors[:, min_idx]

    # Calculate pixel sizes from scale factors
    # From convert_to_mag_matrix: scale = mean_px / px
    # So: px = mean_px / scale
    px_major = mean_px / scale_major
    px_minor = mean_px / scale_minor

    # Normalize eigenvector and compute angle
    norm = np.linalg.norm(eigenvector_major)
    if norm > 1e-10:
        eigenvector_major = eigenvector_major / norm

    angle = math.atan2(eigenvector_major[1], eigenvector_major[0])
    
    # Normalize angle to [0, pi) range since axis orientation is modulo 180 degrees
    # (an axis pointing in opposite direction is the same axis)
    if angle < 0:
        angle += math.pi
    if angle >= math.pi:
        angle -= math.pi

    # Calculate anisotropy percentage
    if abs(mean_px) > 1e-10:
        anisotropy_pct = abs(px_major - px_minor) / mean_px * 100.0
    else:
        anisotropy_pct = 0.0

    return px_major, px_minor, angle, anisotropy_pct


def parse_mag_matrix(matrix_str: str) -> torch.Tensor:
    """Parse magnification matrix from string input.

    Parameters
    ----------
    matrix_str : str
        Matrix as string, e.g., "[[a, b], [c, d]]" or "a b c d" or file path.

    Returns
    -------
    torch.Tensor
        2x2 magnification matrix.
    """
    # Try to read as file first
    try:
        with open(matrix_str, encoding="utf-8") as f:
            content = f.read().strip()
            # Try to evaluate as Python code
            try:
                matrix = eval(content)  # pylint: disable=eval-used
            except (SyntaxError, NameError) as exc:
                # Try parsing as space-separated values
                values = content.split()
                if len(values) == 4:
                    matrix = [
                        [float(values[0]), float(values[1])],
                        [float(values[2]), float(values[3])],
                    ]
                else:
                    raise ValueError(
                        f"Could not parse matrix from file: {matrix_str}"
                    ) from exc
    except (FileNotFoundError, OSError):
        # Not a file, try parsing as string directly
        try:
            matrix = eval(matrix_str)  # pylint: disable=eval-used
        except (SyntaxError, NameError) as exc:
            # Try parsing as space-separated values
            values = matrix_str.split()
            if len(values) == 4:
                matrix = [
                    [float(values[0]), float(values[1])],
                    [float(values[2]), float(values[3])],
                ]
            else:
                raise ValueError(f"Could not parse matrix: {matrix_str}") from exc

    # Convert to tensor
    matrix_tensor = torch.tensor(matrix, dtype=torch.float32)

    # Validate shape
    if matrix_tensor.shape != (2, 2):
        raise ValueError(f"Matrix must be 2x2, got shape {matrix_tensor.shape}")

    return matrix_tensor


def main() -> None:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert anisotropic magnification matrix to pixel size measurements. "
            "The mag matrix transforms an image of mean pixel size to the measured "
            "anisotropic pixel size."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mag_matrix",
        type=str,
        required=True,
        help=(
            "2x2 magnification matrix. Can be: "
            "1) Python list format: '[[a, b], [c, d]]', "
            "2) Space-separated values: 'a b c d', "
            "3) Path to file containing the matrix"
        ),
    )
    parser.add_argument(
        "--mean_px",
        type=float,
        required=True,
        help="Mean pixel size used to construct the mag matrix",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Output angle in degrees (default: radians)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.mean_px <= 0:
        raise ValueError("Mean pixel size must be positive")

    # Parse mag matrix
    try:
        mag_matrix = parse_mag_matrix(args.mag_matrix)
    except Exception as e:
        raise ValueError(f"Failed to parse mag matrix: {e}") from e

    # Convert from mag matrix
    px_major, px_minor, angle, anisotropy_pct = convert_from_mag_matrix(
        mag_matrix=mag_matrix,
        mean_px=args.mean_px,
    )

    # Print results
    print("\nInput Magnification Matrix:")
    print(mag_matrix.numpy() if isinstance(mag_matrix, torch.Tensor) else mag_matrix)
    print(f"\nMean pixel size: {args.mean_px:.6f}")

    print("\nExtracted Pixel Size Measurements:")
    print(f"  Major axis pixel size: {px_major:.6f}")
    print(f"  Minor axis pixel size: {px_minor:.6f}")
    angle_deg = math.degrees(angle)
    if args.degrees:
        print(f"  Major axis angle: {angle_deg:.2f} degrees")
    else:
        print(f"  Major axis angle: {angle:.6f} radians ({angle_deg:.2f} degrees)")
    print(f"  Anisotropy: {anisotropy_pct:.2f}%")

    # Verify by converting back (round-trip test)
    print("\nVerification (round-trip test):")
    # pylint: disable=import-outside-toplevel
    from convert_to_mag_matrix import convert_to_mag_matrix

    reconstructed = convert_to_mag_matrix(
        px_major=px_major,
        px_minor=px_minor,
        angle=angle,
        angle_in_degrees=False,
    )
    diff = torch.abs(mag_matrix - reconstructed).max().item()
    print(f"  Max difference between input and reconstructed matrix: {diff:.2e}")
    if diff < 1e-5:
        print("  Round-trip test passed!")
    else:
        print("  Warning: Round-trip test shows difference > 1e-5")


if __name__ == "__main__":
    main()


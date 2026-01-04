"""Convenience script to convert pixel size measurements to anisotropic mag matrix.

This script converts measured pixel sizes (major axis, minor axis, and angle)
into a 2x2 anisotropic magnification matrix that can be used as input for
CTF refinement. The mag matrix transforms an image of mean pixel size to the
measured anisotropic pixel size.
"""

import argparse
import math

import torch


def convert_to_mag_matrix(
    px_major: float,
    px_minor: float,
    angle: float,
    angle_in_degrees: bool = False,
) -> torch.Tensor:
    """Convert pixel size measurements to anisotropic magnification matrix.

    Parameters
    ----------
    px_major : float
        Measured pixel size on major axis (lowest mag/highest pixel size).
    px_minor : float
        Measured pixel size on minor axis.
    angle : float
        Angle of the major axis (minor is perpendicular to it).
    angle_in_degrees : bool, optional
        Whether the angle is in degrees. Defaults to False (radians).

    Returns
    -------
    torch.Tensor
        2x2 anisotropic magnification matrix that transforms an image of
        mean pixel size to the measured anisotropic pixel size.
    """
    # Convert angle to radians if needed
    if angle_in_degrees:
        theta = math.radians(angle)
    else:
        theta = angle

    # Calculate mean pixel size
    mean_px = (px_major + px_minor) / 2.0

    # Calculate scale factors
    # Larger pixel size means lower magnification, so we scale by mean/measured
    scale_major = mean_px / px_major
    scale_minor = mean_px / px_minor

    # Construct rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=torch.float32
    )

    # Construct scaling matrix (diagonal with scale factors)
    scale = torch.tensor([[scale_major, 0.0], [0.0, scale_minor]], dtype=torch.float32)

    # Construct mag matrix: R * S * R^T
    mag_matrix = rotation @ scale @ rotation.T

    return mag_matrix


def main() -> None:
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert pixel size measurements to anisotropic magnification matrix. "
            "The mag matrix transforms an image of mean pixel size to the measured "
            "anisotropic pixel size."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--px_major",
        type=float,
        required=True,
        help="Measured pixel size on major axis (lowest mag/highest pixel size)",
    )
    parser.add_argument(
        "--px_minor",
        type=float,
        required=True,
        help="Measured pixel size on minor axis",
    )
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Angle of the major axis (minor is perpendicular to it)",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret angle as degrees (default: radians)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.px_major <= 0 or args.px_minor <= 0:
        raise ValueError("Pixel sizes must be positive")
    if args.px_major < args.px_minor:
        print(
            "Warning: px_major is less than px_minor. "
            "px_major should be the larger pixel size (lowest mag)."
        )

    # Convert to mag matrix
    mag_matrix = convert_to_mag_matrix(
        px_major=args.px_major,
        px_minor=args.px_minor,
        angle=args.angle,
        angle_in_degrees=args.degrees,
    )

    # Print results
    print("\nAnisotropic Magnification Matrix:")
    print(mag_matrix.numpy())
    print("\nMatrix as Python code (torch.Tensor):")
    print(f"mag_matrix = torch.tensor({mag_matrix.tolist()}, dtype=torch.float32)")

    # Calculate and display some useful information
    mean_px = (args.px_major + args.px_minor) / 2.0
    anisotropy_pct = abs(args.px_major - args.px_minor) / mean_px * 100.0
    angle_deg = math.degrees(args.angle) if not args.degrees else args.angle

    print("\nSummary:")
    print(f"  Mean pixel size: {mean_px:.6f}")
    print(f"  Major axis pixel size: {args.px_major:.6f}")
    print(f"  Minor axis pixel size: {args.px_minor:.6f}")
    print(f"  Anisotropy: {anisotropy_pct:.2f}%")
    print(f"  Major axis angle: {angle_deg:.2f} degrees")


if __name__ == "__main__":
    main()

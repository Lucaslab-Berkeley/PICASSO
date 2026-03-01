"""Configuration for CTF refinement options for PICASSO."""

from pydantic import model_validator
from typing_extensions import Self

from picasso.utils.custom_types import BaseModelPICASSO


class CtfRefineConfig(BaseModelPICASSO):
    """Configuration for CTF refinement options for PICASSO."""

    refine_anisotropic_mag: bool = False
    refine_astigmatism: bool = False
    refine_beam_tilt: bool = False
    refine_defocus: bool = False
    refine_phase_shift: bool = False
    phase_shift_mode: str = "per_micrograph"
    apply_global_filtering: bool = True
    per_micrograph_iterations: int = 10
    per_dataset_iterations: int = 10
    convergence_threshold: float = 0.0005
    num_convergence_iterations: int = 5
    optimizer: str = "lbfgs"
    lbfgs_max_eval: int = 5
    anisotropic_mag_lr: float = 0.005
    beam_tilt_lr: float = 0.01
    astigmatism_angle_lr: float = 5.0
    astigmatism_magnitude_lr: float = 5.0
    defocus_lr: float = 5.0
    phase_shift_lr: float = 1.0

    @model_validator(mode="after")  # type: ignore
    def validate_optimizer(self) -> Self:
        """Validate optimizer value after initialization.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If optimizer is not 'adam' or 'lbfgs'.
        """
        if self.optimizer not in ("adam", "lbfgs"):
            raise ValueError(
                f"optimizer must be 'adam' or 'lbfgs', got '{self.optimizer}'"
            )
        return self

    @model_validator(mode="after")  # type: ignore
    def validate_phase_shift_mode(self) -> Self:
        """Validate phase_shift_mode value after initialization.

        Returns
        -------
        Self
            The validated instance.        Raises
        ------
        ValueError
            If phase_shift_mode is not 'per_micrograph' or 'per_particle'.
        """
        if self.phase_shift_mode not in ("per_micrograph", "per_particle"):
            raise ValueError(
                f"phase_shift_mode must be 'per_micrograph' or 'per_particle', "
                f"got '{self.phase_shift_mode}'"
            )
        return self

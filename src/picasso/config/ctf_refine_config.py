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
    # Unused: astigmatism is refined as two Cartesian components in Å, both with
    # astigmatism_magnitude_lr. Kept so existing configs still load.
    astigmatism_angle_lr: float = 5.0
    astigmatism_magnitude_lr: float = 5.0
    defocus_lr: float = 5.0
    phase_shift_lr: float = 1.0
    # Quadratic phase shift model (when phase_shift_mode == "quadratic")
    # degrees; if None, use mean from results
    phase_shift_quadratic_C: float | None = None
    phase_shift_quadratic_g: float = 0.0
    phase_shift_quadratic_k: float = 0.0
    phase_shift_quadratic_alpha: float = 0.0  # radians
    # Two-axis: C + g1*s + k1*s^2 + g2*t + k2*t^2 (needs quadratic mode)
    phase_shift_quadratic_two_axes: bool = False
    phase_shift_quadratic_g2: float = 0.0
    phase_shift_quadratic_k2: float = 0.0
    # Memory for keeping each micrograph's inputs between passes instead of
    # re-reading and re-filtering it. None: 40% of each GPU's memory / of the
    # available host memory. 0 disables that tier.
    setup_cache_gpu_gb: float | None = None
    setup_cache_host_gb: float | None = None
    # With the L-BFGS optimizer: refine a micrograph's astigmatism, defocus and
    # phase shift with one optimizer (True) or one per parameter group in turn.
    joint_per_micrograph_lbfgs: bool = True

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
            If phase_shift_mode is not 'per_micrograph', 'per_particle', or 'quadratic'.
        """
        if self.phase_shift_mode not in (
            "per_micrograph",
            "per_particle",
            "quadratic",
        ):
            raise ValueError(
                "phase_shift_mode must be 'per_micrograph', 'per_particle', or "
                f"'quadratic', got '{self.phase_shift_mode}'"
            )
        return self

    @model_validator(mode="after")  # type: ignore
    def validate_phase_shift_quadratic_two_axes(self) -> Self:
        """Require quadratic mode when two-axis phase model is enabled."""
        if self.phase_shift_quadratic_two_axes and self.phase_shift_mode != "quadratic":
            raise ValueError(
                "phase_shift_quadratic_two_axes requires "
                "phase_shift_mode == 'quadratic', "
                f"got phase_shift_mode='{self.phase_shift_mode}'"
            )
        return self

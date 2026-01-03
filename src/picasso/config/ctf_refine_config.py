"""Configuration for CTF refinement options for PICASSO."""

from picasso.utils.custom_types import BaseModelPICASSO


class CtfRefineConfig(BaseModelPICASSO):
    """Configuration for CTF refinement options for PICASSO."""

    refine_anisotropic_mag: bool = False
    refine_astigmatism: bool = False
    refine_beam_tilt: bool = False
    apply_global_filtering: bool = True
    per_micrograph_iterations: int = 10
    per_dataset_iterations: int = 10
    convergence_threshold: float = 0.0005

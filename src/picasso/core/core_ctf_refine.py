"""CTF refinement with Adam optimizers.

The refinement loop lives in ``picasso.core.core_ctf_refine_driver``; this module
keeps the historical entry point.
"""

from typing import Any

from picasso.core.core_ctf_refine_driver import refine_ctf
from picasso.core.core_ctf_refine_utils import output_mag_matrix


def core_refine_ctf(**kwargs: Any) -> dict[str, Any]:
    """Refine the CTF parameters for the micrographs using Adam.

    Accepts the keyword arguments of
    ``picasso.core.core_ctf_refine_driver.refine_ctf`` (except
    ``optimizer_type``) and returns its result.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments forwarded to ``refine_ctf``.

    Returns
    -------
    dict[str, Any]
        Refined CTF parameters and loss histories.
    """
    return refine_ctf(optimizer_type="adam", **kwargs)


__all__ = ["core_refine_ctf", "output_mag_matrix"]

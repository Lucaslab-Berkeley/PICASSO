"""Core functions for PICASSO."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core_ctf_estimate import CtfEstimateCoreResult, core_estimate_ctf
    from .core_ctf_refine import core_refine_ctf
    from .core_ctf_refine_lbfgs import core_refine_ctf as core_refine_ctf_lbfgs

__all__ = [
    "CtfEstimateCoreResult",
    "core_estimate_ctf",
    "core_refine_ctf",
    "core_refine_ctf_lbfgs",
]


def __dir__() -> list[str]:
    """Include lazy exports in dir()."""
    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str) -> Any:
    """Import core callables on first access so estimate does not load refine."""
    if name in {"CtfEstimateCoreResult", "core_estimate_ctf"}:
        from .core_ctf_estimate import CtfEstimateCoreResult, core_estimate_ctf

        return {
            "CtfEstimateCoreResult": CtfEstimateCoreResult,
            "core_estimate_ctf": core_estimate_ctf,
        }[name]
    if name == "core_refine_ctf":
        from .core_ctf_refine import core_refine_ctf

        return core_refine_ctf
    if name == "core_refine_ctf_lbfgs":
        from .core_ctf_refine_lbfgs import core_refine_ctf as core_refine_ctf_lbfgs

        return core_refine_ctf_lbfgs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

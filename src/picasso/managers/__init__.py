"""Managers for PICASSO."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ctf_estimate_manager import CTFEstimateManager
    from .ctf_refine_manager import CTFRefineManager

__all__ = [
    "CTFEstimateManager",
    "CTFRefineManager",
]


def __dir__() -> list[str]:
    """Include lazy exports in dir()."""
    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str) -> Any:
    """Import managers on first access so estimate does not load refine."""
    if name == "CTFEstimateManager":
        from .ctf_estimate_manager import CTFEstimateManager

        return CTFEstimateManager
    if name == "CTFRefineManager":
        from .ctf_refine_manager import CTFRefineManager

        return CTFRefineManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

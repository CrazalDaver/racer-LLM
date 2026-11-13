"""Public interface for the single configurable IBR policy."""

from .trajectory_ibr import (
    IBR_STRATEGY_SPACES,
    IteratedBestResponsePolicy,
)

__all__ = [
    "IteratedBestResponsePolicy",
    "IBR_STRATEGY_SPACES",
]

"""Supporting pieces for the readable top-level simulation flow.

Most callers should import :func:`simulation.run_simulation`.  This package
keeps construction, observations, environment mechanics, and recording
together without crowding the repository root.
"""

from .config import LLM_OBSERVATION_VARIANTS, SimulationConfig

__all__ = ["LLM_OBSERVATION_VARIANTS", "SimulationConfig"]

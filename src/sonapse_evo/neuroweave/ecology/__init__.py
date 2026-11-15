"""Embodied environment expansion (additional selectable environments; existing tasks are kept)."""
from .arena import EcologyArena, EcologyConfig
from .presets import ECOLOGY_PRESETS, ENVIRONMENTS, MultiAgentEcology, list_environments, make_environment

__all__ = ["EcologyArena", "EcologyConfig", "ECOLOGY_PRESETS", "ENVIRONMENTS", "MultiAgentEcology", "list_environments",
           "make_environment"]

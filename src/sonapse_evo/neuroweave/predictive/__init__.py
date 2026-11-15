"""Predictive processing: ensemble world model, hierarchical horizons, action-grounded self-model."""
from .world_model import PredictiveWorldModel, pooled_features
from .horizons import HierarchicalPrediction
from .self_model import EfferenceSelfModel

__all__ = ["PredictiveWorldModel", "pooled_features", "HierarchicalPrediction", "EfferenceSelfModel"]

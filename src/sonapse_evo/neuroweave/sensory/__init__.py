"""Expanded sensory system: sixteen channels, local feature extraction, hub projection and cross-modal fusion.

Privacy rule: raw camera frames and raw audio are never stored on any object that is serialised or
exported.  Only low-dimensional derived features persist.
"""
from .channels import (CHANNEL_SPECS, EXTENDED_FEATURES, SENSORY_CHANNELS, ChannelMode, ChannelSpec, SensoryManifest)
from .visual import VisualFeatureExtractor, extract_visual_features
from .auditory import AudioFeatureExtractor
from .simulated import SimulatedSensors
from .hub import MultimodalSensory
from .fusion import CrossModalFusion

__all__ = ["SENSORY_CHANNELS", "CHANNEL_SPECS", "EXTENDED_FEATURES", "ChannelMode", "ChannelSpec", "SensoryManifest",
           "VisualFeatureExtractor", "extract_visual_features", "AudioFeatureExtractor", "SimulatedSensors",
           "MultimodalSensory", "CrossModalFusion"]

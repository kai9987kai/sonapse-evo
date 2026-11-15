"""Representation and communication mechanisms (structural encoder, attention, FlyHash, NeuroWave-X2)."""
from .structural import StructuralEncoder, compute_fingerprints, FINGERPRINT_FEATURES
from .attention import ConnectivityAttention
from .flyhash import FlyHashProjection, flyhash_code
from .neurowave import (
    PACKET_CONTRACTS, PACKET_VERSION, NeuroWaveBus, RepresentationKind, SignalPacket, vsa_bind, vsa_bundle,
    vsa_similarity,
)

__all__ = [
    "StructuralEncoder", "compute_fingerprints", "FINGERPRINT_FEATURES", "ConnectivityAttention",
    "FlyHashProjection", "flyhash_code", "NeuroWaveBus", "RepresentationKind", "SignalPacket",
    "PACKET_CONTRACTS", "PACKET_VERSION", "vsa_bind", "vsa_bundle", "vsa_similarity",
]

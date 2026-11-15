"""Hardware-inspired execution experiments (software only: no RTL, no bitstream, no measured FPGA results)."""
from .quantization import QuantisedSynapses, from_q8_8, quantise_int8, to_q8_8
from .tiled import TiledExecution
from .fpga import build_fpga_manifest, validate_fpga_manifest, FPGA_FORMAT
from .profiler import available_backends, compact_state, profile_substrate, scaling_profile, state_bytes

__all__ = ["QuantisedSynapses", "quantise_int8", "to_q8_8", "from_q8_8", "TiledExecution", "build_fpga_manifest",
           "validate_fpga_manifest", "FPGA_FORMAT", "profile_substrate", "scaling_profile", "state_bytes",
           "compact_state", "available_backends"]

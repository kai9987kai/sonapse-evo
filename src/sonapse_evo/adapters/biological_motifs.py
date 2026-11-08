from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


class BiologicalMotifLibrary:
    """Library of verified biological connectome motifs obeying Dale's Law."""

    @staticmethod
    def get_cx_ring_attractor(num_wedges: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Central Complex 16-wedge ring attractor compass motif.
        
        Local mutual excitation between adjacent wedges, global ring inhibition.
        Returns:
            (W_m, signs)
        """
        signs = np.ones(num_wedges, dtype=np.float32)
        # Every 4th wedge is an inhibitory interneuron
        for i in range(3, num_wedges, 4):
            signs[i] = -1.0

        W_m = np.zeros((num_wedges, num_wedges), dtype=np.float32)
        for i in range(num_wedges):
            left = (i - 1) % num_wedges
            right = (i + 1) % num_wedges
            W_m[left, i] = 0.25
            W_m[right, i] = 0.25

        return W_m, signs

    @staticmethod
    def get_lateral_inhibition_motif(size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Feedforward lateral inhibition motif."""
        signs = np.ones(size, dtype=np.float32)
        # Half excitatory, half inhibitory
        signs[size // 2:] = -1.0
        W_m = np.zeros((size, size), dtype=np.float32)
        for i in range(size // 2):
            for j in range(size // 2, size):
                W_m[j, i] = 0.20  # E excites I
                W_m[i, j] = 0.30  # I inhibits E
        return W_m, signs

    @staticmethod
    def get_cpg_rhythm_generator() -> Tuple[np.ndarray, np.ndarray]:
        """Half-center oscillator CPG rhythm generator (2-unit reciprocal inhibition)."""
        signs = np.array([-1.0, -1.0], dtype=np.float32)
        W_m = np.array([[0.0, 0.40], [0.40, 0.0]], dtype=np.float32)
        return W_m, signs

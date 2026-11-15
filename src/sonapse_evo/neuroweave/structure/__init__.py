"""Structural evolution: utility-guided synaptic turnover and neurogenesis, with an audit ledger."""
from .ledger import StructuralLedger
from .structural_plasticity import StructuralPlasticity
from .neurogenesis import Neurogenesis, grow_substrate

__all__ = ["StructuralLedger", "StructuralPlasticity", "Neurogenesis", "grow_substrate"]

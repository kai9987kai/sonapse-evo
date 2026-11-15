"""Region layouts: independently configurable neuron populations.

A :class:`RegionLayout` partitions the population into contiguous regions, each with a functional role
(``sensory``, ``association``, ``memory``, ``motor``) and its own inhibitory fraction.  Three factories:

``legacy_layout``      the existing three equal regions ("flywire", "manc", "optic-lobe")
``ten_region_layout``  visual, auditory, olfactory, gustatory, thermosensory, mechanosensory, interoceptive,
                       association, memory, motor   (the NEUROWEAVE-LAB organisation)
``custom_layout``      any number of regions with heterogeneous sizes and excitatory/inhibitory mixes
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

ROLES: Tuple[str, ...] = ("sensory", "association", "memory", "motor")
ROLE_ORDER: Dict[str, int] = {role: i for i, role in enumerate(ROLES)}


@dataclass(frozen=True)
class RegionSpec:
    name: str
    role: str
    fraction: float
    inhibitory_fraction: float = 0.2

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown role '{self.role}'; expected one of {ROLES}")
        if not (0.0 < self.fraction) or not (0.0 <= self.inhibitory_fraction < 1.0):
            raise ValueError("fraction must be positive and inhibitory_fraction in [0, 1)")


class RegionLayout:
    def __init__(self, specs: Sequence[RegionSpec], total_neurons: int, sign_rule: str = "per_region"):
        if total_neurons < len(specs):
            raise ValueError("fewer neurons than regions")
        self.specs: Tuple[RegionSpec, ...] = tuple(specs)
        self.total_neurons = int(total_neurons)
        self.sign_rule = sign_rule
        share = np.array([s.fraction for s in self.specs], dtype=np.float64)
        raw = share / share.sum() * self.total_neurons
        sizes = np.maximum(1, np.floor(raw).astype(int))
        while sizes.sum() < self.total_neurons:                       # largest remainder
            sizes[int(np.argmax(raw - sizes))] += 1
        while sizes.sum() > self.total_neurons:
            sizes[int(np.argmax(sizes))] -= 1
        self.sizes: Tuple[int, ...] = tuple(int(s) for s in sizes)
        self.starts: Tuple[int, ...] = tuple(int(s) for s in np.concatenate([[0], np.cumsum(sizes)[:-1]]))
        self.assignments = np.repeat(np.arange(len(self.specs)), sizes).astype(np.int64)

    # -- lookups -------------------------------------------------------------
    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(s.role for s in self.specs)

    @property
    def num_regions(self) -> int:
        return len(self.specs)

    def region_id(self, region: Union[int, str]) -> int:
        return self.names.index(region) if isinstance(region, str) else int(region)

    def indices(self, region: Union[int, str]) -> np.ndarray:
        rid = self.region_id(region)
        return np.arange(self.starts[rid], self.starts[rid] + self.sizes[rid])

    def slices(self) -> Dict[str, slice]:
        return {s.name: slice(self.starts[i], self.starts[i] + self.sizes[i]) for i, s in enumerate(self.specs)}

    def region_of(self, neuron: int) -> int:
        return int(self.assignments[int(neuron)])

    def role_of(self, region: Union[int, str]) -> str:
        return self.specs[self.region_id(region)].role

    def regions_with_role(self, role: str) -> List[int]:
        return [i for i, s in enumerate(self.specs) if s.role == role]

    @property
    def sensory_region_ids(self) -> List[int]:
        return self.regions_with_role("sensory")

    @property
    def motor_offset(self) -> int:
        motors = self.regions_with_role("motor")
        return self.starts[motors[0]] if motors else 0

    def signs(self) -> np.ndarray:
        """+1 excitatory / -1 inhibitory per neuron (a property of the SOURCE neuron)."""
        if self.sign_rule == "global_fifth":             # convention used by the existing code base
            return np.where(np.arange(self.total_neurons) % 5 == 0, -1.0, 1.0)
        signs = np.ones(self.total_neurons)
        for i, spec in enumerate(self.specs):
            if spec.inhibitory_fraction > 0:
                stride = max(2, int(round(1.0 / spec.inhibitory_fraction)))
                signs[self.starts[i]: self.starts[i] + self.sizes[i]: stride] = -1.0
        return signs

    def population_summary(self) -> List[Dict[str, Any]]:
        signs = self.signs()
        out = []
        for i, spec in enumerate(self.specs):
            sl = slice(self.starts[i], self.starts[i] + self.sizes[i])
            out.append({"region": spec.name, "role": spec.role, "neurons": self.sizes[i],
                        "excitatory": int((signs[sl] > 0).sum()), "inhibitory": int((signs[sl] < 0).sum())})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"total_neurons": self.total_neurons, "sign_rule": self.sign_rule,
                "regions": [{"name": s.name, "role": s.role, "fraction": s.fraction, "inhibitory_fraction": s.inhibitory_fraction,
                             "size": self.sizes[i], "start": self.starts[i]} for i, s in enumerate(self.specs)]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionLayout":
        specs = [RegionSpec(r["name"], r["role"], float(r.get("size", r["fraction"])), float(r.get("inhibitory_fraction", 0.2)))
                 for r in data["regions"]]
        return cls(specs, int(data["total_neurons"]), data.get("sign_rule", "per_region"))


def legacy_layout(total_neurons: int = 768) -> RegionLayout:
    specs = [RegionSpec("flywire", "sensory", 1.0), RegionSpec("manc", "association", 1.0), RegionSpec("optic-lobe", "motor", 1.0)]
    layout = RegionLayout(specs, total_neurons, sign_rule="global_fifth")
    per = total_neurons // 3                                            # match TopologyState's integer split exactly
    layout.sizes = (per, per, total_neurons - 2 * per)
    layout.starts = (0, per, 2 * per)
    layout.assignments = np.repeat(np.arange(3), layout.sizes).astype(np.int64)
    return layout


TEN_REGIONS: Tuple[Tuple[str, str, float], ...] = (
    ("visual", "sensory", 0.08), ("auditory", "sensory", 0.08), ("olfactory", "sensory", 0.08), ("gustatory", "sensory", 0.08),
    ("thermosensory", "sensory", 0.08), ("mechanosensory", "sensory", 0.08), ("interoceptive", "sensory", 0.08),
    ("association", "association", 0.20), ("memory", "memory", 0.14), ("motor", "motor", 0.10),
)


def ten_region_layout(total_neurons: int = 768) -> RegionLayout:
    return RegionLayout([RegionSpec(n, r, f) for n, r, f in TEN_REGIONS], total_neurons, sign_rule="global_fifth")


def custom_layout(specs: Sequence[Union[RegionSpec, Dict[str, Any]]], total_neurons: int) -> RegionLayout:
    built = [s if isinstance(s, RegionSpec) else RegionSpec(**s) for s in specs]
    return RegionLayout(built, total_neurons, sign_rule="per_region")


__all__ = ["ROLES", "ROLE_ORDER", "RegionSpec", "RegionLayout", "legacy_layout", "ten_region_layout", "custom_layout", "TEN_REGIONS"]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


ALLOWED_COORDS = ("t", "x", "y", "z")


@dataclass(frozen=True)
class CoordinateSpec:
    coords: tuple[str, ...]
    include_time: bool
    spatial_axes: tuple[str, ...]

    @property
    def rope_dims(self) -> int:
        return len(self.coords)


@dataclass(frozen=True)
class PositionBank:
    rope_positions: np.ndarray
    positions_4d: np.ndarray


@dataclass(frozen=True)
class RequirementCheck:
    ok: bool
    rule: str


@dataclass(frozen=True)
class RunConfig:
    encoders: tuple[str, ...]
    dim: int
    num_vectors: int
    seed: int
    theta_base: float
    coords_spec: CoordinateSpec
    grid_size: int
    centered_coords: bool
    t_values: np.ndarray
    z_value: float
    position_chunk_size: int
    save_dir: Path | None = None
    save_encoded: bool = False
    encoder_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def params_for(self, encoder_name: str) -> dict[str, Any]:
        params = self.encoder_params.get(encoder_name, {})
        return params if isinstance(params, dict) else {}

    def param(self, encoder_name: str, key: str, default: Any) -> Any:
        return self.params_for(encoder_name).get(key, default)


CacheT = Any
InvariantMetrics = dict[str, float | int]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    validate_config: Callable[[RunConfig], RequirementCheck]
    precompute: Callable[[RunConfig, PositionBank], CacheT]
    apply: Callable[[np.ndarray, CacheT, int], np.ndarray]
    check_invariants: Callable[[np.ndarray, np.ndarray], InvariantMetrics]

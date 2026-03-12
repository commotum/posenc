from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ENCODER_NAMES = ("axial", "spiral", "monster", "f-monster", "ape")
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
    monster_positions: np.ndarray


@dataclass(frozen=True)
class RequirementCheck:
    ok: bool
    rule: str


@dataclass(frozen=True)
class ExperimentConfig:
    encoders: tuple[str, ...]
    dim: int
    num_vectors: int
    seed: int
    theta_base: float
    coords_spec: CoordinateSpec
    num_directions: int
    top_delta: float
    span: float
    grid_size: int
    centered_coords: bool
    t_values: np.ndarray
    z_value: float
    position_chunk_size: int
    save_dir: Path | None
    save_encoded: bool
    raw_coords: str
    config_path: Path | None = None
    config_payload: dict[str, Any] | None = None

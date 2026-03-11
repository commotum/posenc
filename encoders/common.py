from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.types import ExperimentConfig, PositionBank, RequirementCheck

CacheT = Any
InvariantMetrics = dict[str, float | int]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    validate_config: Callable[[ExperimentConfig], RequirementCheck]
    precompute: Callable[[ExperimentConfig, PositionBank], CacheT]
    apply: Callable[[np.ndarray, CacheT, int], np.ndarray]
    check_invariants: Callable[[np.ndarray, np.ndarray], InvariantMetrics]

"""Template contract for adding a new positional encoder module.

Required exports:
- NAME: str
- Cache dataclass
- validate_config(cfg) -> RequirementCheck
- precompute(cfg, bank) -> Cache
- apply(vectors, cache, chunk_size) -> np.ndarray
- check_invariants(vectors, encoded) -> dict[str, float | int]
- SPEC: EncoderSpec
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import EncoderSpec, PositionBank, RequirementCheck, RunConfig


NAME = "replace_me"


@dataclass(frozen=True)
class Cache:
    example: np.ndarray


def validate_config(cfg: RunConfig) -> RequirementCheck:
    return RequirementCheck(ok=True, rule="define compatibility rule")


def precompute(cfg: RunConfig, bank: PositionBank) -> Cache:
    raise NotImplementedError


def apply(vectors: np.ndarray, cache: Cache, chunk_size: int) -> np.ndarray:
    raise NotImplementedError


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float | int]:
    raise NotImplementedError


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

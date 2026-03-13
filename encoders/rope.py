from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.frequencies import base_frequencies
from core.types import ExperimentConfig, PositionBank, RequirementCheck
from encoders.common import EncoderSpec


NAME = "rope"


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, C)
    sequence_positions: np.ndarray  # (P,)
    inv_freq: np.ndarray  # (D // 2,)
    cos_phase: np.ndarray  # (P, D // 2)
    sin_phase: np.ndarray  # (P, D // 2)


def validate_config(cfg: ExperimentConfig) -> RequirementCheck:
    ok = cfg.dim % 2 == 0
    return RequirementCheck(ok=ok, rule="dim % 2 == 0")


def precompute(cfg: ExperimentConfig, bank: PositionBank) -> Cache:
    if cfg.dim % 2 != 0:
        raise ValueError("RoPE requires dim divisible by 2 for complex pairs.")

    num_positions = bank.rope_positions.shape[0]
    sequence_positions = np.arange(num_positions, dtype=np.float64)
    inv_freq = base_frequencies(cfg.dim // 2, cfg.theta_base)
    phase = sequence_positions[:, None] * inv_freq[None, :]

    return Cache(
        positions=bank.rope_positions,
        sequence_positions=sequence_positions,
        inv_freq=inv_freq,
        cos_phase=np.cos(phase),
        sin_phase=np.sin(phase),
    )


def _chunk_slices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]

    slices: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        slices.append((start, end))
        start = end
    return slices


def apply(vectors: np.ndarray, cache: Cache, chunk_size: int) -> np.ndarray:
    num_vectors, dim = vectors.shape
    num_positions = cache.positions.shape[0]

    x_even = vectors[:, 0::2]
    x_odd = vectors[:, 1::2]

    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)
    for start, end in _chunk_slices(num_positions, chunk_size):
        cos_phase = cache.cos_phase[start:end][None, :, :]
        sin_phase = cache.sin_phase[start:end][None, :, :]

        r0 = x_even[:, None, :] * cos_phase - x_odd[:, None, :] * sin_phase
        r1 = x_even[:, None, :] * sin_phase + x_odd[:, None, :] * cos_phase

        out[:, start:end, 0::2] = r0
        out[:, start:end, 1::2] = r1
    return out


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float]:
    original_norms = np.linalg.norm(vectors, axis=1)
    encoded_norms = np.linalg.norm(encoded, axis=2)
    return {
        "max_abs_euclidean_norm_error": float(np.max(np.abs(encoded_norms - original_norms[:, None]))),
    }


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

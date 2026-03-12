from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import ExperimentConfig, PositionBank, RequirementCheck
from encoders.common import EncoderSpec


NAME = "ape"


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, C)
    pe: np.ndarray  # (P, D)


def validate_config(cfg: ExperimentConfig) -> RequirementCheck:
    ok = cfg.dim % 2 == 0
    return RequirementCheck(ok=ok, rule="dim % 2 == 0")


def precompute(cfg: ExperimentConfig, bank: PositionBank) -> Cache:
    if cfg.dim % 2 != 0:
        raise ValueError("APE requires dim divisible by 2 for sin/cos pairs.")

    num_positions = bank.rope_positions.shape[0]
    positions = np.arange(num_positions, dtype=np.float64)[:, None]
    dim_indices = np.arange(cfg.dim, dtype=np.float64)[None, :]

    angle_rates = cfg.theta_base ** (-(2.0 * np.floor(dim_indices / 2.0)) / float(cfg.dim))
    angle_rads = positions * angle_rates

    pe = angle_rads.copy()
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    return Cache(positions=bank.rope_positions, pe=pe)


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
    num_positions = cache.pe.shape[0]

    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)
    for start, end in _chunk_slices(num_positions, chunk_size):
        out[:, start:end, :] = vectors[:, None, :] + cache.pe[None, start:end, :]
    return out


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float | int]:
    delta = encoded - vectors[:, None, :]
    reference = delta[0]
    consistency_error = float(np.max(np.abs(delta - reference[None, :, :])))

    pe_norms = np.linalg.norm(reference, axis=1)
    return {
        "max_abs_broadcast_consistency_error": consistency_error,
        "mean_pe_norm": float(np.mean(pe_norms)),
        "max_pe_norm": float(np.max(pe_norms)),
    }


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

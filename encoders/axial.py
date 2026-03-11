from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.frequencies import base_frequencies
from core.types import ExperimentConfig, PositionBank, RequirementCheck
from encoders.common import EncoderSpec


NAME = "axial"


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, C)
    freqs: np.ndarray  # (F,)
    cos_axes: np.ndarray  # (P, C, F)
    sin_axes: np.ndarray  # (P, C, F)


def validate_config(cfg: ExperimentConfig) -> RequirementCheck:
    ok = cfg.dim % (2 * cfg.coords_spec.rope_dims) == 0
    return RequirementCheck(
        ok=ok,
        rule=f"dim % (2 * len(coords)) == 0; len(coords)={cfg.coords_spec.rope_dims}",
    )


def precompute(cfg: ExperimentConfig, bank: PositionBank) -> Cache:
    rope_positions = bank.rope_positions
    coord_dims = rope_positions.shape[1]
    if cfg.dim % (2 * coord_dims) != 0:
        raise ValueError("Axial factorization requires dim divisible by 2 * len(coords).")

    pair_count = cfg.dim // (2 * coord_dims)
    freqs = base_frequencies(pair_count, cfg.theta_base)
    phase = rope_positions[:, :, None] * freqs[None, None, :]
    return Cache(
        positions=rope_positions,
        freqs=freqs,
        cos_axes=np.cos(phase),
        sin_axes=np.sin(phase),
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
    num_positions, coord_dims = cache.positions.shape
    pair_count = cache.freqs.size

    groups = vectors.reshape(num_vectors, coord_dims, pair_count, 2)
    g0 = groups[:, :, :, 0]
    g1 = groups[:, :, :, 1]

    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)
    for start, end in _chunk_slices(num_positions, chunk_size):
        cos_axes = cache.cos_axes[start:end][None, :, :, :]
        sin_axes = cache.sin_axes[start:end][None, :, :, :]

        r0 = g0[:, None, :, :] * cos_axes - g1[:, None, :, :] * sin_axes
        r1 = g0[:, None, :, :] * sin_axes + g1[:, None, :, :] * cos_axes
        out[:, start:end, :] = np.stack((r0, r1), axis=-1).reshape(num_vectors, end - start, dim)
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

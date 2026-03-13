from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.math import chunk_slices, max_abs_euclidean_norm_error, spiral_frequency_sets
from core.types import EncoderSpec, PositionBank, RequirementCheck, RunConfig


NAME = "spiral"


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, C)
    direction_vectors: np.ndarray  # (G, C)
    frequency_sets: np.ndarray  # (G, F)
    projected: np.ndarray  # (P, G)
    cos_phase: np.ndarray  # (P, G, F)
    sin_phase: np.ndarray  # (P, G, F)


def _num_directions(cfg: RunConfig) -> int:
    raw = cfg.param(NAME, "num_directions", cfg.coords_spec.rope_dims)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spiral param 'num_directions' must be an integer.") from exc
    if value <= 0:
        raise ValueError("Spiral param 'num_directions' must be positive.")
    return value


def validate_config(cfg: RunConfig) -> RequirementCheck:
    try:
        num_directions = _num_directions(cfg)
    except ValueError as exc:
        return RequirementCheck(ok=False, rule=str(exc))

    ok = num_directions == cfg.coords_spec.rope_dims and cfg.dim % (2 * num_directions) == 0
    return RequirementCheck(
        ok=ok,
        rule=(
            "encoder_params.spiral.num_directions == len(coords) and "
            f"dim % (2 * num_directions) == 0; num_directions={num_directions}"
        ),
    )


def _orthonormalize_rows(matrix: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for idx in range(matrix.shape[0]):
        vec = matrix[idx].astype(np.float64).copy()
        for row in rows:
            vec -= np.dot(vec, row) * row
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            raise ValueError("Failed to build stable spiral direction vectors.")
        rows.append(vec / norm)
    return np.stack(rows, axis=0)


def _direction_vectors(coord_dims: int) -> np.ndarray:
    if coord_dims == 1:
        return np.asarray([[1.0]], dtype=np.float64)
    if coord_dims == 2:
        return _orthonormalize_rows(np.asarray([[1.0, 1.0], [-1.0, 1.0]], dtype=np.float64))
    if coord_dims == 3:
        return _orthonormalize_rows(
            np.asarray(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, -2.0],
                ],
                dtype=np.float64,
            )
        )
    if coord_dims == 4:
        return _orthonormalize_rows(
            np.asarray(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
    raise ValueError("This script supports up to 4 coordinates in --coords.")


def precompute(cfg: RunConfig, bank: PositionBank) -> Cache:
    rope_positions = bank.rope_positions
    coord_dims = rope_positions.shape[1]
    num_directions = _num_directions(cfg)
    if num_directions != coord_dims:
        raise ValueError("Spiral requires num_directions == len(coords).")

    frequency_sets = spiral_frequency_sets(cfg.dim, num_directions, cfg.theta_base)
    direction_vectors = _direction_vectors(coord_dims)
    projected = rope_positions @ direction_vectors.T
    phase = projected[:, :, None] * frequency_sets[None, :, :]
    return Cache(
        positions=rope_positions,
        direction_vectors=direction_vectors,
        frequency_sets=frequency_sets,
        projected=projected,
        cos_phase=np.cos(phase),
        sin_phase=np.sin(phase),
    )


def apply(vectors: np.ndarray, cache: Cache, chunk_size: int) -> np.ndarray:
    num_vectors, dim = vectors.shape
    num_positions = cache.positions.shape[0]
    num_groups, group_pairs = cache.frequency_sets.shape

    groups = vectors.reshape(num_vectors, num_groups, group_pairs, 2)
    g0 = groups[:, :, :, 0]
    g1 = groups[:, :, :, 1]

    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)
    for start, end in chunk_slices(num_positions, chunk_size):
        cos_phase = cache.cos_phase[start:end][None, :, :, :]
        sin_phase = cache.sin_phase[start:end][None, :, :, :]

        r0 = g0[:, None, :, :] * cos_phase - g1[:, None, :, :] * sin_phase
        r1 = g0[:, None, :, :] * sin_phase + g1[:, None, :, :] * cos_phase
        out[:, start:end, :] = np.stack((r0, r1), axis=-1).reshape(num_vectors, end - start, dim)
    return out


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float]:
    return {
        "max_abs_euclidean_norm_error": max_abs_euclidean_norm_error(vectors, encoded),
    }


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

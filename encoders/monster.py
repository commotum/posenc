from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.frequencies import base_frequencies
from core.types import ExperimentConfig, MONSTER_SLICE, PositionBank, RequirementCheck
from encoders.common import EncoderSpec


NAME = "monster"


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, 4) with columns (t, x, y, z)
    inv_freq: np.ndarray  # (F,)
    ch: np.ndarray  # (P, F)
    sh: np.ndarray  # (P, F)
    c_axes: np.ndarray  # (P, F, 3)
    s_axes: np.ndarray  # (P, F, 3)


def validate_config(cfg: ExperimentConfig) -> RequirementCheck:
    ok = cfg.dim % MONSTER_SLICE == 0
    return RequirementCheck(ok=ok, rule="dim % 12 == 0")


def precompute(cfg: ExperimentConfig, bank: PositionBank) -> Cache:
    positions_4d = bank.monster_positions
    if cfg.dim % MONSTER_SLICE != 0:
        raise ValueError("MonSTER requires dim divisible by 12.")

    num_freq = cfg.dim // MONSTER_SLICE
    inv_freq = base_frequencies(num_freq, cfg.theta_base)
    unit = 1.0 / float(cfg.top_delta)

    phi = positions_4d[:, 0:1] * unit * inv_freq[None, :]
    thx = positions_4d[:, 1:2] * unit * inv_freq[None, :]
    thy = positions_4d[:, 2:3] * unit * inv_freq[None, :]
    thz = positions_4d[:, 3:4] * unit * inv_freq[None, :]

    return Cache(
        positions=positions_4d,
        inv_freq=inv_freq,
        ch=np.cosh(phi),
        sh=np.sinh(phi),
        c_axes=np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=2),
        s_axes=np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=2),
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
    num_freq = cache.inv_freq.size

    base = vectors.reshape(num_vectors, num_freq, 3, 4)
    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)

    for start, end in _chunk_slices(num_positions, chunk_size):
        pos_count = end - start
        state = np.broadcast_to(base[None, :, :, :, :], (pos_count, num_vectors, num_freq, 3, 4)).copy()

        ch = cache.ch[start:end][:, None, :, None]
        sh = cache.sh[start:end][:, None, :, None]
        c_axes = cache.c_axes[start:end][:, None, :, :]
        s_axes = cache.s_axes[start:end][:, None, :, :]

        time_components = state[:, :, :, :, 0]
        aligned_spatial = np.empty_like(time_components)
        aligned_spatial[:, :, :, 0] = state[:, :, :, 0, 1]
        aligned_spatial[:, :, :, 1] = state[:, :, :, 1, 2]
        aligned_spatial[:, :, :, 2] = state[:, :, :, 2, 3]

        boosted_time = ch * time_components - sh * aligned_spatial
        boosted_space = -sh * time_components + ch * aligned_spatial

        state[:, :, :, :, 0] = boosted_time
        state[:, :, :, 0, 1] = boosted_space[:, :, :, 0]
        state[:, :, :, 1, 2] = boosted_space[:, :, :, 1]
        state[:, :, :, 2, 3] = boosted_space[:, :, :, 2]

        x_u = state[:, :, :, 0, 2].copy()
        x_v = state[:, :, :, 0, 3].copy()
        state[:, :, :, 0, 2] = c_axes[:, :, :, 0] * x_u - s_axes[:, :, :, 0] * x_v
        state[:, :, :, 0, 3] = s_axes[:, :, :, 0] * x_u + c_axes[:, :, :, 0] * x_v

        y_u = state[:, :, :, 1, 1].copy()
        y_v = state[:, :, :, 1, 3].copy()
        state[:, :, :, 1, 1] = c_axes[:, :, :, 1] * y_u - s_axes[:, :, :, 1] * y_v
        state[:, :, :, 1, 3] = s_axes[:, :, :, 1] * y_u + c_axes[:, :, :, 1] * y_v

        z_u = state[:, :, :, 2, 1].copy()
        z_v = state[:, :, :, 2, 2].copy()
        state[:, :, :, 2, 1] = c_axes[:, :, :, 2] * z_u - s_axes[:, :, :, 2] * z_v
        state[:, :, :, 2, 2] = s_axes[:, :, :, 2] * z_u + c_axes[:, :, :, 2] * z_v

        out[:, start:end, :] = state.reshape(pos_count, num_vectors, dim).transpose(1, 0, 2)
    return out


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float]:
    eta = np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    original = vectors.reshape(vectors.shape[0], -1, 4)
    encoded_blocks = encoded.reshape(encoded.shape[0], encoded.shape[1], -1, 4)
    base_norms = np.sum(original * eta[None, None, :] * original, axis=2)
    transformed_norms = np.sum(encoded_blocks * eta[None, None, None, :] * encoded_blocks, axis=3)
    return {
        "max_abs_minkowski_norm_error": float(np.max(np.abs(transformed_norms - base_norms[:, None, :]))),
    }


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

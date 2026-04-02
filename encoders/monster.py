from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.math import base_frequencies, chunk_slices, max_abs_minkowski_norm_error
from core.types import EncoderSpec, PositionBank, RequirementCheck, RunConfig


NAME = "monster"
DEFAULT_L = 1.0 / 4096.0


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, 4) with columns (t, x, y, z)
    inv_freq: np.ndarray  # (F,)
    ch: np.ndarray  # (P, F)
    sh: np.ndarray  # (P, F)
    c_axes: np.ndarray  # (P, F, 3)
    s_axes: np.ndarray  # (P, F, 3)


def _params(cfg: RunConfig) -> float:
    raw_l = cfg.param(NAME, "L", DEFAULT_L)
    try:
        l_value = float(raw_l)
    except (TypeError, ValueError) as exc:
        raise ValueError("MonSTER param 'L' must be numeric.") from exc
    if l_value <= 0:
        raise ValueError("MonSTER param 'L' must be positive.")
    return l_value


def validate_config(cfg: RunConfig) -> RequirementCheck:
    try:
        l_value = _params(cfg)
    except ValueError as exc:
        return RequirementCheck(ok=False, rule=str(exc))

    ok = cfg.dim % 12 == 0
    return RequirementCheck(
        ok=ok,
        rule=f"dim % 12 == 0; L={l_value}",
    )


def precompute(cfg: RunConfig, bank: PositionBank) -> Cache:
    positions_4d = bank.positions_4d
    if cfg.dim % 12 != 0:
        raise ValueError("MonSTER requires dim divisible by 12.")

    l_value = _params(cfg)
    num_freq = cfg.dim // 12
    inv_freq = base_frequencies(num_freq, cfg.theta_base)

    phi = (positions_4d[:, 0:1] * l_value) * inv_freq[None, :]
    spatial_angles = positions_4d[:, 1:4, None] * inv_freq[None, None, :]  # (P,3,F)
    spatial_angles = np.transpose(spatial_angles, (0, 2, 1))  # (P,F,3)

    return Cache(
        positions=positions_4d,
        inv_freq=inv_freq,
        ch=np.cosh(phi),
        sh=np.sinh(phi),
        c_axes=np.cos(spatial_angles),
        s_axes=np.sin(spatial_angles),
    )


def apply(vectors: np.ndarray, cache: Cache, chunk_size: int) -> np.ndarray:
    num_vectors, dim = vectors.shape
    num_positions = cache.positions.shape[0]
    num_freq = cache.inv_freq.size

    base = vectors.reshape(num_vectors, num_freq, 3, 4)
    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)

    for start, end in chunk_slices(num_positions, chunk_size):
        pos_count = end - start
        state = np.broadcast_to(base[None, :, :, :, :], (pos_count, num_vectors, num_freq, 3, 4)).copy()

        ch = cache.ch[start:end][:, None, :, None]
        sh = cache.sh[start:end][:, None, :, None]
        c_axes = cache.c_axes[start:end][:, None, :, :]
        s_axes = cache.s_axes[start:end][:, None, :, :]

        # Step 1: boost along each axis' aligned spatial component.
        comp_idx = np.array([1, 2, 3], dtype=np.int64)[None, None, None, :, None]
        t = state[..., 0]  # (P,N,F,3)
        aligned = np.take_along_axis(state, comp_idx, axis=4)[..., 0]  # (P,N,F,3)

        t1 = ch * t - sh * aligned
        x1 = -sh * t + ch * aligned
        state[..., 0] = t1
        np.put_along_axis(state, comp_idx, x1[..., None], axis=4)

        # Step 2: rotate in orthogonal spatial planes.
        pair_idx = np.array([[2, 3], [1, 3], [1, 2]], dtype=np.int64)[None, None, None, :, :]
        pair_vals = np.take_along_axis(state, pair_idx, axis=4)  # (P,N,F,3,2)
        u = pair_vals[..., 0]
        v = pair_vals[..., 1]

        u2 = c_axes * u - s_axes * v
        v2 = s_axes * u + c_axes * v
        rotated = np.stack((u2, v2), axis=-1)
        np.put_along_axis(state, pair_idx, rotated, axis=4)

        out[:, start:end, :] = state.reshape(pos_count, num_vectors, dim).transpose(1, 0, 2)
    return out


def check_invariants(vectors: np.ndarray, encoded: np.ndarray) -> dict[str, float]:
    return {
        "max_abs_minkowski_norm_error": max_abs_minkowski_norm_error(vectors, encoded),
    }


SPEC = EncoderSpec(
    name=NAME,
    validate_config=validate_config,
    precompute=precompute,
    apply=apply,
    check_invariants=check_invariants,
)

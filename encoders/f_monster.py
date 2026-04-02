from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.math import base_frequencies, chunk_slices, max_abs_minkowski_norm_error
from core.types import EncoderSpec, PositionBank, RequirementCheck, RunConfig


NAME = "f-monster"
DEFAULT_L = 1.0 / 4096.0


@dataclass(frozen=True)
class Cache:
    positions: np.ndarray  # (P, 4) with columns (t, x, y, z)
    inv_freq: np.ndarray  # (F,)
    axis: np.ndarray  # (F, 3)
    ch: np.ndarray  # (P, F)
    sh: np.ndarray  # (P, F)
    c: np.ndarray  # (P, F)
    s: np.ndarray  # (P, F)


def _params(cfg: RunConfig) -> float:
    raw_l = cfg.param(NAME, "L", DEFAULT_L)
    try:
        l_value = float(raw_l)
    except (TypeError, ValueError) as exc:
        raise ValueError("F-MonSTER param 'L' must be numeric.") from exc
    if l_value <= 0:
        raise ValueError("F-MonSTER param 'L' must be positive.")
    return l_value


def validate_config(cfg: RunConfig) -> RequirementCheck:
    try:
        l_value = _params(cfg)
    except ValueError as exc:
        return RequirementCheck(ok=False, rule=str(exc))

    ok = cfg.dim % 4 == 0
    return RequirementCheck(
        ok=ok,
        rule=f"dim % 4 == 0; L={l_value}",
    )


def _fibonacci_sphere(num_points: int) -> np.ndarray:
    i = np.arange(num_points, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / num_points
    r = np.sqrt(1.0 - z * z)
    theta = i * phi

    out = np.empty((num_points, 3), dtype=np.float64)
    out[:, 0] = np.cos(theta) * r
    out[:, 1] = np.sin(theta) * r
    out[:, 2] = z

    # Defensive normalization in case of FP drift.
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def precompute(cfg: RunConfig, bank: PositionBank) -> Cache:
    positions_4d = bank.positions_4d
    if cfg.dim % 4 != 0:
        raise ValueError("F-MonSTER requires dim divisible by 4.")

    l_value = _params(cfg)
    num_freq = cfg.dim // 4
    inv_freq = base_frequencies(num_freq, cfg.theta_base)

    axis = _fibonacci_sphere(num_freq)  # (F,3)
    t = positions_4d[:, 0:1]  # (P,1)
    spatial = positions_4d[:, 1:4]  # (P,3)

    phi = (t * l_value) * inv_freq[None, :]  # (P,F)
    proj = spatial @ axis.T  # (P,F)
    theta = proj * inv_freq[None, :]  # (P,F)

    return Cache(
        positions=positions_4d,
        inv_freq=inv_freq,
        axis=axis,
        ch=np.cosh(phi),
        sh=np.sinh(phi),
        c=np.cos(theta),
        s=np.sin(theta),
    )


def apply(vectors: np.ndarray, cache: Cache, chunk_size: int) -> np.ndarray:
    num_vectors, dim = vectors.shape
    num_positions = cache.positions.shape[0]
    num_freq = cache.inv_freq.size

    base = vectors.reshape(num_vectors, num_freq, 4)
    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)

    axis = cache.axis[None, None, :, :]  # (1,1,F,3)

    for start, end in chunk_slices(num_positions, chunk_size):
        pos_count = end - start
        state = np.broadcast_to(base[None, :, :, :], (pos_count, num_vectors, num_freq, 4)).copy()

        ch = cache.ch[start:end][:, None, :]  # (P,1,F)
        sh = cache.sh[start:end][:, None, :]  # (P,1,F)
        c = cache.c[start:end][:, None, :]  # (P,1,F)
        s = cache.s[start:end][:, None, :]  # (P,1,F)

        t = state[:, :, :, 0]  # (P,N,F)
        spatial = state[:, :, :, 1:]  # (P,N,F,3)

        # Step 1: boost along each block's own axis.
        proj = np.sum(spatial * axis, axis=-1)  # (P,N,F)
        t1 = ch * t - sh * proj
        spatial1 = spatial + (((ch - 1.0) * proj - sh * t)[..., None] * axis)

        # Step 2: rotate around the same axis.
        proj1 = np.sum(spatial1 * axis, axis=-1)  # (P,N,F)
        cross = np.cross(axis, spatial1)  # (P,N,F,3)
        spatial2 = (
            c[..., None] * spatial1
            + s[..., None] * cross
            + (1.0 - c)[..., None] * proj1[..., None] * axis
        )

        state[:, :, :, 0] = t1
        state[:, :, :, 1:] = spatial2

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

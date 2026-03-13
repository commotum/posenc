from __future__ import annotations

import numpy as np


def base_frequencies(num_freqs: int, theta_base: float) -> np.ndarray:
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive.")
    return theta_base ** (-np.arange(num_freqs, dtype=np.float64) / float(num_freqs))


def spiral_frequency_sets(embed_dim: int, num_groups: int, theta_base: float) -> np.ndarray:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if embed_dim % (2 * num_groups) != 0:
        raise ValueError("Spiral factorization requires dim divisible by 2 * num_groups.")

    group_pairs = embed_dim // (2 * num_groups)
    freqs = base_frequencies(num_groups * group_pairs, theta_base)
    return freqs.reshape(num_groups, group_pairs)


def chunk_slices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]

    slices: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        slices.append((start, end))
        start = end
    return slices


def max_abs_euclidean_norm_error(vectors: np.ndarray, encoded: np.ndarray) -> float:
    original_norms = np.linalg.norm(vectors, axis=1)
    encoded_norms = np.linalg.norm(encoded, axis=2)
    return float(np.max(np.abs(encoded_norms - original_norms[:, None])))


def max_abs_minkowski_norm_error(vectors: np.ndarray, encoded: np.ndarray) -> float:
    eta = np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    original = vectors.reshape(vectors.shape[0], -1, 4)
    encoded_blocks = encoded.reshape(encoded.shape[0], encoded.shape[1], -1, 4)
    base_norms = np.sum(original * eta[None, None, :] * original, axis=2)
    transformed_norms = np.sum(encoded_blocks * eta[None, None, None, :] * encoded_blocks, axis=3)
    return float(np.max(np.abs(transformed_norms - base_norms[:, None, :])))

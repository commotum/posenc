from __future__ import annotations

import numpy as np


def base_frequencies(num_freqs: int, theta_base: float) -> np.ndarray:
    return theta_base ** (-np.arange(num_freqs, dtype=np.float64) / num_freqs)


def spiral_frequency_sets(embed_dim: int, num_groups: int, theta_base: float) -> np.ndarray:
    if embed_dim % (2 * num_groups) != 0:
        raise ValueError("Spiral factorization requires dim divisible by 2 * num_groups.")
    group_pairs = embed_dim // (2 * num_groups)
    freqs = base_frequencies(num_groups * group_pairs, theta_base)
    return freqs.reshape(num_groups, group_pairs)

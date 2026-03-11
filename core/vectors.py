from __future__ import annotations

import numpy as np


def random_vectors(num_vectors: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(0.0, 1.0, size=(num_vectors, dim)).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero_norm = norms.squeeze(-1) == 0.0
    if np.any(zero_norm):
        vectors[zero_norm] = 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms * np.sqrt(dim)
    return vectors


def verify_vectors(vectors: np.ndarray, dim: int) -> dict[str, float]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D tensor for vectors, got shape {vectors.shape}.")
    if vectors.shape[1] != dim:
        raise ValueError(f"Vector dim mismatch: expected {dim}, got {vectors.shape[1]}.")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Vectors contain non-finite values.")
    target = np.sqrt(dim)
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "target_norm": float(target),
        "mean_norm": float(np.mean(norms)),
        "max_abs_norm_error": float(np.max(np.abs(norms - target))),
    }

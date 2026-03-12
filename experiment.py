from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from core.io import maybe_save
from core.positions import build_position_bank
from core.types import ENCODER_NAMES, ExperimentConfig, RequirementCheck
from core.vectors import random_vectors, verify_vectors
from encoders.common import EncoderSpec
from encoders.registry import all_specs, resolve_specs


@dataclass(frozen=True)
class ExperimentArtifacts:
    summary: dict[str, object]
    vectors: np.ndarray
    encoded: dict[str, np.ndarray]


def requirement_report(cfg: ExperimentConfig) -> dict[str, RequirementCheck]:
    specs = all_specs()
    report: dict[str, RequirementCheck] = {}
    for name in ENCODER_NAMES:
        report[name] = specs[name].validate_config(cfg)
    return report


def enforce_requested_requirements(
    requested_encoders: tuple[str, ...],
    report: dict[str, RequirementCheck],
) -> None:
    failing = [name for name in requested_encoders if not report[name].ok]
    if failing:
        details = "; ".join(f"{name}: {report[name].rule}" for name in failing)
        raise ValueError(f"Requested encoders are incompatible with current settings -> {details}")


def run_experiment(cfg: ExperimentConfig) -> ExperimentArtifacts:
    selected_specs: list[EncoderSpec] = resolve_specs(list(cfg.encoders))

    t0 = perf_counter()
    vectors = random_vectors(cfg.num_vectors, cfg.dim, cfg.seed)
    vector_stats = verify_vectors(vectors, cfg.dim)
    t1 = perf_counter()

    bank = build_position_bank(
        cfg.coords_spec,
        cfg.grid_size,
        cfg.centered_coords,
        cfg.t_values,
        cfg.z_value,
    )

    caches: dict[str, object] = {}
    for spec in selected_specs:
        caches[spec.name] = spec.precompute(cfg, bank)
    t2 = perf_counter()

    encoded: dict[str, np.ndarray] = {}
    verification: dict[str, dict[str, float | int]] = {}
    for spec in selected_specs:
        out = spec.apply(vectors, caches[spec.name], cfg.position_chunk_size)
        encoded[spec.name] = out
        metrics = spec.check_invariants(vectors, out)
        verification[spec.name] = {
            "num_positions": int(out.shape[1]),
            **metrics,
        }
    t3 = perf_counter()

    summary: dict[str, object] = {
        "encoders": list(cfg.encoders),
        "config": {
            "dim": cfg.dim,
            "num_vectors": cfg.num_vectors,
            "seed": cfg.seed,
            "theta_base": cfg.theta_base,
            "coords": list(cfg.coords_spec.coords),
            "rope_coordinate_dims": cfg.coords_spec.rope_dims,
            "spatial_dimensions": len(cfg.coords_spec.spatial_axes),
            "include_time": bool(cfg.coords_spec.include_time),
            "num_directions": cfg.num_directions,
            "top_delta": cfg.top_delta,
            "span": cfg.span,
            "grid_size": cfg.grid_size,
            "centered_coords": bool(cfg.centered_coords),
            "t_values": cfg.t_values.tolist(),
            "z_value": cfg.z_value,
            "position_chunk_size": cfg.position_chunk_size,
        },
        "vector_stats": vector_stats,
        "positions": {
            "rope_positions": int(bank.rope_positions.shape[0]),
            "monster_positions": int(bank.monster_positions.shape[0]),
        },
        "verification": verification,
        "timing_seconds": {
            "generate_vectors": round(t1 - t0, 6),
            "build_caches": round(t2 - t1, 6),
            "apply_encoders": round(t3 - t2, 6),
            "total": round(t3 - t0, 6),
        },
    }

    maybe_save(cfg.save_dir, vectors, summary, encoded, cfg.save_encoded)
    return ExperimentArtifacts(summary=summary, vectors=vectors, encoded=encoded)

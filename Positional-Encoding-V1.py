from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


MONSTER_SLICE = 12
ENCODER_NAMES = ("axial", "spiral", "monster")
ALLOWED_COORDS = ("t", "x", "y", "z")


@dataclass(frozen=True)
class CoordinateSpec:
    coords: tuple[str, ...]
    include_time: bool
    spatial_axes: tuple[str, ...]

    @property
    def rope_dims(self) -> int:
        return len(self.coords)


@dataclass(frozen=True)
class AxialCache:
    positions: np.ndarray  # (P, C)
    freqs: np.ndarray  # (F,)
    cos_axes: np.ndarray  # (P, C, F)
    sin_axes: np.ndarray  # (P, C, F)


@dataclass(frozen=True)
class SpiralCache:
    positions: np.ndarray  # (P, C)
    direction_vectors: np.ndarray  # (G, C)
    frequency_sets: np.ndarray  # (G, F)
    projected: np.ndarray  # (P, G)
    cos_phase: np.ndarray  # (P, G, F)
    sin_phase: np.ndarray  # (P, G, F)


@dataclass(frozen=True)
class MonsterCache:
    positions: np.ndarray  # (P, 4) with columns (t, x, y, z)
    inv_freq: np.ndarray  # (F,)
    ch: np.ndarray  # (P, F)
    sh: np.ndarray  # (P, F)
    c_axes: np.ndarray  # (P, F, 3)
    s_axes: np.ndarray  # (P, F, 3)


def parse_t_values(raw: str) -> np.ndarray:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one value for --t-values.")
    return np.asarray([float(v) for v in values], dtype=np.float64)


def parse_coords(raw: str) -> CoordinateSpec:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not tokens:
        raise ValueError("Expected at least one coordinate in --coords.")

    seen: set[str] = set()
    for token in tokens:
        if token not in ALLOWED_COORDS:
            allowed = ", ".join(ALLOWED_COORDS)
            raise ValueError(f"Invalid coordinate '{token}'. Allowed coordinates: {allowed}.")
        if token in seen:
            raise ValueError(f"Duplicate coordinate '{token}' in --coords.")
        seen.add(token)

    spatial_axes = tuple(axis for axis in tokens if axis != "t")
    return CoordinateSpec(
        coords=tokens,
        include_time=("t" in seen),
        spatial_axes=spatial_axes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified positional encoder pipeline: generate vectors, precompute vectorized caches, "
            "and apply Axial/Spiral/MonSTER blockwise."
        )
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["all"],
        choices=[*ENCODER_NAMES, "all"],
        help="Which encoders to run. Use 'all' to run every encoder.",
    )
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--num-vectors", type=int, default=1, help="Number of random vectors to generate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--theta-base", type=float, default=10_000.0, help="Base frequency theta.")
    parser.add_argument(
        "--coords",
        type=str,
        default="x,y",
        help="Coordinate tuple used by Axial/Spiral, e.g. 'x', 'x,y', 't,x,y', or 't,x,y,z'.",
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=None,
        help=(
            "Spiral direction groups. Defaults to len(coords). "
            "For the requested factorization behavior this must equal len(coords)."
        ),
    )
    parser.add_argument("--top-delta", type=float, default=1024.0, help="MonSTER top_delta denominator.")
    parser.add_argument("--grid-size", type=int, default=16, help="Grid side length for spatial axes.")
    parser.add_argument(
        "--centered-coords",
        action="store_true",
        help="Use centered coordinates instead of integer indices for spatial axes.",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default="0",
        help="Comma-separated t coordinates, e.g. '-8,-4,0,4,8'.",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=0.0,
        help="Default fixed z coordinate when z is not included in --coords.",
    )
    parser.add_argument(
        "--position-chunk-size",
        type=int,
        default=128,
        help="Positions per block during apply stage. <=0 means all at once.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory for vectors, metadata, and optional encoded tensors.",
    )
    parser.add_argument(
        "--save-encoded",
        action="store_true",
        help="If set, save encoded tensors for each encoder to .npy files.",
    )
    args = parser.parse_args()

    if args.num_vectors <= 0:
        parser.error("--num-vectors must be positive.")
    if args.dim <= 0:
        parser.error("--dim must be positive.")
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive.")
    if args.top_delta <= 0:
        parser.error("--top-delta must be positive.")

    try:
        args.coords_spec = parse_coords(args.coords)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        args.t_values = parse_t_values(args.t_values)
    except ValueError as exc:
        parser.error(str(exc))

    if args.num_directions is None:
        args.num_directions = args.coords_spec.rope_dims
    if args.num_directions <= 0:
        parser.error("--num-directions must be positive.")
    if args.num_directions != args.coords_spec.rope_dims:
        parser.error(
            "For this factorization mode, --num-directions must equal len(--coords). "
            f"Received num_directions={args.num_directions}, len(coords)={args.coords_spec.rope_dims}."
        )
    return args


def resolve_encoders(raw_encoders: list[str]) -> list[str]:
    if "all" in raw_encoders:
        return list(ENCODER_NAMES)
    return [name for name in ENCODER_NAMES if name in raw_encoders]


def requirement_report(
    dim: int,
    coord_spec: CoordinateSpec,
    num_directions: int,
) -> dict[str, tuple[bool, str]]:
    report: dict[str, tuple[bool, str]] = {}

    axial_ok = dim % (2 * coord_spec.rope_dims) == 0
    report["axial"] = (
        axial_ok,
        f"dim % (2 * len(coords)) == 0; len(coords)={coord_spec.rope_dims}",
    )

    spiral_ok = (
        num_directions == coord_spec.rope_dims
        and dim % (2 * num_directions) == 0
    )
    report["spiral"] = (
        spiral_ok,
        (
            "num_directions == len(coords) and dim % (2 * num_directions) == 0; "
            f"num_directions={num_directions}"
        ),
    )

    monster_ok = dim % MONSTER_SLICE == 0
    report["monster"] = (
        monster_ok,
        "dim % 12 == 0",
    )
    return report


def enforce_requested_requirements(
    requested_encoders: list[str],
    report: dict[str, tuple[bool, str]],
) -> None:
    failing = [name for name in requested_encoders if not report[name][0]]
    if failing:
        details = "; ".join(f"{name}: {report[name][1]}" for name in failing)
        raise ValueError(f"Requested encoders are incompatible with current settings -> {details}")


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


def make_spatial_axis_values(grid_size: int, centered: bool) -> np.ndarray:
    if centered:
        return np.arange(grid_size, dtype=np.float64) - ((grid_size - 1) / 2.0)
    return np.arange(grid_size, dtype=np.float64)


def build_positions_for_coords(
    coord_spec: CoordinateSpec,
    spatial_values: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    axis_values: list[np.ndarray] = []
    for axis in coord_spec.coords:
        if axis == "t":
            axis_values.append(t_values)
        else:
            axis_values.append(spatial_values)

    mesh = np.meshgrid(*axis_values, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, coord_spec.rope_dims)


def build_monster_positions(
    coord_spec: CoordinateSpec,
    spatial_values: np.ndarray,
    t_values: np.ndarray,
    z_value: float,
) -> np.ndarray:
    t_axis = t_values if coord_spec.include_time else np.asarray([0.0], dtype=np.float64)
    x_axis = spatial_values if "x" in coord_spec.spatial_axes else np.asarray([0.0], dtype=np.float64)
    y_axis = spatial_values if "y" in coord_spec.spatial_axes else np.asarray([0.0], dtype=np.float64)
    z_axis = spatial_values if "z" in coord_spec.spatial_axes else np.asarray([z_value], dtype=np.float64)

    mesh = np.meshgrid(t_axis, x_axis, y_axis, z_axis, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 4)


def base_frequencies(num_freqs: int, theta_base: float) -> np.ndarray:
    return theta_base ** (-np.arange(num_freqs, dtype=np.float64) / num_freqs)


def spiral_frequency_sets(embed_dim: int, num_groups: int, theta_base: float) -> np.ndarray:
    if embed_dim % (2 * num_groups) != 0:
        raise ValueError("Spiral factorization requires dim divisible by 2 * num_groups.")
    group_pairs = embed_dim // (2 * num_groups)
    freqs = base_frequencies(num_groups * group_pairs, theta_base)
    return freqs.reshape(num_groups, group_pairs)


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


def spiral_direction_vectors(coord_dims: int) -> np.ndarray:
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


def build_axial_cache(
    dim: int,
    theta_base: float,
    rope_positions: np.ndarray,
) -> AxialCache:
    coord_dims = rope_positions.shape[1]
    if dim % (2 * coord_dims) != 0:
        raise ValueError("Axial factorization requires dim divisible by 2 * len(coords).")
    pair_count = dim // (2 * coord_dims)
    freqs = base_frequencies(pair_count, theta_base)
    phase = rope_positions[:, :, None] * freqs[None, None, :]
    return AxialCache(
        positions=rope_positions,
        freqs=freqs,
        cos_axes=np.cos(phase),
        sin_axes=np.sin(phase),
    )


def build_spiral_cache(
    dim: int,
    num_directions: int,
    theta_base: float,
    rope_positions: np.ndarray,
) -> SpiralCache:
    coord_dims = rope_positions.shape[1]
    if num_directions != coord_dims:
        raise ValueError("For this mode, Spiral directions must match coordinate dimensionality.")
    frequency_sets = spiral_frequency_sets(dim, num_directions, theta_base)
    direction_vectors = spiral_direction_vectors(coord_dims)
    projected = rope_positions @ direction_vectors.T
    phase = projected[:, :, None] * frequency_sets[None, :, :]
    return SpiralCache(
        positions=rope_positions,
        direction_vectors=direction_vectors,
        frequency_sets=frequency_sets,
        projected=projected,
        cos_phase=np.cos(phase),
        sin_phase=np.sin(phase),
    )


def build_monster_cache(
    dim: int,
    theta_base: float,
    top_delta: float,
    positions_4d: np.ndarray,
) -> MonsterCache:
    if dim % MONSTER_SLICE != 0:
        raise ValueError("MonSTER requires dim divisible by 12.")

    num_freq = dim // MONSTER_SLICE
    inv_freq = base_frequencies(num_freq, theta_base)
    unit = 1.0 / float(top_delta)

    phi = positions_4d[:, 0:1] * unit * inv_freq[None, :]
    thx = positions_4d[:, 1:2] * unit * inv_freq[None, :]
    thy = positions_4d[:, 2:3] * unit * inv_freq[None, :]
    thz = positions_4d[:, 3:4] * unit * inv_freq[None, :]

    return MonsterCache(
        positions=positions_4d,
        inv_freq=inv_freq,
        ch=np.cosh(phi),
        sh=np.sinh(phi),
        c_axes=np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=2),
        s_axes=np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=2),
    )


def chunk_slices(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total)]
    slices: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        slices.append((start, end))
        start = end
    return slices


def apply_axial_blockwise(vectors: np.ndarray, cache: AxialCache, chunk_size: int) -> np.ndarray:
    num_vectors, dim = vectors.shape
    num_positions, coord_dims = cache.positions.shape
    pair_count = cache.freqs.size

    groups = vectors.reshape(num_vectors, coord_dims, pair_count, 2)
    g0 = groups[:, :, :, 0]
    g1 = groups[:, :, :, 1]

    out = np.empty((num_vectors, num_positions, dim), dtype=np.float64)
    for start, end in chunk_slices(num_positions, chunk_size):
        cos_axes = cache.cos_axes[start:end][None, :, :, :]
        sin_axes = cache.sin_axes[start:end][None, :, :, :]

        r0 = g0[:, None, :, :] * cos_axes - g1[:, None, :, :] * sin_axes
        r1 = g0[:, None, :, :] * sin_axes + g1[:, None, :, :] * cos_axes
        out[:, start:end, :] = np.stack((r0, r1), axis=-1).reshape(num_vectors, end - start, dim)
    return out


def apply_spiral_blockwise(vectors: np.ndarray, cache: SpiralCache, chunk_size: int) -> np.ndarray:
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


def apply_monster_blockwise(vectors: np.ndarray, cache: MonsterCache, chunk_size: int) -> np.ndarray:
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


def euclidean_norm_error(vectors: np.ndarray, encoded: np.ndarray) -> float:
    original_norms = np.linalg.norm(vectors, axis=1)
    encoded_norms = np.linalg.norm(encoded, axis=2)
    return float(np.max(np.abs(encoded_norms - original_norms[:, None])))


def monster_minkowski_norm_error(vectors: np.ndarray, encoded: np.ndarray) -> float:
    eta = np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    original = vectors.reshape(vectors.shape[0], -1, 4)
    encoded_blocks = encoded.reshape(encoded.shape[0], encoded.shape[1], -1, 4)
    base_norms = np.sum(original * eta[None, None, :] * original, axis=2)
    transformed_norms = np.sum(encoded_blocks * eta[None, None, None, :] * encoded_blocks, axis=3)
    return float(np.max(np.abs(transformed_norms - base_norms[:, None, :])))


def maybe_save(
    save_dir: Path | None,
    vectors: np.ndarray,
    metadata: dict[str, object],
    encoded: dict[str, np.ndarray],
    save_encoded: bool,
) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "vectors.npy", vectors)
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if save_encoded:
        for name, tensor in encoded.items():
            np.save(save_dir / f"encoded_{name}.npy", tensor)


def main() -> None:
    args = parse_args()
    encoders = resolve_encoders(args.encoders)
    coord_spec: CoordinateSpec = args.coords_spec
    checks = requirement_report(args.dim, coord_spec, args.num_directions)

    print("Encoder compatibility checks:")
    for name in ENCODER_NAMES:
        ok, rule = checks[name]
        print(f"  - {name:<7} {'OK' if ok else 'FAIL'} | {rule}")
    enforce_requested_requirements(encoders, checks)

    t0 = perf_counter()
    vectors = random_vectors(args.num_vectors, args.dim, args.seed)
    vector_stats = verify_vectors(vectors, args.dim)
    t1 = perf_counter()

    spatial_values = make_spatial_axis_values(args.grid_size, args.centered_coords)
    rope_positions = build_positions_for_coords(coord_spec, spatial_values, args.t_values)
    monster_positions = build_monster_positions(coord_spec, spatial_values, args.t_values, args.z_value)

    caches: dict[str, AxialCache | SpiralCache | MonsterCache] = {}
    if "axial" in encoders:
        caches["axial"] = build_axial_cache(args.dim, args.theta_base, rope_positions)
    if "spiral" in encoders:
        caches["spiral"] = build_spiral_cache(
            args.dim,
            args.num_directions,
            args.theta_base,
            rope_positions,
        )
    if "monster" in encoders:
        caches["monster"] = build_monster_cache(
            args.dim,
            args.theta_base,
            args.top_delta,
            monster_positions,
        )
    t2 = perf_counter()

    encoded: dict[str, np.ndarray] = {}
    verification: dict[str, dict[str, float | int]] = {}

    if "axial" in encoders:
        axial_out = apply_axial_blockwise(vectors, caches["axial"], args.position_chunk_size)  # type: ignore[arg-type]
        encoded["axial"] = axial_out
        verification["axial"] = {
            "num_positions": int(axial_out.shape[1]),
            "max_abs_euclidean_norm_error": euclidean_norm_error(vectors, axial_out),
        }

    if "spiral" in encoders:
        spiral_out = apply_spiral_blockwise(vectors, caches["spiral"], args.position_chunk_size)  # type: ignore[arg-type]
        encoded["spiral"] = spiral_out
        verification["spiral"] = {
            "num_positions": int(spiral_out.shape[1]),
            "max_abs_euclidean_norm_error": euclidean_norm_error(vectors, spiral_out),
        }

    if "monster" in encoders:
        monster_out = apply_monster_blockwise(vectors, caches["monster"], args.position_chunk_size)  # type: ignore[arg-type]
        encoded["monster"] = monster_out
        verification["monster"] = {
            "num_positions": int(monster_out.shape[1]),
            "max_abs_minkowski_norm_error": monster_minkowski_norm_error(vectors, monster_out),
        }
    t3 = perf_counter()

    summary: dict[str, object] = {
        "encoders": encoders,
        "config": {
            "dim": args.dim,
            "num_vectors": args.num_vectors,
            "seed": args.seed,
            "theta_base": args.theta_base,
            "coords": list(coord_spec.coords),
            "rope_coordinate_dims": coord_spec.rope_dims,
            "spatial_dimensions": len(coord_spec.spatial_axes),
            "include_time": bool(coord_spec.include_time),
            "num_directions": args.num_directions,
            "top_delta": args.top_delta,
            "grid_size": args.grid_size,
            "centered_coords": bool(args.centered_coords),
            "t_values": args.t_values.tolist(),
            "z_value": args.z_value,
            "position_chunk_size": args.position_chunk_size,
        },
        "vector_stats": vector_stats,
        "positions": {
            "rope_positions": int(rope_positions.shape[0]),
            "monster_positions": int(monster_positions.shape[0]),
        },
        "verification": verification,
        "timing_seconds": {
            "generate_vectors": round(t1 - t0, 6),
            "build_caches": round(t2 - t1, 6),
            "apply_encoders": round(t3 - t2, 6),
            "total": round(t3 - t0, 6),
        },
    }
    maybe_save(args.save_dir, vectors, summary, encoded, args.save_encoded)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Iterable

import numpy as np

from core.types import ALLOWED_COORDS, CoordinateSpec, PositionBank


def parse_t_values(raw: str | Iterable[float]) -> np.ndarray:
    if isinstance(raw, str):
        values = [token.strip() for token in raw.split(",") if token.strip()]
        if not values:
            raise ValueError("Expected at least one value for --t-values.")
        return np.asarray([float(v) for v in values], dtype=np.float64)

    values = [float(v) for v in raw]
    if not values:
        raise ValueError("Expected at least one value for --t-values.")
    return np.asarray(values, dtype=np.float64)


def parse_coords(raw: str | Iterable[str]) -> CoordinateSpec:
    if isinstance(raw, str):
        tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    else:
        tokens = tuple(str(token).strip() for token in raw if str(token).strip())

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


def build_position_bank(
    coord_spec: CoordinateSpec,
    grid_size: int,
    centered_coords: bool,
    t_values: np.ndarray,
    z_value: float,
) -> PositionBank:
    spatial_values = make_spatial_axis_values(grid_size, centered_coords)
    rope_positions = build_positions_for_coords(coord_spec, spatial_values, t_values)
    monster_positions = build_monster_positions(coord_spec, spatial_values, t_values, z_value)
    return PositionBank(rope_positions=rope_positions, monster_positions=monster_positions)

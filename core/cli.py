from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.positions import parse_coords, parse_t_values
from core.types import ENCODER_NAMES, ExperimentConfig


DEFAULTS: dict[str, Any] = {
    "encoders": ["all"],
    "dim": 768,
    "num_vectors": 1,
    "seed": 0,
    "theta_base": 10_000.0,
    "coords": "x,y",
    "num_directions": None,
    "top_delta": 1024.0,
    "span": 2.0 * 3.141592653589793,
    "grid_size": 16,
    "centered_coords": False,
    "t_values": "0",
    "z_value": 0.0,
    "position_chunk_size": 128,
    "save_dir": None,
    "save_encoded": False,
}

COMMON_CONFIG_KEYS = {
    "encoders",
    "dim",
    "num_vectors",
    "seed",
    "theta_base",
    "coords",
    "num_directions",
    "top_delta",
    "span",
    "grid_size",
    "centered_coords",
    "t_values",
    "z_value",
    "position_chunk_size",
    "save_dir",
    "save_encoded",
}


def load_config_payload(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a JSON object.")
    return payload


def _normalize_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(defaults)

    if isinstance(normalized.get("coords"), (list, tuple)):
        normalized["coords"] = ",".join(str(token) for token in normalized["coords"])

    if isinstance(normalized.get("t_values"), (list, tuple)):
        normalized["t_values"] = ",".join(str(token) for token in normalized["t_values"])

    if isinstance(normalized.get("encoders"), tuple):
        normalized["encoders"] = list(normalized["encoders"])

    if isinstance(normalized.get("save_dir"), str):
        normalized["save_dir"] = Path(normalized["save_dir"])

    return normalized


def merged_defaults(config_payload: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(DEFAULTS)

    common = config_payload.get("common")
    if common is None:
        common = {}
    if common and not isinstance(common, dict):
        raise ValueError("Config key 'common' must be a JSON object when provided.")

    for key, value in config_payload.items():
        if key == "encoders" and isinstance(value, dict):
            continue
        if key in COMMON_CONFIG_KEYS and key not in common:
            common[key] = value

    for key in COMMON_CONFIG_KEYS:
        if key in common:
            defaults[key] = common[key]

    encoder_cfg = config_payload.get("encoders")
    if encoder_cfg is not None and not isinstance(encoder_cfg, (dict, list)):
        raise ValueError("Config key 'encoders' must be either a list of names or an object.")

    if isinstance(encoder_cfg, list):
        defaults["encoders"] = encoder_cfg

    if isinstance(encoder_cfg, dict):
        if "names" in encoder_cfg:
            names = encoder_cfg["names"]
            if not isinstance(names, list):
                raise ValueError("Config key 'encoders.names' must be a list.")
            defaults["encoders"] = names

        spiral_cfg = encoder_cfg.get("spiral")
        monster_cfg = encoder_cfg.get("monster")

        if spiral_cfg is not None:
            if not isinstance(spiral_cfg, dict):
                raise ValueError("Config key 'encoders.spiral' must be a JSON object.")
            if "num_directions" in spiral_cfg:
                defaults["num_directions"] = spiral_cfg["num_directions"]

        if monster_cfg is not None:
            if not isinstance(monster_cfg, dict):
                raise ValueError("Config key 'encoders.monster' must be a JSON object.")
            if "top_delta" in monster_cfg:
                defaults["top_delta"] = monster_cfg["top_delta"]

    return _normalize_defaults(defaults)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified positional encoder pipeline: generate vectors, precompute vectorized caches, "
            "and apply Axial/Spiral/MonSTER blockwise."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config file. CLI values override config defaults. "
            "Supports top-level/common keys and encoder-specific overrides."
        ),
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=defaults["encoders"],
        choices=[*ENCODER_NAMES, "all"],
        help="Which encoders to run. Use 'all' to run every encoder.",
    )
    parser.add_argument("--dim", type=int, default=defaults["dim"], help="Embedding dimension.")
    parser.add_argument(
        "--num-vectors",
        type=int,
        default=defaults["num_vectors"],
        help="Number of random vectors to generate.",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="RNG seed.")
    parser.add_argument(
        "--theta-base",
        type=float,
        default=defaults["theta_base"],
        help="Base frequency theta.",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default=defaults["coords"],
        help="Coordinate tuple used by Axial/Spiral, e.g. 'x', 'x,y', 't,x,y', or 't,x,y,z'.",
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=defaults["num_directions"],
        help=(
            "Spiral direction groups. Defaults to len(coords). "
            "For the requested factorization behavior this must equal len(coords)."
        ),
    )
    parser.add_argument(
        "--top-delta",
        type=float,
        default=defaults["top_delta"],
        help="MonSTER top_delta denominator.",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=defaults["span"],
        help="Angular span used by MonSTER-family encoders; unit = span / top_delta.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=defaults["grid_size"],
        help="Grid side length for spatial axes.",
    )
    parser.add_argument(
        "--centered-coords",
        action=argparse.BooleanOptionalAction,
        default=defaults["centered_coords"],
        help="Use centered coordinates instead of integer indices for spatial axes.",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default=defaults["t_values"],
        help="Comma-separated t coordinates, e.g. '-8,-4,0,4,8'.",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=defaults["z_value"],
        help="Default fixed z coordinate when z is not included in --coords.",
    )
    parser.add_argument(
        "--position-chunk-size",
        type=int,
        default=defaults["position_chunk_size"],
        help="Positions per block during apply stage. <=0 means all at once.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=defaults["save_dir"],
        help="Optional output directory for vectors, metadata, and optional encoded tensors.",
    )
    parser.add_argument(
        "--save-encoded",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_encoded"],
        help="If set, save encoded tensors for each encoder to .npy files.",
    )
    return parser


def resolve_encoder_names(raw_encoders: list[str]) -> list[str]:
    if "all" in raw_encoders:
        return list(ENCODER_NAMES)
    return [name for name in ENCODER_NAMES if name in raw_encoders]


def parse_args() -> tuple[ExperimentConfig, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_payload = load_config_payload(pre_args.config)
    defaults = merged_defaults(config_payload)
    parser = build_parser(defaults)
    args = parser.parse_args()

    if args.num_vectors <= 0:
        parser.error("--num-vectors must be positive.")
    if args.dim <= 0:
        parser.error("--dim must be positive.")
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive.")
    if args.top_delta <= 0:
        parser.error("--top-delta must be positive.")
    if args.span <= 0:
        parser.error("--span must be positive.")

    try:
        coord_spec = parse_coords(args.coords)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        t_values = parse_t_values(args.t_values)
    except ValueError as exc:
        parser.error(str(exc))

    num_directions = args.num_directions
    if num_directions is None:
        num_directions = coord_spec.rope_dims
    if num_directions <= 0:
        parser.error("--num-directions must be positive.")
    if num_directions != coord_spec.rope_dims:
        parser.error(
            "For this factorization mode, --num-directions must equal len(--coords). "
            f"Received num_directions={num_directions}, len(coords)={coord_spec.rope_dims}."
        )

    encoders = resolve_encoder_names(args.encoders)
    cfg = ExperimentConfig(
        encoders=tuple(encoders),
        dim=args.dim,
        num_vectors=args.num_vectors,
        seed=args.seed,
        theta_base=args.theta_base,
        coords_spec=coord_spec,
        num_directions=num_directions,
        top_delta=args.top_delta,
        span=args.span,
        grid_size=args.grid_size,
        centered_coords=bool(args.centered_coords),
        t_values=t_values,
        z_value=args.z_value,
        position_chunk_size=args.position_chunk_size,
        save_dir=args.save_dir,
        save_encoded=bool(args.save_encoded),
        raw_coords=args.coords,
        config_path=args.config,
        config_payload=config_payload or None,
    )
    return cfg, config_payload

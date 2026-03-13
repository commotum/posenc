from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.positions import parse_coords, parse_t_values
from core.types import RunConfig
from encoders import encoder_names


DEFAULTS: dict[str, Any] = {
    "encoders": ["all"],
    "dim": 768,
    "num_vectors": 1,
    "seed": 0,
    "theta_base": 10_000.0,
    "coords": "x,y",
    "grid_size": 16,
    "centered_coords": False,
    "t_values": "0",
    "z_value": 0.0,
    "position_chunk_size": 128,
    "save_dir": None,
    "save_encoded": False,
    "encoder_params": {},
}

CONFIG_KEYS = set(DEFAULTS.keys())


def load_config_payload(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a JSON object.")
    return payload


def _normalize_encoder_params(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config key 'encoder_params' must be a JSON object.")

    normalized: dict[str, dict[str, Any]] = {}
    for encoder_name, params in raw.items():
        if not isinstance(params, dict):
            raise ValueError(f"Config key 'encoder_params.{encoder_name}' must be a JSON object.")
        normalized[str(encoder_name)] = {str(k): v for k, v in params.items()}
    return normalized


def _normalize_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(defaults)

    if isinstance(normalized.get("coords"), (list, tuple)):
        normalized["coords"] = ",".join(str(token) for token in normalized["coords"])

    if isinstance(normalized.get("t_values"), (list, tuple)):
        normalized["t_values"] = ",".join(str(token) for token in normalized["t_values"])

    encoders_value = normalized.get("encoders")
    if isinstance(encoders_value, tuple):
        normalized["encoders"] = list(encoders_value)
    elif not isinstance(encoders_value, list):
        raise ValueError("Config key 'encoders' must be a list.")

    if isinstance(normalized.get("save_dir"), str):
        normalized["save_dir"] = Path(normalized["save_dir"])

    normalized["encoder_params"] = _normalize_encoder_params(normalized.get("encoder_params", {}))
    return normalized


def merged_defaults(config_payload: dict[str, Any]) -> dict[str, Any]:
    unknown_keys = sorted(set(config_payload.keys()) - CONFIG_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config keys: {joined}.")

    defaults = dict(DEFAULTS)
    for key in CONFIG_KEYS:
        if key in config_payload:
            defaults[key] = config_payload[key]
    return _normalize_defaults(defaults)


def build_parser(defaults: dict[str, Any], available_encoders: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Positional encoder experiment harness: generate vectors, precompute encoder caches, "
            "apply encoders blockwise, and verify invariants."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config file using one top-level schema. "
            "CLI values override common keys; encoder-specific values come from encoder_params."
        ),
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=defaults["encoders"],
        choices=[*available_encoders, "all"],
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
        help="Coordinate tuple used by encoders, e.g. 'x', 'x,y', 't,x,y', or 't,x,y,z'.",
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


def resolve_encoder_names(raw_encoders: list[str], available_encoders: tuple[str, ...]) -> list[str]:
    if "all" in raw_encoders:
        return list(available_encoders)
    return [name for name in available_encoders if name in raw_encoders]


def parse_args() -> RunConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    try:
        config_payload = load_config_payload(pre_args.config)
        defaults = merged_defaults(config_payload)
    except ValueError as exc:
        pre_parser.error(str(exc))

    available = encoder_names()
    parser = build_parser(defaults, available)
    args = parser.parse_args()

    if args.num_vectors <= 0:
        parser.error("--num-vectors must be positive.")
    if args.dim <= 0:
        parser.error("--dim must be positive.")
    if args.grid_size <= 0:
        parser.error("--grid-size must be positive.")

    try:
        coord_spec = parse_coords(args.coords)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        t_values = parse_t_values(args.t_values)
    except ValueError as exc:
        parser.error(str(exc))

    encoders = resolve_encoder_names(args.encoders, available)
    if not encoders:
        parser.error("At least one encoder must be selected.")

    return RunConfig(
        encoders=tuple(encoders),
        dim=args.dim,
        num_vectors=args.num_vectors,
        seed=args.seed,
        theta_base=args.theta_base,
        coords_spec=coord_spec,
        grid_size=args.grid_size,
        centered_coords=bool(args.centered_coords),
        t_values=t_values,
        z_value=args.z_value,
        position_chunk_size=args.position_chunk_size,
        save_dir=args.save_dir,
        save_encoded=bool(args.save_encoded),
        encoder_params=defaults["encoder_params"],
    )

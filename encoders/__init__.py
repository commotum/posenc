"""Encoder implementations and explicit registry."""

from __future__ import annotations

from core.types import EncoderSpec
from encoders.ape import SPEC as APE_SPEC
from encoders.axial import SPEC as AXIAL_SPEC
from encoders.f_monster import SPEC as F_MONSTER_SPEC
from encoders.monster import SPEC as MONSTER_SPEC
from encoders.rope import SPEC as ROPE_SPEC
from encoders.spiral import SPEC as SPIRAL_SPEC

_REGISTRY: dict[str, EncoderSpec] = {
    ROPE_SPEC.name: ROPE_SPEC,
    AXIAL_SPEC.name: AXIAL_SPEC,
    SPIRAL_SPEC.name: SPIRAL_SPEC,
    MONSTER_SPEC.name: MONSTER_SPEC,
    F_MONSTER_SPEC.name: F_MONSTER_SPEC,
    APE_SPEC.name: APE_SPEC,
}


def encoder_names() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def all_specs() -> dict[str, EncoderSpec]:
    return dict(_REGISTRY)


def get_spec(name: str) -> EncoderSpec:
    return _REGISTRY[name]


def resolve_specs(names: list[str]) -> list[EncoderSpec]:
    return [get_spec(name) for name in names]

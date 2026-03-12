# PosEnc

Compact research harness for comparing positional encoding mechanisms on high-dimensional vectors.

## What this repo does

Given random normalized vectors, this project:
1. Builds coordinate grids over `t, x, y, z`.
2. Precomputes encoder-specific caches (vectorized NumPy).
3. Applies selected positional encoders blockwise.
4. Verifies encoder-specific invariants.
5. Emits a JSON summary (+ optional `.npy` artifacts).

This is experiment infrastructure, not model training/inference code.

## Implemented encoders

- `axial`: Multi-axis rotary encoding (RoPE-like), independent rotations per coordinate axis/frequency pair.
- `spiral`: Rotary encoding on projected coordinate directions (mixed/orthogonal direction vectors).
- `monster`: Triad MonSTER transform over 12D blocks (boost + axis-plane rotations), with Minkowski-form check.
- `f-monster`: Fibonacci-axis MonSTER over 4D blocks (boost + rotation around isotropic per-block axes), with Minkowski-form check.
- `ape`: Fixed sinusoidal Absolute Positional Encoding (Transformer-style `sin`/`cos`) added to vectors.

## Current architecture

- Thin CLI entrypoint: `main.py`
- Argument/config parsing: `core/cli.py`
- Experiment orchestration: `experiment.py`
- Shared utilities:
  - `core/types.py` (config + contracts)
  - `core/positions.py` (coordinate parsing/grid generation)
  - `core/frequencies.py` (frequency helpers)
  - `core/vectors.py` (random normalized vectors + vector checks)
  - `core/io.py` (artifact saving)
- Encoders:
  - `encoders/common.py` (encoder interface)
  - `encoders/registry.py` (explicit registry)
  - `encoders/{axial,spiral,monster,f_monster,ape}.py`
  - `encoders/template.py` (new-encoder template)
- Analysis space: `analysis/`
- Regression baselines: `baselines/`
- Tests: `tests/`

## Encoder contract (for new variants)

Each encoder module follows the same exports:
- `NAME`
- `Cache` dataclass
- `validate_config(cfg) -> RequirementCheck`
- `precompute(cfg, bank) -> Cache`
- `apply(vectors, cache, chunk_size) -> encoded`
- `check_invariants(vectors, encoded) -> metrics`
- `SPEC` (`EncoderSpec`) for registry wiring

## Quick start

Run all current encoders:

```bash
uv run python main.py --encoders all --dim 120 --num-vectors 2 --coords t,x,y --t-values=-1,0,1 --grid-size 3
```

Run APE only:

```bash
uv run python main.py --encoders ape --dim 64 --num-vectors 2 --coords x --grid-size 5
```

Save outputs:

```bash
uv run python main.py --encoders axial spiral monster ape --save-dir out --save-encoded
```

MonSTER-family with span scaling (`unit = span / top_delta`):

```bash
uv run python main.py --encoders monster f-monster --dim 120 --span 6.283185307179586 --top-delta 1024
```

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

## JSON config support

`--config <path.json>` sets defaults; CLI flags still override.

Supported pattern (example):

```json
{
  "common": {
    "dim": 120,
    "coords": "t,x,y",
    "t_values": [-1, 0, 1]
  },
  "encoders": {
    "names": ["axial", "monster", "ape"],
    "monster": { "top_delta": 2048 },
    "spiral": { "num_directions": 3 }
  }
}
```

## Verification and outputs

Summary includes:
- compatibility checks
- effective config
- vector stats
- position counts
- per-encoder verification metrics
- stage timings

Optionally saved files:
- `vectors.npy`
- `metadata.json`
- `encoded_<encoder>.npy` (when `--save-encoded`)

## Legacy context files

`Big-Picture.md` and `Positional-Encoding-V1.py` are still present as historical references/baseline sources.  
This README is the condensed successor intended for future sessions.

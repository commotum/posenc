# PosEnc

Compact research harness for comparing positional encoders on high-dimensional vectors.

## Implemented encoders

- `rope`: standard 1D sequence RoPE over embedding pairs.
- `axial`: multi-axis rotary encoding with independent axis rotations.
- `spiral`: rotary encoding over projected coordinate directions.
- `monster`: triad MonSTER transform over 12D blocks.
- `f-monster`: Fibonacci-axis MonSTER transform over 4D blocks.
- `ape`: fixed sinusoidal absolute positional encoding.

## Project layout

- `main.py`: CLI entrypoint.
- `experiment.py`: experiment orchestration and artifact saving.
- `core/`
  - `cli.py`: argument + config parsing.
  - `positions.py`: coordinate parsing and position-bank construction.
  - `types.py`: shared runtime/config/spec dataclasses.
  - `math.py`: shared frequency/chunk/norm helpers.
  - `vectors.py`: random vector generation and checks.
- `encoders/`: one module per encoder + package registry.
- `notebooks/compare.ipynb`: output comparison notebook.
- `tests/`: parse + smoke coverage.

## Runtime config model

The runtime config is a single schema (`RunConfig`) with a common surface and optional per-encoder overrides:

```python
@dataclass(frozen=True)
class RunConfig:
    encoders: tuple[str, ...]
    dim: int
    num_vectors: int
    seed: int
    theta_base: float
    coords_spec: CoordinateSpec
    grid_size: int
    centered_coords: bool
    t_values: np.ndarray
    z_value: float
    position_chunk_size: int
    save_dir: Path | None = None
    save_encoded: bool = False
    encoder_params: dict[str, dict[str, Any]] = field(default_factory=dict)
```

### JSON config schema

One top-level shape only:

```json
{
  "dim": 120,
  "coords": ["t", "x", "y"],
  "t_values": [-1, 0, 1],
  "encoders": ["axial", "spiral", "monster", "f-monster"],
  "encoder_params": {
    "spiral": {"num_directions": 3},
    "monster": {"top_delta": 1024, "span": 6.283185307179586},
    "f-monster": {"top_delta": 1024, "span": 6.283185307179586}
  }
}
```

## Quick start

Run all encoders from CLI defaults:

```bash
uv run python main.py --encoders all --dim 120 --num-vectors 2 --coords t,x,y --t-values=-1,0,1 --grid-size 3
```

Run with JSON config:

```bash
uv run python main.py --config config.json
```

Save outputs:

```bash
uv run python main.py --encoders rope axial spiral monster f-monster ape --save-dir out --save-encoded
```

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

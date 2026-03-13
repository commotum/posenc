# PosEnc Cleanup To-Do

## Goal
Reduce refactor residue while preserving one-file-per-encoder structure and current behavior.

## Phase 1: Museum Pass
- [x] Delete `baselines/`
- [x] Delete `Positional-Encoding-V1.py`
- [x] Delete `Big-Picture.md`
- [x] Delete parity test (`tests/test_parity_with_v1.py`)
- [x] Keep this `To-Do.md` as the only temporary planning artifact

## Phase 2: Config Simplification
- [x] Replace `ExperimentConfig` with a single runtime `RunConfig`
- [x] Add `encoder_params: dict[str, dict[str, Any]]` to runtime config
- [x] Simplify JSON config loading to one schema (no `common`, no `encoders.names`, no top-level aliases)
- [x] Keep CLI focused on common fields; encoder-specific values come from `encoder_params`
- [x] Update encoder modules to read per-encoder params via shared config accessors

## Phase 3: Small-Module Collapse
- [x] Move `EncoderSpec` into `core/types.py`
- [x] Make `encoders/__init__.py` the canonical registry (`all_specs`, `get_spec`, `resolve_specs`)
- [x] Remove `encoders/common.py` and `encoders/registry.py`
- [x] Add `core/math.py` with shared helpers:
  - [x] `base_frequencies`
  - [x] `spiral_frequency_sets`
  - [x] `chunk_slices`
  - [x] norm-error helpers
- [x] Remove duplicated `_chunk_slices` from encoders
- [x] Inline or eliminate thin IO abstraction if only used once

## Phase 4: Naming + Structure Cleanup
- [x] Rename `PositionBank.monster_positions` to `positions_4d`
- [x] Stop storing encoder name constants in `core/types.py`; derive names from encoder registry
- [x] Convert `analysis/` into `notebooks/`
- [x] Keep one useful notebook: `notebooks/compare.ipynb`
- [x] Remove package-style `analysis/__init__.py`

## Phase 5: Tests + Docs
- [x] Keep only parse/validation test and end-to-end smoke test
- [x] Delete outdated config/parity tests
- [x] Update README for new config schema and project layout
- [x] Update `pyproject.toml` description
- [x] Move notebook-only deps to optional extras

## Final Validation
- [x] `uv run python -m unittest discover -s tests -v` passes
- [x] `uv run python main.py --encoders all --dim 120 --num-vectors 2 --coords t,x,y --t-values=-1,0,1 --grid-size 3` runs successfully
- [x] Repo tree reflects simplified architecture

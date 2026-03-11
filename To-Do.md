# Positional Encoding Refactor To-Do

## 1) Lock in a baseline before refactor
- [x] Run the current script with representative settings and save baseline outputs (`metadata.json`, optional encoded tensors).
- [x] Record baseline invariants and timing values for `axial`, `spiral`, and `monster`.
- [x] Keep 2-3 canonical CLI invocations to use for regression checks during the refactor.

## 2) Create package structure for a plugin-style harness
- [x] Add folders: `core/`, `encoders/`, `analysis/`.
- [x] Keep `main.py` as a thin CLI entrypoint.
- [x] Add `experiment.py` as the orchestrator (`run_experiment(config)`).
- [x] Add `encoders/registry.py` for explicit encoder registration (no dynamic import magic).

## 3) Move shared utilities out of the monolith
- [x] Move random vector generation and vector checks into `core/vectors.py`.
- [x] Move coordinate parsing and position-grid builders into `core/positions.py`.
- [x] Move shared frequency helpers (e.g., base frequencies) into `core/frequencies.py`.
- [x] Move save helpers (`vectors.npy`, `metadata.json`, `encoded_*.npy`) into `core/io.py`.

## 4) Define shared data contracts
- [x] Add an `ExperimentConfig` dataclass (common args + encoder options).
- [x] Add a shared position bank type/object so encoders receive one position payload.
- [x] Add a requirement-check structure for encoder compatibility reporting.
- [x] Keep naming split clear: `validate_config` (compatibility) vs `check_invariants` (post-apply correctness).

## 5) Standardize encoder interface
- [x] Define a single encoder contract each module must expose:
  - [x] `NAME`
  - [x] `Cache` dataclass
  - [x] `validate_config(cfg)`
  - [x] `precompute(cfg, bank)`
  - [x] `apply(vectors, cache, chunk_size)`
  - [x] `check_invariants(vectors, encoded)`
- [x] Add a code template file for new encoders (code first, markdown checklist second).

## 6) Split current encoders into modules
- [x] Create `encoders/axial.py` and move axial cache/apply/invariant logic.
- [x] Create `encoders/spiral.py` and move spiral cache/apply/invariant logic.
- [x] Create `encoders/monster.py` and move monster cache/apply/invariant logic.
- [x] Preserve current math exactly during extraction to avoid silent behavior changes.

## 7) Rebuild orchestration flow
- [x] In `experiment.py`, resolve selected encoders from registry.
- [x] Run per-encoder `validate_config` and fail early with clear messages.
- [x] Build shared positions once, then call each encoder `precompute`.
- [x] Apply encoders blockwise and collect outputs in a uniform result object.
- [x] Run `check_invariants` per encoder and aggregate verification summary.
- [x] Preserve current summary schema (`encoders`, `config`, `vector_stats`, `positions`, `verification`, `timing_seconds`).

## 8) Keep CLI thin and scalable
- [x] Keep core/common CLI options in `main.py` (`--dim`, `--coords`, `--encoders`, etc.).
- [x] Add optional experiment config file support (`--config path`) for encoder-specific overrides.
- [x] Keep code defaults as source of truth; config files override per run.
- [x] Avoid turning `main.py` into an encoder-specific flag graveyard.

## 9) Add analysis hooks without coupling to encoding math
- [ ] Put plotting/analysis helpers (heatmaps, comparisons, diagnostics) in `analysis/`.
- [x] Ensure analysis consumes experiment outputs instead of touching encoder internals.
- [x] Make outputs easy to load from notebooks/scripts for follow-up experiments.

## 10) Add regression tests and guardrails
- [x] Add tests for parser/config validation and coordinate compatibility rules.
- [x] Add tests that compare refactored outputs to the baseline for fixed seeds/settings.
- [x] Add invariant tests:
  - [x] Euclidean norm preservation for `axial` and `spiral`.
  - [x] Minkowski-form preservation for `monster`.
- [x] Add shape and dtype tests for caches and encoded outputs.
- [x] Add one end-to-end smoke test for multi-encoder runs.

## 11) Future-proofing rule for encoder growth
- [x] Keep one file per encoder by default.
- [ ] Promote an encoder to its own folder only when complexity warrants it (multiple apply modes, large helpers, custom diagnostics, large constants/presets).
- [ ] Keep registry-facing API unchanged when promoting file -> folder.

# Baseline Artifacts (from `Positional-Encoding-V1.py`)

These baseline outputs were generated on 2026-03-10 before the modular refactor and are used for regression checks.

## Canonical CLI Invocations

1. `case1_xy_all`
- `uv run python Positional-Encoding-V1.py --encoders all --dim 120 --num-vectors 2 --seed 0 --coords x,y --grid-size 3 --position-chunk-size 5 --save-dir baselines/case1_xy_all --save-encoded`

2. `case2_txy_all`
- `uv run python Positional-Encoding-V1.py --encoders all --dim 120 --num-vectors 2 --seed 0 --coords t,x,y --t-values=-1,0,1 --grid-size 3 --position-chunk-size 5 --save-dir baselines/case2_txy_all --save-encoded`

3. `case3_txyz_monster`
- `uv run python Positional-Encoding-V1.py --encoders monster --dim 120 --num-vectors 2 --seed 0 --coords t,x,y,z --t-values=-1,0,1 --grid-size 2 --position-chunk-size 0 --save-dir baselines/case3_txyz_monster --save-encoded`

## Snapshot Metrics

- `case1_xy_all`: `total=0.009917s`, axial/spiral Euclidean max error `0.0`, monster Minkowski max error `3.552713678800501e-15`
- `case2_txy_all`: `total=0.008696s`, axial Euclidean max error `0.0`, spiral Euclidean max error `1.7763568394002505e-15`, monster Minkowski max error `3.552713678800501e-15`
- `case3_txyz_monster`: `total=0.008205s`, monster Minkowski max error `2.6645352591003757e-15`

Each case directory contains:
- `metadata.json`
- `vectors.npy`
- `encoded_*.npy`
- `run.log`

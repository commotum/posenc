---
title: "Positional Encoding Comparison"
source: "https://chatgpt.com/c/69b0b582-cf30-8324-9aa2-74ba4b4161d2"
author:
  - "[[ChatGPT]]"
published:
created: 2026-03-10
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Tell me what you think this program is about: from \_\_future\_\_ import annotations import argparse import json from dataclasses import dataclass from pathlib import Path from time import perf\_counter import numpy as np MONSTER\_SLICE = 12 ENCODER\_NAMES = ("axial", "spiral", "monster") ALLOWED\_COORDS = ("t", "x", "y", "z") @dataclass(frozen=True) class CoordinateSpec: coords: tuple\[str,...\] include\_time: bool spatial\_axes: tuple\[str,...\] @property def rope\_dims(self) -> int: return len(self.coords) @dataclass(frozen=True) class AxialCache: positions: np.ndarray # (P, C) freqs: np.ndarray # (F,) cos\_axes: np.ndarray # (P, C, F) sin\_axes: np.ndarray # (P, C, F) @dataclass(frozen=True) class SpiralCache: positions: np.ndarray # (P, C) direction\_vectors: np.ndarray # (G, C) frequency\_sets: np.ndarray # (G, F) projected: np.ndarray # (P, G) cos\_phase: np.ndarray # (P, G, F) sin\_phase: np.ndarray # (P, G, F) @dataclass(frozen=True) class MonsterCache: positions: np.ndarray # (P, 4) with columns (t, x, y, z) inv\_freq: np.ndarray # (F,) ch: np.ndarray # (P, F) sh: np.ndarray # (P, F) c\_axes: np.ndarray # (P, F, 3) s\_axes: np.ndarray # (P, F, 3) def parse\_t\_values(raw: str) -> np.ndarray: values = \[token.strip() for token in raw.split(",") if token.strip()\] if not values: raise ValueError("Expected at least one value for --t-values.") return np.asarray(\[float(v) for v in values\], dtype=np.float64) def parse\_coords(raw: str) -> CoordinateSpec: tokens = tuple(token.strip() for token in raw.split(",") if token.strip()) if not tokens: raise ValueError("Expected at least one coordinate in --coords.") seen: set\[str\] = set() for token in tokens: if token not in ALLOWED\_COORDS: allowed = ", ".join(ALLOWED\_COORDS) raise ValueError(f"Invalid coordinate '{token}'. Allowed coordinates: {allowed}.") if token in seen: raise ValueError(f"Duplicate coordinate '{token}' in --coords.") seen.add(token) spatial\_axes = tuple(axis for axis in tokens if axis!= "t") return CoordinateSpec( coords=tokens, include\_time=("t" in seen), spatial\_axes=spatial\_axes, ) def parse\_args() -> argparse.Namespace: parser = argparse.ArgumentParser( description=( "Unified positional encoder pipeline: generate vectors, precompute vectorized caches, " "and apply Axial/Spiral/MonSTER blockwise." ) ) parser.add\_argument( "--encoders", nargs="+", default=\["all"\], choices=\[\*ENCODER\_NAMES, "all"\], help="Which encoders to run. Use 'all' to run every encoder.", ) parser.add\_argument("--dim", type=int, default=768, help="Embedding dimension.") parser.add\_argument("--num-vectors", type=int, default=1, help="Number of random vectors to generate.") parser.add\_argument("--seed", type=int, default=0, help="RNG seed.") parser.add\_argument("--theta-base", type=float, default=10\_000.0, help="Base frequency theta.") parser.add\_argument( "--coords", type=str, default="x,y", help="Coordinate tuple used by Axial/Spiral, e.g. 'x', 'x,y', 't,x,y', or 't,x,y,z'.", ) parser.add\_argument( "--num-directions", type=int, default=None, help=( "Spiral direction groups. Defaults to len(coords). " "For the requested factorization behavior this must equal len(coords)." ), ) parser.add\_argument("--top-delta", type=float, default=1024.0, help="MonSTER top\_delta denominator.") parser.add\_argument("--grid-size", type=int, default=16, help="Grid side length for spatial axes.") parser.add\_argument( "--centered-coords", action="store\_true", help="Use centered coordinates instead of integer indices for spatial axes.", ) parser.add\_argument( "--t-values", type=str, default="0", help="Comma-separated t coordinates, e.g. '-8,-4,0,4,8'.", ) parser.add\_argument( "--z-value", type=float, default=0.0, help="Default fixed z coordinate when z is not included in --coords.", ) parser.add\_argument( "--position-chunk-size", type=int, default=128, help="Positions per block during apply stage. <=0 means all at once.", ) parser.add\_argument( "--save-dir", type=Path, default=None, help="Optional output directory for vectors, metadata, and optional encoded tensors.", ) parser.add\_argument( "--save-encoded", action="store\_true", help="If set, save encoded tensors for each encoder to.npy files.", ) args = parser.parse\_args() if args.num\_vectors <= 0: parser.error("--num-vectors must be positive.") if args.dim <= 0: parser.error("--dim must be positive.") if args.grid\_size <= 0: parser.error("--grid-size must be positive.") if args.top\_delta <= 0: parser.error("--top-delta must be positive.") try: args.coords\_spec = parse\_coords(args.coords) except ValueError as exc: parser.error(str(exc)) try: args.t\_values = parse\_t\_values(args.t\_values) except ValueError as exc: parser.error(str(exc)) if args.num\_directions is None: args.num\_directions = args.coords\_spec.rope\_dims if args.num\_directions <= 0: parser.error("--num-directions must be positive.") if args.num\_directions!= args.coords\_spec.rope\_dims: parser.error( "For this factorization mode, --num-directions must equal len(--coords). " f"Received num\_directions={args.num\_directions}, len(coords)={args.coords\_spec.rope\_dims}." ) return args def resolve\_encoders(raw\_encoders: list\[str\]) -> list\[str\]: if "all" in raw\_encoders: return list(ENCODER\_NAMES) return \[name for name in ENCODER\_NAMES if name in raw\_encoders\] def requirement\_report( dim: int, coord\_spec: CoordinateSpec, num\_directions: int, ) -> dict\[str, tuple\[bool, str\]\]: report: dict\[str, tuple\[bool, str\]\] = {} axial\_ok = dim % (2 \* coord\_spec.rope\_dims) == 0 report\["axial"\] = ( axial\_ok, f"dim % (2 \* len(coords)) == 0; len(coords)={coord\_spec.rope\_dims}", ) spiral\_ok = ( num\_directions == coord\_spec.rope\_dims and dim % (2 \* num\_directions) == 0 ) report\["spiral"\] = ( spiral\_ok, ( "num\_directions == len(coords) and dim % (2 \* num\_directions) == 0; " f"num\_directions={num\_directions}" ), ) monster\_ok = dim % MONSTER\_SLICE == 0 report\["monster"\] = ( monster\_ok, "dim % 12 == 0", ) return report def enforce\_requested\_requirements( requested\_encoders: list\[str\], report: dict\[str, tuple\[bool, str\]\], ) -> None: failing = \[name for name in requested\_encoders if not report\[name\]\[0\]\] if failing: details = "; ".join(f"{name}: {report\[name\]\[1\]}" for name in failing) raise ValueError(f"Requested encoders are incompatible with current settings -> {details}") def random\_vectors(num\_vectors: int, dim: int, seed: int) -> np.ndarray: rng = np.random.default\_rng(seed) vectors = rng.normal(0.0, 1.0, size=(num\_vectors, dim)).astype(np.float64) norms = np.linalg.norm(vectors, axis=1, keepdims=True) zero\_norm = norms.squeeze(-1) == 0.0 if np.any(zero\_norm): vectors\[zero\_norm\] = 1.0 norms = np.linalg.norm(vectors, axis=1, keepdims=True) vectors = vectors / norms \* np.sqrt(dim) return vectors def verify\_vectors(vectors: np.ndarray, dim: int) -> dict\[str, float\]: if vectors.ndim!= 2: raise ValueError(f"Expected 2D tensor for vectors, got shape {vectors.shape}.") if vectors.shape\[1\]!= dim: raise ValueError(f"Vector dim mismatch: expected {dim}, got {vectors.shape\[1\]}.") if not np.all(np.isfinite(vectors)): raise ValueError("Vectors contain non-finite values.") target = np.sqrt(dim) norms = np.linalg.norm(vectors, axis=1) return { "target\_norm": float(target), "mean\_norm": float(np.mean(norms)), "max\_abs\_norm\_error": float(np.max(np.abs(norms - target))), } def make\_spatial\_axis\_values(grid\_size: int, centered: bool) -> np.ndarray: if centered: return np.arange(grid\_size, dtype=np.float64) - ((grid\_size - 1) / 2.0) return np.arange(grid\_size, dtype=np.float64) def build\_positions\_for\_coords( coord\_spec: CoordinateSpec, spatial\_values: np.ndarray, t\_values: np.ndarray, ) -> np.ndarray: axis\_values: list\[np.ndarray\] = \[\] for axis in coord\_spec.coords: if axis == "t": axis\_values.append(t\_values) else: axis\_values.append(spatial\_values) mesh = np.meshgrid(\*axis\_values, indexing="ij") return np.stack(mesh, axis=-1).reshape(-1, coord\_spec.rope\_dims) def build\_monster\_positions( coord\_spec: CoordinateSpec, spatial\_values: np.ndarray, t\_values: np.ndarray, z\_value: float, ) -> np.ndarray: t\_axis = t\_values if coord\_spec.include\_time else np.asarray(\[0.0\], dtype=np.float64) x\_axis = spatial\_values if "x" in coord\_spec.spatial\_axes else np.asarray(\[0.0\], dtype=np.float64) y\_axis = spatial\_values if "y" in coord\_spec.spatial\_axes else np.asarray(\[0.0\], dtype=np.float64) z\_axis = spatial\_values if "z" in coord\_spec.spatial\_axes else np.asarray(\[z\_value\], dtype=np.float64) mesh = np.meshgrid(t\_axis, x\_axis, y\_axis, z\_axis, indexing="ij") return np.stack(mesh, axis=-1).reshape(-1, 4) def base\_frequencies(num\_freqs: int, theta\_base: float) -> np.ndarray: return theta\_base \*\* (-np.arange(num\_freqs, dtype=np.float64) / num\_freqs) def spiral\_frequency\_sets(embed\_dim: int, num\_groups: int, theta\_base: float) -> np.ndarray: if embed\_dim % (2 \* num\_groups)!= 0: raise ValueError("Spiral factorization requires dim divisible by 2 \* num\_groups.") group\_pairs = embed\_dim // (2 \* num\_groups) freqs = base\_frequencies(num\_groups \* group\_pairs, theta\_base) return freqs.reshape(num\_groups, group\_pairs) def \_orthonormalize\_rows(matrix: np.ndarray) -> np.ndarray: rows: list\[np.ndarray\] = \[\] for idx in range(matrix.shape\[0\]): vec = matrix\[idx\].astype(np.float64).copy() for row in rows: vec -= np.dot(vec, row) \* row norm = np.linalg.norm(vec) if norm < 1e-12: raise ValueError("Failed to build stable spiral direction vectors.") rows.append(vec / norm) return np.stack(rows, axis=0) def spiral\_direction\_vectors(coord\_dims: int) -> np.ndarray: if coord\_dims == 1: return np.asarray(\[\[1.0\]\], dtype=np.float64) if coord\_dims == 2: return \_orthonormalize\_rows(np.asarray(\[\[1.0, 1.0\], \[-1.0, 1.0\]\], dtype=np.float64)) if coord\_dims == 3: return \_orthonormalize\_rows( np.asarray( \[ \[1.0, 1.0, 1.0\], \[1.0, -1.0, 0.0\], \[1.0, 1.0, -2.0\], \], dtype=np.float64, ) ) if coord\_dims == 4: return \_orthonormalize\_rows( np.asarray( \[ \[1.0, 1.0, 1.0, 1.0\], \[1.0, -1.0, 1.0, -1.0\], \[1.0, 1.0, -1.0, -1.0\], \[1.0, -1.0, -1.0, 1.0\], \], dtype=np.float64, ) ) raise ValueError("This script supports up to 4 coordinates in --coords.") def build\_axial\_cache( dim: int, theta\_base: float, rope\_positions: np.ndarray, ) -> AxialCache: coord\_dims = rope\_positions.shape\[1\] if dim % (2 \* coord\_dims)!= 0: raise ValueError("Axial factorization requires dim divisible by 2 \* len(coords).") pair\_count = dim // (2 \* coord\_dims) freqs = base\_frequencies(pair\_count, theta\_base) phase = rope\_positions\[:,:, None\] \* freqs\[None, None,:\] return AxialCache( positions=rope\_positions, freqs=freqs, cos\_axes=np.cos(phase), sin\_axes=np.sin(phase), ) def build\_spiral\_cache( dim: int, num\_directions: int, theta\_base: float, rope\_positions: np.ndarray, ) -> SpiralCache: coord\_dims = rope\_positions.shape\[1\] if num\_directions!= coord\_dims: raise ValueError("For this mode, Spiral directions must match coordinate dimensionality.") frequency\_sets = spiral\_frequency\_sets(dim, num\_directions, theta\_base) direction\_vectors = spiral\_direction\_vectors(coord\_dims) projected = rope\_positions @ direction\_vectors.T phase = projected\[:,:, None\] \* frequency\_sets\[None,:,:\] return SpiralCache( positions=rope\_positions, direction\_vectors=direction\_vectors, frequency\_sets=frequency\_sets, projected=projected, cos\_phase=np.cos(phase), sin\_phase=np.sin(phase), ) def build\_monster\_cache( dim: int, theta\_base: float, top\_delta: float, positions\_4d: np.ndarray, ) -> MonsterCache: if dim % MONSTER\_SLICE!= 0: raise ValueError("MonSTER requires dim divisible by 12.") num\_freq = dim // MONSTER\_SLICE inv\_freq = base\_frequencies(num\_freq, theta\_base) unit = 1.0 / float(top\_delta) phi = positions\_4d\[:, 0:1\] \* unit \* inv\_freq\[None,:\] thx = positions\_4d\[:, 1:2\] \* unit \* inv\_freq\[None,:\] thy = positions\_4d\[:, 2:3\] \* unit \* inv\_freq\[None,:\] thz = positions\_4d\[:, 3:4\] \* unit \* inv\_freq\[None,:\] return MonsterCache( positions=positions\_4d, inv\_freq=inv\_freq, ch=np.cosh(phi), sh=np.sinh(phi), c\_axes=np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=2), s\_axes=np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=2), ) def chunk\_slices(total: int, chunk\_size: int) -> list\[tuple\[int, int\]\]: if chunk\_size <= 0 or chunk\_size >= total: return \[(0, total)\] slices: list\[tuple\[int, int\]\] = \[\] start = 0 while start < total: end = min(total, start + chunk\_size) slices.append((start, end)) start = end return slices def apply\_axial\_blockwise(vectors: np.ndarray, cache: AxialCache, chunk\_size: int) -> np.ndarray: num\_vectors, dim = vectors.shape num\_positions, coord\_dims = cache.positions.shape pair\_count = cache.freqs.size groups = vectors.reshape(num\_vectors, coord\_dims, pair\_count, 2) g0 = groups\[:,:,:, 0\] g1 = groups\[:,:,:, 1\] out = np.empty((num\_vectors, num\_positions, dim), dtype=np.float64) for start, end in chunk\_slices(num\_positions, chunk\_size): cos\_axes = cache.cos\_axes\[start:end\]\[None,:,:,:\] sin\_axes = cache.sin\_axes\[start:end\]\[None,:,:,:\] r0 = g0\[:, None,:,:\] \* cos\_axes - g1\[:, None,:,:\] \* sin\_axes r1 = g0\[:, None,:,:\] \* sin\_axes + g1\[:, None,:,:\] \* cos\_axes out\[:, start:end,:\] = np.stack((r0, r1), axis=-1).reshape(num\_vectors, end - start, dim) return out def apply\_spiral\_blockwise(vectors: np.ndarray, cache: SpiralCache, chunk\_size: int) -> np.ndarray: num\_vectors, dim = vectors.shape num\_positions = cache.positions.shape\[0\] num\_groups, group\_pairs = cache.frequency\_sets.shape groups = vectors.reshape(num\_vectors, num\_groups, group\_pairs, 2) g0 = groups\[:,:,:, 0\] g1 = groups\[:,:,:, 1\] out = np.empty((num\_vectors, num\_positions, dim), dtype=np.float64) for start, end in chunk\_slices(num\_positions, chunk\_size): cos\_phase = cache.cos\_phase\[start:end\]\[None,:,:,:\] sin\_phase = cache.sin\_phase\[start:end\]\[None,:,:,:\] r0 = g0\[:, None,:,:\] \* cos\_phase - g1\[:, None,:,:\] \* sin\_phase r1 = g0\[:, None,:,:\] \* sin\_phase + g1\[:, None,:,:\] \* cos\_phase out\[:, start:end,:\] = np.stack((r0, r1), axis=-1).reshape(num\_vectors, end - start, dim) return out def apply\_monster\_blockwise(vectors: np.ndarray, cache: MonsterCache, chunk\_size: int) -> np.ndarray: num\_vectors, dim = vectors.shape num\_positions = cache.positions.shape\[0\] num\_freq = cache.inv\_freq.size base = vectors.reshape(num\_vectors, num\_freq, 3, 4) out = np.empty((num\_vectors, num\_positions, dim), dtype=np.float64) for start, end in chunk\_slices(num\_positions, chunk\_size): pos\_count = end - start state = np.broadcast\_to(base\[None,:,:,:,:\], (pos\_count, num\_vectors, num\_freq, 3, 4)).copy() ch = cache.ch\[start:end\]\[:, None,:, None\] sh = cache.sh\[start:end\]\[:, None,:, None\] c\_axes = cache.c\_axes\[start:end\]\[:, None,:,:\] s\_axes = cache.s\_axes\[start:end\]\[:, None,:,:\] time\_components = state\[:,:,:,:, 0\] aligned\_spatial = np.empty\_like(time\_components) aligned\_spatial\[:,:,:, 0\] = state\[:,:,:, 0, 1\] aligned\_spatial\[:,:,:, 1\] = state\[:,:,:, 1, 2\] aligned\_spatial\[:,:,:, 2\] = state\[:,:,:, 2, 3\] boosted\_time = ch \* time\_components - sh \* aligned\_spatial boosted\_space = -sh \* time\_components + ch \* aligned\_spatial state\[:,:,:,:, 0\] = boosted\_time state\[:,:,:, 0, 1\] = boosted\_space\[:,:,:, 0\] state\[:,:,:, 1, 2\] = boosted\_space\[:,:,:, 1\] state\[:,:,:, 2, 3\] = boosted\_space\[:,:,:, 2\] x\_u = state\[:,:,:, 0, 2\].copy() x\_v = state\[:,:,:, 0, 3\].copy() state\[:,:,:, 0, 2\] = c\_axes\[:,:,:, 0\] \* x\_u - s\_axes\[:,:,:, 0\] \* x\_v state\[:,:,:, 0, 3\] = s\_axes\[:,:,:, 0\] \* x\_u + c\_axes\[:,:,:, 0\] \* x\_v y\_u = state\[:,:,:, 1, 1\].copy() y\_v = state\[:,:,:, 1, 3\].copy() state\[:,:,:, 1, 1\] = c\_axes\[:,:,:, 1\] \* y\_u - s\_axes\[:,:,:, 1\] \* y\_v state\[:,:,:, 1, 3\] = s\_axes\[:,:,:, 1\] \* y\_u + c\_axes\[:,:,:, 1\] \* y\_v z\_u = state\[:,:,:, 2, 1\].copy() z\_v = state\[:,:,:, 2, 2\].copy() state\[:,:,:, 2, 1\] = c\_axes\[:,:,:, 2\] \* z\_u - s\_axes\[:,:,:, 2\] \* z\_v state\[:,:,:, 2, 2\] = s\_axes\[:,:,:, 2\] \* z\_u + c\_axes\[:,:,:, 2\] \* z\_v out\[:, start:end,:\] = state.reshape(pos\_count, num\_vectors, dim).transpose(1, 0, 2) return out def euclidean\_norm\_error(vectors: np.ndarray, encoded: np.ndarray) -> float: original\_norms = np.linalg.norm(vectors, axis=1) encoded\_norms = np.linalg.norm(encoded, axis=2) return float(np.max(np.abs(encoded\_norms - original\_norms\[:, None\]))) def monster\_minkowski\_norm\_error(vectors: np.ndarray, encoded: np.ndarray) -> float: eta = np.array(\[-1.0, 1.0, 1.0, 1.0\], dtype=np.float64) original = vectors.reshape(vectors.shape\[0\], -1, 4) encoded\_blocks = encoded.reshape(encoded.shape\[0\], encoded.shape\[1\], -1, 4) base\_norms = np.sum(original \* eta\[None, None,:\] \* original, axis=2) transformed\_norms = np.sum(encoded\_blocks \* eta\[None, None, None,:\] \* encoded\_blocks, axis=3) return float(np.max(np.abs(transformed\_norms - base\_norms\[:, None,:\]))) def maybe\_save( save\_dir: Path | None, vectors: np.ndarray, metadata: dict\[str, object\], encoded: dict\[str, np.ndarray\], save\_encoded: bool, ) -> None: if save\_dir is None: return save\_dir.mkdir(parents=True, exist\_ok=True) np.save(save\_dir / "vectors.npy", vectors) (save\_dir / "metadata.json").write\_text(json.dumps(metadata, indent=2)) if save\_encoded: for name, tensor in encoded.items(): np.save(save\_dir / f"encoded\_{name}.npy", tensor) def main() -> None: args = parse\_args() encoders = resolve\_encoders(args.encoders) coord\_spec: CoordinateSpec = args.coords\_spec checks = requirement\_report(args.dim, coord\_spec, args.num\_directions) print("Encoder compatibility checks:") for name in ENCODER\_NAMES: ok, rule = checks\[name\] print(f" - {name:<7} {'OK' if ok else 'FAIL'} | {rule}") enforce\_requested\_requirements(encoders, checks) t0 = perf\_counter() vectors = random\_vectors(args.num\_vectors, args.dim, args.seed) vector\_stats = verify\_vectors(vectors, args.dim) t1 = perf\_counter() spatial\_values = make\_spatial\_axis\_values(args.grid\_size, args.centered\_coords) rope\_positions = build\_positions\_for\_coords(coord\_spec, spatial\_values, args.t\_values) monster\_positions = build\_monster\_positions(coord\_spec, spatial\_values, args.t\_values, args.z\_value) caches: dict\[str, AxialCache | SpiralCache | MonsterCache\] = {} if "axial" in encoders: caches\["axial"\] = build\_axial\_cache(args.dim, args.theta\_base, rope\_positions) if "spiral" in encoders: caches\["spiral"\] = build\_spiral\_cache( args.dim, args.num\_directions, args.theta\_base, rope\_positions, ) if "monster" in encoders: caches\["monster"\] = build\_monster\_cache( args.dim, args.theta\_base, args.top\_delta, monster\_positions, ) t2 = perf\_counter() encoded: dict\[str, np.ndarray\] = {} verification: dict\[str, dict\[str, float | int\]\] = {} if "axial" in encoders: axial\_out = apply\_axial\_blockwise(vectors, caches\["axial"\], args.position\_chunk\_size) # type: ignore\[arg-type\] encoded\["axial"\] = axial\_out verification\["axial"\] = { "num\_positions": int(axial\_out.shape\[1\]), "max\_abs\_euclidean\_norm\_error": euclidean\_norm\_error(vectors, axial\_out), } if "spiral" in encoders: spiral\_out = apply\_spiral\_blockwise(vectors, caches\["spiral"\], args.position\_chunk\_size) # type: ignore\[arg-type\] encoded\["spiral"\] = spiral\_out verification\["spiral"\] = { "num\_positions": int(spiral\_out.shape\[1\]), "max\_abs\_euclidean\_norm\_error": euclidean\_norm\_error(vectors, spiral\_out), } if "monster" in encoders: monster\_out = apply\_monster\_blockwise(vectors, caches\["monster"\], args.position\_chunk\_size) # type: ignore\[arg-type\] encoded\["monster"\] = monster\_out verification\["monster"\] = { "num\_positions": int(monster\_out.shape\[1\]), "max\_abs\_minkowski\_norm\_error": monster\_minkowski\_norm\_error(vectors, monster\_out), } t3 = perf\_counter() summary: dict\[str, object\] = { "encoders": encoders, "config": { "dim": args.dim, "num\_vectors": args.num\_vectors, "seed": args.seed, "theta\_base": args.theta\_base, "coords": list(coord\_spec.coords), "rope\_coordinate\_dims": coord\_spec.rope\_dims, "spatial\_dimensions": len(coord\_spec.spatial\_axes), "include\_time": bool(coord\_spec.include\_time), "num\_directions": args.num\_directions, "top\_delta": args.top\_delta, "grid\_size": args.grid\_size, "centered\_coords": bool(args.centered\_coords), "t\_values": args.t\_values.tolist(), "z\_value": args.z\_value, "position\_chunk\_size": args.position\_chunk\_size, }, "vector\_stats": vector\_stats, "positions": { "rope\_positions": int(rope\_positions.shape\[0\]), "monster\_positions": int(monster\_positions.shape\[0\]), }, "verification": verification, "timing\_seconds": { "generate\_vectors": round(t1 - t0, 6), "build\_caches": round(t2 - t1, 6), "apply\_encoders": round(t3 - t2, 6), "total": round(t3 - t0, 6), }, } maybe\_save(args.save\_dir, vectors, summary, encoded, args.save\_encoded) print(json.dumps(summary, indent=2)) if \_\_name\_\_ == "\_\_main\_\_": main()

---

It looks like a benchmarking / experimentation script for **multidimensional positional encodings** on embedding vectors.

More specifically, it seems built to compare three ways of applying position-dependent transforms to vectors:

- **axial**
- **spiral**
- **monster**

### What it does

The program:

1. **Generates random embedding vectors**
	- shape: `(num_vectors, dim)`
	- normalized so each vector has norm about `sqrt(dim)`
2. **Builds position grids**
	- positions can use coordinates from `t, x, y, z`
	- examples:
		- just spatial: `x,y`
		- spacetime: `t,x,y,z`
	- it creates all coordinate combinations over a grid
3. **Precomputes caches**
	- cosine/sine tables for axial and spiral
	- hyperbolic + trigonometric tables for monster
4. **Applies each encoder blockwise**
	- transforms every input vector at every position
	- output shape is roughly `(num_vectors, num_positions, dim)`
5. **Checks invariants**
	- axial and spiral preserve **Euclidean norm**
	- monster preserves a **Minkowski-style norm** instead of ordinary Euclidean norm
6. **Prints a JSON summary**
	- settings
	- compatibility checks
	- number of positions
	- norm-preservation errors
	- timings

---

## What the three encoders seem to mean

### 1\. Axial

This is the most recognizable one.

It splits the embedding dimension across coordinate axes and applies independent **2D rotations** per axis/frequency pair using:

- `cos(position * frequency)`
- `sin(position * frequency)`

That is basically a generalized **multi-axis RoPE-style positional encoding**.

Interpretation:

- each coordinate axis (`x`, `y`, `t`, etc.) gets its own slice of the embedding
- position rotates those slices
- norm stays the same because rotation preserves Euclidean length

---

### 2\. Spiral

This one is similar in spirit, but instead of using raw axes directly, it:

- builds a set of **direction vectors**
- projects coordinates onto those directions
- applies rotations using those projected values

So instead of encoding along pure axes like `(x, y, z)`, it encodes along mixed directions like diagonals / orthogonal combinations.

Interpretation:

- a more coupled positional encoding
- probably intended to capture geometry better than strictly axis-aligned encoding
- still norm-preserving in ordinary Euclidean space

---

### 3\. MonSTER

This is the unusual one.

It treats positions as **4D `(t, x, y, z)`** and applies a combination of:

- **hyperbolic transforms** in the time-related part (`cosh`, `sinh`)
- ordinary **rotations** in spatial subcomponents (`cos`, `sin`)

That strongly suggests it is trying to mimic something like a **Lorentzian / spacetime-inspired transform**, where time is treated differently from space.

The giveaway is here:

- `monster_minkowski_norm_error`
- metric `[-1, 1, 1, 1]`

That is the standard signature of a **Minkowski metric** from relativity.

Interpretation:

- this encoder is not just a positional encoding in the usual transformer sense
- it seems like a **spacetime-geometry-inspired embedding transform**
- it preserves a relativistic-style quadratic form rather than Euclidean norm

---

## What the author is probably studying

My best guess is that this script is for **researching or prototyping positional encodings for high-dimensional embeddings over spatial or spacetime coordinates**.

Possible goals:

- compare standard axis-based rotary encoding vs more coupled directional encoding
- experiment with encoding positions in 2D/3D/4D
- test whether a spacetime-inspired encoding behaves nicely
- verify preservation properties numerically
- benchmark memory/time tradeoffs using cached blockwise application

So this feels like a **numerical experiment harness** for geometric positional encoding methods, possibly for:

- transformers over images or video
- 3D data
- spatiotemporal models
- physics-informed ML
- novel RoPE variants

---

## Why the structure looks research-oriented

A few clues:

- strict divisibility requirements on `dim`
- explicit cache dataclasses
- support for `t,x,y,z`
- chunked application over positions
- norm-invariance verification
- optional saving of tensors and metadata
- JSON output for experiment logging

This looks less like production inference code and more like:

- a prototype,
- benchmark,
- ablation script,
- or validation harness.

---

## In one sentence

This program is a **testbed for generating random embeddings and applying three different geometry-aware positional encoders—two Euclidean rotational ones and one spacetime/Minkowski-inspired one—over coordinate grids, while checking that each transform preserves its intended norm.**

A couple of names translated into plain English:

- **axial** = rotate embedding slices independently by coordinate axis
- **spiral** = rotate embedding slices along mixed coordinate directions
- **monster** = apply a relativity-flavored spacetime transform to embedding blocks

I can also walk through the `monster` transform line by line, since that is the least obvious part.

---

You are exactly right, it is a research harness. However, right now it's only a single program, a single python script. If I were to add more positional encoders it might get messy? So how could I still keep this tight but while adding a per-positional-encoding-variant (file or folder)? I mean would you simply create 1. get\_vec.py - a script for generating a random normalized vector given the dim 2. main.py - CLI takes args for --encoders (list the encoders you wish to compare), --n-dim (int for the number of dims in your high dimensional embedding vector), --coords (the spatial and temporal dimensions we'll be aligning them to) 3. and a base template in markdown outlining the following as a prompt: Base Template Each Positional Encoder File Must: 1. accept args of dim, and the expected coords 2. Implement a verify function that ensures the given coords, and dim are compatible with the encoder 3. Implement a precompute function that efficiently builds up the cache tables in a vectorized numpy fashion 4. returns the tables And then we go back into main, it gets all the cache tables, computes them all and then we can do what we want from there? like plot attention heat maps or do further analysis, whatever.

2/2

---

A clean split is possible without making it heavyweight: keep `main.py` thin, use a small plugin-style package for the math, and make each encoder follow an executable Python contract. I’d avoid a top-level `get_vec.py` unless you truly run it on its own often, and I’d start encoders as simple modules, only promoting the complicated ones into folders later.

A registry plus a shared interface is what keeps `main` small as the encoder count grows. I’d also let each encoder register its own CLI options so `main.py` doesn’t turn into an argument graveyard.

Yes. I would do almost that, but with one important change: make the contract a **Python interface**, not a markdown prompt. The markdown can still exist as human docs, but the real template should be code.

The shape I would use is:

## What I would change from your version

`get_vec.py` should probably **not** be a script. Make it `core/vectors.py` and expose:

- `random_vectors(num_vectors, dim, seed)`
- `verify_vectors(vectors, dim)`

That keeps it reusable from both CLI and notebooks.

`main.py` should stay very thin. It should:

1. parse CLI
2. build one config object
3. call `run_experiment(config)`
4. optionally call analysis/plotting functions

That way you can import the same harness from a notebook without dragging `argparse` through everything.

Also, keep `--dim`, not `--n-dim`. `--dim` is already clear.

---

## The contract each encoder should expose

I would make every encoder module export the same small surface:

- `NAME`
- `Cache` dataclass
- `validate_config(cfg) -> RequirementCheck`
- `precompute(cfg, bank) -> Cache`
- `apply(vectors, cache, chunk_size) -> encoded`
- `check_invariants(vectors, encoded) -> dict`

That is enough to keep `main` ignorant of encoder internals.

Two naming tweaks matter:

- Use `validate_config`, not `verify`, for “is this dim/coord combo legal?”
- Use `check_invariants` for numerical checks after apply

That avoids mixing “configuration validity” with “math behaved as expected.”

---

## A good minimal base interface

This is intentionally simple. No inheritance hierarchy, no plugin framework, no metaclass machinery.

For research code, simple is better.

---

## Shared types worth having

You will benefit from two shared dataclasses:

The big idea is: **encoders should receive one config object and one shared position bank**, not a growing list of raw arguments.

That keeps signatures stable as the harness grows.

---

## What a new encoder file should look like

This is the real template. Much better than a markdown prompt.

Then `axial.py`, `spiral.py`, and `monster.py` just implement that shape.

Your current `AxialCache`, `SpiralCache`, and `MonsterCache` move directly into those files.

---

## Registry

Keep encoder discovery explicit:

Explicit registry is better than dynamic import magic for a research harness. Easier to debug, easier to read.

---

## What experiment.py should do

`main.py` parses args. `experiment.py` does the actual work.

Something like:

That gives you one uniform object to hand to:

- attention heatmap code
- benchmark code
- save/load code
- comparison plots
- notebook analysis

---

## File or folder per encoder?

Start with **one file per encoder**:

- `encoders/axial.py`
- `encoders/spiral.py`
- `encoders/monster.py`

Only promote an encoder to a folder when it becomes noticeably bigger than the others. For example:

That way `registry.py` does not change. It still imports `SPEC`.

So the rule is:

- simple encoder: one file
- complicated encoder: one package, same outward interface

---

## One more practical improvement

Right now you already have encoder-specific parameters:

- spiral: `num_directions`
- monster: `top_delta`

As that grows, the CLI will become cluttered. The clean way to handle that is:

- keep a few common CLI args in `main.py`
- allow an optional JSON or YAML config for per-encoder options

Example shape:

That keeps the harness scalable without turning `argparse` into a wall of flags.

---

## Bottom line

Yes, your instinct is right. The clean version is:

- shared utilities in `core/`
- one encoder module per positional encoding variant
- a small explicit registry
- thin CLI
- `run_experiment()` as the actual orchestrator
- analysis code in a separate `analysis/` package

And I would make the encoder template a real Python module, not just markdown. The markdown can be the checklist, but the code template is what will keep the project tight when you add encoder four, five, and six.

---

---

Yes, I’d still keep each encoder as a single `.py` file; YAML alone wouldn’t justify turning it into a folder.

Yes, I would still keep each encoder implementation as a **single Python file**.

A YAML file does not usually justify turning an encoder into a folder. The thing that should trigger a folder is **implementation complexity**, not configuration.

A clean layout would be:

## What I would avoid

I would avoid making the YAML the primary definition of the encoder.

For example, I would **not** do this as the main source of truth:

unless that encoder has grown large enough to deserve its own package.

The main risk is drift:

- code says one thing
- YAML defaults say another
- CLI help says a third thing

That gets annoying fast in research code.

## Better pattern

Keep the encoder’s **supported parameters and defaults in code**, near the implementation.

Example:

Then let YAML provide **overrides** for experiments.

Example experiment YAML:

That is usually cleaner than “one YAML per encoder.”

## When a per-encoder YAML does make sense

A separate YAML per encoder can be useful when you want **named presets**.

For example:

That is useful for:

- reproducible experiments
- sharing presets
- running sweeps
- comparing standardized setups

In that case, I would still keep the encoder implementation in one file, and keep YAMLs in a separate `configs/` area, not beside the code.

## Rule of thumb

Keep it as a single file per encoder until one of these happens:

- the encoder needs multiple apply modes
- it has substantial helper math
- it has custom plotting / diagnostics tied only to that encoder
- it has lots of preset tables or learned constants
- the file starts feeling crowded, roughly 250–400+ lines with mixed responsibilities

Then promote it to a folder:

## My actual recommendation

For your case, I would do:

- **one Python file per encoder**
- **one experiment YAML per run/comparison**
- per-encoder params nested inside that experiment YAML
- defaults and validation live in code

So, not “one YAML per encoder” first.

More like:

- code defines what an encoder is
- YAML defines which experiment you are running

That keeps the harness tight and makes adding encoder 4 or 5 straightforward.

A good next step would be for me to sketch the exact `RunConfig` + `EncoderSpec` shape and one concrete `axial.py` example.
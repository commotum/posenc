# Vectorized NumPy version of the "Fast-scalar MonSTER Triad" (no Python loops over frequencies)
from __future__ import annotations
import numpy as np

# =============================================================================
# 1) Minkowski helpers
# =============================================================================
ETA4 = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)

def minkowski_dot_big_vec(u: np.ndarray, v: np.ndarray) -> float:
    """
    Vectorized big Minkowski inner product for (dim,) row vectors,
    where the metric is block-diagonal with copies of ETA4 on each 4-D chunk.
    """
    U = u.reshape(-1, 4)
    V = v.reshape(-1, 4)
    return np.sum((U @ ETA4) * V)


# =============================================================================
# 2) Vectorized scalar-table builder
# =============================================================================
class TriadMonSTERFastVec:
    """
    Vectorized cache of scalar tables for fast absolute/relative transforms.
    Frequencies: lam_j = base^{-j / F}, j=0..F-1, where F = dim // 12.

    Angles/Rapidities:
        phi_j   = (t * unit) * lam_j
        thx_j   = (x * unit) * lam_j
        thy_j   = (y * unit) * lam_j
        thz_j   = (z * unit) * lam_j

    Forward returns a dict with shapes:
        ch, sh:     (F,)          # cosh/sinh(phi)
        c_axes, s_axes: (F,3)     # cos/sin for X,Y,Z axes respectively
    """
    def __init__(
        self,
        dim: int = 768,
        base: float = 10000.0,
        top_delta: int = 1024,
        span: float = 2.0 * np.pi,
    ):
        if dim % 12 != 0:
            raise ValueError(f"dim must be divisible by 12, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // 12
        self.base     = float(base)
        self.span     = float(span)
        self.unit     = self.span / float(top_delta)  # global unit (argument units per step)
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)
        self._cache = {}

    def forward(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t, x, y, z = s
        lam = self.inv_freq  # (F,)
        u = self.unit

        phi  = (t * u) * lam
        thx  = (x * u) * lam
        thy  = (y * u) * lam
        thz  = (z * u) * lam

        ch = np.cosh(phi)              # (F,)
        sh = np.sinh(phi)              # (F,)
        c_axes = np.stack((np.cos(thx), np.cos(thy), np.cos(thz)), axis=1)  # (F,3) -> X,Y,Z
        s_axes = np.stack((np.sin(thx), np.sin(thy), np.sin(thz)), axis=1)  # (F,3)

        out = {"ch": ch, "sh": sh, "c": c_axes, "s": s_axes}
        self._cache[key] = out
        return out


# =============================================================================
# 3) Vectorized apply (no loops over frequencies)
# =============================================================================
def apply_monster_triad_fast_vec(emb: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Apply triad transforms to a full embedding using only vectorized broadcasting.
    Args:
        emb   : (dim,) row vector.
        tables: dict with "ch","sh","c","s" from TriadMonSTERFastVec.forward.
        dim   : total embedding dimension (multiple of 12).
    Returns:
        (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    F = dim // 12

    # Reshape into (F, 3, 4): freq buckets × {X,Y,Z} × [t,x,y,z]
    V = emb.reshape(F, 3, 4).astype(np.float64, copy=False)
    out = V.copy()

    # Broadcasted scalars
    ch = tables["ch"]          # (F,)
    sh = tables["sh"]          # (F,)
    c_axes = tables["c"]       # (F,3)
    s_axes = tables["s"]       # (F,3)

    # --------------------
    # Step 1: Boost along each axis' spatial component
    # --------------------
    # Indices for the spatial component aligned with axis: X->1, Y->2, Z->3
    comp_idx = np.array([1, 2, 3], dtype=np.int64)[None, :, None]  # (1,3,1)
    t = out[:, :, 0]                        # (F,3)
    x_axis = np.take_along_axis(out, comp_idx, axis=2)[..., 0]  # (F,3)

    t1 = ch[:, None] * t - sh[:, None] * x_axis
    x1 = -sh[:, None] * t + ch[:, None] * x_axis

    out[:, :, 0] = t1
    np.put_along_axis(out, comp_idx, x1[..., None], axis=2)

    # --------------------
    # Step 2: Rotate in the orthogonal spatial 2D planes
    # --------------------
    # For axis X: rotate (y,z) -> indices (2,3)
    # For axis Y: rotate (x,z) -> indices (1,3)
    # For axis Z: rotate (x,y) -> indices (1,2)
    pair_idx = np.array([[2, 3], [1, 3], [1, 2]], dtype=np.int64)[None, :, :]  # (1,3,2)

    pair_vals = np.take_along_axis(out, pair_idx, axis=2)  # (F,3,2)
    u = pair_vals[..., 0]  # first in the pair
    v = pair_vals[..., 1]  # second in the pair

    cu = c_axes  # (F,3)
    su = s_axes  # (F,3)

    u2 = cu * u - su * v
    v2 = su * u + cu * v

    rotated = np.stack((u2, v2), axis=-1)  # (F,3,2)
    np.put_along_axis(out, pair_idx, rotated, axis=2)

    return out.reshape(dim,)


# =============================================================================
# 4) Demo / Sanity checks
# =============================================================================
if __name__ == "__main__":
    np.random.seed(0)
    DIM = 768
    monster = TriadMonSTERFastVec(dim=DIM, base=10000.0, top_delta=1024)

    # Absolute 4D positions (in "steps")
    s_q  = np.array([ 700.0,  500.0, -300.0,  200.0], dtype=np.float64)  # (t,x,y,z)
    s_k  = np.array([ -40.0,  -20.0,   60.0,  -10.0], dtype=np.float64)
    dskq = s_k - s_q

    # Tables
    T_abs_q = monster.forward(s_q)
    T_abs_k = monster.forward(s_k)
    T_rel   = monster.forward(dskq)

    # Random embeddings
    q = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)
    k = np.random.uniform(-0.6, 0.6, size=DIM).astype(np.float64)

    # Apply absolute maps
    q_abs = apply_monster_triad_fast_vec(q, T_abs_q, dim=DIM)
    k_abs = apply_monster_triad_fast_vec(k, T_abs_k, dim=DIM)

    # RoPE-style identity check
    lhs = minkowski_dot_big_vec(q_abs, k_abs)
    k_rel = apply_monster_triad_fast_vec(k, T_rel, dim=DIM)
    rhs = minkowski_dot_big_vec(q, k_rel)

    print("RoPE-style identity holds? ", np.allclose(lhs, rhs, rtol=1e-12, atol=1e-12))
    print(f"lhs: {lhs:+.12f}  rhs: {rhs:+.12f}")

    # Per-4D Minkowski norm preservation
    Q_blocks = q.reshape(-1, 4)
    Q_abs_blocks = q_abs.reshape(-1, 4)
    norms_before = np.sum((Q_blocks @ ETA4) * Q_blocks, axis=1)
    norms_after  = np.sum((Q_abs_blocks @ ETA4) * Q_abs_blocks, axis=1)
    ok = np.allclose(norms_before, norms_after, rtol=1e-11, atol=1e-12)
    max_err = np.max(np.abs(norms_before - norms_after))
    print("Per-4D Minkowski norms preserved? ", ok, "| max abs err:", max_err)

    print("NUM_FREQ:", DIM // 12, " | DIM:", DIM, " | BLOCK dim:", 12)

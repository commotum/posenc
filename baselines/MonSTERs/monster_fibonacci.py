
# Vectorized NumPy version of fast MonSTER with
# isotropic Fibonacci-sphere axes (no Python loops over frequencies)
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


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = i * phi

    out = np.empty((n, 3), dtype=np.float64)
    out[:, 0] = np.cos(theta) * r
    out[:, 1] = np.sin(theta) * r
    out[:, 2] = z
    return out


# =============================================================================
# 2) Vectorized scalar-table builder
# =============================================================================
class FibonacciMonSTERFastVec:
    """
    Vectorized cache of scalar tables for fast absolute/relative transforms.
    Frequencies: lam_j = base^{-j / F}, j=0..F-1, where F = dim // 4.

    Each 4-D block gets:
        1) its own frequency lam_j
        2) its own unit spatial axis a_j sampled from a Fibonacci sphere

    For block j:
        phi_j   = (t * unit) * lam_j
        theta_j = ((a_j · [x,y,z]) * unit) * lam_j

    Forward returns a dict with shapes:
        ch, sh, c, s:  (F,)   # cosh/sinh(phi), cos/sin(theta)
        axis:          (F,3)  # one unit axis per 4-D block
    """
    def __init__(
        self,
        dim: int = 768,
        base: float = 10000.0,
        top_delta: int = 1024,
        span: float = 2.0 * np.pi,
    ):
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}.")
        self.dim      = dim
        self.num_freq = dim // 4
        self.base     = float(base)
        self.span     = float(span)
        self.unit     = self.span / float(top_delta)  # global unit (argument units per step)
        j = np.arange(self.num_freq, dtype=np.float64)
        self.inv_freq = self.base ** (- j / self.num_freq)

        # One isotropic axis per 4-D block.
        axes = fibonacci_sphere(self.num_freq)
        # Re-normalize defensively in case of accumulated FP error.
        self.axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

        self._cache = {}

    def forward(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError("position must be a 4D vector (t, x, y, z).")
        key = (s[0], s[1], s[2], s[3], self.unit, self.base)
        if key in self._cache:
            return self._cache[key]

        t = s[0]
        spatial = s[1:]
        lam = self.inv_freq  # (F,)
        u = self.unit

        phi = (t * u) * lam

        # Project the spatial position onto each block's Fibonacci axis.
        proj = self.axes @ spatial  # (F,)
        theta = (u * lam) * proj    # (F,)

        ch = np.cosh(phi)   # (F,)
        sh = np.sinh(phi)   # (F,)
        c = np.cos(theta)   # (F,)
        s = np.sin(theta)   # (F,)

        out = {
            "ch": ch,
            "sh": sh,
            "c": c,
            "s": s,
            "axis": self.axes,
        }
        self._cache[key] = out
        return out


# =============================================================================
# 3) Vectorized apply (no loops over frequencies)
# =============================================================================
def apply_monster_fibonacci_fast_vec(emb: np.ndarray, tables: dict, dim: int = 768) -> np.ndarray:
    """
    Apply isotropic Fibonacci-axis transforms to a full embedding
    using only vectorized broadcasting.

    Args:
        emb   : (dim,) row vector.
        tables: dict with "ch","sh","c","s","axis" from FibonacciMonSTERFastVec.forward.
        dim   : total embedding dimension (multiple of 4).

    Returns:
        (dim,) transformed row vector.
    """
    if emb.shape != (dim,):
        raise ValueError(f"embedding must be shape ({dim},), got {emb.shape}")
    F = dim // 4

    # Reshape into (F, 4): freq blocks × [t,x,y,z]
    V = emb.reshape(F, 4).astype(np.float64, copy=False)
    out = V.copy()

    # Broadcasted scalars / axes
    ch = tables["ch"]    # (F,)
    sh = tables["sh"]    # (F,)
    c = tables["c"]      # (F,)
    s = tables["s"]      # (F,)
    axis = tables["axis"]  # (F,3)

    t = out[:, 0]        # (F,)
    spatial = out[:, 1:] # (F,3)

    # --------------------
    # Step 1: Boost along each block's own Fibonacci axis
    # --------------------
    proj = np.sum(spatial * axis, axis=-1)  # (F,) = a · x

    t1 = ch * t - sh * proj
    spatial1 = spatial + (
        (((ch - 1.0) * proj - sh * t)[:, None] * axis)
    )

    # --------------------
    # Step 2: Rotate the spatial part around the same axis
    # --------------------
    proj1 = np.sum(spatial1 * axis, axis=-1)  # (F,) = a · x'
    cross = np.cross(axis, spatial1)          # (F,3) = a × x'

    spatial2 = (
        c[:, None] * spatial1
        + s[:, None] * cross
        + (1.0 - c)[:, None] * proj1[:, None] * axis
    )

    out[:, 0] = t1
    out[:, 1:] = spatial2

    return out.reshape(dim,)


# =============================================================================
# 4) Demo / Sanity checks
# =============================================================================
if __name__ == "__main__":
    np.random.seed(0)
    DIM = 768
    monster = FibonacciMonSTERFastVec(dim=DIM, base=10000.0, top_delta=1024)

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
    q_abs = apply_monster_fibonacci_fast_vec(q, T_abs_q, dim=DIM)
    k_abs = apply_monster_fibonacci_fast_vec(k, T_abs_k, dim=DIM)

    # RoPE-style identity check
    lhs = minkowski_dot_big_vec(q_abs, k_abs)
    k_rel = apply_monster_fibonacci_fast_vec(k, T_rel, dim=DIM)
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

    print("NUM_FREQ:", DIM // 4, " | DIM:", DIM, " | BLOCK dim:", 4)
    print("NUM_AXES:", DIM // 4)

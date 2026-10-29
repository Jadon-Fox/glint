"""GLINT host encode/expand in numpy, f32-exact to cuda/glint_encode.cu.
Same op order as the kernels: block absmax in f32, v = w / absmax in f32
(absmax 0 -> v 0), nearest16 with first-minimum-wins, lo-then-hi nibbles.
So a host pin is byte-identical to a CUDA pin. This is the host-vs-CUDA
plug byte gate reference, and a fast --allow-cpu path. Optional: numpy.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from glint_codec import CELLS, PARENT
PARENT_F32 = np.asarray(PARENT, dtype=np.float32)
CELLS_F32 = np.asarray(CELLS, dtype=np.float32)
CELLS_FLAT = CELLS_F32.reshape(-1) # ascending across rows: each row sits in its parent's Voronoi cell
# Elements per chunk. Multiple of every sane blocksize and even.
CHUNK = 1 << 22
def dense_to_f32(dtype: str, raw: bytes, n: int) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2", count=n)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).astype(np.float32, copy=False)
    raise ValueError(f"dense unpack does not support dtype {dtype}")
def f32_to_bf16_bytes(w: np.ndarray) -> bytes:
    """Round to nearest even, NaN stays NaN (same as torch .to(bfloat16))."""
    u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32)
    rounded = (u + (np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1)))) >> 16
    nan = np.isnan(w)
    out = np.where(nan, (u >> 16) | np.uint32(0x40), rounded).astype("<u2")
    return out.tobytes()
def block_absmax(w: np.ndarray, blocksize: int) -> np.ndarray:
    n = w.size
    nb = (n + blocksize - 1) // blocksize
    a = np.abs(w)
    a = np.where(np.isnan(a), np.float32(0), a)  # fabsf(NaN) > m is false in k_absmax
    pad = nb * blocksize - n
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=np.float32)])
    return a.reshape(nb, blocksize).max(axis=1)
def _nearest(v: np.ndarray, flat: np.ndarray, lo: np.ndarray | int, width: int = 16) -> np.ndarray:
    """Index in [lo, lo+width) of the nearest flat[] entry, minus lo. Same answer as
    nearest16(): flat rows are strictly ascending, f32 subtraction is monotone, so
    the scan's first minimum is one of the two neighbours of the insertion point,
    and a tie goes to the lower index."""
    pos = np.searchsorted(flat, v, side="left").astype(np.int64)
    pos = np.clip(pos, lo + 1, lo + width - 1)
    left = flat[pos - 1]
    right = flat[pos]
    take_right = np.abs(v - right) < np.abs(v - left)
    idx = pos - 1 + take_right - lo
    return np.where(np.isfinite(v), idx, 0).astype(np.uint8) # NaN/inf: every d < bd is false
def pack_nibbles(idx: np.ndarray) -> np.ndarray:
    if idx.size & 1:
        idx = np.concatenate([idx, np.zeros(1, dtype=np.uint8)])
    return (idx[0::2] & 15) | ((idx[1::2] & 15) << 4)
def unpack_nibbles(packed: bytes | np.ndarray, n: int) -> np.ndarray:
    b = np.frombuffer(packed, dtype=np.uint8) if isinstance(packed, (bytes, bytearray)) else packed
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 15
    out[1::2] = b >> 4
    return out[:n]
def encode_f32(w: np.ndarray, blocksize: int = 64) -> Tuple[bytes, bytes, np.ndarray]:
    """w: flat f32. Returns packed hole bytes, packed plug bytes, f32 absmax."""
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    n = w.size
    absmax = block_absmax(w, blocksize)
    hole_pk = np.empty((n + 1) // 2, dtype=np.uint8)
    plug_pk = np.empty((n + 1) // 2, dtype=np.uint8)
    step = max(blocksize, (CHUNK // blocksize) * blocksize)
    if step & 1:
        step *= 2
    with np.errstate(divide="ignore", invalid="ignore"):
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            s = np.repeat(absmax[i0 // blocksize : (i1 + blocksize - 1) // blocksize], blocksize)
            s = s[: i1 - i0]
            v = np.where(s == 0, np.float32(0), w[i0:i1] / s).astype(np.float32, copy=False)
            h = _nearest(v, PARENT_F32, 0)
            p = _nearest(v, CELLS_FLAT, h.astype(np.int64) * 16)
            hole_pk[i0 // 2 : (i1 + 1) // 2] = pack_nibbles(h)
            plug_pk[i0 // 2 : (i1 + 1) // 2] = pack_nibbles(p)
    return hole_pk.tobytes(), plug_pk.tobytes(), absmax
def expand(
    hole_pk: bytes, plug_pk: bytes, absmax: np.ndarray, n: int, blocksize: int = 64, cells: np.ndarray | None = None
) -> np.ndarray:
    """f32 expand, same as k_expand: cells[h][p] * absmax[i / bs]."""
    tab = CELLS_F32 if cells is None else np.asarray(cells, dtype=np.float32).reshape(16, 16)
    h = unpack_nibbles(hole_pk, n)
    p = unpack_nibbles(plug_pk, n)
    s = np.repeat(np.asarray(absmax, dtype=np.float32), blocksize)[:n]
    return tab[h, p] * s
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d))) if d.size else 0.0

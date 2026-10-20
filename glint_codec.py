"""GLINT — Gaussian Location INside Tile.
Parent 16-cell tessellation (same geometry as QLoRA NF4 table — geometry only).
Plug = 4-bit equal-mass child inside that cell.
W = CELLS[parent][plug] * absmax[block]
Not NF8. Not uniform INT8. Not IEEE bitplanes.
"""
from __future__ import annotations
import math
from typing import List, Sequence, Tuple
# Parent tessellation (QLoRA NF4 measured table). Geometry, not the product name.
PARENT: Tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
NIBBLE_LO_THEN_HI = 0
def qweight_nbytes(n: int) -> int:
    return (n + 1) // 2
def pack_nibbles(first: int, second: int) -> int:
    return (first & 15) | ((second & 15) << 4)
def extract_nibble(byte: int, which: int) -> int:
    return (byte & 15) if which == 0 else ((byte >> 4) & 15)
def pack_indices(idx: Sequence[int]) -> bytearray:
    n = len(idx)
    out = bytearray(qweight_nbytes(n))
    for i in range(0, n, 2):
        b = int(idx[i]) & 15
        s = int(idx[i + 1]) & 15 if i + 1 < n else 0
        out[i // 2] = pack_nibbles(b, s)
    return out
def unpack_indices(packed: bytes | bytearray, n: int) -> List[int]:
    out = [0] * n
    for i in range(n):
        out[i] = extract_nibble(packed[i // 2], i & 1)
    return out
def pack_f32(vals: Sequence[float]) -> bytes:
    import struct
    return struct.pack("<" + "f" * len(vals), *[float(x) for x in vals])
def unpack_f32(buf: bytes) -> List[float]:
    import struct
    n = len(buf) // 4
    return list(struct.unpack("<" + "f" * n, buf[: n * 4]))
def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        d = float(a[i]) - float(b[i])
        s += d * d
    return math.sqrt(s / n)
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577459574091e02, -3.066479806614736e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )
def _voronoi() -> List[float]:
    b = [-1.0]
    for i in range(len(PARENT) - 1):
        b.append(0.5 * (PARENT[i] + PARENT[i + 1]))
    b.append(1.0)
    return b
def _subdivide(lo: float, hi: float, n: int = 16) -> List[float]:
    p0 = _norm_cdf(lo * 3.0)
    p1 = _norm_cdf(hi * 3.0)
    if p1 <= p0:
        p1 = p0 + 1e-12
    out = []
    for i in range(n):
        p = p0 + (p1 - p0) * (i + 0.5) / n
        z = _norm_ppf(p) / 3.0
        z = lo if z < lo else (hi if z > hi else z)
        out.append(z)
    return out
def build_cells() -> List[List[float]]:
    bounds = _voronoi()
    return [_subdivide(bounds[k], bounds[k + 1], 16) for k in range(16)]
CELLS: List[List[float]] = build_cells()
def cells_flat() -> List[float]:
    out: List[float] = []
    for row in CELLS:
        out.extend(row)
    return out
def nearest_parent(v: float) -> int:
    best, bd = 0, abs(v - PARENT[0])
    for k in range(1, 16):
        d = abs(v - PARENT[k])
        if d < bd:
            best, bd = k, d
    return best
def encode(weights: Sequence[float], blocksize: int = 64) -> Tuple[List[int], List[int], List[float]]:
    n = len(weights)
    nb = (n + blocksize - 1) // blocksize
    absmax = [0.0] * nb
    for i, w in enumerate(weights):
        a = abs(float(w))
        b = i // blocksize
        if a > absmax[b]:
            absmax[b] = a
    hole = [0] * n
    plug = [0] * n
    for i, w in enumerate(weights):
        s = absmax[i // blocksize] or 1.0
        v = float(w) / s
        k = nearest_parent(v)
        hole[i] = k
        sub = CELLS[k]
        best_j, bd = 0, abs(v - sub[0])
        for j in range(1, 16):
            d = abs(v - sub[j])
            if d < bd:
                best_j, bd = j, d
        plug[i] = best_j
    return hole, plug, absmax
def decode(
    hole: Sequence[int], plug: Sequence[int], absmax: Sequence[float], blocksize: int = 64
) -> List[float]:
    out = [0.0] * len(hole)
    for i in range(len(hole)):
        out[i] = CELLS[hole[i]][plug[i]] * absmax[i // blocksize]
    return out
def decode_parent_only(hole: Sequence[int], absmax: Sequence[float], blocksize: int = 64) -> List[float]:
    out = [0.0] * len(hole)
    for i in range(len(hole)):
        out[i] = PARENT[hole[i]] * absmax[i // blocksize]
    return out
def write_c_header(path: str) -> None:
    lines = [
        "/* Generated by glint_codec.py. GLINT_CELLS[hole][plug]. */",
        "#ifndef GLINT_CELLS_H_",
        "#define GLINT_CELLS_H_",
        "static const float GLINT_CELLS[16][16] = {",
    ]
    for row in CELLS:
        lines.append(" { " + ", ".join(f"{x:.9e}f" for x in row) + " },")
    lines += ["};", "#endif", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))

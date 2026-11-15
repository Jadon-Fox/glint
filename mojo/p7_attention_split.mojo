# Project 7 Mojo/MAX: position-parallel single-token attention (GLINT #116/#117 speed work).
# The reference kernel (attention_one) runs one block per query head and walks every cached position sequentially with
# ~8 barriers per position, so cost grows linearly with context (1.7 ms at position 100 -> 55 ms at 2000, summed over 32 layers).
# Here the positions of one head are split over NCH blocks; inside a block each of 4 warps takes every 4th position and keeps
# its own online-softmax state (max, denominator, 4 accumulator values per lane); the warps are merged in shared memory, and a
# second kernel merges the NCH chunk partials. Same math and the same bf16 rounding of the softmax weight
# (bf16r(exp(score - running_max))), but the summation / rescaling ORDER differs, so results are not bit-identical to
# attention_one. Validate on the recorded gate before relying on it.
# Launch A: grid_dim=(NCH, NQ), block_dim=128.  Launch B: grid_dim=NQ, block_dim=128.
from std.math import exp, sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import warp
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from p7_forward import bf16r

comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime NCH = 8

def attention_split_partial[
    ro: ImmOrigin, ko: ImmOrigin, vo: ImmOrigin, mo: MutOrigin, do: MutOrigin, ao: MutOrigin
](
    rotated: ImmPointer[Float32, ro],
    cache_k: ImmPointer[Float32, ko],
    cache_v: ImmPointer[Float32, vo],
    pm: MutPointer[Float32, mo],
    pd: MutPointer[Float32, do],
    pacc: MutPointer[Float32, ao],
    position: Int32,
):
    var hq = block_idx.y
    var c = block_idx.x
    var tid = thread_idx.x
    var lane = tid & 31
    var w = tid >> 5
    var hkv = hq // 3
    var npos = Int(position) + 1
    var ch = (npos + NCH - 1) // NCH
    var lo = Int(c) * ch
    var hi = lo + ch
    if hi > npos:
        hi = npos
    var isc = Float32(1) / sqrt(Float32(HD))
    var q0 = rotated[unsafe_offset=hq * HD + lane * 4] * isc
    var q1 = rotated[unsafe_offset=hq * HD + lane * 4 + 1] * isc
    var q2 = rotated[unsafe_offset=hq * HD + lane * 4 + 2] * isc
    var q3 = rotated[unsafe_offset=hq * HD + lane * 4 + 3] * isc
    var m = Float32(-1e30)
    var dsum = Float32(0)
    var a0 = Float32(0)
    var a1 = Float32(0)
    var a2 = Float32(0)
    var a3 = Float32(0)
    for tk in range(lo + Int(w), hi, 4):
        var base = (tk * NKV + Int(hkv)) * HD + Int(lane) * 4
        var part = q0 * cache_k[unsafe_offset=base] + q1 * cache_k[unsafe_offset=base + 1] + q2 * cache_k[unsafe_offset=base + 2] + q3 * cache_k[unsafe_offset=base + 3]
        var score = warp.sum(part)
        var mn = score if score > m else m
        var alpha = exp(m - mn)
        var beta = exp(score - mn)
        dsum = dsum * alpha + beta
        m = mn
        var bw = bf16r(beta)
        a0 = a0 * alpha + bw * cache_v[unsafe_offset=base]
        a1 = a1 * alpha + bw * cache_v[unsafe_offset=base + 1]
        a2 = a2 * alpha + bw * cache_v[unsafe_offset=base + 2]
        a3 = a3 * alpha + bw * cache_v[unsafe_offset=base + 3]
    var sm = stack_allocation[4, Float32, address_space=AddressSpace.SHARED]()
    var sd = stack_allocation[4, Float32, address_space=AddressSpace.SHARED]()
    var sa = stack_allocation[512, Float32, address_space=AddressSpace.SHARED]()
    if lane == 0:
        sm[unsafe_offset=w] = m
        sd[unsafe_offset=w] = dsum
    sa[unsafe_offset=Int(w) * HD + Int(lane) * 4] = a0
    sa[unsafe_offset=Int(w) * HD + Int(lane) * 4 + 1] = a1
    sa[unsafe_offset=Int(w) * HD + Int(lane) * 4 + 2] = a2
    sa[unsafe_offset=Int(w) * HD + Int(lane) * 4 + 3] = a3
    barrier()
    var mm = sm[unsafe_offset=0]
    for j in range(1, 4):
        var v = sm[unsafe_offset=j]
        if v > mm:
            mm = v
    var dd = Float32(0)
    var aa = Float32(0)
    for j in range(4):
        var f = exp(sm[unsafe_offset=j] - mm)
        dd += sd[unsafe_offset=j] * f
        aa += sa[unsafe_offset=j * HD + Int(tid)] * f
    var slot = Int(hq) * NCH + Int(c)
    pacc[unsafe_offset=slot * HD + Int(tid)] = aa
    if tid == 0:
        pm[unsafe_offset=slot] = mm
        pd[unsafe_offset=slot] = dd

def attention_split_combine[
    mo: ImmOrigin, do: ImmOrigin, ao: ImmOrigin, so: MutOrigin, bo: MutOrigin
](
    pm: ImmPointer[Float32, mo],
    pd: ImmPointer[Float32, do],
    pacc: ImmPointer[Float32, ao],
    dst: MutPointer[Float32, so],
    out_bf16: MutPointer[BFloat16, bo],
):
    var hq = Int(block_idx.x)
    var d = Int(thread_idx.x)
    var mm = pm[unsafe_offset=hq * NCH]
    for j in range(1, NCH):
        var v = pm[unsafe_offset=hq * NCH + j]
        if v > mm:
            mm = v
    var dd = Float32(0)
    var aa = Float32(0)
    for j in range(NCH):
        var f = exp(pm[unsafe_offset=hq * NCH + j] - mm)
        dd += pd[unsafe_offset=hq * NCH + j] * f
        aa += pacc[unsafe_offset=(hq * NCH + j) * HD + d] * f
    var o = aa / dd if dd > Float32(0) else Float32(0)
    dst[unsafe_offset=hq * HD + d] = o
    out_bf16[unsafe_offset=hq * HD + d] = o.cast[DType.bfloat16]()

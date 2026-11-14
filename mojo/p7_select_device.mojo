# Project 7 Mojo/MAX: device-side activation selection for the selected-plug decoder (GLINT #117).
# Replaces the per-linear host round trip (D2H of the linear input, 32768-bin host histogram, index upload, host sync) with kernels
# that stay on the stream. Same selection rule as the host path and the CUDA act rule: with key = bf16 bits & 0x7FFF, keep every
# channel whose key >= T, where T is the LARGEST value such that at least nsel channels have key >= T (so ties at T are all kept).
# Indices are written in ascending channel order (the correction sums in that order, so results match the host path exactly).
# The count stays on the device (cnt) and is also logged per linear (cnt_log) for byte accounting.
# select_topk: grid_dim=1, block_dim=1024, K <= 8192.  gather_rows_d: grid_dim=K, block_dim=128.
from std.gpu import thread_idx, block_idx, global_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace

comptime SB = 1024

def select_topk[
    xo: ImmOrigin, io: MutOrigin, co: MutOrigin, lo: MutOrigin
](
    x: ImmPointer[UInt16, xo],
    idx: MutPointer[Int32, io],
    cnt: MutPointer[Int32, co],
    cnt_log: MutPointer[Int32, lo],
    slot: Int32, kdim: Int32, nsel: Int32,
):
    var tid = Int(thread_idx.x)
    var kk = Int(kdim)
    var per = (kk + SB - 1) // SB
    var lo_i = tid * per
    var hi_i = lo_i + per
    if hi_i > kk:
        hi_i = kk
    var red = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var thr = 0
    for bi in range(15):
        var cand = thr | (1 << (14 - bi))
        var c = 0
        for i in range(lo_i, hi_i):
            if (Int(x[unsafe_offset=i]) & 32767) >= cand:
                c += 1
        red[unsafe_offset=tid] = Int32(c)
        barrier()
        comptime for step in range(10):
            comptime off = 512 >> step
            if tid < off:
                red[unsafe_offset=tid] = red[unsafe_offset=tid] + red[unsafe_offset=tid + off]
            barrier()
        var total = Int(red[unsafe_offset=0])
        barrier()
        if total >= Int(nsel):
            thr = cand
    var mine = 0
    for i in range(lo_i, hi_i):
        if (Int(x[unsafe_offset=i]) & 32767) >= thr:
            mine += 1
    red[unsafe_offset=tid] = Int32(mine)
    barrier()
    comptime for step in range(10):
        comptime off = 1 << step
        var v = Int32(0)
        if tid >= off:
            v = red[unsafe_offset=tid - off]
        barrier()
        red[unsafe_offset=tid] = red[unsafe_offset=tid] + v
        barrier()
    var pos = Int(red[unsafe_offset=tid]) - mine
    for i in range(lo_i, hi_i):
        if (Int(x[unsafe_offset=i]) & 32767) >= thr:
            idx[unsafe_offset=pos] = Int32(i)
            pos += 1
    if tid == SB - 1:
        var total = red[unsafe_offset=SB - 1]
        cnt[unsafe_offset=0] = total
        cnt_log[unsafe_offset=Int(slot)] = total

# One block per candidate selected row: copies the K-major plug row of channel idx[i] (rowwords uint32 words) from pinned host
# memory into the staging buffer; blocks beyond the device-side count exit immediately.
def gather_rows_d[
    po: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin
](plugk: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], stage: MutPointer[UInt32, so], rowwords: Int32):
    var i = Int(block_idx.x)
    if i >= Int(cnt[unsafe_offset=0]):
        return
    var rw = Int(rowwords)
    var src = Int(idx[unsafe_offset=i]) * rw
    for w in range(Int(thread_idx.x), rw, 128):
        stage[unsafe_offset=i * rw + w] = plugk[unsafe_offset=src + w]

# Same arithmetic and channel order as apply_corr in p7_glint_select_decode.mojo, count read from device memory.
def apply_corr_d[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, yo: MutOrigin
](
    hole: ImmPointer[UInt8, ho],
    scales: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    cnt: ImmPointer[Int32, co],
    y: MutPointer[Float32, yo],
    n: Int32, kdim: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var kk = Int(kdim)
    var half = Int(n) // 2
    var cn = Int(cnt[unsafe_offset=0])
    var acc = Float32(0)
    for i in range(cn):
        var c = Int(idx[unsafe_offset=i])
        var j = r * kk + c
        var h = Int((hole[unsafe_offset=j >> 1] >> UInt8(4 * (j & 1))) & UInt8(15))
        var pb = stage[unsafe_offset=i * half + (r >> 1)]
        var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
        acc += (delta[unsafe_offset=h * 16 + p] * scales[unsafe_offset=j // 64]) * x[unsafe_offset=c].cast[DType.float32]()
    y[unsafe_offset=r] = y[unsafe_offset=r] + acc

# K-major scale copy: scales_k[(c // 64) * n + r] = scales[r * (kdim // 64) + c // 64]. Launch: grid_dim=ceildiv(n * kdim // 64, 256), block_dim=256.
def transpose_scales[
    io: ImmOrigin, oo: MutOrigin
](src: ImmPointer[Float32, io], dst: MutPointer[Float32, oo], n: Int32, kdim: Int32):
    var g = global_idx.x
    var cb_n = Int(kdim) // 64
    if g >= Int(n) * cb_n:
        return
    var r = g // cb_n
    var cb = g - r * cb_n
    dst[unsafe_offset=cb * Int(n) + r] = src[unsafe_offset=g]

# Correction with K-major hole nibbles (hole_k[c * n/2 + r/2], low nibble first, same layout as the K-major plug staging) and
# K-major scales, so every per-channel read is coalesced across rows. Identical arithmetic and channel order to apply_corr_d.
def apply_corr_k[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, yo: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    cnt: ImmPointer[Int32, co],
    y: MutPointer[Float32, yo],
    n: Int32, kdim: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var nn = Int(n)
    var half = nn // 2
    var cn = Int(cnt[unsafe_offset=0])
    var acc = Float32(0)
    for i in range(cn):
        var c = Int(idx[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
        var pb = stage[unsafe_offset=i * half + (r >> 1)]
        var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
        acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    y[unsafe_offset=r] = y[unsafe_offset=r] + acc

# 16-byte-wide variant of gather_rows_d: every thread moves 4 words per iteration, which is what the zero-copy PCIe read needs to
# get near link rate (rowwords must be a multiple of 4: 384, 640 and 2048 here). Same data movement, same staging layout.
def gather_rows_d4[
    po: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin
](plugk: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], stage: MutPointer[UInt32, so], rowwords: Int32):
    var i = Int(block_idx.x)
    if i >= Int(cnt[unsafe_offset=0]):
        return
    var rw = Int(rowwords)
    var src = Int(idx[unsafe_offset=i]) * rw
    var dst = i * rw
    for w4 in range(Int(thread_idx.x), rw // 4, 128):
        var v = plugk.unsafe_offset(src + 4 * w4).load[width=4, alignment=16]()
        stage.unsafe_offset(dst + 4 * w4).store[alignment=16](v)

# Split-over-channels correction: the selected channels are divided into KS contiguous chunks, one grid row each, so 8x more
# blocks run than with one thread per output row walking every channel. Each chunk sums its channels in ascending order into
# part[s * n + r]; combine_corr adds the KS partials in a fixed order and then the row of y. Deterministic, but a different
# summation order from apply_corr_k (partial sums per chunk), so results are validated on the gate, not assumed identical.
comptime KS = 8

def apply_corr_ks[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var cn = Int(cnt[unsafe_offset=0])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
        var pb = stage[unsafe_offset=i * half + (r >> 1)]
        var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
        acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

def combine_corr[
    po: ImmOrigin, yo: MutOrigin
](part: ImmPointer[Float32, po], y: MutPointer[Float32, yo], n: Int32):
    var r = global_idx.x
    if r >= Int(n):
        return
    var acc = part[unsafe_offset=r]
    for s in range(1, KS):
        acc = acc + part[unsafe_offset=s * Int(n) + r]
    y[unsafe_offset=r] = y[unsafe_offset=r] + acc

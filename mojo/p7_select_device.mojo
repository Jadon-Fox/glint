# Project 7 Mojo/MAX: device-side activation selection for the selected-plug decoder (GLINT #117).
# Replaces the per-linear host round trip (D2H of the linear input, 32768-bin host histogram, index upload, host sync) with kernels
# that stay on the stream. Same selection rule as the host path and the CUDA act rule: with key = bf16 bits & 0x7FFF, keep every
# channel whose key >= T, where T is the LARGEST value such that at least nsel channels have key >= T (so ties at T are all kept).
# Indices are written in ascending channel order (the correction sums in that order, so results match the host path exactly).
# The count stays on the device (cnt) and is also logged per linear (cnt_log) for byte accounting.
# select_topk: grid_dim=1, block_dim=1024, K <= 8192.  gather_rows_d: grid_dim=K, block_dim=128.
from std.gpu import thread_idx, block_idx, global_idx, grid_dim
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
    slot: Int32, kdim: Int32, nsel: Int32, tkey: Int32, nfloor: Int32,
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
    # Price mode (G-fold selection, GFOLD_SELECTION_2026-11-15.md): tkey > 0 is a fixed |x| key threshold for this linear. nsel then
    # acts only as the cap that bounds the staging buffers: the kept set is every channel with key >= max(rank threshold, tkey).
    var thr_cap = thr
    if Int(tkey) > thr:
        thr = Int(tkey)
    # Relational price (H-G2r): tkey < 0 means tau = -tkey / 1024, and the threshold is tau * mean|x| of THIS input vector.
    if Int(tkey) < 0:
        var redf = stack_allocation[SB, Float32, address_space=AddressSpace.SHARED]()
        var acc = Float32(0)
        for i in range(lo_i, hi_i):
            var k = Int(x[unsafe_offset=i]) & 32767
            var ex = k >> 7
            if ex > 0:
                var f = Float32(128 + (k & 127))
                var n = ex - 134
                if n > 40:
                    n = 40
                if n < -60:
                    n = -60
                while n > 0:
                    f = f * 2.0
                    n -= 1
                while n < 0:
                    f = f * 0.5
                    n += 1
                acc += f
        redf[unsafe_offset=tid] = acc
        barrier()
        comptime for step in range(10):
            comptime off = 512 >> step
            if tid < off:
                redf[unsafe_offset=tid] = redf[unsafe_offset=tid] + redf[unsafe_offset=tid + off]
            barrier()
        var t = redf[unsafe_offset=0] / Float32(kk) * Float32(-Int(tkey)) / 1024.0
        barrier()
        var rk = 1
        if t > 0.0:
            var e2 = 0
            while t >= 2.0 and e2 < 100:
                t = t * 0.5
                e2 += 1
            while t < 1.0 and e2 > -120:
                t = t * 2.0
                e2 -= 1
            rk = ((e2 + 127) << 7) | Int((t - 1.0) * 128.0)
            if rk < 1:
                rk = 1
            if rk > 32767:
                rk = 32767
        if rk > thr:
            thr = rk
    # Floor (2026-11-16): with a price threshold active, every linear still keeps at least its nfloor largest channels, i.e. the
    # kept set is {key >= max(cap threshold, min(price threshold, floor-rank threshold))}.
    if Int(nfloor) > 0 and Int(tkey) != 0:
        var thf = 0
        for bi in range(15):
            var cand = thf | (1 << (14 - bi))
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
            if total >= Int(nfloor):
                thf = cand
        if thf < thr:
            thr = thf
        if thr < thr_cap:
            thr = thr_cap
    # nsel = 0 selects nothing (coast on the fold alone); without this the bisection keeps the largest key.
    if Int(nsel) == 0:
        thr = 32768
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

# Selection-frequency profile (read-only diagnostic, "prof" mode): hist[off + channel] += 1 for every channel selected by the current linear.
def accum_hist[
    io: ImmOrigin, co: ImmOrigin, ho: MutOrigin
](idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], hist: MutPointer[Int32, ho], off: Int32):
    var n = Int(cnt[unsafe_offset=0])
    for i in range(Int(thread_idx.x), n, 128):
        var c = Int(off) + Int(idx[unsafe_offset=i])
        hist[unsafe_offset=c] = hist[unsafe_offset=c] + 1

# Persistent-grid variant of gather_rows_d4 for the skeleton prefetch: a fixed small grid (GB blocks) loops over the selected rows, so a
# PCIe-bound gather occupies few SMs and the compute kernels running alongside are not starved. Same data movement and staging layout.
comptime GB = 16
def gather_rows_p4[
    po: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin
](plugk: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], stage: MutPointer[UInt32, so], rowwords: Int32):
    var rw = Int(rowwords)
    var n = Int(cnt[unsafe_offset=0])
    for i in range(Int(block_idx.x), n, GB):
        var src = Int(idx[unsafe_offset=i]) * rw
        var dst = i * rw
        for w4 in range(Int(thread_idx.x), rw // 4, 128):
            var v = plugk.unsafe_offset(src + 4 * w4).load[width=4, alignment=16]()
            stage.unsafe_offset(dst + 4 * w4).store[alignment=16](v)

# Token-to-token selection overlap (read-only diagnostic, "ovl" mode): counts how many of this call's selected channels were also selected
# by the same linear on the previous token (stamp = position + 1000; the flag array holds the stamp of the last selection of each channel).
def overlap_count[
    io: ImmOrigin, co: ImmOrigin, fo: MutOrigin, ho: MutOrigin
](idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], flags: MutPointer[Int32, fo], hits: MutPointer[Int32, ho], off: Int32, stamp: Int32, slot: Int32):
    if Int(thread_idx.x) != 0:
        return
    var n = Int(cnt[unsafe_offset=0])
    var h = 0
    for i in range(n):
        var c = Int(off) + Int(idx[unsafe_offset=i])
        if flags[unsafe_offset=c] == stamp:
            h += 1
        flags[unsafe_offset=c] = stamp + 1
    hits[unsafe_offset=Int(slot)] = Int32(h)

# Coverage probe (read-only, "ovl" mode): of the channels the REAL activation selects (idx), how many were selected by the skeleton on this
# token (flags == stamp + 1) and how many only on the previous token (flags == stamp). Does not modify flags. hits[2 * slot + {0, 1}].
def overlap_probe[
    io: ImmOrigin, co: ImmOrigin, fo: ImmOrigin, ho: MutOrigin
](idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], flags: ImmPointer[Int32, fo], hits: MutPointer[Int32, ho], off: Int32, stamp: Int32, slot: Int32):
    if Int(thread_idx.x) != 0:
        return
    var n = Int(cnt[unsafe_offset=0])
    var a = 0
    var b = 0
    for i in range(n):
        var f = flags[unsafe_offset=Int(off) + Int(idx[unsafe_offset=i])]
        if f == stamp + 1:
            a += 1
        elif f == stamp:
            b += 1
    hits[unsafe_offset=2 * Int(slot)] = Int32(a)
    hits[unsafe_offset=2 * Int(slot) + 1] = Int32(b)

# Exact-miss fixup ("fix" mode). Channel-addressed staging: the plug row of channel c lives at stage[c * rowwords], so any selection list can
# read it. A row is copied only if the channel was not selected on the previous token or earlier on this one (flags[off + c] < stamp, with
# stamp = position + 1000); the call then marks it (flags = stamp + 1). Plug rows never change, so a skipped row is still valid in the staging
# buffer; the flag only emulates a one-token-lookback cache. Persistent grid of GB blocks, same as gather_rows_p4.
def gather_rows_fix[
    po: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin, fo: MutOrigin
](plugk: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], stage: MutPointer[UInt32, so], flags: MutPointer[Int32, fo], rowwords: Int32, off: Int32, stamp: Int32, win: Int32):
    var rw = Int(rowwords)
    var n = Int(cnt[unsafe_offset=0])
    var tid = Int(thread_idx.x)
    var need = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
    for i in range(Int(block_idx.x), n, Int(grid_dim.x)):
        var c = Int(idx[unsafe_offset=i])
        if tid == 0:
            var f = flags[unsafe_offset=Int(off) + c]
            if f < stamp - win:
                need[unsafe_offset=0] = 1
            else:
                need[unsafe_offset=0] = 0
            flags[unsafe_offset=Int(off) + c] = stamp + 1
        barrier()
        if need[unsafe_offset=0] != 0:
            var base = c * rw
            for w4 in range(tid, rw // 4, 128):
                var v = plugk.unsafe_offset(base + 4 * w4).load[width=4, alignment=16]()
                stage.unsafe_offset(base + 4 * w4).store[alignment=16](v)
        barrier()

# apply_corr_ks reading channel-addressed staging: stage[c * (n / 2) + r / 2] instead of stage[i * (n / 2) + r / 2].
def apply_corr_ksc[
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
        var pb = stage[unsafe_offset=c * half + (r >> 1)]
        var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
        acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

# Ephemeral plug correction (operator, 2026-11-15): apply_corr_ksc with the plug rows read straight from pinned HOST memory.
# The plain kernel reads one byte per thread, which over the link is one tiny transfer per byte. Here four threads per block
# each pull one aligned 16-byte vector (32 output rows) into shared memory, then all 128 threads compute from it, so host
# reads are the same 16-byte pieces the gather kernel uses. Nothing of the plug is written to device memory beyond the block's
# 64-byte scratch. block_dim must be 128 and n a multiple of 128. Same arithmetic and summation order as apply_corr_ksc.
def apply_corr_kse[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    plug_host: ImmPointer[UInt8, go],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32,
):
    var r = global_idx.x
    var tid = Int(thread_idx.x)
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var cn = Int(cnt[unsafe_offset=0])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var sh = stack_allocation[64, UInt8, address_space=AddressSpace.SHARED]()
    var acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        if (tid & 31) == 0 and r < nn:
            var v = plug_host.unsafe_offset(c * half + (r >> 1)).load[width=16, alignment=16]()
            sh.unsafe_offset(tid >> 1).store[alignment=1](v)
        barrier()
        if r < nn:
            var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
            var p = Int((sh[unsafe_offset=tid >> 1] >> UInt8(4 * (r & 1))) & UInt8(15))
            acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
        barrier()
    if r < nn:
        part[unsafe_offset=s * nn + r] = acc

# Reuse-distance probe (read-only, "ovl" mode): for each channel this call selects, the number of tokens since the same linear last selected it
# (flags hold last-selected position + 1001; stamp = position + 1000, so age = stamp - flag + 1, >= 1). hits[16 * slot + age - 1] += 1 for ages 1..16.
# Run BEFORE overlap_count, which marks the flags for this token.
def overlap_age[
    io: ImmOrigin, co: ImmOrigin, fo: ImmOrigin, ho: MutOrigin
](idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], flags: ImmPointer[Int32, fo], hits: MutPointer[Int32, ho], off: Int32, stamp: Int32, slot: Int32):
    if Int(thread_idx.x) != 0:
        return
    var n = Int(cnt[unsafe_offset=0])
    for i in range(n):
        var f = Int(flags[unsafe_offset=Int(off) + Int(idx[unsafe_offset=i])])
        if f > 0:
            var age = Int(stamp) - f + 1
            if age >= 1 and age <= 16:
                var k = 16 * Int(slot) + age - 1
                hits[unsafe_offset=k] = hits[unsafe_offset=k] + 1

# ---- Bounded row pool ("pool" mode) ----------------------------------------------------------------------------------------------------
# Replaces the full-plug channel-addressed staging buffer with CAP rows per linear. slot_of[c] is the pool slot holding channel c's plug row (or -1),
# slot_chan[s] the channel in slot s (or -1), stamp[s] the last token (position + 1) that used slot s. A row stays resident across tokens until its
# slot is reused for a newer channel; a slot is reusable only if the current token has not used it (stamp < cur), so rows of the current
# selection are never evicted. Three single-purpose kernels per list: lookup (mark hits, compact the misses), free (compact the reusable
# slots), assign (give each miss a free slot and copy its row from host memory).

# One block of SB threads. Hits refresh their slot's stamp; misses are compacted into miss_idx in list order. log[slot] = number of misses.
def pool_lookup[
    io: ImmOrigin, co: ImmOrigin, so: ImmOrigin, to: MutOrigin, mo: MutOrigin, no: MutOrigin, lo: MutOrigin
](idx: ImmPointer[Int32, io], cnt: ImmPointer[Int32, co], slot_of: ImmPointer[Int32, so], stamp: MutPointer[Int32, to],
  miss_idx: MutPointer[Int32, mo], miss_cnt: MutPointer[Int32, no], log: MutPointer[Int32, lo], cur: Int32, slot: Int32):
    var tid = Int(thread_idx.x)
    var n = Int(cnt[unsafe_offset=0])
    var per = (n + SB - 1) // SB
    var lo_i = tid * per
    var hi_i = lo_i + per
    if hi_i > n:
        hi_i = n
    var red = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var m = 0
    for i in range(lo_i, hi_i):
        var c = Int(idx[unsafe_offset=i])
        var s = Int(slot_of[unsafe_offset=c])
        if s >= 0:
            stamp[unsafe_offset=s] = cur
        else:
            m += 1
    red[unsafe_offset=tid] = Int32(m)
    barrier()
    comptime for step in range(10):
        comptime off = 1 << step
        var v = Int32(0)
        if tid >= off:
            v = red[unsafe_offset=tid - off]
        barrier()
        red[unsafe_offset=tid] = red[unsafe_offset=tid] + v
        barrier()
    var pos = Int(red[unsafe_offset=tid]) - m
    for i in range(lo_i, hi_i):
        var c = Int(idx[unsafe_offset=i])
        if Int(slot_of[unsafe_offset=c]) < 0:
            miss_idx[unsafe_offset=pos] = Int32(c)
            pos += 1
    if tid == SB - 1:
        var total = red[unsafe_offset=SB - 1]
        miss_cnt[unsafe_offset=0] = total
        log[unsafe_offset=Int(slot)] = total

# One block of SB threads. Compacts the slots with stamp < cur (reusable) into free_idx in ascending slot order.
def pool_free[
    to: ImmOrigin, fo: MutOrigin, no: MutOrigin
](stamp: ImmPointer[Int32, to], cap: Int32, cur: Int32, free_idx: MutPointer[Int32, fo], free_cnt: MutPointer[Int32, no]):
    var tid = Int(thread_idx.x)
    var n = Int(cap)
    var per = (n + SB - 1) // SB
    var lo_i = tid * per
    var hi_i = lo_i + per
    if hi_i > n:
        hi_i = n
    var red = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var m = 0
    for s in range(lo_i, hi_i):
        if stamp[unsafe_offset=s] < cur:
            m += 1
    red[unsafe_offset=tid] = Int32(m)
    barrier()
    comptime for step in range(10):
        comptime off = 1 << step
        var v = Int32(0)
        if tid >= off:
            v = red[unsafe_offset=tid - off]
        barrier()
        red[unsafe_offset=tid] = red[unsafe_offset=tid] + v
        barrier()
    var pos = Int(red[unsafe_offset=tid]) - m
    for s in range(lo_i, hi_i):
        if stamp[unsafe_offset=s] < cur:
            free_idx[unsafe_offset=pos] = Int32(s)
            pos += 1
    if tid == SB - 1:
        free_cnt[unsafe_offset=0] = red[unsafe_offset=SB - 1]

# GB blocks of 128 threads. Miss j takes free slot j: its previous occupant (if any) is un-mapped, the slot is bound to the new channel, stamped, and
# the plug row is copied from host memory. A miss with no free slot sets overflow[0] = 1 and is skipped (the run is then invalid; the count is reported).
def pool_assign[
    po: ImmOrigin, mo: ImmOrigin, mc: ImmOrigin, fo: ImmOrigin, fc: ImmOrigin,
    so: MutOrigin, co: MutOrigin, to: MutOrigin, lo: MutOrigin, vo: MutOrigin
](plugk: ImmPointer[UInt32, po], miss_idx: ImmPointer[Int32, mo], miss_cnt: ImmPointer[Int32, mc],
  free_idx: ImmPointer[Int32, fo], free_cnt: ImmPointer[Int32, fc],
  slot_of: MutPointer[Int32, so], slot_chan: MutPointer[Int32, co], stamp: MutPointer[Int32, to],
  pool: MutPointer[UInt32, lo], overflow: MutPointer[Int32, vo], rowwords: Int32, cur: Int32):
    var rw = Int(rowwords)
    var nm = Int(miss_cnt[unsafe_offset=0])
    var nf = Int(free_cnt[unsafe_offset=0])
    var tid = Int(thread_idx.x)
    for j in range(Int(block_idx.x), nm, GB):
        if j >= nf:
            if tid == 0:
                overflow[unsafe_offset=0] = 1
            continue
        var c = Int(miss_idx[unsafe_offset=j])
        var s = Int(free_idx[unsafe_offset=j])
        if tid == 0:
            var old = Int(slot_chan[unsafe_offset=s])
            if old >= 0:
                slot_of[unsafe_offset=old] = -1
            slot_chan[unsafe_offset=s] = Int32(c)
            slot_of[unsafe_offset=c] = Int32(s)
            stamp[unsafe_offset=s] = cur
        var src = c * rw
        var dst = s * rw
        for w4 in range(tid, rw // 4, 128):
            var v = plugk.unsafe_offset(src + 4 * w4).load[width=4, alignment=16]()
            pool.unsafe_offset(dst + 4 * w4).store[alignment=16](v)

# apply_corr_ks reading the slot-indexed pool: the plug row of channel c is pool[slot_of[c] * (n / 2) + r / 2]. Channels without a slot are skipped.
def apply_corr_ksp[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, slo: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    slot_of: ImmPointer[Int32, slo],
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
        var sl = Int(slot_of[unsafe_offset=c])
        if sl >= 0:
            var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
            var pb = stage[unsafe_offset=sl * half + (r >> 1)]
            var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
            acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

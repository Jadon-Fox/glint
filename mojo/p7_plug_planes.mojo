# Two-tier plug on the link (GLINT_TIER2_REAL): the K-major 4-bit plug is split into two 2-bit planes, HI (top 2 bits of every
# nibble) and LO (low 2 bits), each K-major with n/4 bytes per channel row; row r sits at byte r >> 2, bits 2 * (r & 3).
# Wide-set channels fetch only their HI row, top-set channels also their LO row, so link bytes are |top| * n/2 + (|wide| - |top|) * n/4.
from std.gpu import thread_idx, block_idx, global_idx, grid_dim
from std.gpu.primitives import warp
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from p7_select_device import KS, SB

# src: K-major plug, row c holds n nibbles at c * n/2 (low nibble first). Launch: grid_dim=ceildiv(kdim * n/4, 256), block_dim=256.
def split_planes[
    io: ImmOrigin, ho: MutOrigin, lo: MutOrigin
](src: ImmPointer[UInt8, io], hi: MutPointer[UInt8, ho], lo_p: MutPointer[UInt8, lo], n: Int32, kdim: Int32):
    var g = global_idx.x
    var quarter = Int(n) // 4
    if g >= Int(kdim) * quarter:
        return
    var c = g // quarter
    var b = g - c * quarter
    var base = c * (Int(n) // 2) + 2 * b
    var hv = UInt8(0)
    var lv = UInt8(0)
    comptime for k in range(4):
        var v = (src[unsafe_offset=base + (k >> 1)] >> UInt8(4 * (k & 1))) & UInt8(15)
        hv |= (v >> UInt8(2)) << UInt8(2 * k)
        lv |= (v & UInt8(3)) << UInt8(2 * k)
    hi[unsafe_offset=g] = hv
    lo_p[unsafe_offset=g] = lv

# apply_corr_ksc with the plug value read from channel-addressed plane staging (hi_s[c * n/4 + r/4]). use_lo = 0: p = HI << 2, which
# indexes the prefix table at the first cell of its group (every cell of a group holds the same value). use_lo = 1: p = (HI << 2) | LO,
# the full nibble. Same arithmetic, chunking and summation order as apply_corr_ksc.
def apply_corr_ksc_planes[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, lg: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    hi_s: ImmPointer[UInt8, go],
    lo_s: ImmPointer[UInt8, lg],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32, use_lo: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var quarter = nn // 4
    var sh = UInt8(2 * (r & 3))
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
        var p = Int((hi_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3)) << 2
        if use_lo != 0:
            p |= Int((lo_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3))
        acc += (delta[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

# GLINT_TIER2_FUSED: both selections of a two-tier linear in one launch. select_topk's rank rule (no price, no floor) run for nsel_t
# (top) and nsel_w (wide) side by side: each set is every channel whose key >= T, T the largest value with at least nsel channels
# at or above it; nsel = 0 selects nothing. Both index lists are ascending and equal what two select_topk calls write.
# cnt_w[1] receives the top threshold, which apply_corr_ksc_tier uses for top-set membership. grid_dim=1, block_dim=SB.
def select_topk2[
    xo: ImmOrigin, wo: MutOrigin, cwo: MutOrigin, to: MutOrigin, cto: MutOrigin, lo: MutOrigin
](
    x: ImmPointer[UInt16, xo],
    idx_w: MutPointer[Int32, wo],
    cnt_w: MutPointer[Int32, cwo],
    idx_t: MutPointer[Int32, to],
    cnt_t: MutPointer[Int32, cto],
    cnt_log: MutPointer[Int32, lo],
    slot_w: Int32, slot_t: Int32, kdim: Int32, nsel_w: Int32, nsel_t: Int32,
):
    var tid = Int(thread_idx.x)
    var kk = Int(kdim)
    var per = (kk + SB - 1) // SB
    var lo_i = tid * per
    var hi_i = lo_i + per
    if hi_i > kk:
        hi_i = kk
    var rw = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var rt = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var thw = 0
    var tht = 0
    for bi in range(15):
        var cw = thw | (1 << (14 - bi))
        var ct = tht | (1 << (14 - bi))
        var nw = 0
        var nt = 0
        for i in range(lo_i, hi_i):
            var k = Int(x[unsafe_offset=i]) & 32767
            if k >= cw:
                nw += 1
            if k >= ct:
                nt += 1
        rw[unsafe_offset=tid] = Int32(nw)
        rt[unsafe_offset=tid] = Int32(nt)
        barrier()
        comptime for step in range(10):
            comptime off = 512 >> step
            if tid < off:
                rw[unsafe_offset=tid] = rw[unsafe_offset=tid] + rw[unsafe_offset=tid + off]
                rt[unsafe_offset=tid] = rt[unsafe_offset=tid] + rt[unsafe_offset=tid + off]
            barrier()
        var tw = Int(rw[unsafe_offset=0])
        var tt = Int(rt[unsafe_offset=0])
        barrier()
        if tw >= Int(nsel_w):
            thw = cw
        if tt >= Int(nsel_t):
            tht = ct
    if Int(nsel_w) == 0:
        thw = 32768
    if Int(nsel_t) == 0:
        tht = 32768
    var mw = 0
    var mt = 0
    for i in range(lo_i, hi_i):
        var k = Int(x[unsafe_offset=i]) & 32767
        if k >= thw:
            mw += 1
        if k >= tht:
            mt += 1
    rw[unsafe_offset=tid] = Int32(mw)
    rt[unsafe_offset=tid] = Int32(mt)
    barrier()
    comptime for step in range(10):
        comptime off = 1 << step
        var vw = Int32(0)
        var vt = Int32(0)
        if tid >= off:
            vw = rw[unsafe_offset=tid - off]
            vt = rt[unsafe_offset=tid - off]
        barrier()
        rw[unsafe_offset=tid] = rw[unsafe_offset=tid] + vw
        rt[unsafe_offset=tid] = rt[unsafe_offset=tid] + vt
        barrier()
    var pw = Int(rw[unsafe_offset=tid]) - mw
    var pt = Int(rt[unsafe_offset=tid]) - mt
    for i in range(lo_i, hi_i):
        var k = Int(x[unsafe_offset=i]) & 32767
        if k >= thw:
            idx_w[unsafe_offset=pw] = Int32(i)
            pw += 1
        if k >= tht:
            idx_t[unsafe_offset=pt] = Int32(i)
            pt += 1
    if tid == SB - 1:
        var tw = rw[unsafe_offset=SB - 1]
        var tt = rt[unsafe_offset=SB - 1]
        cnt_w[unsafe_offset=0] = tw
        cnt_w[unsafe_offset=1] = Int32(tht)
        cnt_t[unsafe_offset=0] = tt
        cnt_log[unsafe_offset=Int(slot_w)] = tw
        cnt_log[unsafe_offset=Int(slot_t)] = tt

# GLINT_TIER2_FUSED: both two-tier corrections in one pass over the wide list. Every wide channel adds its prefix-table term
# (d_pre at HI << 2); a channel whose key >= the top threshold (cnt[1], from select_topk2) also adds its difference-table term
# (d_dif at the full nibble). Equals apply_corr_ksc_planes(use_lo=0, wide) + apply_corr_ksc_planes(use_lo=1, top) up to summation order.
def apply_corr_ksc_tier[
    ho: ImmOrigin, so: ImmOrigin, dp: ImmOrigin, dd: ImmOrigin, xo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, go: ImmOrigin, lg: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    d_pre: ImmPointer[Float32, dp],
    d_dif: ImmPointer[Float32, dd],
    x: ImmPointer[BFloat16, xo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    hi_s: ImmPointer[UInt8, go],
    lo_s: ImmPointer[UInt8, lg],
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
    var quarter = nn // 4
    var sh = UInt8(2 * (r & 3))
    var cn = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
        var p = Int((hi_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3)) << 2
        var d = d_pre[unsafe_offset=h * 16 + p]
        if (Int(x16[unsafe_offset=c]) & 32767) >= tht:
            p |= Int((lo_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3))
            d += d_dif[unsafe_offset=h * 16 + p]
        acc += (d * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

# GLINT_TIER2_FUSED: one launch for both plane gathers. Walks the wide list (cnt[0] rows); every channel's HI row goes to hi_s, and a
# channel whose key >= the top threshold (cnt[1]) also has its LO row copied to lo_s. Same rows as gather_rows_fix with win = 0 over
# the wide list (HI) and the top list (LO); the fetch stamps are not written (nothing reads them on the two-tier path).
# rowwords = n / 16 (one plane row), a multiple of 4. grid_dim = GB blocks, block_dim = 128.
def gather_planes2[
    ho: ImmOrigin, lo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, co: ImmOrigin, hs: MutOrigin, ls: MutOrigin
](
    hi_p: ImmPointer[UInt32, ho],
    lo_p: ImmPointer[UInt32, lo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    cnt: ImmPointer[Int32, co],
    hi_s: MutPointer[UInt32, hs],
    lo_s: MutPointer[UInt32, ls],
    rowwords: Int32,
):
    var rw = Int(rowwords)
    var n = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var tid = Int(thread_idx.x)
    for i in range(Int(block_idx.x), n, Int(grid_dim.x)):
        var c = Int(idx[unsafe_offset=i])
        var base = c * rw
        for w4 in range(tid, rw // 4, 128):
            var v = hi_p.unsafe_offset(base + 4 * w4).unsafe_load[width=4, alignment=16]()
            hi_s.unsafe_offset(base + 4 * w4).unsafe_store[alignment=16](v)
        if (Int(x16[unsafe_offset=c]) & 32767) >= tht:
            for w4 in range(tid, rw // 4, 128):
                var v = lo_p.unsafe_offset(base + 4 * w4).unsafe_load[width=4, alignment=16]()
                lo_s.unsafe_offset(base + 4 * w4).unsafe_store[alignment=16](v)

# GLINT_TIER2_FUSED=x: both two-tier corrections in one launch with the unfused arithmetic. Block (·, s) computes chunk s of the
# prefix correction over the wide list into part_w and chunk s of the difference correction over the top list into part_t, each with
# exactly the chunking and summation order of apply_corr_ksc_planes; combine_corr2 then adds them in the unfused order.
def apply_corr_ksc_tier_x[
    ho: ImmOrigin, so: ImmOrigin, dp: ImmOrigin, dd: ImmOrigin, xo: ImmOrigin, wo: ImmOrigin, cwo: ImmOrigin, to: ImmOrigin, cto: ImmOrigin, go: ImmOrigin, lg: ImmOrigin, pw: MutOrigin, pt: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    d_pre: ImmPointer[Float32, dp],
    d_dif: ImmPointer[Float32, dd],
    x: ImmPointer[BFloat16, xo],
    idx_w: ImmPointer[Int32, wo],
    cnt_w: ImmPointer[Int32, cwo],
    idx_t: ImmPointer[Int32, to],
    cnt_t: ImmPointer[Int32, cto],
    hi_s: ImmPointer[UInt8, go],
    lo_s: ImmPointer[UInt8, lg],
    part_w: MutPointer[Float32, pw],
    part_t: MutPointer[Float32, pt],
    n: Int32, kdim: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var quarter = nn // 4
    var sh = UInt8(2 * (r & 3))
    var hsel = r >> 1
    var hsh = UInt8(4 * (r & 1))
    var cn = Int(cnt_w[unsafe_offset=0])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx_w[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + hsel] >> hsh) & UInt8(15))
        var p = Int((hi_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3)) << 2
        acc += (d_pre[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part_w[unsafe_offset=s * nn + r] = acc
    cn = Int(cnt_t[unsafe_offset=0])
    chunk = (cn + KS - 1) // KS
    lo = s * chunk
    hi = lo + chunk
    if hi > cn:
        hi = cn
    acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx_t[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + hsel] >> hsh) & UInt8(15))
        var p = Int((hi_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3)) << 2
        p |= Int((lo_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3))
        acc += (d_dif[unsafe_offset=h * 16 + p] * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part_t[unsafe_offset=s * nn + r] = acc

# Two combine_corr calls in one: y = (y + sum_s part_w) + sum_s part_t, same order as combine_corr(part_w) then combine_corr(part_t).
def combine_corr2[
    wo: ImmOrigin, to: ImmOrigin, yo: MutOrigin
](part_w: ImmPointer[Float32, wo], part_t: ImmPointer[Float32, to], y: MutPointer[Float32, yo], n: Int32):
    var r = global_idx.x
    if r >= Int(n):
        return
    var aw = part_w[unsafe_offset=r]
    var at = part_t[unsafe_offset=r]
    for s in range(1, KS):
        aw = aw + part_w[unsafe_offset=s * Int(n) + r]
        at = at + part_t[unsafe_offset=s * Int(n) + r]
    var v = y[unsafe_offset=r] + aw
    y[unsafe_offset=r] = v + at

# Timing variants of apply_corr_ksc_tier (GLINT_TIER2_CV). d_top[h * 16 + p] = d_pre[h * 16 + (p & 12)] + d_dif[h * 16 + p], built once,
# so a top channel needs one table read instead of two. mode 0: global tables (bit-identical to apply_corr_ksc_tier). mode 1: both
# tables staged in shared memory per block (bit-identical). mode 2: prefix term only, no LO read (timing ablation, wrong values).
# mode 3: no table read, d = h (timing ablation, wrong values). block_dim must be 128 and n a multiple of 128.
def apply_corr_ksc_tier_v[
    MODE: Int, ho: ImmOrigin, so: ImmOrigin, dp: ImmOrigin, dd: ImmOrigin, xo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, go: ImmOrigin, lg: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    d_pre: ImmPointer[Float32, dp],
    d_top: ImmPointer[Float32, dd],
    x: ImmPointer[BFloat16, xo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    hi_s: ImmPointer[UInt8, go],
    lo_s: ImmPointer[UInt8, lg],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32,
):
    var tbl = stack_allocation[512, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    comptime if MODE == 1:
        for j in range(tid, 256, 128):
            tbl[j] = d_pre[unsafe_offset=j]
            tbl[256 + j] = d_top[unsafe_offset=j]
        barrier()
    var r = global_idx.x
    if r >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var quarter = nn // 4
    var sh = UInt8(2 * (r & 3))
    var cn = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var acc = Float32(0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var h = Int((hole_k[unsafe_offset=c * half + (r >> 1)] >> UInt8(4 * (r & 1))) & UInt8(15))
        var p = Int((hi_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3)) << 2
        var d: Float32
        comptime if MODE == 3:
            d = Float32(h)
        elif MODE == 4:
            d = Float32(h + p)
        elif MODE == 5:
            d = d_pre[unsafe_offset=h * 17]
        elif MODE == 2:
            d = d_pre[unsafe_offset=h * 16 + p]
        elif MODE == 1:
            if (Int(x16[unsafe_offset=c]) & 32767) >= tht:
                p |= Int((lo_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3))
                d = tbl[256 + h * 16 + p]
            else:
                d = tbl[h * 16 + p]
        else:
            if (Int(x16[unsafe_offset=c]) & 32767) >= tht:
                p |= Int((lo_s[unsafe_offset=c * quarter + (r >> 2)] >> sh) & UInt8(3))
                d = d_top[unsafe_offset=h * 16 + p]
            else:
                d = d_pre[unsafe_offset=h * 16 + p]
        acc += (d * scales_k[unsafe_offset=(c >> 6) * nn + r]) * x[unsafe_offset=c].cast[DType.float32]()
    part[unsafe_offset=s * nn + r] = acc

# Row-blocked corrections (GLINT_CORR_R = R): one thread owns R consecutive output rows, reads them with vector loads, and keeps the
# delta tables in shared memory, so a channel costs a few wide loads per thread instead of a few byte loads per row. Every row still
# sums its channels in list order within chunk s, so apply_corr_ksc_r equals apply_corr_ksc and apply_corr_ksc_tier_r equals
# apply_corr_ksc_tier bit for bit (d_top[h * 16 + p] = d_pre[h * 16 + (p & 12)] + d_dif[h * 16 + p], the same f32 add).
# R in {2, 4, 8}; block_dim = 128; grid = (n / R / 128, KS); n a multiple of 128 * R.
def apply_corr_ksc_r[
    R: Int, ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, po: MutOrigin
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
    var tbl = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    for j in range(tid, 256, 128):
        tbl[j] = delta[unsafe_offset=j]
    barrier()
    var r0 = global_idx.x * R
    if r0 >= Int(n):
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
    var acc = SIMD[DType.float32, R](0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var xs = x[unsafe_offset=c].cast[DType.float32]()
        var hb = hole_k.unsafe_offset(c * half + (r0 >> 1)).load[width = R // 2, alignment = R // 2]()
        var pb = stage.unsafe_offset(c * half + (r0 >> 1)).load[width = R // 2, alignment = R // 2]()
        var sc = scales_k.unsafe_offset((c >> 6) * nn + r0).load[width=R, alignment = 4 * R if R < 4 else 16]()
        comptime for k in range(R):
            var h = Int((hb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
            var p = Int((pb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
            acc[k] += (tbl[h * 16 + p] * sc[k]) * xs
    part.unsafe_offset(s * nn + r0).store[alignment = 4 * R](acc)

def apply_corr_ksc_tier_r[
    R: Int, ho: ImmOrigin, so: ImmOrigin, dp: ImmOrigin, dd: ImmOrigin, xo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, go: ImmOrigin, lg: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    d_pre: ImmPointer[Float32, dp],
    d_top: ImmPointer[Float32, dd],
    x: ImmPointer[BFloat16, xo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    hi_s: ImmPointer[UInt8, go],
    lo_s: ImmPointer[UInt8, lg],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32,
):
    var tbl = stack_allocation[512, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    for j in range(tid, 256, 128):
        tbl[j] = d_pre[unsafe_offset=j]
        tbl[256 + j] = d_top[unsafe_offset=j]
    barrier()
    var r0 = global_idx.x * R
    if r0 >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var quarter = nn // 4
    var cn = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    comptime QW = R // 4 if R >= 4 else 1
    var q0 = r0 & 3
    var acc = SIMD[DType.float32, R](0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var xs = x[unsafe_offset=c].cast[DType.float32]()
        var hb = hole_k.unsafe_offset(c * half + (r0 >> 1)).load[width = R // 2, alignment = R // 2]()
        var qb = hi_s.unsafe_offset(c * quarter + (r0 >> 2)).load[width=QW, alignment=QW]()
        var sc = scales_k.unsafe_offset((c >> 6) * nn + r0).load[width=R, alignment = 4 * R if R < 4 else 16]()
        if (Int(x16[unsafe_offset=c]) & 32767) >= tht:
            var lb = lo_s.unsafe_offset(c * quarter + (r0 >> 2)).load[width=QW, alignment=QW]()
            comptime for k in range(R):
                var h = Int((hb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
                var p = (Int((qb[(q0 + k) >> 2] >> UInt8(2 * ((q0 + k) & 3))) & UInt8(3)) << 2) | Int((lb[(q0 + k) >> 2] >> UInt8(2 * ((q0 + k) & 3))) & UInt8(3))
                acc[k] += (tbl[256 + h * 16 + p] * sc[k]) * xs
        else:
            comptime for k in range(R):
                var h = Int((hb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
                var p = Int((qb[(q0 + k) >> 2] >> UInt8(2 * ((q0 + k) & 3))) & UInt8(3)) << 2
                acc[k] += (tbl[h * 16 + p] * sc[k]) * xs
    part.unsafe_offset(s * nn + r0).store[alignment = 4 * R](acc)

# GLINT_TIER2_FUSED=m: the planes exist to save link bytes; in VRAM the correction wants the plain 4-bit stage layout. This gather
# pulls the same plane rows over the link as gather_planes2 (HI for every wide channel, LO for top channels) and writes each channel's
# row as 4-bit nibbles (HI << 2) | LO into the channel-addressed stage slot (LO = 0 for wide-only channels), so apply_corr_ksc_tier_mr
# reads one nibble per row like apply_corr_ksc. rowwords = n / 16 (one plane row in u32), a multiple of 4. grid_dim = GB, block_dim = 128.
def gather_planes_merge[
    ho: ImmOrigin, lo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin
](
    hi_p: ImmPointer[UInt32, ho],
    lo_p: ImmPointer[UInt32, lo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    cnt: ImmPointer[Int32, co],
    stage: MutPointer[UInt32, so],
    rowwords: Int32,
):
    var rw = Int(rowwords)
    var n = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var tid = Int(thread_idx.x)
    for i in range(Int(block_idx.x), n, Int(grid_dim.x)):
        var c = Int(idx[unsafe_offset=i])
        var top = (Int(x16[unsafe_offset=c]) & 32767) >= tht
        for w4 in range(tid, rw // 4, 128):
            var hv = hi_p.unsafe_offset(c * rw + 4 * w4).unsafe_load[width=4, alignment=16]()
            var lv = SIMD[DType.uint32, 4](0)
            if top:
                lv = lo_p.unsafe_offset(c * rw + 4 * w4).unsafe_load[width=4, alignment=16]()
            var o = SIMD[DType.uint32, 8](0)
            comptime for q in range(4):
                var a = UInt32(0)
                var b = UInt32(0)
                comptime for k in range(16):
                    var nib = (((hv[q] >> UInt32(2 * k)) & UInt32(3)) << UInt32(2)) | ((lv[q] >> UInt32(2 * k)) & UInt32(3))
                    comptime if k < 8:
                        a |= nib << UInt32(4 * k)
                    else:
                        b |= nib << UInt32(4 * (k - 8))
                o[2 * q] = a
                o[2 * q + 1] = b
            stage.unsafe_offset(c * 2 * rw + 8 * w4).unsafe_store[alignment=16](o.slice[4, offset=0]())
            stage.unsafe_offset(c * 2 * rw + 8 * w4 + 4).unsafe_store[alignment=16](o.slice[4, offset=4]())

# GLINT_TIER2_FUSED=m correction: apply_corr_ksc_r over the merged stage with the table picked per channel (d_top for top channels,
# d_pre otherwise; both in shared memory). Equals apply_corr_ksc_tier bit for bit. R in {2, 4, 8}; grid = (n / R / 128, KS).
def apply_corr_ksc_tier_mr[
    R: Int, ho: ImmOrigin, so: ImmOrigin, dp: ImmOrigin, dd: ImmOrigin, xo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, go: ImmOrigin, co: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt8, ho],
    scales_k: ImmPointer[Float32, so],
    d_pre: ImmPointer[Float32, dp],
    d_top: ImmPointer[Float32, dd],
    x: ImmPointer[BFloat16, xo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    cnt: ImmPointer[Int32, co],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32,
):
    var tbl = stack_allocation[512, Float32, address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    for j in range(tid, 256, 128):
        tbl[j] = d_pre[unsafe_offset=j]
        tbl[256 + j] = d_top[unsafe_offset=j]
    barrier()
    var r0 = global_idx.x * R
    if r0 >= Int(n):
        return
    var s = Int(block_idx.y)
    var nn = Int(n)
    var half = nn // 2
    var cn = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var chunk = (cn + KS - 1) // KS
    var lo = s * chunk
    var hi = lo + chunk
    if hi > cn:
        hi = cn
    var acc = SIMD[DType.float32, R](0)
    for i in range(lo, hi):
        var c = Int(idx[unsafe_offset=i])
        var xs = x[unsafe_offset=c].cast[DType.float32]()
        var tb = 256 if (Int(x16[unsafe_offset=c]) & 32767) >= tht else 0
        var hb = hole_k.unsafe_offset(c * half + (r0 >> 1)).load[width = R // 2, alignment = R // 2]()
        var pb = stage.unsafe_offset(c * half + (r0 >> 1)).load[width = R // 2, alignment = R // 2]()
        var sc = scales_k.unsafe_offset((c >> 6) * nn + r0).load[width=R, alignment = 4 * R if R < 4 else 16]()
        comptime for k in range(R):
            var h = Int((hb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
            var p = Int((pb[k >> 1] >> UInt8(4 * (k & 1))) & UInt8(15))
            acc[k] += (tbl[tb + h * 16 + p] * sc[k]) * xs
    part.unsafe_offset(s * nn + r0).store[alignment = 4 * R](acc)

# GLINT_SEL_FAST=1: select_topk (rank rule, tkey = nfloor = 0) and select_topk2 with the same thresholds, fewer barriers. Each thread
# keeps its <= 8 keys in registers; a bisection round counts both sets at once (wide count in the low 16 bits, top in the high 16;
# kdim <= 8192 keeps both below 2^16), sums them with a warp shuffle, and the 32 warp sums meet in a double-buffered shared array,
# so a round costs one barrier instead of eleven. The thresholds are the same integers, so the index lists and counts are identical.
# TWO = False writes select_topk's outputs (idx_w, cnt_w[0], cnt_log[slot_w]); TWO = True writes select_topk2's. grid 1, block SB.
def select_rank_f[
    TWO: Bool, xo: ImmOrigin, wo: MutOrigin, cwo: MutOrigin, to: MutOrigin, cto: MutOrigin, lo: MutOrigin
](
    x: ImmPointer[UInt16, xo],
    idx_w: MutPointer[Int32, wo],
    cnt_w: MutPointer[Int32, cwo],
    idx_t: MutPointer[Int32, to],
    cnt_t: MutPointer[Int32, cto],
    cnt_log: MutPointer[Int32, lo],
    slot_w: Int32, slot_t: Int32, kdim: Int32, nsel_w: Int32, nsel_t: Int32,
):
    var tid = Int(thread_idx.x)
    var lane = tid & 31
    var wid = tid >> 5
    var kk = Int(kdim)
    var per = (kk + SB - 1) // SB
    var lo_i = tid * per
    var hi_i = lo_i + per
    if hi_i > kk:
        hi_i = kk
    var keys = SIMD[DType.int32, 8](-1)
    comptime for j in range(8):
        if lo_i + j < hi_i:
            keys[j] = Int32(Int(x[unsafe_offset=lo_i + j]) & 32767)
    var red = stack_allocation[64, Int32, address_space=AddressSpace.SHARED]()
    var thw = 0
    var tht = 0
    for bi in range(15):
        var cw = Int32(thw | (1 << (14 - bi)))
        var ct = Int32(tht | (1 << (14 - bi)))
        var v = Int32(0)
        comptime for j in range(8):
            if keys[j] >= cw:
                v += 1
            comptime if TWO:
                if keys[j] >= ct:
                    v += 65536
        v = warp.sum(v)
        var buf = (bi & 1) * 32
        if lane == 0:
            red[buf + wid] = v
        barrier()
        var tot = Int32(0)
        comptime for w in range(32):
            tot += red[buf + w]
        if Int(tot & 65535) >= Int(nsel_w):
            thw = Int(cw)
        comptime if TWO:
            if Int(tot >> 16) >= Int(nsel_t):
                tht = Int(ct)
    if Int(nsel_w) == 0:
        thw = 32768
    if Int(nsel_t) == 0:
        tht = 32768
    var rw = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var rt = stack_allocation[SB, Int32, address_space=AddressSpace.SHARED]()
    var mw = 0
    var mt = 0
    comptime for j in range(8):
        if Int(keys[j]) >= thw:
            mw += 1
        if Int(keys[j]) >= tht:
            mt += 1
    rw[unsafe_offset=tid] = Int32(mw)
    rt[unsafe_offset=tid] = Int32(mt)
    barrier()
    comptime for step in range(10):
        comptime off = 1 << step
        var vw = Int32(0)
        var vt = Int32(0)
        if tid >= off:
            vw = rw[unsafe_offset=tid - off]
            vt = rt[unsafe_offset=tid - off]
        barrier()
        rw[unsafe_offset=tid] = rw[unsafe_offset=tid] + vw
        rt[unsafe_offset=tid] = rt[unsafe_offset=tid] + vt
        barrier()
    var pw = Int(rw[unsafe_offset=tid]) - mw
    var pt = Int(rt[unsafe_offset=tid]) - mt
    comptime for j in range(8):
        if Int(keys[j]) >= thw:
            idx_w[unsafe_offset=pw] = Int32(lo_i + j)
            pw += 1
        comptime if TWO:
            if Int(keys[j]) >= tht:
                idx_t[unsafe_offset=pt] = Int32(lo_i + j)
                pt += 1
    if tid == SB - 1:
        var tw = rw[unsafe_offset=SB - 1]
        cnt_w[unsafe_offset=0] = tw
        cnt_log[unsafe_offset=Int(slot_w)] = tw
        comptime if TWO:
            var tt = rt[unsafe_offset=SB - 1]
            cnt_w[unsafe_offset=1] = Int32(tht)
            cnt_t[unsafe_offset=0] = tt
            cnt_log[unsafe_offset=Int(slot_t)] = tt

# GLINT_BIRD=1 helper: o = a + b (f32), for the predicted residual x + fold-only output.
def add_vec[ao: ImmOrigin, bo: ImmOrigin, oo: MutOrigin](a: ImmPointer[Float32, ao], b: ImmPointer[Float32, bo], o: MutPointer[Float32, oo], n: Int32):
    var i = global_idx.x
    if i < Int(n):
        o[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]

# GLINT_BIRD=1 on GLINT_TIER2_FUSED=m: gather_planes_merge with fetch stamps, so a predicted gather and the exact gather of the same
# linear share the staging slot. hflags[off + c] marks the HI row fetched at stamp, tflags[off + c] the LO row. A wide channel is
# fetched when its HI row is not from this linear (hflags < stamp) or when it is a top channel and its LO row is not (tflags < stamp);
# a fetch writes both planes when top, HI with LO = 0 otherwise. Stale LO bits on a wide-only channel are harmless: d_pre holds one
# value per prefix group, so the LO bits do not change the lookup. Persistent grid of GB blocks, block_dim 128.
def gather_planes_merge_f[
    ho: ImmOrigin, lo: ImmOrigin, ko: ImmOrigin, io: ImmOrigin, co: ImmOrigin, so: MutOrigin, fo: MutOrigin, to: MutOrigin
](
    hi_p: ImmPointer[UInt32, ho],
    lo_p: ImmPointer[UInt32, lo],
    x16: ImmPointer[UInt16, ko],
    idx: ImmPointer[Int32, io],
    cnt: ImmPointer[Int32, co],
    stage: MutPointer[UInt32, so],
    hflags: MutPointer[Int32, fo],
    tflags: MutPointer[Int32, to],
    rowwords: Int32, off: Int32, stamp: Int32,
):
    var rw = Int(rowwords)
    var n = Int(cnt[unsafe_offset=0])
    var tht = Int(cnt[unsafe_offset=1])
    var tid = Int(thread_idx.x)
    var need = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
    for i in range(Int(block_idx.x), n, Int(grid_dim.x)):
        var c = Int(idx[unsafe_offset=i])
        var top = (Int(x16[unsafe_offset=c]) & 32767) >= tht
        if tid == 0:
            var fh = hflags[unsafe_offset=Int(off) + c]
            var ft = tflags[unsafe_offset=Int(off) + c]
            var nd = fh < stamp or (top and ft < stamp)
            need[unsafe_offset=0] = 1 if nd else 0
            if nd:
                hflags[unsafe_offset=Int(off) + c] = stamp + 1
                if top:
                    tflags[unsafe_offset=Int(off) + c] = stamp + 1
        barrier()
        if need[unsafe_offset=0] != 0:
            for w4 in range(tid, rw // 4, 128):
                var hv = hi_p.unsafe_offset(c * rw + 4 * w4).unsafe_load[width=4, alignment=16]()
                var lv = SIMD[DType.uint32, 4](0)
                if top:
                    lv = lo_p.unsafe_offset(c * rw + 4 * w4).unsafe_load[width=4, alignment=16]()
                var o = SIMD[DType.uint32, 8](0)
                comptime for q in range(4):
                    var a = UInt32(0)
                    var b = UInt32(0)
                    comptime for k in range(16):
                        var nib = (((hv[q] >> UInt32(2 * k)) & UInt32(3)) << UInt32(2)) | ((lv[q] >> UInt32(2 * k)) & UInt32(3))
                        comptime if k < 8:
                            a |= nib << UInt32(4 * k)
                        else:
                            b |= nib << UInt32(4 * (k - 8))
                    o[2 * q] = a
                    o[2 * q + 1] = b
                stage.unsafe_offset(c * 2 * rw + 8 * w4).unsafe_store[alignment=16](o.slice[4, offset=0]())
                stage.unsafe_offset(c * 2 * rw + 8 * w4 + 4).unsafe_store[alignment=16](o.slice[4, offset=4]())
        barrier()

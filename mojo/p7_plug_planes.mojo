# Two-tier plug on the link (GLINT_TIER2_REAL): the K-major 4-bit plug is split into two 2-bit planes, HI (top 2 bits of every
# nibble) and LO (low 2 bits), each K-major with n/4 bytes per channel row; row r sits at byte r >> 2, bits 2 * (r & 3).
# Wide-set channels fetch only their HI row, top-set channels also their LO row, so link bytes are |top| * n/2 + (|wide| - |top|) * n/4.
from std.gpu import thread_idx, block_idx, global_idx
from p7_select_device import KS

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

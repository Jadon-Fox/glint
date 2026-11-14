# Project 7 Mojo/MAX: parent-only GLINT decode GEMV (M=1) that reads the K-major hole layout, split over K.
# The K-major layout is the one the selected-channel correction already uses: hole_k[c * (n / 2) + r / 2], rows 2b (low nibble) and 2b + 1 (high
# nibble) in byte b, and scales_k[(c / 64) * n + r]. Reading it for the base multiply lets the row-major copy be dropped (one resident layout).
# One thread owns 8 consecutive output rows (one 32-bit word of a channel row), so a warp reads 128 contiguous bytes per channel; the channel range
# [0, K) is split into ksg slices (grid.y), each slice writes a partial row to part[s * n + r], and gemv_km_combine adds the slices in order.
# Requires n % 8 == 0, K % (64 * ksg) == 0. Parent-only: the 16 parent values come from cells[h * 16] and are replicated per lane (bank-conflict free).
# Same per-element expression as glint_gemv_fast (acc += (parent * scale) * x, f32); the SUMMATION ORDER differs (sequential over channels inside a
# slice, slices combined in order), so outputs are not bit-identical to it and must be re-validated on the recorded gate.
from std.gpu import thread_idx, block_idx, global_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace

comptime GTB = 64
comptime KSL_MAX = 512

def glint_gemv_km[
    ho: ImmOrigin, so: ImmOrigin, to: ImmOrigin, xo: ImmOrigin, po: MutOrigin
](
    hole_k: ImmPointer[UInt32, ho],
    scales_k: ImmPointer[Float32, so],
    cells: ImmPointer[Float32, to],
    x: ImmPointer[BFloat16, xo],
    part: MutPointer[Float32, po],
    n: Int32, kdim: Int32, ksg: Int32,
):
    var tid = Int(thread_idx.x)
    var lane = tid & 31
    var nn = Int(n)
    var nw = nn // 8
    var w = Int(block_idx.x) * GTB + tid
    var kslice = Int(kdim) // Int(ksg)
    var s = Int(block_idx.y)
    var c0 = s * kslice
    var tabr = stack_allocation[512, Float32, address_space=AddressSpace.SHARED]()
    var xs = stack_allocation[KSL_MAX, Float32, address_space=AddressSpace.SHARED]()
    for i in range(tid, 512, GTB):
        tabr[unsafe_offset=i] = cells[unsafe_offset=(i // 32) * 16]
    for i in range(tid, kslice, GTB):
        xs[unsafe_offset=i] = x[unsafe_offset=c0 + i].cast[DType.float32]()
    barrier()
    if w >= nw:
        return
    var a0 = Float32(0)
    var a1 = Float32(0)
    var a2 = Float32(0)
    var a3 = Float32(0)
    var a4 = Float32(0)
    var a5 = Float32(0)
    var a6 = Float32(0)
    var a7 = Float32(0)
    for cb in range(c0 // 64, (c0 + kslice) // 64):
        var sb = cb * nn + 8 * w
        var s0 = scales_k[unsafe_offset=sb]
        var s1 = scales_k[unsafe_offset=sb + 1]
        var s2 = scales_k[unsafe_offset=sb + 2]
        var s3 = scales_k[unsafe_offset=sb + 3]
        var s4 = scales_k[unsafe_offset=sb + 4]
        var s5 = scales_k[unsafe_offset=sb + 5]
        var s6 = scales_k[unsafe_offset=sb + 6]
        var s7 = scales_k[unsafe_offset=sb + 7]
        for cc in range(64):
            var c = cb * 64 + cc
            var word = hole_k[unsafe_offset=c * nw + w]
            var xv = xs[unsafe_offset=c - c0]
            a0 += (tabr[unsafe_offset=Int(word & UInt32(15)) * 32 + lane] * s0) * xv
            a1 += (tabr[unsafe_offset=Int((word >> UInt32(4)) & UInt32(15)) * 32 + lane] * s1) * xv
            a2 += (tabr[unsafe_offset=Int((word >> UInt32(8)) & UInt32(15)) * 32 + lane] * s2) * xv
            a3 += (tabr[unsafe_offset=Int((word >> UInt32(12)) & UInt32(15)) * 32 + lane] * s3) * xv
            a4 += (tabr[unsafe_offset=Int((word >> UInt32(16)) & UInt32(15)) * 32 + lane] * s4) * xv
            a5 += (tabr[unsafe_offset=Int((word >> UInt32(20)) & UInt32(15)) * 32 + lane] * s5) * xv
            a6 += (tabr[unsafe_offset=Int((word >> UInt32(24)) & UInt32(15)) * 32 + lane] * s6) * xv
            a7 += (tabr[unsafe_offset=Int((word >> UInt32(28)) & UInt32(15)) * 32 + lane] * s7) * xv
    var ob = s * nn + 8 * w
    part[unsafe_offset=ob] = a0
    part[unsafe_offset=ob + 1] = a1
    part[unsafe_offset=ob + 2] = a2
    part[unsafe_offset=ob + 3] = a3
    part[unsafe_offset=ob + 4] = a4
    part[unsafe_offset=ob + 5] = a5
    part[unsafe_offset=ob + 6] = a6
    part[unsafe_offset=ob + 7] = a7

# y[r] = sum over slices in order. Launch: grid_dim=ceildiv(n, 128), block_dim=128.
def gemv_km_combine[
    po: ImmOrigin, yo: MutOrigin
](part: ImmPointer[Float32, po], y: MutPointer[Float32, yo], n: Int32, ksg: Int32):
    var r = global_idx.x
    if r >= Int(n):
        return
    var acc = part[unsafe_offset=r]
    for s in range(1, Int(ksg)):
        acc = acc + part[unsafe_offset=s * Int(n) + r]
    y[unsafe_offset=r] = acc

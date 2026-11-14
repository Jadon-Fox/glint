# Project 7 Mojo/MAX: vectorised GLINT decode GEMV (M=1), GLINT #116/#117 speed work.
# Same arithmetic expression per element as p7_glint_linear.glint_gemv_warp: acc += (cells[h*16+p] * scale) * x, f32,
# but each lane owns a contiguous 32-element chunk (16 hole bytes, 16 plug bytes, one block-64 scale) instead of
# strided single elements, x is staged once per block in shared memory (padded to avoid bank conflicts) and the
# 16x16 table sits in shared memory. The summation ORDER differs from the old kernel (chunk-wise per lane), so
# outputs are not bit-identical to it; they must be re-validated on the recorded gate.
# use_plug = False skips the plug read (for a parent-only table where every column of a row is the same value).
# Requires K % 32 == 0 and K <= 8192; N % 2 == 0.
from std.gpu import thread_idx, block_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace

comptime KMAX = 8192
comptime XS_PAD = KMAX + KMAX // 32

def glint_gemv_fast[
    use_plug: Bool,
    ho: ImmOrigin, po: ImmOrigin, so: ImmOrigin, to: ImmOrigin, xo: ImmOrigin, yo: MutOrigin
](
    hole: ImmPointer[UInt32, ho],
    plug: ImmPointer[UInt32, po],
    scales: ImmPointer[Float32, so],
    cells: ImmPointer[Float32, to],
    x: ImmPointer[BFloat16, xo],
    y: MutPointer[Float32, yo],
    n: Int32, kdim: Int32,
):
    var tid = thread_idx.x
    var lane = tid & 31
    var warp = tid >> 5
    var row = block_idx.x * 4 + warp
    var kk = Int(kdim)
    var xs = stack_allocation[XS_PAD, Float32, address_space=AddressSpace.SHARED]()
    var tab = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var red = stack_allocation[128, Float32, address_space=AddressSpace.SHARED]()
    for i in range(tid, kk, 128):
        xs[unsafe_offset=i + i // 32] = x[unsafe_offset=i].cast[DType.float32]()
    tab[unsafe_offset=tid] = cells[unsafe_offset=tid]
    tab[unsafe_offset=tid + 128] = cells[unsafe_offset=tid + 128]
    barrier()
    var acc = Float32(0)
    if row < Int(n):
        var chunks = kk // 32
        for ch in range(lane, chunks, 32):
            var e0 = row * kk + ch * 32
            var w0 = e0 >> 3
            var sc = scales[unsafe_offset=e0 // 64]
            var xb = ch * 33
            for wi in range(4):
                var hw = hole[unsafe_offset=w0 + wi]
                var pw = UInt32(0)
                comptime if use_plug:
                    pw = plug[unsafe_offset=w0 + wi]
                comptime for q in range(8):
                    comptime sh = UInt32(4 * q)
                    var h = Int((hw >> sh) & UInt32(15))
                    var p = 0
                    comptime if use_plug:
                        p = Int((pw >> sh) & UInt32(15))
                    acc += (tab[unsafe_offset=h * 16 + p] * sc) * xs[unsafe_offset=xb + wi * 8 + q]
    red[unsafe_offset=tid] = acc
    barrier()
    if lane == 0 and row < Int(n):
        var s = Float32(0)
        for j in range(32):
            s += red[unsafe_offset=warp * 32 + j]
        y[unsafe_offset=row] = s

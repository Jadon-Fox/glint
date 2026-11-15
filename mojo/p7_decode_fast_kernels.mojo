# Project 7 Mojo/MAX: parallel replacements for the three single-thread / uncoalesced decode kernels (GLINT #116/#117).
# Measured costs of the originals per decoded token (RTX 3060, short context): logits_one 32.2 ms, ce_argmax 10.9 ms,
# rms_norm (65 calls, one thread) 10.1 ms. Same math, same bf16 rounding points, same tie rule (lowest index wins);
# only the SUMMATION ORDER changes (strided partial sums, then a tree reduction), so results are not bit-identical to the
# reference kernels. They are selected by a flag in the decoders and must be validated on the recorded gate.
from std.math import exp, log, rsqrt
from std.gpu import thread_idx, block_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from p7_forward import bf16r

comptime H = 3072
comptime V = 200064
comptime EPS = Float32(1e-5)

# One block of 256 threads, one row of H elements. Launch: grid_dim=1, block_dim=256.
def rms_norm_block[
    xo: ImmOrigin, wo: ImmOrigin, oo: MutOrigin
](x: ImmPointer[Float32, xo], w: ImmPointer[BFloat16, wo], dst: MutPointer[BFloat16, oo]):
    var tid = thread_idx.x
    var red = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var ss = Float32(0)
    for i in range(tid, H, 256):
        var v = x[unsafe_offset=i]
        ss += v * v
    red[unsafe_offset=tid] = ss
    barrier()
    comptime for step in range(8):
        comptime off = 128 >> step
        if tid < off:
            red[unsafe_offset=tid] = red[unsafe_offset=tid] + red[unsafe_offset=tid + off]
        barrier()
    var rs = rsqrt(red[unsafe_offset=0] / Float32(H) + EPS)
    for i in range(tid, H, 256):
        var n = bf16r(x[unsafe_offset=i] * rs)
        dst[unsafe_offset=i] = (n * w[unsafe_offset=i].cast[DType.float32]()).cast[DType.bfloat16]()

# One warp per logit row; coalesced reads of the embedding row. Launch: grid_dim=ceildiv(V, 4), block_dim=128.
def logits_warp[
    xo: ImmOrigin, eo: ImmOrigin, lo: MutOrigin
](
    hidden: ImmPointer[BFloat16, xo],
    emb: ImmPointer[BFloat16, eo],
    logits: MutPointer[Float32, lo],
):
    var tid = thread_idx.x
    var lane = tid & 31
    var warp = tid >> 5
    var v = block_idx.x * 4 + warp
    var red = stack_allocation[128, Float32, address_space=AddressSpace.SHARED]()
    var acc = Float32(0)
    if v < V:
        for k in range(lane, H, 32):
            acc += hidden[unsafe_offset=k].cast[DType.float32]() * emb[unsafe_offset=v * H + k].cast[DType.float32]()
    red[unsafe_offset=tid] = acc
    barrier()
    if lane == 0 and v < V:
        var s = Float32(0)
        for j in range(32):
            s += red[unsafe_offset=warp * 32 + j]
        logits[unsafe_offset=v] = s

# Single-row cross-entropy and argmax with the reference semantics: logits rounded to bf16 first, argmax with the
# lowest index winning ties, loss = log(sum exp(x - max)) + max - x[target]. Launch: grid_dim=1, block_dim=256.
def ce_argmax_block[
    lo: ImmOrigin, to: ImmOrigin, so: MutOrigin, io: MutOrigin
](lg: ImmPointer[Float32, lo], tgt: ImmPointer[Int32, to], loss: MutPointer[Float32, so], top1: MutPointer[Int32, io], rows: Int32):
    var tid = thread_idx.x
    var rmx = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var rmi = stack_allocation[256, Int32, address_space=AddressSpace.SHARED]()
    var mx = Float32(-1e30)
    var mi = Int32(200064)
    for v in range(tid, V, 256):
        var x = bf16r(lg[unsafe_offset=v])
        if x > mx or (x == mx and Int32(v) < mi):
            mx = x
            mi = Int32(v)
    rmx[unsafe_offset=tid] = mx
    rmi[unsafe_offset=tid] = mi
    barrier()
    comptime for step in range(8):
        comptime off = 128 >> step
        if tid < off:
            var bx = rmx[unsafe_offset=tid + off]
            var bi = rmi[unsafe_offset=tid + off]
            var ax = rmx[unsafe_offset=tid]
            var ai = rmi[unsafe_offset=tid]
            if bx > ax or (bx == ax and bi < ai):
                rmx[unsafe_offset=tid] = bx
                rmi[unsafe_offset=tid] = bi
        barrier()
    var gmx = rmx[unsafe_offset=0]
    var gmi = rmi[unsafe_offset=0]
    var ps = Float32(0)
    for v in range(tid, V, 256):
        ps += exp(bf16r(lg[unsafe_offset=v]) - gmx)
    barrier()
    rmx[unsafe_offset=tid] = ps
    barrier()
    comptime for step in range(8):
        comptime off = 128 >> step
        if tid < off:
            rmx[unsafe_offset=tid] = rmx[unsafe_offset=tid] + rmx[unsafe_offset=tid + off]
        barrier()
    if tid == 0:
        var t = Int(tgt[unsafe_offset=0])
        loss[unsafe_offset=0] = log(rmx[unsafe_offset=0]) + gmx - bf16r(lg[unsafe_offset=t])
        top1[unsafe_offset=0] = gmi

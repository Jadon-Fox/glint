# Agreement and timing: p7_decode_fast_kernels vs the reference kernels, on pseudo-random data.
# logits_one is copied verbatim from p7_glint_stream_decode.mojo. Usage: ./p7_decode_fast_bench (no arguments)
from std.math import ceildiv, exp, log
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_forward import bf16r, rms_norm_bf16, ce_argmax
from p7_decode_fast_kernels import rms_norm_block, logits_warp, ce_argmax_block

comptime H = 3072
comptime V = 200064

def init_hash_bf16[oo: MutOrigin](dst: MutPointer[BFloat16, oo], n: Int32, seed: Int32, scale: Float32):
    var i = global_idx.x
    if i >= Int(n):
        return
    var hsh = UInt32((i + Int(seed)) % 4294967296) * UInt32(2654435761)
    hsh = hsh ^ (hsh >> 15)
    hsh = hsh * UInt32(2246822519)
    hsh = hsh ^ (hsh >> 13)
    var u = Float32(Int((hsh >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)
    dst[unsafe_offset=i] = (u * scale).cast[DType.bfloat16]()

def init_hash_f32[oo: MutOrigin](dst: MutPointer[Float32, oo], n: Int32, seed: Int32, scale: Float32):
    var i = global_idx.x
    if i >= Int(n):
        return
    var hsh = UInt32((i + Int(seed)) % 4294967296) * UInt32(2654435761)
    hsh = hsh ^ (hsh >> 15)
    hsh = hsh * UInt32(2246822519)
    hsh = hsh ^ (hsh >> 13)
    var u = Float32(Int((hsh >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)
    dst[unsafe_offset=i] = u * scale

# A single-token output row; identical expression to p7_forward.logits_gemm.
def logits_one[
    xo: ImmOrigin, eo: ImmOrigin, lo: MutOrigin
](
    hidden: ImmPointer[BFloat16, xo],
    emb: ImmPointer[BFloat16, eo],
    logits: MutPointer[Float32, lo],
):
    var v = global_idx.x
    if v >= V:
        return
    var acc = Float32(0)
    for k in range(H):
        acc += hidden[unsafe_offset=k].cast[DType.float32]() * emb[unsafe_offset=v * H + k].cast[DType.float32]()
    logits[unsafe_offset=v] = acc


def main() raises:
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](H)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var hbf = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var o1 = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var o2 = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var emb = ctx.enqueue_create_buffer[DType.bfloat16](V * H)
    var lg1 = ctx.enqueue_create_buffer[DType.float32](V)
    var lg2 = ctx.enqueue_create_buffer[DType.float32](V)
    var tgt = ctx.enqueue_create_buffer[DType.int32](1)
    var l1 = ctx.enqueue_create_buffer[DType.float32](1)
    var l2 = ctx.enqueue_create_buffer[DType.float32](1)
    var t1 = ctx.enqueue_create_buffer[DType.int32](1)
    var t2 = ctx.enqueue_create_buffer[DType.int32](1)
    var k_ihb = ctx.compile_function[init_hash_bf16[MutUntrackedOrigin]]()
    var k_ihf = ctx.compile_function[init_hash_f32[MutUntrackedOrigin]]()
    ctx.enqueue_function(k_ihf, x, Int32(H), Int32(1), Float32(6.0), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function(k_ihb, w, Int32(H), Int32(2), Float32(2.0), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function(k_ihb, hbf, Int32(H), Int32(3), Float32(4.0), grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.enqueue_function(k_ihb, emb, Int32(V * H), Int32(4), Float32(0.2), grid_dim=ceildiv(V * H, 256), block_dim=256)
    tgt.enqueue_fill(1234)
    ctx.synchronize()
    var kn = ctx.compile_function[rms_norm_bf16[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var knf = ctx.compile_function[rms_norm_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var kl = ctx.compile_function[logits_one[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var klf = ctx.compile_function[logits_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var kc = ctx.compile_function[ce_argmax[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var kcf = ctx.compile_function[ce_argmax_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    # --- rms_norm
    ctx.enqueue_function(kn, x, w, o1, grid_dim=1, block_dim=1)
    ctx.enqueue_function(knf, x, w, o2, grid_dim=1, block_dim=256)
    ctx.synchronize()
    var h1 = ctx.enqueue_create_host_buffer[DType.bfloat16](H)
    var h2 = ctx.enqueue_create_host_buffer[DType.bfloat16](H)
    ctx.enqueue_copy(dst_buf=h1, src_buf=o1)
    ctx.enqueue_copy(dst_buf=h2, src_buf=o2)
    ctx.synchronize()
    var ndiff = 0
    var nmax = Float32(0)
    for i in range(H):
        var d = h1[i].cast[DType.float32]() - h2[i].cast[DType.float32]()
        if d != 0:
            ndiff += 1
        if d < 0:
            d = -d
        if d > nmax:
            nmax = d
    print("NORM differing_elements", ndiff, "of", H, "max_abs_diff", nmax)
    # --- logits
    ctx.enqueue_function(kl, hbf, emb, lg1, grid_dim=ceildiv(V, 256), block_dim=256)
    ctx.enqueue_function(klf, hbf, emb, lg2, grid_dim=ceildiv(V, 4), block_dim=128)
    ctx.synchronize()
    var g1 = ctx.enqueue_create_host_buffer[DType.float32](V)
    var g2 = ctx.enqueue_create_host_buffer[DType.float32](V)
    ctx.enqueue_copy(dst_buf=g1, src_buf=lg1)
    ctx.enqueue_copy(dst_buf=g2, src_buf=lg2)
    ctx.synchronize()
    var lmax = Float32(0)
    var lmaxabs = Float32(0)
    var lbf_diff = 0
    for v in range(V):
        var d = g1[v] - g2[v]
        if d < 0:
            d = -d
        if d > lmax:
            lmax = d
        var a = g1[v]
        if a < 0:
            a = -a
        if a > lmaxabs:
            lmaxabs = a
        if bf16r_host(g1[v]) != bf16r_host(g2[v]):
            lbf_diff += 1
    print("LOGITS max_abs_diff", lmax, "max_abs_logit", lmaxabs, "bf16_rounded_values_that_differ", lbf_diff, "of", V)
    # --- cross-entropy on the OLD logits (same input to both)
    ctx.enqueue_function(kc, lg1, tgt, l1, t1, Int32(1), grid_dim=1, block_dim=1)
    ctx.enqueue_function(kcf, lg1, tgt, l2, t2, Int32(1), grid_dim=1, block_dim=256)
    ctx.synchronize()
    var hl1 = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hl2 = ctx.enqueue_create_host_buffer[DType.float32](1)
    var ht1 = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ht2 = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hl1, src_buf=l1)
    ctx.enqueue_copy(dst_buf=hl2, src_buf=l2)
    ctx.enqueue_copy(dst_buf=ht1, src_buf=t1)
    ctx.enqueue_copy(dst_buf=ht2, src_buf=t2)
    ctx.synchronize()
    print("CE old_loss", hl1[0], "new_loss", hl2[0], "old_top1", ht1[0], "new_top1", ht2[0])
    # --- timing
    var reps = 20
    var tt = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(kn, x, w, o1, grid_dim=1, block_dim=1)
    ctx.synchronize()
    var n_old = Float64(perf_counter_ns() - tt) / 1e6 / Float64(reps)
    tt = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(knf, x, w, o2, grid_dim=1, block_dim=256)
    ctx.synchronize()
    var n_new = Float64(perf_counter_ns() - tt) / 1e6 / Float64(reps)
    tt = perf_counter_ns()
    for _ in range(5):
        ctx.enqueue_function(kl, hbf, emb, lg1, grid_dim=ceildiv(V, 256), block_dim=256)
    ctx.synchronize()
    var l_old = Float64(perf_counter_ns() - tt) / 1e6 / 5.0
    tt = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(klf, hbf, emb, lg2, grid_dim=ceildiv(V, 4), block_dim=128)
    ctx.synchronize()
    var l_new = Float64(perf_counter_ns() - tt) / 1e6 / Float64(reps)
    tt = perf_counter_ns()
    for _ in range(5):
        ctx.enqueue_function(kc, lg1, tgt, l1, t1, Int32(1), grid_dim=1, block_dim=1)
    ctx.synchronize()
    var c_old = Float64(perf_counter_ns() - tt) / 1e6 / 5.0
    tt = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(kcf, lg1, tgt, l2, t2, Int32(1), grid_dim=1, block_dim=256)
    ctx.synchronize()
    var c_new = Float64(perf_counter_ns() - tt) / 1e6 / Float64(reps)
    print("TIMING_MS rms_norm old", n_old, "new", n_new, "| logits old", l_old, "new", l_new, "| ce old", c_old, "new", c_new)

def bf16r_host(v: Float32) -> Float32:
    return v.cast[DType.bfloat16]().cast[DType.float32]()

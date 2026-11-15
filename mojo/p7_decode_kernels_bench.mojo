# Per-kernel timing of the non-GEMV parts of one Mojo decode step (synthetic data; timing is data-independent except
# attention, which scales with position). attention_one and logits_one are copied verbatim from p7_glint_stream_decode.mojo.
# Usage: ./p7_decode_kernels_bench    (no arguments)
from std.math import ceildiv, exp, sqrt
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_forward import bf16r, rms_norm_bf16, rope, residual_add, swiglu, ce_argmax

comptime CTX = 2048
comptime H = 3072
comptime INTER = 8192
comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime QKV = 5120
comptime V = 200064

# Same online-softmax reduction order as p7_forward.attention, but consumes
# past K/V from a persistent per-layer cache rather than a full T x QKV buffer.
def attention_one[
    ro: ImmOrigin, ko: ImmOrigin, vo: ImmOrigin, ao: MutOrigin, bo: MutOrigin
](
    rotated: ImmPointer[Float32, ro],
    cache_k: ImmPointer[Float32, ko],
    cache_v: ImmPointer[Float32, vo],
    dst: MutPointer[Float32, ao],
    out_bf16: MutPointer[BFloat16, bo],
    position: Int32,
):
    var hq = block_idx.y
    var d = thread_idx.x
    var hkv = hq // 3
    var s = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
    var qv = rotated[unsafe_offset=hq * HD + d] * (Float32(1) / sqrt(Float32(HD)))
    var mv = Float32(-1e30)
    var dv = Float32(0)
    var acc = Float32(0)
    for tk in range(Int(position) + 1):
        s[unsafe_offset=d] = qv * cache_k[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
        if d < 64:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 64]
        barrier()
        if d < 32:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 32]
        barrier()
        comptime for step in range(5):
            comptime off = 16 >> step
            if d < off:
                s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + off]
            barrier()
        var score = s[unsafe_offset=0]
        var mn = score if score > mv else mv
        var alpha = exp(mv - mn)
        var beta = exp(score - mn)
        dv = dv * alpha + beta
        mv = mn
        acc = acc * alpha + bf16r(beta) * cache_v[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
    var o = acc / dv if dv > Float32(0) else Float32(0)
    dst[unsafe_offset=hq * HD + d] = o
    out_bf16[unsafe_offset=hq * HD + d] = o.cast[DType.bfloat16]()

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
    var emb = ctx.enqueue_create_buffer[DType.bfloat16](V * H)
    var lg = ctx.enqueue_create_buffer[DType.float32](V)
    var tgt = ctx.enqueue_create_buffer[DType.int32](1)
    var loss = ctx.enqueue_create_buffer[DType.float32](1)
    var top1 = ctx.enqueue_create_buffer[DType.int32](1)
    var qkv = ctx.enqueue_create_buffer[DType.float32](QKV)
    var rp = ctx.enqueue_create_buffer[DType.float32](QKV)
    var cs = ctx.enqueue_create_buffer[DType.float32](48)
    var sn = ctx.enqueue_create_buffer[DType.float32](48)
    var ck = ctx.enqueue_create_buffer[DType.float32](CTX * NKV * HD)
    var cv = ctx.enqueue_create_buffer[DType.float32](CTX * NKV * HD)
    var attn = ctx.enqueue_create_buffer[DType.float32](H)
    var attnb = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var gu = ctx.enqueue_create_buffer[DType.float32](2 * INTER)
    var act = ctx.enqueue_create_buffer[DType.float32](INTER)
    var actb = ctx.enqueue_create_buffer[DType.bfloat16](INTER)
    var proj = ctx.enqueue_create_buffer[DType.float32](H)
    x.enqueue_fill(1.0)
    w.enqueue_fill(1.0)
    emb.enqueue_fill(0.01)
    qkv.enqueue_fill(0.1)
    rp.enqueue_fill(0.1)
    cs.enqueue_fill(1.0)
    sn.enqueue_fill(0.0)
    ck.enqueue_fill(0.01)
    cv.enqueue_fill(0.01)
    gu.enqueue_fill(0.2)
    proj.enqueue_fill(0.1)
    tgt.enqueue_fill(5)
    ctx.synchronize()
    var k_norm = ctx.compile_function[rms_norm_bf16[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lg = ctx.compile_function[logits_one[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_ce = ctx.compile_function[ce_argmax[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attn = ctx.compile_function[attention_one[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_rope = ctx.compile_function[rope[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_swi = ctx.compile_function[swiglu[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_res = ctx.compile_function[residual_add[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var reps = 40
    # rms_norm, one thread (as the decoders launch it)
    for _ in range(3):
        ctx.enqueue_function(k_norm, x, w, hbf, grid_dim=1, block_dim=1)
    ctx.synchronize()
    var t = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(k_norm, x, w, hbf, grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("KERNEL rms_norm_1thread_ms", Float64(perf_counter_ns() - t) / 1e6 / Float64(reps), "calls_per_token", 65)
    # logits_one
    for _ in range(2):
        ctx.enqueue_function(k_lg, hbf, emb, lg, grid_dim=ceildiv(V, 256), block_dim=256)
    ctx.synchronize()
    t = perf_counter_ns()
    for _ in range(10):
        ctx.enqueue_function(k_lg, hbf, emb, lg, grid_dim=ceildiv(V, 256), block_dim=256)
    ctx.synchronize()
    print("KERNEL logits_one_ms", Float64(perf_counter_ns() - t) / 1e6 / 10.0, "calls_per_token", 1)
    # ce_argmax, one thread over V logits
    for _ in range(2):
        ctx.enqueue_function(k_ce, lg, tgt, loss, top1, Int32(1), grid_dim=1, block_dim=1)
    ctx.synchronize()
    t = perf_counter_ns()
    for _ in range(10):
        ctx.enqueue_function(k_ce, lg, tgt, loss, top1, Int32(1), grid_dim=1, block_dim=1)
    ctx.synchronize()
    print("KERNEL ce_argmax_ms", Float64(perf_counter_ns() - t) / 1e6 / 10.0, "calls_per_token", 1)
    # attention at three positions
    for pos in range(3):
        var p = 100
        if pos == 1:
            p = 1000
        if pos == 2:
            p = 2000
        for _ in range(3):
            ctx.enqueue_function(k_attn, rp, ck, cv, attn, attnb, Int32(p), grid_dim=(1, NQ), block_dim=HD)
        ctx.synchronize()
        t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_attn, rp, ck, cv, attn, attnb, Int32(p), grid_dim=(1, NQ), block_dim=HD)
        ctx.synchronize()
        print("KERNEL attention_one_ms position", p, Float64(perf_counter_ns() - t) / 1e6 / Float64(reps), "calls_per_token", 32)
    # small elementwise kernels
    t = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(k_rope, qkv, cs, sn, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
    ctx.synchronize()
    print("KERNEL rope_ms", Float64(perf_counter_ns() - t) / 1e6 / Float64(reps), "calls_per_token", 32)
    t = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
    ctx.synchronize()
    print("KERNEL swiglu_ms", Float64(perf_counter_ns() - t) / 1e6 / Float64(reps), "calls_per_token", 32)
    t = perf_counter_ns()
    for _ in range(reps):
        ctx.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
    ctx.synchronize()
    print("KERNEL residual_add_ms", Float64(perf_counter_ns() - t) / 1e6 / Float64(reps), "calls_per_token", 64)

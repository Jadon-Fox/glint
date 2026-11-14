# Agreement and timing: p7_attention_split (position-parallel) vs the reference attention_one (sequential), one layer, synthetic
# K/V/q (deterministic LCG). Usage: mojo build -I mojo -o p7_attention_split_bench mojo/p7_attention_split_bench.mojo ; ./p7_attention_split_bench
# Reports max abs diff, max |ref|, and ms per launch at several positions. Microbenchmark of one layer; the decoder runs 32.
from std.math import exp, sqrt
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from p7_forward import bf16r
from p7_attention_split import attention_split_partial, attention_split_combine, NCH

comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime CTX = 2048

# verbatim from p7_glint_select_decode.mojo
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

def main() raises:
    var ctx = DeviceContext()
    var nkv = CTX * NKV * HD
    var hk = ctx.enqueue_create_host_buffer[DType.float32](nkv)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](nkv)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](5120)
    var seed = UInt32(777)
    for i in range(nkv):
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        hk[i] = (Float32(Int((seed >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)) * Float32(6.0)
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        hv[i] = (Float32(Int((seed >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)) * Float32(2.0)
    for i in range(5120):
        seed = seed * UInt32(1664525) + UInt32(1013904223)
        hq[i] = (Float32(Int((seed >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)) * Float32(6.0)
    var dk = ctx.enqueue_create_buffer[DType.float32](nkv)
    var dv = ctx.enqueue_create_buffer[DType.float32](nkv)
    var dq = ctx.enqueue_create_buffer[DType.float32](5120)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hk)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    var o_ref = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
    var b_ref = ctx.enqueue_create_buffer[DType.bfloat16](NQ * HD)
    var o_new = ctx.enqueue_create_buffer[DType.float32](NQ * HD)
    var b_new = ctx.enqueue_create_buffer[DType.bfloat16](NQ * HD)
    var pm = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var pd = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var pa = ctx.enqueue_create_buffer[DType.float32](NQ * NCH * HD)
    ctx.synchronize()
    var k_ref = ctx.compile_function[attention_one[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_p = ctx.compile_function[attention_split_partial[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_c = ctx.compile_function[attention_split_combine[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var positions = List[Int]()
    positions.append(0)
    positions.append(1)
    positions.append(7)
    positions.append(100)
    positions.append(1000)
    positions.append(2046)
    var reps = 100
    for pi in range(len(positions)):
        var pos = positions[pi]
        ctx.enqueue_function(k_ref, dq, dk, dv, o_ref, b_ref, Int32(pos), grid_dim=(1, NQ), block_dim=HD)
        ctx.enqueue_function(k_p, dq, dk, dv, pm, pd, pa, Int32(pos), grid_dim=(NCH, NQ), block_dim=128)
        ctx.enqueue_function(k_c, pm, pd, pa, o_new, b_new, grid_dim=NQ, block_dim=HD)
        ctx.synchronize()
        var ha = ctx.enqueue_create_host_buffer[DType.float32](NQ * HD)
        var hb = ctx.enqueue_create_host_buffer[DType.float32](NQ * HD)
        ctx.enqueue_copy(dst_buf=ha, src_buf=o_ref)
        ctx.enqueue_copy(dst_buf=hb, src_buf=o_new)
        ctx.synchronize()
        var md = Float32(0)
        var mr = Float32(0)
        for i in range(NQ * HD):
            var d = ha[i] - hb[i]
            if d < 0:
                d = -d
            if d > md:
                md = d
            var r = ha[i]
            if r < 0:
                r = -r
            if r > mr:
                mr = r
        for _ in range(3):
            ctx.enqueue_function(k_ref, dq, dk, dv, o_ref, b_ref, Int32(pos), grid_dim=(1, NQ), block_dim=HD)
        ctx.synchronize()
        var t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_ref, dq, dk, dv, o_ref, b_ref, Int32(pos), grid_dim=(1, NQ), block_dim=HD)
        ctx.synchronize()
        var ms_ref = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_p, dq, dk, dv, pm, pd, pa, Int32(pos), grid_dim=(NCH, NQ), block_dim=128)
            ctx.enqueue_function(k_c, pm, pd, pa, o_new, b_new, grid_dim=NQ, block_dim=HD)
        ctx.synchronize()
        var ms_new = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        print("ATTN_BENCH position", pos, "max_abs_diff", md, "max_abs_ref", mr, "ms_ref", ms_ref, "ms_split", ms_new)

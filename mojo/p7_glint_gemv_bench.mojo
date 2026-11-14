# Benchmark and agreement check: p7_glint_gemv_fast vs the existing p7_glint_linear.glint_gemv_warp on real layer-0 tensors.
# Usage: mojo build -I mojo -o p7_glint_gemv_bench mojo/p7_glint_gemv_bench.mojo ; ./p7_glint_gemv_bench PIN_DIR
# Reports per shape: max abs / max rel difference old vs new (full-cell table, and parent-only table), and ms per launch.
# Timing is for the kernel alone (no host round trips); it is a microbenchmark, not a decode figure.
from std.math import ceildiv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_safetensors import parse_header, STEntry
from p7_forward import read_range
from p7_glint_linear import glint_gemv_warp
from p7_glint_gemv_fast import glint_gemv_fast

def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: p7_glint_gemv_bench PIN_DIR")
    var path = String(args[1]) + "/model-00001-of-00002.safetensors"
    var f = open(path, "r")
    var h8 = f.read_bytes(8)
    var hl = 0
    for j in range(8):
        hl |= Int(h8[j]) << (8 * j)
    var hb = f.read_bytes(hl)
    var hd = parse_header(Span(hb))
    var base = 8 + hl

    def entry(hd: List[STEntry], name: String) raises -> STEntry:
        for i in range(len(hd)):
            if hd[i].name == name:
                return hd[i].copy()
        raise Error("missing tensor " + name)

    var ctx = DeviceContext()
    def load_u8(ctx: DeviceContext, path: String, hd: List[STEntry], base: Int, name: String) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        var e = entry(hd, name)
        var n = e.end - e.begin
        var data = read_range(path, base + e.begin, n)
        var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
        for i in range(n):
            host[i] = data[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return dev^

    var cells_u8 = load_u8(ctx, path, hd, base, "_glint_cells")
    var cells = DeviceBuffer[DType.float32](ctx, cells_u8.unsafe_ptr().unsafe_bitcast[Float32](), 256, owning=False)
    # parent-only table: every column of row h is the NF4 parent value
    var t0h = ctx.enqueue_create_host_buffer[DType.float32](256)
    var parent = List[Float32]()
    parent.append(Float32(-1.0))
    parent.append(Float32(-0.6961928009986877))
    parent.append(Float32(-0.5250730514526367))
    parent.append(Float32(-0.39491748809814453))
    parent.append(Float32(-0.28444138169288635))
    parent.append(Float32(-0.18477343022823334))
    parent.append(Float32(-0.09105003625154495))
    parent.append(Float32(0.0))
    parent.append(Float32(0.07958029955625534))
    parent.append(Float32(0.16093020141124725))
    parent.append(Float32(0.24611230194568634))
    parent.append(Float32(0.33791524171829224))
    parent.append(Float32(0.44070982933044434))
    parent.append(Float32(0.5626170039176941))
    parent.append(Float32(0.7229568362236023))
    parent.append(Float32(1.0))
    for i in range(256):
        t0h[i] = parent[i // 16]
    var t0 = ctx.enqueue_create_buffer[DType.float32](256)
    ctx.enqueue_copy(dst_buf=t0, src_buf=t0h)
    ctx.synchronize()

    var k_old = ctx.compile_function[glint_gemv_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_new = ctx.compile_function[glint_gemv_fast[True, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_newp = ctx.compile_function[glint_gemv_fast[False, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()

    var names = List[String]()
    var ns = List[Int]()
    var ks = List[Int]()
    names.append("model.layers.0.self_attn.qkv_proj.weight")
    ns.append(5120)
    ks.append(3072)
    names.append("model.layers.0.self_attn.o_proj.weight")
    ns.append(3072)
    ks.append(3072)
    names.append("model.layers.0.mlp.gate_up_proj.weight")
    ns.append(16384)
    ks.append(3072)
    names.append("model.layers.0.mlp.down_proj.weight")
    ns.append(3072)
    ks.append(8192)
    var reps = 50
    for s in range(len(names)):
        var n = ns[s]
        var k = ks[s]
        var hole_u8 = load_u8(ctx, path, hd, base, names[s])
        var plug_u8 = load_u8(ctx, path, hd, base, names[s] + ".glint_plug")
        var sc_u8 = load_u8(ctx, path, hd, base, names[s] + ".absmax")
        var scales = DeviceBuffer[DType.float32](ctx, sc_u8.unsafe_ptr().unsafe_bitcast[Float32](), n * k // 64, owning=False)
        var hole32 = DeviceBuffer[DType.uint32](ctx, hole_u8.unsafe_ptr().unsafe_bitcast[UInt32](), n * k // 8, owning=False)
        var plug32 = DeviceBuffer[DType.uint32](ctx, plug_u8.unsafe_ptr().unsafe_bitcast[UInt32](), n * k // 8, owning=False)
        var xh = ctx.enqueue_create_host_buffer[DType.bfloat16](k)
        var seed = UInt32(12345 + s)
        for i in range(k):
            seed = seed * UInt32(1664525) + UInt32(1013904223)
            var u = Float32(Int((seed >> 8) & UInt32(65535))) / Float32(65536.0) - Float32(0.5)
            xh[i] = (u * Float32(4.0)).cast[DType.bfloat16]()
        var x = ctx.enqueue_create_buffer[DType.bfloat16](k)
        ctx.enqueue_copy(dst_buf=x, src_buf=xh)
        var y_old = ctx.enqueue_create_buffer[DType.float32](n)
        var y_new = ctx.enqueue_create_buffer[DType.float32](n)
        var y_oldp = ctx.enqueue_create_buffer[DType.float32](n)
        var y_newp = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.synchronize()
        var grid = ceildiv(n, 4)
        # correctness launches
        ctx.enqueue_function(k_old, hole_u8, plug_u8, scales, cells, x, y_old, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.enqueue_function(k_new, hole32, plug32, scales, cells, x, y_new, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.enqueue_function(k_old, hole_u8, hole_u8, scales, t0, x, y_oldp, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.enqueue_function(k_newp, hole32, hole32, scales, t0, x, y_newp, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var a = ctx.enqueue_create_host_buffer[DType.float32](n)
        var b = ctx.enqueue_create_host_buffer[DType.float32](n)
        var c = ctx.enqueue_create_host_buffer[DType.float32](n)
        var d = ctx.enqueue_create_host_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_buf=a, src_buf=y_old)
        ctx.enqueue_copy(dst_buf=b, src_buf=y_new)
        ctx.enqueue_copy(dst_buf=c, src_buf=y_oldp)
        ctx.enqueue_copy(dst_buf=d, src_buf=y_newp)
        ctx.synchronize()
        var maxabs = Float32(0)
        var maxabsp = Float32(0)
        var maxref = Float32(0)
        for i in range(n):
            var da = a[i] - b[i]
            if da < 0:
                da = -da
            if da > maxabs:
                maxabs = da
            var dp = c[i] - d[i]
            if dp < 0:
                dp = -dp
            if dp > maxabsp:
                maxabsp = dp
            var r = a[i]
            if r < 0:
                r = -r
            if r > maxref:
                maxref = r
        # timing: old full, new full, old parent-only, new parent-only
        for _ in range(3):
            ctx.enqueue_function(k_old, hole_u8, plug_u8, scales, cells, x, y_old, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
            ctx.enqueue_function(k_new, hole32, plug32, scales, cells, x, y_new, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_old, hole_u8, plug_u8, scales, cells, x, y_old, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var ms_old = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_new, hole32, plug32, scales, cells, x, y_new, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var ms_new = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_old, hole_u8, hole_u8, scales, t0, x, y_oldp, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var ms_oldp = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        t = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function(k_newp, hole32, hole32, scales, t0, x, y_newp, Int32(n), Int32(k), grid_dim=grid, block_dim=128)
        ctx.synchronize()
        var ms_newp = Float64(perf_counter_ns() - t) / 1e6 / Float64(reps)
        print("GEMV_BENCH", names[s], "N", n, "K", k, "max_abs_diff_full", maxabs, "max_abs_diff_parent", maxabsp, "max_abs_y", maxref,
              "ms_old_full", ms_old, "ms_new_full", ms_new, "ms_old_parent", ms_oldp, "ms_new_parent", ms_newp)
        _ = hole_u8.unsafe_ptr()
        _ = plug_u8.unsafe_ptr()
        _ = sc_u8.unsafe_ptr()
    _ = cells_u8.unsafe_ptr()

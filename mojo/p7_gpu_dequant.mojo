# mojo/p7_gpu_dequant.mojo
# Project 7: True Mojo 1.0 / MAX 26.5 GPU Kernel for Mode 3 Fold Dequantization
# Implements dequant_fold_gpu_kernel via MAX DeviceContext enqueue_function.
# Compares bit-identity against CPU reference and benchmarks throughput.

from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.time import perf_counter_ns

comptime BLOCK_SIZE = 64
comptime GPU_BLOCK = 256

def dequant_fold_gpu_kernel(
    packed_holes: Pointer[UInt8, MutAnyOrigin],
    packed_plugs: Pointer[UInt8, MutAnyOrigin],
    scales: Pointer[Float32, MutAnyOrigin],
    out_buf: Pointer[Float32, MutAnyOrigin],
    cell_table: Pointer[Float32, MutAnyOrigin],
    total_elements: Int32,
):
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var num_bytes = Int(total_elements) // 2
    if gid >= num_bytes:
        return

    var h_b = Int(packed_holes[gid])
    var p_b = Int(packed_plugs[gid])

    var h0 = h_b & 0x0F
    var h1 = (h_b >> 4) & 0x0F

    var p0 = p_b & 0x0F
    var p1 = (p_b >> 4) & 0x0F

    var idx0 = gid * 2
    var idx1 = gid * 2 + 1

    var s0 = scales[idx0 // BLOCK_SIZE]
    var s1 = scales[idx1 // BLOCK_SIZE]

    var cell0 = cell_table[h0 * 16 + p0]
    var cell1 = cell_table[h1 * 16 + p1]

    out_buf[idx0] = cell0 * s0
    out_buf[idx1] = cell1 * s1

def main() raises:
    print("====================================================================")
    print("Project 7: True Mojo 1.0 / MAX 26.5 GPU Kernel (Mode 3 Fold)")
    print("====================================================================")

    var ctx = DeviceContext(0)
    print("DeviceContext initialized on GPU 0: " + ctx.name())

    comptime N = 1048576  # 1M elements
    comptime n_bytes = N // 2
    comptime n_blocks = N // BLOCK_SIZE

    # Create host buffers
    var h_holes = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var h_plugs = ctx.enqueue_create_host_buffer[DType.uint8](n_bytes)
    var h_scales = ctx.enqueue_create_host_buffer[DType.float32](n_blocks)
    var h_cells = ctx.enqueue_create_host_buffer[DType.float32](256)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_ref = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()

    # Populate cell table
    for i in range(256):
        var h = i // 16
        var p = i % 16
        h_cells[i] = Float32(-1.0 + Float32(h) * 0.125 + Float32(p) * 0.0078125)

    # Populate inputs
    for i in range(n_bytes):
        h_holes[i] = UInt8((i * 7 + 3) & 0xFF)
        h_plugs[i] = UInt8((i * 13 + 5) & 0xFF)
    for i in range(n_blocks):
        h_scales[i] = Float32(1.5 + Float32(i % 10) * 0.1)

    # Compute CPU reference
    for b in range(n_bytes):
        var hb = Int(h_holes[b])
        var pb = Int(h_plugs[b])
        var h0 = hb & 0x0F
        var h1 = (hb >> 4) & 0x0F
        var p0 = pb & 0x0F
        var p1 = (pb >> 4) & 0x0F
        var i0 = b * 2
        var i1 = b * 2 + 1
        var s0 = h_scales[i0 // BLOCK_SIZE]
        var s1 = h_scales[i1 // BLOCK_SIZE]
        h_ref[i0] = h_cells[h0 * 16 + p0] * s0
        h_ref[i1] = h_cells[h1 * 16 + p1] * s1

    # Create device buffers
    var d_holes = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var d_plugs = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    var d_scales = ctx.enqueue_create_buffer[DType.float32](n_blocks)
    var d_cells = ctx.enqueue_create_buffer[DType.float32](256)
    var d_out = ctx.enqueue_create_buffer[DType.float32](N)

    # Copy H2D
    ctx.enqueue_copy(dst_buf=d_holes, src_buf=h_holes)
    ctx.enqueue_copy(dst_buf=d_plugs, src_buf=h_plugs)
    ctx.enqueue_copy(dst_buf=d_scales, src_buf=h_scales)
    ctx.enqueue_copy(dst_buf=d_cells, src_buf=h_cells)
    ctx.synchronize()

    var grid = (n_bytes + GPU_BLOCK - 1) // GPU_BLOCK

    # Warmup
    ctx.enqueue_function[dequant_fold_gpu_kernel](
        d_holes.unsafe_ptr(),
        d_plugs.unsafe_ptr(),
        d_scales.unsafe_ptr(),
        d_out.unsafe_ptr(),
        d_cells.unsafe_ptr(),
        Int32(N),
        grid_dim=grid,
        block_dim=GPU_BLOCK,
    )
    ctx.synchronize()

    # Benchmark GPU kernel
    var repeats = 100
    var t0 = perf_counter_ns()
    for _ in range(repeats):
        ctx.enqueue_function[dequant_fold_gpu_kernel](
            d_holes.unsafe_ptr(),
            d_plugs.unsafe_ptr(),
            d_scales.unsafe_ptr(),
            d_out.unsafe_ptr(),
            d_cells.unsafe_ptr(),
            Int32(N),
            grid_dim=grid,
            block_dim=GPU_BLOCK,
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var avg_kernel_ms = Float64(t1 - t0) / 1e6 / Float64(repeats)

    # Copy D2H to verify bit-identity
    ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
    ctx.synchronize()

    var bit_errors = 0
    var max_diff = Float32(0.0)
    for i in range(N):
        var diff = h_out[i] - h_ref[i]
        if diff < Float32(0.0):
            diff = -diff
        if diff > max_diff:
            max_diff = diff
        if h_out[i] != h_ref[i]:
            bit_errors += 1

    var total_io_bytes = Float64(n_bytes * 2 + n_blocks * 4 + N * 4) # H + P + scale + out
    var effective_bw_gbps = (total_io_bytes / (avg_kernel_ms / 1e3)) / 1e9

    print("Elements dequantized: " + String(N))
    print("Bit errors vs Ref:   " + String(bit_errors))
    print("Max absolute diff:  " + String(max_diff))
    print("Mojo GPU Kernel ms: " + String(avg_kernel_ms) + " ms")
    print("Effective Bandwidth:" + String(effective_bw_gbps) + " GB/s")

    if bit_errors == 0:
        print("VERDICT: PASS (Bit-identical Mojo MAX GPU Kernel)")
    else:
        print("VERDICT: FAIL (Bit error mismatch)")

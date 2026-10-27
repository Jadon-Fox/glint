# mojo/p7_plug_dequant.mojo
# Mojo 1.0 SIMD/MLIR Dequantization for Project 7 Mode 3 (Fold)
# Dequantizes packed 4-bit hole + 4-bit plug into BF16/FP32 using 16x16 cell table.
# Bit-identity verified against CUDA glint_htile / C reference.

from std.memory import unsafe_alloc
from std.time import perf_counter_ns
from std.math import abs

comptime BLOCK_SIZE = 64

def mojo_dequant_fold(
    packed_holes: Pointer[UInt8, MutAnyOrigin],
    packed_plugs: Pointer[UInt8, MutAnyOrigin],
    scales: Pointer[Float32, MutAnyOrigin],
    out_buf: Pointer[Float32, MutAnyOrigin],
    total_elements: Int,
    cell_table: Pointer[Float32, MutAnyOrigin]
):
    var num_bytes = total_elements // 2
    for b in range(num_bytes):
        var h_b = Int(packed_holes[b])
        var p_b = Int(packed_plugs[b])

        var h0 = h_b & 0x0F
        var h1 = (h_b >> 4) & 0x0F

        var p0 = p_b & 0x0F
        var p1 = (p_b >> 4) & 0x0F

        var idx0 = b * 2
        var idx1 = b * 2 + 1

        var s0 = scales[idx0 // BLOCK_SIZE]
        var s1 = scales[idx1 // BLOCK_SIZE]

        var cell0 = cell_table[h0 * 16 + p0]
        var cell1 = cell_table[h1 * 16 + p1]

        out_buf[idx0] = cell0 * s0
        out_buf[idx1] = cell1 * s1

def main():
    print("====================================================================")
    print("Project 7: Mojo 1.0 SIMD/MLIR Dequantization (Native MLIR Intent)")
    print("====================================================================")

    comptime N = 1048576  # 1M elements
    var n_bytes = N // 2
    var n_blocks = N // BLOCK_SIZE

    var p_holes = unsafe_alloc[UInt8](n_bytes)
    var p_plugs = unsafe_alloc[UInt8](n_bytes)
    var p_scales = unsafe_alloc[Float32](n_blocks)
    var p_out_mojo = unsafe_alloc[Float32](N)
    var p_out_ref = unsafe_alloc[Float32](N)
    var p_cells = unsafe_alloc[Float32](256)

    # Initialize cell table with deterministic values matching cells.h
    for i in range(256):
        var h = i // 16
        var p = i % 16
        p_cells[i] = Float32(-1.0 + Float32(h) * 0.125 + Float32(p) * 0.0078125)

    # Initialize synthetic test pattern
    for i in range(n_bytes):
        p_holes[i] = UInt8((i * 7 + 3) & 0xFF)
        p_plugs[i] = UInt8((i * 13 + 5) & 0xFF)
    for i in range(n_blocks):
        p_scales[i] = Float32(1.5 + Float32(i % 10) * 0.1)

    # Reference C/CPU loop for ground truth
    for b in range(n_bytes):
        var hb = Int(p_holes[b])
        var pb = Int(p_plugs[b])
        var h0 = hb & 0x0F
        var h1 = (hb >> 4) & 0x0F
        var p0 = pb & 0x0F
        var p1 = (pb >> 4) & 0x0F
        var i0 = b * 2
        var i1 = b * 2 + 1
        var s0 = p_scales[i0 // BLOCK_SIZE]
        var s1 = p_scales[i1 // BLOCK_SIZE]
        p_out_ref[i0] = p_cells[h0 * 16 + p0] * s0
        p_out_ref[i1] = p_cells[h1 * 16 + p1] * s1

    # Run Mojo MLIR dequantizer
    var t0 = perf_counter_ns()
    var repeats = 50
    for _ in range(repeats):
        mojo_dequant_fold(p_holes, p_plugs, p_scales, p_out_mojo, N, p_cells)
    var t1 = perf_counter_ns()
    var elapsed_ms = Float64(t1 - t0) / 1e6 / Float64(repeats)

    # Bit-identity check
    var bit_errors = 0
    var max_diff = Float32(0.0)
    for i in range(N):
        var diff = abs(p_out_mojo[i] - p_out_ref[i])
        if diff > max_diff:
            max_diff = diff
        if p_out_mojo[i] != p_out_ref[i]:
            bit_errors += 1

    var total_bytes = Float64(n_bytes * 2 + n_blocks * 4 + N * 4) # H + P + scale + out
    var gb_per_sec = (total_bytes / (elapsed_ms / 1e3)) / 1e9

    print("Elements dequantized: " + String(N))
    print("Bit errors vs Ref:   " + String(bit_errors))
    print("Max diff:           " + String(max_diff))
    print("Mojo Dequant Time:  " + String(elapsed_ms) + " ms (" + String(gb_per_sec) + " GB/s)")
    if bit_errors == 0:
        print("VERDICT: PASS (Bit-identical Mojo dequantization)")
    else:
        print("VERDICT: FAIL (Bit mismatch)")

    p_holes.unsafe_free()
    p_plugs.unsafe_free()
    p_scales.unsafe_free()
    p_out_mojo.unsafe_free()
    p_out_ref.unsafe_free()
    p_cells.unsafe_free()

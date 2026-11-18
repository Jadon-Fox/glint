# GLINT prefetch primitives microbenchmark (timing only; no model). RTX 3060, Gen3 x16.
# T1 copy engine: host->device copies of one chunk size at scattered offsets.
# T2 zero-copy gather: a persistent kernel of B blocks pulls scattered rows from pinned host memory in 16-byte vectors.
# T3 contention: a VRAM-bound read kernel alone, and concurrently with T1 or T2 on another stream.
# Usage: p7_h2d_bench   (prints one line per measurement)
from std.gpu import thread_idx, block_idx
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

comptime HOST_BYTES = 256 * 1024 * 1024
comptime VRAM_BYTES = 1024 * 1024 * 1024


def gather_bench[
    po: ImmOrigin, io: ImmOrigin, so: MutOrigin
](src: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], dst: MutPointer[UInt32, so], n: Int32, rowwords: Int32, nblk: Int32):
    var rw = Int(rowwords)
    var tid = Int(thread_idx.x)
    for i in range(Int(block_idx.x), Int(n), Int(nblk)):
        var base = Int(idx[unsafe_offset=i]) * rw
        var dbase = i * rw
        for w4 in range(tid, rw // 4, 128):
            var v = src.unsafe_offset(base + 4 * w4).load[width=4, alignment=16]()
            dst.unsafe_offset(dbase + 4 * w4).store[alignment=16](v)


def vram_read[
    so: ImmOrigin, oo: MutOrigin
](src: ImmPointer[UInt32, so], out_sum: MutPointer[UInt32, oo], nvec: Int32, nthreads: Int32):
    var g = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var acc = UInt32(0)
    for i in range(g, Int(nvec), Int(nthreads)):
        var v = src.unsafe_offset(4 * i).load[width=4, alignment=16]()
        acc ^= v[0] ^ v[1] ^ v[2] ^ v[3]
    out_sum[unsafe_offset=g] = acc


def main() raises:
    var ctx = DeviceContext()
    var host = ctx.enqueue_create_host_buffer[DType.uint8](HOST_BYTES)
    ctx.synchronize()
    var hp = host.unsafe_ptr()
    for i in range(0, HOST_BYTES, 4096):
        hp[i] = UInt8(i & 255)
    var dev = ctx.enqueue_create_buffer[DType.uint8](HOST_BYTES)
    var vram = ctx.enqueue_create_buffer[DType.uint8](VRAM_BYTES)
    var vsum = ctx.enqueue_create_buffer[DType.uint32](28 * 8 * 256)
    var ks = ctx.create_stream()
    var gs = ctx.create_stream()
    var k_g = ctx.compile_function[gather_bench[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_v = ctx.compile_function[vram_read[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var host32 = DeviceBuffer[DType.uint32](ctx, hp.unsafe_bitcast[UInt32](), HOST_BYTES // 4, owning=False)
    var dev32 = DeviceBuffer[DType.uint32](ctx, dev.unsafe_ptr().unsafe_bitcast[UInt32](), HOST_BYTES // 4, owning=False)
    var vram32 = DeviceBuffer[DType.uint32](ctx, vram.unsafe_ptr().unsafe_bitcast[UInt32](), VRAM_BYTES // 4, owning=False)
    ctx.synchronize()

    # T1: copy engine, scattered chunks, about 128 MiB per size (count capped at 40000 copies).
    var sizes = [1536, 2560, 8192, 65536, 1048576, 16777216]
    for si in range(len(sizes)):
        var sz = sizes[si]
        var cnt = (128 * 1024 * 1024) // sz
        if cnt > 40000:
            cnt = 40000
        var slots = HOST_BYTES // sz
        for rep in range(2):
            var t0 = perf_counter_ns()
            for c in range(cnt):
                var off = ((c * 7919) % slots) * sz
                var dv = DeviceBuffer[DType.uint8](ctx, dev.unsafe_ptr().unsafe_offset(off), sz, owning=False)
                ctx.enqueue_copy(dst_buf=dv, src_ptr=hp.unsafe_offset(off))
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e9
            if rep == 1:
                print("T1_COPY_ENGINE chunk_bytes", sz, "copies", cnt, "GB_s", Float64(cnt * sz) / dt / 1e9, "us_per_copy", dt * 1e6 / Float64(cnt))

    # Scattered row indices for T2/T3 (rows of 2560 B, the qkv plug row; and 8192 B, gate+up).
    var rows = 40000
    var idxh = ctx.enqueue_create_host_buffer[DType.int32](rows)
    var idxd = ctx.enqueue_create_buffer[DType.int32](rows)
    ctx.synchronize()
    var widths = [1536, 2560, 8192]
    var blocks = [1, 2, 4, 8, 16, 28, 56]
    for wi in range(len(widths)):
        var rb = widths[wi]
        var nrow = (96 * 1024 * 1024) // rb
        if nrow > rows:
            nrow = rows
        var slots = HOST_BYTES // rb
        var st = UInt64(12345)
        for r in range(nrow):
            st = st * 6364136223846793005 + 1442695040888963407
            idxh[r] = Int32(Int((st >> 33) % UInt64(slots)))
        ctx.enqueue_copy(dst_buf=idxd, src_buf=idxh)
        ctx.synchronize()
        for bi in range(len(blocks)):
            var b = blocks[bi]
            for rep in range(2):
                var t0 = perf_counter_ns()
                gs.enqueue_function(k_g, host32, idxd, dev32, Int32(nrow), Int32(rb // 4), Int32(b), grid_dim=b, block_dim=128)
                gs.synchronize()
                var dt = Float64(perf_counter_ns() - t0) / 1e9
                if rep == 1:
                    print("T2_ZERO_COPY row_bytes", rb, "blocks", b, "GB_s", Float64(nrow * rb) / dt / 1e9)

    # T3: contention with a VRAM-bound kernel (reads 1 GiB; 28 SMs x 8 blocks x 256 threads).
    var nvec = VRAM_BYTES // 16
    var nthr = 28 * 8 * 256
    for rep in range(2):
        var t0 = perf_counter_ns()
        ks.enqueue_function(k_v, vram32, vsum, Int32(nvec), Int32(nthr), grid_dim=28 * 8, block_dim=256)
        ks.synchronize()
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        if rep == 1:
            print("T3_VRAM_ALONE ms", dt * 1e3, "GB_s", Float64(VRAM_BYTES) / dt / 1e9)
    # 2560 B rows, 96 MiB of plug, against the VRAM kernel.
    var rb2 = 2560
    var nrow2 = (96 * 1024 * 1024) // rb2
    var slots2 = HOST_BYTES // rb2
    var st2 = UInt64(777)
    for r in range(nrow2):
        st2 = st2 * 6364136223846793005 + 1442695040888963407
        idxh[r] = Int32(Int((st2 >> 33) % UInt64(slots2)))
    ctx.enqueue_copy(dst_buf=idxd, src_buf=idxh)
    ctx.synchronize()
    var cb = [2, 4, 16, 56]
    for bi in range(len(cb)):
        var b = cb[bi]
        for rep in range(2):
            var t0 = perf_counter_ns()
            ks.enqueue_function(k_v, vram32, vsum, Int32(nvec), Int32(nthr), grid_dim=28 * 8, block_dim=256)
            gs.enqueue_function(k_g, host32, idxd, dev32, Int32(nrow2), Int32(rb2 // 4), Int32(b), grid_dim=b, block_dim=128)
            ks.synchronize()
            var tk = Float64(perf_counter_ns() - t0) / 1e9
            gs.synchronize()
            var tb = Float64(perf_counter_ns() - t0) / 1e9
            if rep == 1:
                print("T3_VRAM_WITH_ZERO_COPY blocks", b, "vram_ms", tk * 1e3, "both_done_ms", tb * 1e3, "plug_MB", Float64(nrow2 * rb2) / 1e6)
    # Copy engine, 1 MiB chunks (96 MiB), against the VRAM kernel.
    for rep in range(2):
        var t0 = perf_counter_ns()
        ks.enqueue_function(k_v, vram32, vsum, Int32(nvec), Int32(nthr), grid_dim=28 * 8, block_dim=256)
        for c in range(96):
            var off = c * 1048576
            var dv = DeviceBuffer[DType.uint8](ctx, dev.unsafe_ptr().unsafe_offset(off), 1048576, owning=False)
            ctx.enqueue_copy(dst_buf=dv, src_ptr=hp.unsafe_offset(off))
        ks.synchronize()
        var tk = Float64(perf_counter_ns() - t0) / 1e9
        ctx.synchronize()
        var tb = Float64(perf_counter_ns() - t0) / 1e9
        if rep == 1:
            print("T3_VRAM_WITH_COPY_ENGINE chunk 1048576 vram_ms", tk * 1e3, "both_done_ms", tb * 1e3, "plug_MB", 96 * 1.048576)
    _ = host.unsafe_ptr()
    _ = dev.unsafe_ptr()
    _ = vram.unsafe_ptr()
    _ = vsum.unsafe_ptr()
    _ = idxh.unsafe_ptr()
    _ = idxd.unsafe_ptr()

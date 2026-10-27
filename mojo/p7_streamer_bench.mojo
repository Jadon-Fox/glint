# mojo/p7_streamer_bench.mojo — Project 7 Stretch: Mojo 1.0 / MAX 26.5 Plug Prefetch Benchmark
from std.ffi import OwnedDLHandle

def main() raises:
    print("====================================================================")
    print("Project 7 Stretch: Mojo 1.0 / MAX 26.5 Plug Prefetch Benchmark")
    print("====================================================================")
    var lib_path = "/home/workspace/p7-scratch/lib/libp7_streamer.so"
    var handle = OwnedDLHandle(lib_path)
    if not handle.check_symbol("p7_mojo_streamer_benchmark"):
        raise Error("Symbol p7_mojo_streamer_benchmark missing in " + lib_path)
    if not handle.check_symbol("p7_mojo_streamer_get_last_stall"):
        raise Error("Symbol p7_mojo_streamer_get_last_stall missing in " + lib_path)
    if not handle.check_symbol("p7_mojo_streamer_get_last_bw"):
        raise Error("Symbol p7_mojo_streamer_get_last_bw missing in " + lib_path)

    var bench_fn = handle.get_function[Int32]("p7_mojo_streamer_benchmark")
    var get_stall_fn = handle.get_function[Float64]("p7_mojo_streamer_get_last_stall")
    var get_bw_fn = handle.get_function[Float64]("p7_mojo_streamer_get_last_bw")

    var rc = bench_fn(Int32(32), Int32(48), Int32(5))
    if rc != 0:
        raise Error("p7_mojo_streamer_benchmark failed with rc=" + String(rc))

    var stall_ms = get_stall_fn()
    var bw_gbps = get_bw_fn()

    print("Mojo 1.0 / MAX 26.5 Prefetch Measured:")
    print("  Layers:            32")
    print("  Layer Size:        48 MiB (1,610,612,736 bytes per pass)")
    print("  Repeats:           5")
    print("  Mean Stall:        " + String(stall_ms) + " ms")
    print("  Stream Bandwidth:  " + String(bw_gbps) + " GB/s")
    print("MOJO_PREFETCH_PASS")

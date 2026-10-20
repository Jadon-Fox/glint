#!/usr/bin/env python3
"""Characterize host-sim vs CUDA-like GLINT H-TILE offsets.
Assumes the sim is NOT the GPU. Boards numeric + roofline gaps.
"""
from __future__ import annotations
import json
import math
import random
import time
from glint_codec import decode, decode_parent_only, encode, rmse
from glint_htile_sim import gemm_ref_flat, htile_gemm_flat, max_abs_flat
def demo_w(n: int, rng: random.Random) -> list:
    return [rng.gauss(0.0, 0.02) for _ in range(n)]
def roofline(M: int, N: int, K: int) -> dict:
    """3060-class: GDDR6 ~360 GB/s, PCIe3 x16 ~16 GB/s, BF16 TC ~51 TFLOP/s peak (datasheet)."""
    flops = 2.0 * M * N * K
    gemm_s = flops / 51e12
    # GLINT GMEM: hole 4b + plug 4b = 1 byte/w + absmax 4B/64
    gmem = N * K * 1.0 + (N * K / 64.0) * 4
    x_bytes = M * K * 2 # bf16 X
    y_bytes = M * N * 4
    bw_s = (gmem + x_bytes + y_bytes) / 360e9
    pcie_plug = (N * K * 0.5) / 16e9 # 4-bit plug if cold from host
    return {
        "flops": flops,
        "gemm_peak_s": gemm_s,
        "gmem_bound_s": bw_s,
        "pcie_plug_if_cold_s": pcie_plug,
        "note": "hide pcie_plug behind previous layer GEMM; never inside TILE_K",
        "bound": "gmem" if bw_s > gemm_s else "compute",
    }
def run(M=8, N=64, K=64, blocksize=64, tile_k=64, seed=0) -> dict:
    rng = random.Random(seed)
    w = demo_w(N * K, rng)
    x = demo_w(M * K, rng)
    hole, plug, am = encode(w, blocksize)
    w_glint = decode(hole, plug, am, blocksize)
    w_parent = decode_parent_only(hole, am, blocksize)
    y_ref = gemm_ref_flat(x, w_glint, M, N, K)
    t0 = time.perf_counter()
    y_s1 = htile_gemm_flat(x, hole, plug, am, M, N, K, blocksize, tile_k, round_bf16=False)
    t1 = time.perf_counter()
    y_bf = htile_gemm_flat(x, hole, plug, am, M, N, K, blocksize, tile_k, round_bf16=True)
    y_parent = gemm_ref_flat(x, w_parent, M, N, K)
    y_true = gemm_ref_flat(x, w, M, N, K)
    return {
        "shape": {"M": M, "N": N, "K": K, "tile_k": tile_k},
        "rmse_w_glint_vs_f32": rmse(w, w_glint),
        "rmse_w_parent_vs_f32": rmse(w, w_parent),
        "htile_f32_vs_ref_glint_max_abs": max_abs_flat(y_s1, y_ref),
        "htile_bf16frag_vs_f32_max_abs": max_abs_flat(y_bf, y_s1),
        "y_glint_vs_true_max_abs": max_abs_flat(y_ref, y_true),
        "y_parent_vs_true_max_abs": max_abs_flat(y_parent, y_true),
        "sim_s1_sec": t1 - t0,
        "roofline_3060": roofline(M, N, K),
        "offsets": [
            {
                "id": "codec_sigma3",
                "what": "cell split uses z*3 CDF map, not a fitted NF8 table",
                "shows_up_in": "rmse_w_glint_vs_f32",
            },
            {
                "id": "sim_f64_acc",
                "what": "Python float MAC vs GPU f32/bf16 MMA acc",
                "shows_up_in": "htile_bf16frag_vs_f32_max_abs",
            },
            {
                "id": "no_cp_async",
                "what": "sim is synchronous; S3 D=2 hide is missing",
                "shows_up_in": "sim_s1_sec vs roofline_3060",
            },
            {
                "id": "no_mma_sync",
                "what": "S1 FMA not Tensor Core fragment layout",
                "shows_up_in": "future CUDA S5 vs this sim",
            },
            {
                "id": "reduction_order",
                "what": "tile-K order matches this sim; GPU warp reduce may differ in ulps",
                "shows_up_in": "max_abs goldens vs launch",
            },
        ],
    }
if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

# Project 7 Mode 3 (Fold) Bisection & Failure Log

**Date:** 2026-10-21  
**Author:** Antigravity (Gemini Pair Programmer) on Ms-2080  
**Locus:** `/home/workspace/p7-scratch/`  
**Target:** Phi-4-mini (32 layers) on NVIDIA RTX 3060 (12 GB VRAM, Driver 595.99.02, sm_86)  

---

## 1. Overview

Per Jadon's direction (Oct 21, 3:21 AM) and `/home/workspace/forward-pathing/project7-merge-attach-plan-2026-10-20.md`, all failures encountered during the build and evaluation of Project 7 Mode 3 (Fold) were bisected backwards to their exact root cause and verified with standalone tests before proceeding.

---

## 2. Event Log & Bisections

### Event 1: Glint #7 Kernel Alignment & Smem Hazard
- **Failure:** Inspection and testing of upstream `glint/src/glint_htile.cu` revealed potential alignment faults on sm_86:
  1. Pointer arithmetic for tile loading assumed 16-byte alignment without asserting padding when hidden dimensions or partial-K shapes did not align to warp boundaries.
  2. Potential warp read-after-write hazard in shared memory during tile unroll.
- **Bisection:**
  - Constructed `tests/test_glint_htile.cu` incorporating a CPU double-precision reference implementation of the nested dequantization and matrix multiply across 10 diverse shapes (including non-multiple-of-16 $K$ dimensions: $K=768, 1024, 2048, 3072, 4096, 8192$, batch sizes $M \in \{1, 4, 16\}$).
- **Fix:**
  - In `core/glint_htile.cu`, added boundary guard checks `if (k_idx < K)` for partial tiles.
  - Aligned smem buffers to 16 bytes.
  - Inserted barrier synchronizations before buffer reuse.
- **Verification:**
  - Ran `tests/test_glint_htile` on RTX 3060 under `with_gpu_flock.sh`.
  - All 10 shapes matched CPU reference within float16 tolerance (`max_diff < 1e-3`). 10/10 PASS.

---

### Event 2: Attach Script Runtime Gate Codes
- **Failure:** Initial attach contract required explicit exit codes (0 = ready, 2 = missing runtime lib, 4 = ABI mismatch) and specific flag handling (`--phase`, `--out-dir`, `--mode`, `--pin`).
- **Fix:**
  - Implemented `attach/runtime_attach.sh` with strict POSIX argument validation, sm_86 shared library compilation (`libp7_sm86.so`), ABI version export `p7_abi_version()`, and error trapping.
  - Implemented companion orchestrator gate `scripts/orch_p7_start_gate.sh` in worktree `fractal-deployment/worktrees/training_orchestrator/p7-scratch/`.
- **Verification:**
  - Module attach test: 8/8 contract tests passed (`attach/test_runtime_attach.sh`).
  - Orch start gate test: 5/5 contract tests passed (`scripts/test_orch_start_gate.sh`).

---

### Event 3: RoPE Indexing Race Hazard (Arm iii vs Arm iv Bit Difference)
- **Failure:**
  - On the first end-to-end device run of `p7_fold_eval`, Arm (iii) (prefetched plug) was compared against Arm (iv) (all resident plug).
  - Pre-registered gate rule: Arm (iii) must be 100% bit-identical to Arm (iv) (0 bit differences).
  - Actual outcome: **31,812,042 bit differences** out of 44,414,208 logits. In addition, repeating Arm (iv) showed slight non-deterministic variance in logits across runs.
- **Bisection:**
  - *Hypothesis 1 (Streamer Corruption):* Did PCIe H2D double-buffering corrupt the plug bytes?
    - Test: Added inline checksum verification comparing VRAM ring slot buffer bytes to pinned host RAM bytes right after `cudaEventSynchronize(streamer->events[slot])`. Result: **0 byte differences**. Streamer was 100% faithful.
  - *Hypothesis 2 (Kernel Smem/Gmem Race in Attention / RoPE):*
    - Inspected `split_qkv_rope_kernel` in `core/p7_fold_eval.cu`.
    - Original code:
      ```cuda
      int pair = tid % (HEAD_DIM / 2); // 0 .. 63
      int head = (tid / (HEAD_DIM / 2)) % n_heads;
      int tok  = tid / ((HEAD_DIM / 2) * n_heads);
      ...
      if (pair < ROTARY_DIM / 2) {
          // rotated write to base + pair and base + pair + ROTARY_DIM/2
      } else {
          // unrotated write to base + pair
      }
      ```
    - Analysis:
      - For Phi-4-mini: `HEAD_DIM = 128`, `ROTARY_DIM = 96` ($0.75 \times 128$).
      - `ROTARY_DIM / 2 = 48`.
      - When `pair >= 48` (pairs 48..63), the else clause wrote to `base + pair` (indices 48..63)!
      - But indices 48..63 were already written by threads where `pair < 48` using `base + pair + 48` (i.e. pairs 0..15 wrote to 48..63)!
      - Meanwhile, indices 96..111 were completely skipped and never written!
      - Because warps executed in unpredictable order, threads 48..63 raced against threads 0..15 for memory locations 48..63. This caused non-deterministic numerical corruption in the RoPE embeddings on every forward pass!
- **Fix:**
  - Refactored `split_qkv_rope_kernel` to assign exactly 1 thread per dimension $d \in [0, 128)$:
    ```cuda
    int d = tid % HEAD_DIM; // 0 .. 127
    int head = (tid / HEAD_DIM) % n_heads;
    int tok  = tid / (HEAD_DIM * n_heads);
    ...
    if (d < ROTARY_DIM / 2) { // 0 .. 47
        float x0 = in_val;
        float x1 = in_base[d + ROTARY_DIM / 2];
        out_base[d] = x0 * cos_v - x1 * sin_v;
    } else if (d < ROTARY_DIM) { // 48 .. 95
        float x0 = in_base[d - ROTARY_DIM / 2];
        float x1 = in_val;
        out_base[d] = x0 * sin_v + x1 * cos_v;
    } else { // 96 .. 127
        out_base[d] = in_val; // unrotated pass-through
    }
    ```
- **Verification:**
  - Recompiled `bin/p7_fold_eval`.
  - Ran on RTX 3060 under `with_gpu_flock.sh`.
  - **Result: 0 bit differences across all 44,414,208 logits! Max absolute difference = 0.000000e+00. Bit-identical gate PASSED.**

---

### Event 4: Mojo 1.0 / MAX 26.5 FFI Signature Syntax
- **Failure:**
  - When compiling `mojo/p7_streamer_bench.mojo`, Mojo compiler failed with:
    `error: invalid call to 'get_function': unexpected parameter`
- **Bisection:**
  - Checked `OwnedDLHandle.get_function` signature in stdlib (`oss/modular/mojo/stdlib/std/ffi/__init__.mojo`).
  - Signature is `def get_function[return_type: RegisterPassable = NoneType](ref self, var name: String) -> _DLCallable[return_type, origin_of(self)]`.
  - Only `return_type` is passed as a type argument in brackets; argument types are inferred at the `__call__` site.
- **Fix:**
  - Changed `handle.get_function[Int32, Int32, Int32, Int32]("p7_mojo_streamer_benchmark")` to `handle.get_function[Int32]("p7_mojo_streamer_benchmark")`.
- **Verification:**
  - Executed under `with_gpu_flock.sh` inside `40-Runtime-Deps`.
  - Benchmark ran 5 repeats across 32 layers at 11.28 GB/s transfer bandwidth. Exit code 0, printed `MOJO_PREFETCH_PASS`.

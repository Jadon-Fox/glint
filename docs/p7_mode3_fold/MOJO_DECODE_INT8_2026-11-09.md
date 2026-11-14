# Mojo/MAX INT8 KV-cache decode — Phase A execution (2026-11-09)

**Project:** GitHub Project #6, issue #116 (P7-MOJO-D1), R4 #113. **Status:** EXECUTED INT8 TEACHER-FORCED REFERENCE PASS; PHASE A NOT YET FULLY COMPLETE. R5/M6 NOT ACCEPTED.

## Build and provenance

- Source branch: feat/p7-mojo-kv-decode, created inside the existing second GLINT worktree from origin/main acfa3e9; no third worktree. Unrelated root-owned untracked directory preserved.
- Module: mojo/p7_decode.mojo. Imports shared numerical kernels from p7_forward.mojo and quant_act from p7_int8_mma_gemm.mojo; p7_forward.mojo is unchanged. Implements token-by-token f32 KV caches [2048][8][128] per layer, online-softmax attention, layer-local INT8 M=1 GEMV and oracle CE/top-1 scoring.
- Runtime: 40-Runtime-Deps, Mojo 1.0.0 ed45d567, MODULAR_HOME=/root/.pixi/envs/modular/share/max, RTX 3060 12 GiB, driver 595.99.02; commands wrapped by training_orchestrator/scripts/with_gpu_flock.sh.
- Model/checkpoint: microsoft/Phi-4-mini-instruct at cfbefacb99257ffa30c83adab238a50856ac3083; existing bf16_to_int8_pin_v1; evaluation token file and HF oracle from docs/p7_mode3_fold/artifacts/.
- On-disk logs (not committed source): /home/workspace/lakes/p7_glint/mojo_decode_preflight_20261109/.

## Direct executed results

| Path | Input | PPL | Oracle top-1 | Compute timing |
|---|---|---:|---:|---|
| Existing unchanged Mojo prefill | 2047 evaluated positions | 5.48190049338596 | 2001/2047 = 97.7528% | total forward 34.8722 s |
| New Mojo KV-cache INT8 decode | 2047 sequential teacher-forced positions | 5.48190049338596 | 2001/2047 = 97.7528% | decode 387.6101 s; 189.355 ms/token; 5.2811 tok/s |

Readback of **all 2047 individual recorded top-1 and cross-entropy outputs** from an instrumented temporary copy of unchanged Mojo prefill vs p7_decode.mojo: **zero differences** (top1 mismatches 0; nonidentical CE floats 0; max_abs_ce 0.0). Comparison log: decode_prefill_match.txt. Temporary prefill trace was generated outside the repository in the experimental lake and does not modify p7_forward.mojo. A second prefill trace required 42.23 s overall due to per-token printing; it reproduced the exact baseline PPL/top1.

The 2047-token Mojo decode passes the existing teacher-forced top-1 >=95% and PPL <=2% from HF oracle (5.486957) on this input. This validates the tested INT8 numerical path against the prefill outputs; it does **not** establish all-intermediate arithmetic parity, cross-backend bit equality, free-running tracking or M6 acceptance.

## Limitations and outstanding #116 tasks

- M=1 GEMV is a transparent scalar-dot reference implementation. Serial RMSNorm imported from the T=2048 prefill kernel is launched with one thread to avoid out-of-bounds reads. Both are correctness scaffolding, **not** optimized decode kernels.
- The existing byte-by-byte host-weight loader has been preserved; implementing a supported bulk host-buffer transfer with correct ownership/lifetime is still open.
- KL against CUDA BF16 anchor, per-layer intermediate tensor comparison, repeated decode timings and peak VRAM telemetry are not yet established.
- Real GLINT folded + plug pin integration and full decoder scoring remain Phase B. A separate single-linear GLINT kernel is being introduced for this.
- M6 issue #92 still requires its separate free-running tracking rule and >=10 tok/s with all mechanisms active. This is an INT8 M1 decode, not that integrated mode.
- No merger, branch deletion, or Training Orchestrator code modification.

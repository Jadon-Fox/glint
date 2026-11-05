# P7 results ledger entry (DRAFT for TD review, glint #22), updated 2026-11-04

Phi-4-mini-instruct, RTX 3060 sm_86, 2047 evaluation tokens (`artifacts/eval_tokens_2048.txt`, sha in `hf_meta.json`), HF BF16 sdpa oracle PPL 5.486957. KL is KL(BF16 CUDA anchor || arm) per token. Every number below is from a run on this machine in glint `main`; commands are the harness invocations named in each PR.

| Claim | Evidence | Result |
|---|---|---|
| Gate 1 anchor (BF16 CUDA vs HF), threshold 95.0% top-1, PPL within 1.0% | `scripts/run_arm0_anchor.sh`, PR #45 | PPL 5.491616 (+0.085%), top-1 2015/2047 = 98.44%, PASS |
| INT8 pin, plain W8 tile GEMM | PR #36 | PPL 5.501980, top-1 97.85%, KL mean 1.25e-3 |
| INT8 W8A8 tensor-core GEMM | PR #36/#38 | PPL 5.496700, top-1 97.85%, KL mean 1.75e-3, 3.37 s |
| Mode 2 offload: streamed == resident | PR #35 (1.5B, bitwise over 23.9M floats), PR #37 (Phi-4, CE and top-1 hash identical) | linear weights in VRAM 3.188 -> 0.199 GiB |
| Attach gate | PRs glint #40, orch #367 | selftest rc=0, 0 mismatches; forced failure exits 4 |
| Weight maps | PR #41, #49 | INT8 RMSE 1.98e-4 / max 1.10e-2; NF4 RMSE 3.05e-3 / max 0.156 |
| Plug (res4) | PR #42, #43 | RMSE 1.98e-4 -> 1.39e-5; 1.81 GB |
| Plug selection, 8 layers | PR #44 | oracle-chosen KL 1.24e-3, best static 1.48e-3, none 1.75e-3 (oracle is in-sample) |
| Resolution + plugs | PR #46 | KL 4.17e-4, top-1 98.49%, PPL 5.476566, 6.03 s |
| NF4 comparison | PR #49 | PPL 5.833352, top-1 91.06% (fails 95.0%) |
| Activation maps | PR #47, #48, #50 | BF16 rerun byte-identical; INT8 vs BF16 rms mean 0.16%, NF4 2.13% |
| Fold forward on the harness (Mode 3, hole+plug) | PR #52 | PPL 5.479566, top-1 98.05%; H-TILE kernel 136 s vs 3.4 s INT8 |
| All-Mojo forward (no CUDA C / Python) | PRs #55-#69 | INT8 PPL 5.4819, top-1 97.75%; + plugs + resolution PPL 5.4772, top-1 98.39% |
| Stage 0 question list (rules 0.5) | PR #72 | 144 items, list sha256 bf342e75..., reproducible; cal16k exclusion not applied (list not on disk) |
| M4-LOSSY sweep (knob = residual plug bit width) | PR #73, amendment committed before data | weight RMSE 1.39e-5 / 3.24e-5 / 9.70e-5 for b=4/3/2; KL mean 1.240e-3 / 1.195e-3 / 1.144e-3 (non-monotone); ideal bytes 1.81e9 / 1.41e9 / 1.01e9 |
| M4-LOSSLESS (Huffman over glint indices) | PR #73 | 258 streams, 0 round-trip failures, 1.044% smaller |
| Fold only (true parent decode) | PR #80 (draft, not merged) | Phi-4-mini PPL 5.846849, top-1 1869/2047 (91.30%); under cuBLAS PPL 5.833352, 1864/2047, identical to the NF4 arm: the hole is exactly NF4 |
| Hole resident, plug streamed per layer, `glint_pin_v1` | PR #80 (draft) | PPL 5.479566, top-1 2007/2047; logits byte-identical to resident; 1,610,612,736 plug bytes, total stall 0.15 ms (H-TILE ~135 s per forward, so not a speed result) |
| VRAM budget (first N layers' plug resident) | PR #82 (draft) | all-streamed saves 1305 MiB (5298 -> 3993 MiB), logits identical at every N |
| Demand map: plug selected per token by activation | specs #81, results #82 (drafts) | pre-registered rules: not supported at 64-channel blocks; channel granularity 10% of plug keeps top-1 95.55% (idealized bytes); hole-skip fails at >= 25%; per-layer errors add |
| Plug fetch speed (K-major layout) | PR #82 (draft) | block fetch 10-11 GB/s; single channels ~7 GB/s zero-copy; 5% of a layer's plug 0.31-0.35 ms vs 4.35 ms whole |
| Rollout (3 prompts x 16 greedy steps) | PR #82 (draft) | no activation-selected arm tracks the full model (matched prefix 6-7 of 16); fold only 1.7; act vs random not separable at this sample |
| Qwen3-4B-Base Mojo INT8 / +res4 plug / +resolution / both | PRs #77, #83 (drafts) | top-1 97.90 / 98.09 / 98.19 / 98.29%; PPL within 0.07% of the oracle; no KL anchor; not the GLINT pin |

Not claimed: a decode (KV-cache) path and any per-step speed number; a GLINT forward on any model other than Phi-4-mini; the weight-residual resolution (#34) and the INT4 base (#70); a comparison against GPTQ/AWQ/HQQ/llama.cpp; an orchestrator loader for `glint_pin_v1`; Part A (H5-H9 hybrid fidelity; job 26 harness sources and captures are not on this machine, and its real-data run is held for Test Designer review), the per-layer metric e for M4, the equal-bytes comparison against a BF16 model of the same total size, any run on a model other than Phi-4-mini (Mode 2 equivalence also on DeepSeek-R1-Distill-Qwen-1.5B), the 100%-on-confident-positions check of the Gate 1 amendment.

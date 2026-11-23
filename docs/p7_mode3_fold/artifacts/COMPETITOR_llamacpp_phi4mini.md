# GLINT against llama.cpp quantizations, Phi-4-mini, same card (2026-11-04) — glint #79 Phase 5

The operator approved installing llama.cpp and downloading a Phi-4-mini GGUF for this comparison. Installed: llama.cpp commit `5ce06fc` built with CUDA (sm_86), standalone cmake 3.30.5 (the workspace's Python-environment policy U-011 blocked a pip route, which was not worked around). Downloaded: `Phi-4-mini-instruct.BF16.gguf`, 7.68 GB, from `unsloth/Phi-4-mini-instruct-GGUF` on Hugging Face. Every quantization below was made locally from that one file with `llama-quantize` (no importance matrix), so all variants share the same weights. Everything lives under `/home/workspace/lakes/p7_glint/competitors/`. GPTQ, AWQ and HQQ were **not** run: they need Python stacks that the same policy blocks here.

## Fidelity: each quantization against its own BF16 baseline (`llama-perplexity --kl-divergence`, one 2048-token chunk of `SCIENCE_JOURNAL.md`)
| llama.cpp | file | mean KL | same top-1 as BF16 | PPL / PPL(BF16) | decode (tg128) | prefill (pp512) |
|---|---|---|---|---|---|---|
| BF16 | 7.68 GB | 0 | 100% | 1 | 35.2 tok/s | 3931 |
| Q8_0 | 4.08 GB | 1.14e-3 | 98.34% | 1.0002 | 63.2 | 3839 |
| Q6_K | 3.16 GB | 9.80e-3 | 97.17% | 1.0094 | 79.8 | 3244 |
| Q4_K_M | 2.49 GB | 2.09e-2 | 95.80% | 1.0173 | 94.2 | 3767 |
| IQ4_NL | 2.34 GB | 2.69e-2 | 95.70% | 1.0267 | 102.2 | 3858 |
| Q4_0 | 2.33 GB | 4.84e-2 | 93.06% | 1.0411 | 103.7 | 3826 |

## GLINT on the same card (own harness; mean KL(BF16 CUDA anchor || arm) over 2047 teacher-forced positions)
| GLINT arm | resident bits per weight | plug fetched | mean KL | decode (resident / with fetch) |
|---|---|---|---|---|
| fold only (hole = NF4) | 4.5 (4-bit hole + absmax) | 0 | 7.93e-2 | 26.6 tok/s |
| + activation-selected plug, channel granularity | 4.5 + fetch | 5% | 2.07e-2 | 17.9 tok/s |
| | | 10% | 1.52e-2 | 12.5 |
| | | 25% | 7.04e-3 | 8.0 |
| | | 50% | 2.18e-3 | 4.9 |
| hole + plug, full (8.5 bpw: 4+4 bits + absmax) | 8.5 | 100% | 9.5e-4 | 24.8 |
| INT8 W8A8 (pin) | 8.5 | n/a | 1.75e-3 | n/a |
| dense BF16 (cuBLAS decode, our harness) | 16 | n/a | 0 | 32.3 |
KL rows from `DEMAND_MAP_phi4mini.md`; decode rows from `SELFETCH_OPT_phi4mini.md`, `GEMV_DECODE_phi4mini.md` and `DECODE_PATH_phi4mini.md`.

## Reading
- **At the same resident precision (4.5 bits per weight) the folded GLINT form is worse than the llama.cpp 4-bit quantizers on this metric**: fold only KL 7.9e-2 against 4.8e-2 (Q4_0), 2.7e-2 (IQ4_NL) and 2.1e-2 (Q4_K_M). The hole is exactly NF4 (measured), and NF4 per-64 blocks lose to llama.cpp's Q4_0 per-32 blocks, IQ4_NL's non-uniform grid and Q4_K_M's two-level scales here.
- **GLINT's case is the plug, not the fold**: with 5% of the plug fetched the KL (2.07e-2) equals Q4_K_M's (2.09e-2), at 10% it is below it (1.52e-2), at 25% below Q6_K's (7.0e-3 vs 9.8e-3), and the whole plug reaches Q8_0's level (9.5e-4 vs 1.14e-3). The price is that the plug lives in host RAM (8.5 bits per weight in total across VRAM + RAM, against 4.5-8.5 for a llama.cpp file in VRAM) and that the KL figures at 5-25% of the plug are teacher-forced results. In the 12-prompt x 32-step free-running study (PR #96) activation-selected fetch of 5%, 10% and 25% of the plug agrees with the full model on 44%, 56% and 73% of generated tokens, and only 50% of the plug tracks it (99%, at 4.9 tokens/s); a comparable free-running measurement was not made for the llama.cpp quantizations.
- **Speed: llama.cpp's tuned kernels are 4-8x faster on this card.** Its 4-bit decode is 94-104 tokens/s (and BF16 decode 35.2, close to our dense cuBLAS 32.3); our GLINT decode GEMV is an unoptimized prototype at 25-27 tokens/s resident, reading about 50 GB/s for the fold-only arm against a possible ~360, and 12.5-17.9 tokens/s with a 5-10% plug fetch.
- A claim that GLINT beats a llama.cpp quantization at equal VRAM is **not** supported by these numbers; a claim that a streamed plug can raise a 4.5-bit resident model to Q4_K_M / Q6_K / Q8_0 fidelity from host RAM at 12-18 tokens/s on a 12 GB card, with a prototype kernel, is.

## Differences that limit the comparison (stated, not hidden)
- Different protocols: llama.cpp's KL is over the second half of one 2048-token chunk (about 1023 positions) against its own BF16 GGUF; ours is over 2047 positions against our CUDA BF16 forward. Absolute PPL is not comparable (2.69 vs 5.49, different scoring window), so only the KL/same-top columns and ratios are compared, and even these are indicative.
- llama.cpp quantizes the tied embedding too (its files include it); the GLINT rows quantize linears only and keep the embedding BF16.
- One text, one run, quantizations without an importance matrix (imatrix would improve the 4-bit rows).
- Not reviewed by the Test Designer.

# Qwen3-4B-Base forward in Mojo / MAX, INT8 pin (2026-11-04)

`mojo/q3_forward.mojo PIN MODEL_DIR ARTIFACTS_DIR` runs the 2047-token forward pass of Qwen3-4B-Base entirely in Mojo (no CUDA C, no Python),
with every linear through the Mojo W8A8 tensor-core GEMM over the `Qwen3-4B-Base-int8-from-bf16` pin. Built from `mojo/p7_forward.mojo`;
what changed for Qwen3: separate q/k/v and gate/up projections (7 linears per layer), per-head QK RMSNorm (`q_norm`/`k_norm`, head_dim 128)
before RoPE, full-width RoPE with theta 1e6 and no attention scale factor, GQA 32:8, RMS eps 1e-6, vocab 151936 tied to the embedding,
three BF16 shards, 36 layers. INT8 arm only: no plug and no resolution in this file yet.

Oracle: `scripts/generate_oracle_ref_hf.py` (HF `AutoModelForCausalLM`, bf16, sdpa, transformers 4.57.6, torch 2.5.1+cu121) on the first 2048
tokens of the science journal, same protocol as Phi-4-mini. Files in `artifacts/qwen3_4b_base/`: `eval_tokens_2048.txt` (sha256 695c1742...),
`oracle_top1_2048.bin` (sha256 d29bb607...), `oracle_ref_2048.json`. Oracle mean CE 1.507263, PPL 4.514359.

| Path | PPL (HF oracle 4.514359) | top-1 vs HF oracle | time |
|---|---|---|---|
| Mojo INT8 W8A8 (this) | 4.511863 (-0.055%) | 2004/2047 = 97.90% | 12.2 s layers, 30.2 s total |

RTX 3060, Mojo 1.0, one run (the GPU lease serializes runs; timings are single-run). Inside the 1.0% PPL window and above the 95.0% top-1
threshold used for Phi-4-mini Gate 1. Phi-4-mini's Mojo INT8 arm gave 97.75% top-1 on its own text; the two figures are different models on
different token streams, so this is not a ranking between models.

Not checked: bit-identity to a CUDA Qwen3 arm (none exists), KL against a BF16 Mojo/CUDA anchor, sha256 verification of the oracle files inside
Mojo, plug and resolution arms (the Qwen3 plug file has not been built), RoPE tables computed in float64 on the host then rounded to bf16
(HF computes the angles in float32; a few table entries could round differently). The logits and cross-entropy kernels are the same
unoptimized ones as the Phi forward and take most of the 30 s.

Found along the way: the first run died with `CUDA_ERROR_ILLEGAL_ADDRESS` because the non-owning views over the uint8 weight buffers outlived
the backing lists, which Mojo frees after their last use. The keep-alive block at the end of `main` is required (same trap as PR #68).

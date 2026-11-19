# GLINT arms on four models: Phi-4-mini, Qwen3-4B-Base, DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-135M (2026-11-04) — glint #85

`tests/test_llama_arms.cu` (replaces the Qwen3-only `test_q3_arms.cu` of the first version of this PR): one CUDA harness whose dimensions come from `config.json`, for three layouts: Qwen3 (separate q/k/v and gate/up, per-head QK RMSNorm, full-width RoPE), Qwen2 as in DeepSeek-R1-Distill (q/k/v biases, untied `lm_head`, GQA 12:2) and Llama as in SmolLM2 (head_dim 64, GQA 9:3, tied embedding). It follows the Phi-4 harness's method: 2047 tokens of `SCIENCE_JOURNAL.md`, HF bf16 sdpa oracle (`scripts/generate_oracle_ref_hf.py`, run in the container's registered `/opt/venv`), bf16 rounding where HF rounds, shared norm/residual/cross-entropy/KL kernels, cuBLAS GEMMs on bf16 weights. Modes: `bf16` (dense BF16 anchor), `full` (glint pin, `GLINT_CELLS[hole][plug] * absmax`), `fold` (`PARENT[hole] * absmax`); decoded to bf16 one matrix at a time. Embedding, norms, biases and the output head are always the BF16 model's. RTX 3060, one run per arm; KL is against the same model's BF16 arm. Logs: `llama_arms/*.log`; oracle files: `oracle_smollm2/`, `oracle_deepseek15/` (Qwen3's are in PR #77's branch).

| Model | Arm | PPL | vs oracle | top-1 | mean KL vs BF16 arm |
|---|---|---|---|---|---|
| SmolLM2-135M (oracle 6.568475) | BF16 | 6.565252 | -0.05% | 2016 / 2047 (98.48%) | 0 |
| | hole + plug | 6.577237 | +0.13% | 2004 (97.90%) | 1.64e-3 |
| | fold only | 7.465627 | +13.66% | 1736 (84.80%) | 1.55e-1 |
| DeepSeek-R1-Distill-1.5B (10.832062) | BF16 | 10.824034 | -0.07% | 2022 (98.78%) | 0 |
| | hole + plug | 11.051991 | +2.03% | 2010 (98.19%) | 1.95e-3 |
| | fold only | 12.104976 | +11.75% | 1792 (87.54%) | 1.46e-1 |
| Qwen3-4B-Base (4.514359) | BF16 | 4.512231 | -0.05% | 2017 (98.53%) | 0 |
| | hole + plug | 4.518957 | +0.10% | 2009 (98.14%) | 3.78e-4 |
| | fold only | 4.615653 | +2.24% | 1892 (92.43%) | 2.87e-2 |
| Phi-4-mini (5.486957, `FOLD_FORWARD`, `DEMAND_MAP`) | BF16 | 5.4916 | +0.09% | 2015 (98.44%) | 0 |
| | hole + plug | 5.4885 (cuBLAS) | +0.03% | 2010 (98.19%) | 9.5e-4 |
| | fold only | 5.8334 | +6.31% | 1864 (91.06%) | 7.9e-2 |

## Reading
- *Measured:* the harness is correct on every layout on the first run: each BF16 anchor is within 0.08% PPL of the HF oracle with 98.4-98.8% top-1, the quality the Phi-4 anchor has (98.44%). The Qwen3 numbers from the generalized harness reproduce the Qwen3-only version exactly.
- *Measured:* the same pattern holds on four models and three architectures: the 4-bit hole alone (an NF4 code) loses 2-14% PPL and 6-14 points of top-1, and adding the 4-bit plug brings the model back to within 0.1-2.1% PPL and 0.3-0.6 points of the BF16 anchor's top-1, with mean KL at or below 2e-3 in all four.
- *Measured:* sensitivity to the NF4 hole differs by model: fold-only KL is 2.9e-2 for Qwen3-4B but 1.5e-1 for both small models (SmolLM2-135M, DeepSeek-1.5B), so the smaller models gain the most from the plug, as would be expected; the PPL cost of hole + plug is largest on DeepSeek-1.5B (+2.03% against the oracle, KL 1.95e-3), which the KL alone would not suggest.
- The decode, selected-fetch, rollout and competitor studies of this session were run on Phi-4-mini only.

## What this does not show
One text and one run per arm; the embedding and output head are not GLINT-decoded (the pins encode them, unused here); no streaming, decode or selected fetch outside Phi-4-mini; no model above 4B parameters. The Phi-4 rows are from earlier harnesses (the Arm 0 harness), not this one. Not reviewed by the Test Designer.

## Reproduce
`nvcc -O2 -arch=sm_86 -Iinclude -o llamaarms tests/test_llama_arms.cu -lcublas`; `Q3_MODE=bf16 ARM0_LOGITS_OUT=a.bin llamaarms BF16_MODEL_DIR GLINT_PIN_DIR ARTIFACTS`, then `Q3_MODE=full|fold ARM0_LOGITS_REF=a.bin llamaarms ...`. Oracle for a new model: `python scripts/generate_oracle_ref_hf.py MODEL_DIR OUT_DIR`.

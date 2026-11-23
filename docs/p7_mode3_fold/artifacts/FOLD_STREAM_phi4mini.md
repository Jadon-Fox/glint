# Fold-only and streamed-plug arms on the Arm 0 harness, Phi-4-mini, 2047 tokens (2026-11-04)

Two arms that FOLD_FORWARD_phi4mini.md listed as invalid or not run. Both use `Phi-4-mini-glint-from-bf16` (`glint_pin_v1`), the Arm 0 harness `tests/test_arm0_bf16_full.cu`, the oracle in `hf_meta.json` (PPL 5.486957), RTX 3060, 40-Runtime-Deps, under the GPU lease. One run each; the kernels are deterministic.

## Arm F: fold only (true parent decode)
`GLINT_PARENT_ONLY=1` fills every plug column of the kernel's cell table with the parent value, so a weight decodes to `PARENT[hole] * absmax` (`glint_codec.decode_parent_only`; PARENT is the QLoRA NF4 table). The plug nibble is never read for its value. This replaces the earlier "plug zeroed" run, which was invalid because plug 0 is a child cell, not the parent.

| Arm | PPL (oracle 5.486957) | PPL delta | top-1 vs HF oracle | forward |
|---|---|---|---|---|
| fold only (parent decode, H-TILE) | 5.846849 | +6.56% | 1869 / 2047 (91.30%) | 135.1 s |
| NF4 comparison point (cuBLAS, bf16-rounded weights; PR #49) | 5.833352 | +6.31% | 1864 / 2047 (91.06%) | - |
| hole + plug (H-TILE, resident) | 5.479566 | +0.14% | 2007 / 2047 (98.05%) | 137.7 s |

Reading: the hole alone lands on the NF4 point. The 0.23% PPL difference to the NF4 row is the arithmetic: H-TILE accumulates `parent * absmax` in f32, the NF4 arm rounds the dequantized weight to bf16 and uses cuBLAS. Fold only fails the 95.0% top-1 gate, as NF4 did, so the plug (or a streamed correction) has real work to do on this model. Report only, no PASS or FAIL is claimed for an arm that has not been designated for a gate.

## Arm S: hole resident, plug streamed through the 2-slot ring
`--glint-stream PIN`: the hole and absmax stay in VRAM. Each layer's four plug tensors (QKV, O, GATE_UP, DOWN) are staged back to back in pinned host RAM and brought in one layer ahead by `glint_plug_streamer` (copy stream non-blocking, compute on the default stream, acquire at the layer's first linear, release after its last).

| | resident (`--glint`) | streamed (`--glint-stream`) |
|---|---|---|
| PPL | 5.479566 | 5.479566 |
| top-1 | 2007 / 2047 | 2007 / 2047 |
| logits (2047 x 200064 f32, 1,638,124,032 B) | sha256 5adf1bc9ef62ed5545a835f681567861355e14fab8402217ee0d8deeddc7ee2a | same sha256, `cmp` byte-identical |
| forward | 137.7 s | 138.3 s |
| plug bytes moved H2D | 0 | 1,610,612,736 (32 layers x 48 MiB, as planned) |
| total streamer stall | - | 0.152 ms (layer 0: 0.005 ms; layers 1-31: at most 0.005 ms each) |

This is the first whole-model forward on `glint_pin_v1` with the plug streamed. Earlier evidence was layers 0-1 on real weights (PR #31) and the Oct-21 run on the nested-NF8 pin, which is stale (missing longrope factor).

## What this does not show
- **Hiding is trivial at this speed.** A layer takes about 4.3 s in the H-TILE kernel; a 48 MiB plug takes about 4.4 ms. The near-zero stall says nothing about hiding at the INT8 kernel's speed or at decode speed, where a layer's compute is far shorter than a whole plug's transfer. Phase 2 of glint #79 addresses that.
- No KL against the BF16 anchor in this run; no lossy plug; no activation-selected fetch.
- Poisoned-slot and delayed-pass tests stay in `tests/test_streamer_real_weights.cu` (PR #31); this run does not repeat them.

## Reproduce
Build (in the CUDA container): `nvcc -O2 -arch=sm_86 -Iinclude -o bf16full tests/test_arm0_bf16_full.cu cuda/glint_htile.cu cuda/glint_plug_streamer.cu -lcublas -lcrypto`.
Fold only: `GLINT_PARENT_ONLY=1 bf16full --glint PIN MODEL_DIR ARTIFACTS`.
Resident, then streamed against it: `ARM0_LOGITS_OUT=a.bin bf16full --glint PIN MODEL_DIR ARTIFACTS`; `ARM0_LOGITS_OUT=b.bin bf16full --glint-stream PIN MODEL_DIR ARTIFACTS`; `cmp a.bin b.bin`.
Raw outputs: `/home/workspace/lakes/p7_glint/fold_stream_20261104/` (build, resident and streamed logs, both logits files).

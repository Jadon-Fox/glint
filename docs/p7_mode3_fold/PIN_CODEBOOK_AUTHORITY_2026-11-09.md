# GLINT pin/codebook custody — corrected complete-shard audit (2026-11-09)

**GitHub Project #6:** R1 #110, Mojo Phase B #116.
**Status:** VERIFIED PIN TABLE PRESENT AND IDENTICAL TO CURRENT COMPILED C TABLE; CUDA LOADER-AUTHORITY GAP REMAINS. No source-format or CUDA kernel changes made.

## Correction to the initial incomplete inspection

An initial inspection looked at the first few codebook/state-looking names from one Safetensors header, causing a false assertion that the pin contained no cell table. A subsequent **complete two-shard enumeration and read of table bytes** found one tensor:

- Pin: /home/workspace/pins/Phi-4-mini-glint-from-bf16
- Schema: glint_pin_v1; blocksize 64; source checkpoint cfbefacb99257ffa30c83adab238a50856ac3083.
- Table: _glint_cells, F32, shape [16,16], exactly 1024 bytes in model-00001-of-00002.safetensors.
- Raw embedded F32 SHA256: e774eceefa0d565e330f966d8ba1ea64f09208f4769bb6dd512bcd419bcb0365.
- Compiled include/glint_cells.h source SHA256: fbfbffbe8dea811f9400d24820f9ae2579c274025d0ff0a6c7734a6780253a82.
- The 256 C float literals, packed as little-endian F32, have SHA256 e774eceefa0d565e330f966d8ba1ea64f09208f4769bb6dd512bcd419bcb0365.
- **All 1024 F32 bytes are identical between the embedded pin tensor and the compiled C table.**

The pin DOES carry its own table. Earlier commentary that the table was missing is superseded by this verified finding.

## Remaining implementation discrepancy

CUDA tests/test_arm0_bf16_full.cu initializes its c_cells via cells_for_bits, reading compiled GLINT_CELLS. It does not use the pin's actual _glint_cells tensor at that point. Existing numerical fidelity is not disproven: the current pin table and compiled table are identical. But a deliberately changed pin table would not affect the CUDA GEMV unless its initialization path is corrected.

Likewise, the upcoming Mojo loader must read the _glint_cells tensor from the pin, validate F32 [16,16], size, and source/codec provenance, and transfer those values to the GEMV's table storage. No silent compiled default.

## Verified audit command and result

Run scripts/p7_pin_codebook_audit.py PIN_DIR --require-embedded --legacy-header include/glint_cells.h --expected-header-sha256 fbfbffbe8dea811f9400d24820f9ae2579c274025d0ff0a6c7734a6780253a82.

It validates all shards, requires a single embedded F32 table, compares actual byte values against the compiled C constants, and rejects a mismatched header SHA or mismatched table. It does not mutate source pin bytes.

## Negative control / CUDA change (separately reviewed)

1. Use a synthetic tiny Safetensors fixture or a safe copy—not the live multi-GB pin—to change one codebook value, leaving hole/plug indices and scales unchanged.
2. Verify the audit detects the mismatch against a pinned reference table.
3. Before any CUDA code change, demonstrate by source inspection and baseline behavior that current gemv_init reads compiled table. Avoid labeling this as a matched perturbation experiment.
4. Implement a small CUDA table-load path from the actual pin tensor, independently reviewed. Then a positive control must match original output, while a changed table should produce the expected changed arithmetic for the affected code(s). No unrelated H-TILE rewrite.
5. Implement the same in Mojo and compare both against a pinned single-linear reference, then teacher-forced full-model gates.
6. Preserve the existing format choice as an *interim compatibility decision*. R1 still decides the long-term native progressive representation by tests; no implicit NF4/GPTQ authority.

**No actual CUDA table-loading patch or Mojo GLINT parity run is claimed here.**

## Executed Phase B single-linear numerical smoke test

- Module: mojo/p7_glint_linear.mojo; reads the actual _glint_cells tensor from the GLINT pin and tests model.layers.0.self_attn.o_proj.weight (N=3072, K=3072, block 64, low-then-high nibble codes).
- Input vector is deterministic, BF16-exact values x[k] = ((k mod 17) - 8)/64. The GPU computes each output row from loaded hole, plug, scale and codebook values.
- Independently computed a NumPy float64 row-wise reference from the exact pin bytes on the host, using F32 reconstruction weights. Of 3072 output rows: max absolute difference 5.668042462048106e-7; mean absolute difference 5.769420682866233e-8; maximum difference relative to RMS output 6.293470324936779e-6.
- The stated 1e-4 maximum-absolute-error smoke tolerance PASSED. Source logs: /home/workspace/lakes/p7_glint/mojo_decode_preflight_20261109/glint_linear.log and glint_single_linear_match.txt.
- This is **Mojo vs independent CPU reference**, not Mojo-vs-CUDA parity. Fold-only handling, multi-layer GLINT decode, and table-perturbation response on CUDA remain unexecuted.

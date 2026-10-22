# GLINT
**Concept SSOT (operator 2026-09-24):** [`CONCEPT_CANONICAL_2026-09-24.md`](CONCEPT_CANONICAL_2026-09-24.md) — fold/compress → unfold → conditional relational refinement. Card: `/home/workspace/lookup/cards/glint-concept.md`.

**Current repository tip (implemented, narrower):**
**Gaussian Location INside Tile.**
A 4-bit **plug** names which child lives inside a 16-cell Gaussian **parent tile**. Reconstruction is a lookup, then H-TILE GEMM. Not NF8. Not uniform INT8. Not the `nf4-to-int8` program.

Keep **designed / implemented / demonstrated** separate (`lookup/cards/canonicality-precedence.md`). Do not treat this README as proof of the full conceptual architecture or of latency-neutral unfolding.
```
Ŵ = GLINT_CELLS[hole][plug] × absmax
```
Hole = parent cell (VRAM, same pack as H-TILE qpack). 
Plug = location inside that cell (4 bits). Drop it in → the weight is recognized.
```bash
# on the 3060 (nvcc, sm_86 default)
bash scripts/build_glint_cuda.sh
python3 glint_pin.py --src /path/to/bf16 --out ./glint-pin # GPU encode
python3 glint_pin.py --src tiny --out ./t --allow-cpu # tests only
python3 glint_offset.py
python3 test_glint.py

# H-TILE device kernel (object compile smoke; see HTILE_SMEM.md)
nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude -c cuda/glint_htile.cu -o /tmp/glint_htile.o
```

Encode is **CUDA**: raw BF16 bytes go to the GPU; no Python float walk. That was the hour-long path. `libglint_encode.so` required unless `--allow-cpu`.

**H-TILE:** K-contiguous packed loads and shared-memory staging are documented in [`HTILE_SMEM.md`](HTILE_SMEM.md). Device code keeps `GLINT_CELLS` in a `__constant__` table (`cuda/glint_htile.cu`); without that host→symbol copy, `nvcc` device compilation fails on a bare header reference.

**Pitfalls (2026-10-20 fix #4/#6):** Do not add empty-string keys to `pin.json` or offset maps (`''` entries broke `glint_offset.py` and `scripts/glint_auto.sh` asserts). Truncated `=false.` comment debris in headers was cleaned up; if a local branch still has it, sync tip before debugging encode/htile.

INT8 pin stays in [nf4-to-int8](https://github.com/Jadon-Fox/nf4-to-int8) (`--to int8`). Both from BF16; different 256.
Sim is **done** (offset board). Product is CUDA encode + CUDA expand. GPU evidence workflow: [`GROK_BUILD_GPU_TEST.md`](GROK_BUILD_GPU_TEST.md). Orch wire-up: [`GROK_BUILD_ORCH_GLINT_PROMPT.md`](GROK_BUILD_ORCH_GLINT_PROMPT.md).

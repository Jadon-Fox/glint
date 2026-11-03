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
python3 glint_pin.py --src tiny --out ./t --allow-cpu # host encode (numpy f32 if present)
python3 glint_verify.py --pin ./glint-pin --src /path/to/bf16 --parity --cuda --quiet --out VERIFY.json
python3 glint_unpin.py --pin ./glint-pin --out ./glint-bf16 # unfold to dense BF16
python3 glint_offset.py
python3 -m pytest -q # GPU test runs only where the .so loads and a device answers

# H-TILE device kernel (object compile smoke; see HTILE_SMEM.md)
nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude -c cuda/glint_htile.cu -o /tmp/glint_htile.o
```
**Sharded models:** `--src` takes a file, a dir with `model.safetensors`, or a dir with `model.safetensors.index.json`. Sharded in → same shard names out + a GLINT index; `_glint_cells` rides in the first shard. `config.json` and tokenizer files are copied so the pin stays a model dir.

**Host encode = CUDA bytes.** `glint_np.py` does the kernel's f32 math in the kernel's order (f32 absmax, f32 divide, first-minimum nearest, lo-then-hi nibbles), so a host pin is byte-identical to a `libglint_encode.so` pin. `--force-python` keeps the old pure-Python f64 path. Neither replaces the CUDA product encode.

**Gates (`glint_verify.py`, exit 1 on fail):** layout always (hole/plug = ⌈n/2⌉ B, absmax = ⌈n/bs⌉ f32, zero pad nibble, `_glint_cells` = codec). `--src`: per-module RMSE. `--parity`: host re-encode vs pin bytes (the host-vs-CUDA plug byte gate). `--cuda`: GPU expand vs host expand, max_abs < 1e-5. `scripts/glint_auto.sh` runs it with `VERIFY=layout` (default), `parity`, or `off`.

**Files are spec safetensors:** contiguous data, no inter-tensor or tail padding, so the official `safetensors` reader opens pins and unpinned dirs.
Encode is **CUDA**: raw BF16 bytes go to the GPU; no Python float walk. That was the hour-long path. `libglint_encode.so` required unless `--allow-cpu`.

**H-TILE:** K-contiguous packed loads and shared-memory staging are documented in [`HTILE_SMEM.md`](HTILE_SMEM.md). Device code keeps `GLINT_CELLS` in a `__constant__` table (`cuda/glint_htile.cu`); without that host→symbol copy, `nvcc` device compilation fails on a bare header reference.

**Pitfalls (2026-10-20 fix #4/#6):** Do not add empty-string keys to `pin.json` or offset maps (`''` entries broke `glint_offset.py` and `scripts/glint_auto.sh` asserts). Truncated `=false.` comment debris in headers was cleaned up; if a local branch still has it, sync tip before debugging encode/htile.

INT8 pin stays in [nf4-to-int8](https://github.com/Jadon-Fox/nf4-to-int8) (`--to int8`). Both from BF16; different 256.
Sim is **done** (offset board). Product is CUDA encode + CUDA expand. GPU evidence workflow: [`GROK_BUILD_GPU_TEST.md`](GROK_BUILD_GPU_TEST.md). Orch wire-up: [`GROK_BUILD_ORCH_GLINT_PROMPT.md`](GROK_BUILD_ORCH_GLINT_PROMPT.md).

**Project 7 attach (INT8 W8A8 + plug ring, orchestrator gate):** [`attach/CONTRACT.md`](attach/CONTRACT.md) — `training_orchestrator/scripts/orch_p7_start_gate.sh` runs `runtime_attach.sh` then GPU `p7_attach_check` / `p7_abi_selftest`. Build kernels: `bash scripts/build_p7_kernels.sh`. Mojo forward / plug measurement notes (not attach proof): [`docs/p7_mode3_fold/`](docs/p7_mode3_fold/) (`MOJO_FORWARD_phi4mini.md`, `GATE_RULES.md`).

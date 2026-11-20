# GLINT (Project 6)
**Project Board:** [`users/Jadon-Fox/projects/6`](https://github.com/users/Jadon-Fox/projects/6) *(Note: legacy `p7_*` filenames/symbols in this repo are historical mislabels; GLINT is **Project 6**, distinct from MMLRPV Project 5 / former Board 7).*  
**Concept SSOT (operator 2026-09-24 / 2026-11-14):** [`CONCEPT_CANONICAL_2026-09-24.md`](CONCEPT_CANONICAL_2026-09-24.md) — **system RAM weight residency + MLIR prefetch + an asynchronous thought pattern for the LLM** so execution runs at **GPU speed** without speed reduction bounded by **PCIe lanes** (`fold/compress → unfold → conditional relational refinement`). Card: `/home/workspace/lookup/cards/glint-concept.md`.

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

## Project 7 attach (training_orchestrator)

Orch does not treat a pin directory as armed until **`orch_p7_start_gate.sh`** runs `attach/runtime_attach.sh` and passes **`lib/p7_attach_check`** (INT8 W8A8 GEMM, resident vs streamed). Build kernels first:

```bash
bash scripts/build_p7_kernels.sh
```

Runbook (host): `training_orchestrator/docs/designs/2026-11-07-ORCH-P7-START-GATE.md`. ABI file: `attach/ABI_VERSION`.

## GPU regression matrix (`retest_all.sh`)

One-shot rebuild + full step matrix after driver/firmware or tip changes. Writes `firmware.txt`, per-step logs, and `summary.tsv`; exits **0** only if every step passed.

```bash
bash scripts/retest_all.sh /home/workspace/lakes/glint_retest_YYYYMMDD
# Rerun subset into an existing dir:
RETEST_APPEND=1 RETEST_ONLY='arm0_int8_mma' bash scripts/retest_all.sh /path/existing
```

Requires GPU lease (`with_gpu_flock.sh`), container `P7_CONTAINER` (default `40-Runtime-Deps`), and host paths under `/home/workspace/pins` and `/opt/models/.hf_cache/...` as wired in the script. Narrated results: [`docs/p7_mode3_fold/artifacts/RETEST_2026-11-03.md`](docs/p7_mode3_fold/artifacts/RETEST_2026-11-03.md). **Qwen3-4B-Base** INT8 + GLINT pins and `glint_verify --parity` live in that addendum; Arm 0 forward/PPL numbers in `retest_all.sh` remain Phi-4-shaped unless extended.

Mode-3 fold design notes: `docs/p7_mode3_fold/`.

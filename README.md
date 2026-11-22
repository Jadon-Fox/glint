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

## Mode-3 fold / lean Mojo decode (Project 6 / `p7_*`)

Transient plug decode (issue #117, G-fold selection) lives under [`docs/p7_mode3_fold/`](docs/p7_mode3_fold/). The **production binary** is `mojo/p7_glint_lean.mojo` (requires `GLINT_EPHEMERAL=2`). Older entrypoints (`p7_glint_decode.mojo`, `p7_glint_select_decode.mojo`) are still in-tree; the lean decoder is the bit-identical fast path documented in [`GFOLD_SELECTION_2026-11-15.md`](docs/p7_mode3_fold/GFOLD_SELECTION_2026-11-15.md).

| Doc | Covers |
|---|---|
| [`MOJO_GLINT_DECODE_2026-11-09.md`](docs/p7_mode3_fold/MOJO_GLINT_DECODE_2026-11-09.md) | Resident vs streamed decode, selection reference path, stream ablation |
| [`GFOLD_SELECTION_2026-11-15.md`](docs/p7_mode3_fold/GFOLD_SELECTION_2026-11-15.md) | G-fold rule, landing gates, two-tier plug, birddog, measured ms/token tables |
| [`GATE_RULES.md`](docs/p7_mode3_fold/GATE_RULES.md) | Amendment / landing predicates for rollout vs teacher-forced |
| [`CAUSAL_DECODE_SCHEDULER_2026-11-09.md`](docs/p7_mode3_fold/CAUSAL_DECODE_SCHEDULER_2026-11-09.md) | Scheduler scaffold (`p7_refine.mojo`) — protocol only |

**Build and identity check (host with LXC `40-Runtime-Deps` + GPU lease):**

```bash
scripts/p7_lean_retest.sh /tmp/p7_lean_out "35" 128   # builds /tmp/p7_lean, 128-step 35% teacher-forced identity gate
```

**Proposed landing preset** (two-tier 12/28, all bit-identical speed paths on; see GFOLD § landing): `scripts/p7_glint_preset.sh /tmp/p7_lean 128`

**Invocation** (from `mojo/p7_glint_lean.mojo` header): after `mojo build -I mojo -o BIN mojo/p7_glint_lean.mojo`,

```bash
GLINT_EPHEMERAL=2 ./BIN PIN MODEL_DIR docs/p7_mode3_fold/artifacts STEPS PERCENT 0 MODE 0 0 0
```

`MODE` containing `roll` runs free-running rollout (12 prompts × 32 greedy steps); otherwise teacher-forced. Positional args 6, 8, 9, 10 are kept for CLI compatibility and must be `0`.

**Environment switches** (read in `p7_glint_lean.mojo`; unset = defaults in source):

| Variable | Role |
|---|---|
| `GLINT_EPHEMERAL` | Must be `2` for the lean decoder |
| `GLINT_PLUG_BITS` | Partial plug bit width (`2` with real two-tier) |
| `GLINT_TIER2_PCT` | Second-tier channel ceiling (percent); with `GLINT_TIER2_REAL=1` moves real 2-bit planes |
| `GLINT_TIER2_REAL` | `1` = split plug into HI/LO pinned planes (`mojo/p7_plug_planes.mojo`) |
| `GLINT_TIER2_FUSED` | `0`, `sel`, `x`, `1`, or `m` — fused gather/correction on top of `TIER2_REAL` (`m` = row-blocked corrections) |
| `GLINT_CORR_R` | Row-blocked correction tile: one int or `Rqkv,Ro,Rgu,Rdown` (each `0`, `2`, `4`, or `8`) |
| `GLINT_SEL_FAST` | `1` = `select_rank_f` / `select_topk2` (rank rule only; identical selections) |
| `GLINT_GEMV_U` | `2`, `4`, or `8` = fold GEMV with that many channel loads in flight (`glint_gemv_km_u`) |
| `GLINT_BIRD` / `GLINT_BIRD_X` | Look-ahead prefetch from fold output; optional wider predicted set (percent) |
| `GLINT_ORACLE` / `GLINT_ORACLE_REUSE` | Timing study: pass A records truth; pass B reuses within *W* tokens (incompatible with `TIER2_REAL`) |
| `GLINT_REUSE` / `GLINT_REUSE_STAT` | Real decode cross-token row reuse within *W* tokens (incompatible with `TIER2_REAL`) |
| `GLINT_CLASS_PCT`, `GLINT_GB` | Per-class plug share and gather blocks (`q,o,gu,d` triples) |
| `GLINT_ROLL_OFFSET`, `GLINT_MARGIN`, `GLINT_ALT_FILE` | Rollout prompt offset, logit margin logging, alt-step triplets file |
| `GLINT_ONE_STREAM`, `GLINT_BYTE_LOG`, `GLINT_TIME_ABL` | Stream merge, byte accounting, timing ablation probes |

Do not treat README tables as measured evidence; numbers and gates are in the lake paths cited in `GFOLD_SELECTION_2026-11-15.md`.

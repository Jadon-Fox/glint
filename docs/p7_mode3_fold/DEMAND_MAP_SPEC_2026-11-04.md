# Demand map: how much plug does a step need? (spec and decision rules, written before any data)

Date: 2026-11-04. Author: Claude (coding agent) from the operator's statements of 2026-11-04 (session 168e7895) and glint #79. Status: **agent-proposed; open to strike by the operator and the Test Designer before any result is used for a claim.** No demand-map measurement exists in this repository at the time this file is committed.

## Question (operator's premise, stated as a test)
The model does not need the whole model to keep producing the answer. For Phi-4-mini, how much of the plug (the part that lifts the folded model from the NF4 point to BF16 fidelity) must be present at each step for the output to stay close to the full model, and can activations choose it?

## Method: split each linear into parent plus correction
A GLINT weight is `GLINT_CELLS[hole][plug] * absmax`. Write it as `PARENT[hole]*absmax + D`, with `D = (GLINT_CELLS[hole][plug] - PARENT[hole]) * absmax`. Linearity gives, for a linear with input `x`:

`y_sel = H_parent(x) + H_delta(mask(x))`

`H_parent` is the existing H-TILE kernel with the parent table (`GLINT_PARENT_ONLY`, PR #80). `H_delta` is the same kernel with a delta table (cell minus parent). `mask(x)` zeros the input channels whose plug correction is not fetched. For the full mask the result equals the full decode up to f32 summation order; for the empty mask it equals the fold-only arm. Selection is per token (row of `x`) and per linear.

Arithmetic (amended before any data, same day): the H-TILE kernel needs about 135 s per forward and the sweep has about 50 arms (about 4 hours), so the sweep harness decodes parent and correction to bf16 per linear on the fly (transient, one linear at a time) and multiplies with cuBLAS, as the NF4 and INT8 arms do. `H_parent` and `H_delta` above then mean "parent weights" and "correction weights" in bf16, each rounded separately. Consequences, fixed now: the `f = 1` and `f = 0` arms are re-measured in this harness and are the references for the sweep; S1 and S2 below compare against those re-measured arms exactly (an all-ones mask and an all-zeros mask are exact), and against the H-TILE numbers within 0.5% PPL, which is a cross-implementation tolerance sized from the 0.23% gap already seen between the NF4 cuBLAS arm and the fold-only H-TILE arm.

Unit of selection (block granularity is the primary result; channel granularity is run for `act` and `random` only and only after the block sweep is reported):
- **block**: 64 consecutive input channels (a K-block), the realistic fetch unit (all rows' nibbles for those channels).
- **channel**: single input channels, comparable to DecDEC (arXiv 2412.20185).

Scores (a selection rule picks the top fraction `f` of units per token per linear):
- **act**: L2 norm of the token's activations over the unit (the proposal under test).
- **random**: uniformly random units at the same `f`, fixed seed 42 (control).
- **static**: units ranked by their mean activation norm over the first 256 tokens, the same set for every later token (a PowerInfer-style hot set; calibration tokens are inside the evaluation text, so this is report-only).
- **oracle**: units ranked by `norm(x_unit) * norm(D columns)` using the true correction (upper reference, uses information a runtime does not have).

## Arms and fractions
`f` in {1, 0.5, 0.25, 0.1, 0.05, 0.01, 0}, for rules act, random, static, oracle, at both granularities, all 32 layers, all four linears. Model Phi-4-mini, the 2047-token text, oracle PPL 5.486957, glint pin `Phi-4-mini-glint-from-bf16`, teacher-forced.

Sanity gates (an arm that fails one is INVALID, not a result):
- S1: `f = 1` equals the harness's own all-ones-mask arm exactly, and is within 0.5% PPL of the H-TILE hole+plug run (PPL 5.479566, top-1 2007/2047).
- S2: `f = 0` equals the harness's own all-zeros-mask arm exactly, and is within 0.5% PPL of the H-TILE fold-only run (PPL 5.846849, top-1 1869/2047).

## Readouts (every arm MEASURED or INVALID, never PASS/FAIL by itself)
1. PPL and its delta to the oracle; top-1 vs the HF oracle.
2. KL(BF16 CUDA anchor || arm) per token: mean, p99, max. The anchor logits come from `bf16full` with `ARM0_LOGITS_OUT`, run once on the same tokens. (Known: the anchor is itself about 0.0015 KL from HF; reported, not corrected.)
3. Bytes fetched: selected units' plug nibbles over total plug nibbles, per layer and overall. absmax and hole stay resident and are not counted.
4. **Churn**: for consecutive tokens, the Jaccard overlap of the selected sets at the same linear, per layer.
5. **Predictability**: recall of the true `act` set at fraction `f` when the set is replaced by the previous token's set at the same linear.
6. Per-layer sensitivity: for `f` = 0.1, the PPL change when only one layer is restricted (others full), to locate where demand concentrates.

## Decision rules (fixed now)
Let `A` = the full hole+plug arm (PPL 5.479566, top-1 2007/2047; its KL against the anchor is measured in the same sweep). Call fraction `f` *sufficient for rule R* when, at that `f`: top-1 >= 95.0% (the operator-set bar, GATE_RULES C.1) **and** PPL within 2.0% of `A` **and** KL mean <= 2x the KL mean of `A`.
- **Premise supported on this model**: some `f* <= 0.25` is sufficient for `act` at block granularity, **and** at `f*` the `act` rule beats `random` by at least 1.0 percentage point of top-1 or 0.5% of PPL.
- **Premise falsified on this model (this granularity, this text)**: `act` is not sufficient at `f = 0.5`, or `act` is within noise of `random` at every `f` in [0.05, 0.5]. (Noise: two `random` seeds, 42 and 43, define the spread.)
- **Decode-hiding feasible**: reported only as the smallest sufficient `f` and its bytes; the bandwidth argument (a layer's decode compute ~0.2 ms against ~4.4 ms for a whole 48 MiB plug at the measured 11.28 GB/s) is arithmetic, not a measurement, and sets the target of about 5% bytes fetched.
- No outcome here alters the codec, the pin, or a ledger row without the operator's lock.

## What this does not measure (stated, not hidden)
- Teacher-forced prefill selects per token independently, so it does not include the compounding of errors when the degraded model feeds its own tokens back. A rollout (greedy generation from a prompt, comparing token sequences and KL along the generated path against the full arm) is Phase 2b and needs its own spec amendment.
- It does not test the streamed weight-residual (resolution) mechanism, only the plug. Streaming selected rows is a speed question for Phase 3.
- It measures fidelity against bytes, not time. No speed claim follows.
- One model, one text. Qwen3 and the others come after the arm set exists for them.

## Implementation notes (not results)
The delta table is `GLINT_CELLS[h][p] - PARENT[h]`. The kernel keeps its table in `__constant__` memory, so the harness swaps tables between the two launches of a linear. The mask is applied to `x` on the device before the delta launch. Per-token top-f selection is done on the device per linear. Activations for the oracle rule come from the true decoded `D`.

## Amendment 2 (rollout, written before any rollout data; 2026-11-04)
Teacher-forced prefill hides compounding. Rollout measures it with greedy generation by repeated full forwards (causal attention makes the logits at the last real position independent of what follows, so a padded 2048-token buffer is exact; this is not a KV-cache decode path, which does not exist in this repository).
- **Setup:** P = 3 prompts, each the first PL = 64 tokens of the evaluation text at offsets 0, 512 and 1024 placed at position 0; S = 16 greedy steps. 48 generated tokens per arm.
- **Arms:** BF16 CUDA anchor (reference noise floor), full plug (f = 1), fold only (f = 0), `act` channel f = 0.25, `act` channel f = 0.1, `act` block f = 0.25, `random` channel f = 0.1 (seed 42).
- **Metrics:** for each arm against the f = 1 arm: mean matched-prefix length (tokens before the first difference, per prompt) and same-position agreement over all 48 tokens; and agreement in steps 9-16 minus steps 1-8 (the drop). The anchor against f = 1 gives the noise floor: two near-identical models still diverge at near-ties.
- **Rules (fixed now):** an arm *tracks the full model through rollout* when its mean matched-prefix is at least 0.75 of the anchor's and its same-position agreement is at least 0.9 of the anchor's. An arm shows *compounding* when its drop exceeds the anchor's drop by more than 10 percentage points. Fold only and `random` are controls expected to fail both.
- **Limits:** 3 prompts of one text, 16 steps; matched-prefix lengths are integers with large variance; no claim beyond Phi-4-mini on this text.

## Amendment 3 (rollout through the decode path, larger set; written before any data; 2026-11-04)
Amendment 2 used 3 prompts and 16 steps, which could not separate activation selection from random. The KV-cache decode path (glint #84) makes a rollout cheap, so this amendment replaces the sample size and adds a per-layer policy and a random control inside the real selected-fetch path.
- **Setup:** P = 12 prompts (the 64 tokens of the evaluation text starting at offsets 0, 128, ..., 1408), S = 32 greedy steps each, 384 generated tokens per arm, generated by `ARM0_DECODE_ROLL` (decode steps through the KV cache, no padded forward). Reference = the full model (selected-fetch path with f = 1, i.e. the GEMV on hole + plug).
- **Arms:** BF16 CUDA anchor decode (noise floor); fold only (f = 0); selected-channel fetch by |x| at uniform f = 0.05, 0.10, 0.25, 0.50; the same with the channel chosen by a seeded hash instead of |x| (random control) at f = 0.10 and 0.25; and **per-layer policy A**: f = 0.50 in layers {0, 1, 2, 29, 30, 31} and f = 0.05 elsewhere (chosen from the per-layer sensitivity result of the demand map, before this run).
- **Metrics:** against the full model's generations: mean matched prefix (of 32), same-position agreement over all 384 tokens, agreement in steps 1-16 vs 17-32, and the fraction of prompts that match for all 32 tokens. The anchor against the full model gives the noise floor.
- **Rules (fixed now):** an arm *tracks the full model* when its mean matched prefix is at least 0.75 of the anchor's and its agreement is at least 0.9 of the anchor's. *Activation selection is separable from random* at a given f when its agreement exceeds the random control's by at least 10 percentage points. M6's acceptance (glint #92) is met only by an arm that tracks the full model **and** decodes at >= 10 tokens/s in the timing runs of `SELFETCH_OPT_phi4mini.md`.
- **Limits:** one text, 12 prompts, 32 steps; the decode path's arithmetic differs from the padded-forward rollout (m = 1 kernels), so numbers are compared within this amendment, not with Amendment 2.

## Amendment 4 (coarser plug code, the M5 compression direction; written before any data; 2026-11-04)
Amendment 3 found that only about half the plug (about 805 MB per token) lets the activation-selected arm track the full model, which the bus cannot deliver at 10 tokens/s. This amendment tests the operator's M5 ("the compression fix") applied to the GLINT plug: a child index coded in fewer than 4 bits.
- **Method (fidelity emulation, bytes analytic):** `GLINT_PLUG_BITS=b` replaces the 16 x 16 cell table by one in which the 2^(4-b) sub-cells that share a b-bit code (adjacent child indices within an NF4 cell) all take the mean of their cell values (the merged bin's reconstruction level). The kernels, the selection and everything else are unchanged, so this measures what a b-bit plug would reconstruct; bytes per token are counted as `f x (b / 4) x` the 4-bit figure, assuming a b-bit packed plug store that this repository does not implement.
- **Arms** (12 prompts x 32 steps, reference = the 4-bit full-model generations of Amendment 3, same prompts and rules): `act` with b = 2 at f = 0.50 and f = 0.25 (bytes 25% and 12.5% of the 4-bit full plug), b = 3 at f = 0.50 (37.5%), and the full plug at b = 2 and b = 3 (f = 1; the cost of the coarser code alone). Teacher-forced KL against the BF16 anchor for f = 1 at b = 4, 3, 2.
- **Rules (fixed now):** a coarser code *improves the frontier* when an arm **tracks the full model** (matched prefix >= 20.0 and agreement >= 82.5%, the thresholds of Amendment 3) while moving **at most half the bytes** of the 4-bit tracking arm (`act` 50%, about 805 MB per token), i.e. <= 403 MB per token. The 4-bit tracking arm, the anchor and fold only are re-used from Amendment 3 unchanged.
- **Limits:** emulation of reconstruction only (no packed 2-/3-bit store, no kernel that reads it), one text, 12 prompts, 32 steps, one run per arm; a plug coded in fewer bits would also change the codec's encoder, which is outside this test.

## Amendment 5 (tiered plug, from the research brief; written before any data; 2026-11-05)
`RESEARCH_BRIEF_2026-11-05.md` recommends testing SpAx's three-tier idea on top of the resident 4-bit base. Amendment 4 found a 3-bit plug code nearly free at full plug and a 2-bit code unusable, so the cheap tier is 3-bit.
- **Method:** per token and per linear, the top `k1` input channels by |x| receive the 4-bit plug correction; the next `k2` channels (ranks `k1 + 1 .. k1 + k2`) receive a 3-bit-code correction (the merged-bin emulation of Amendment 4 with b = 3, held in a second constant table); every other channel is base-only. Everything else is as in Amendment 3 (decode-path rollout, 12 prompts x 32 steps, reference = the 4-bit full model). Bytes per token are analytic: `(k1 + 0.75 x k2)` times the 1.61 GB 4-bit plug.
- **Arms:** T-A `k1` = 10%, `k2` = 30% (32.5% of the plug, about 523 MB); T-B `k1` = 15%, `k2` = 30% (37.5%, about 604 MB, the same bytes as the 3-bit `act` 50% arm of Amendment 4); T-C (informational) `k1` = 25%, `k2` = 25% (43.75%, about 705 MB). Reference arms re-used unchanged: 3-bit `act` 50% (604 MB, 86.2% agreement), 4-bit `act` 25% (403 MB, 72.7%), 4-bit `act` 50% (805 MB, 99.0%).
- **Rules (fixed now):** the tiered plug *improves the frontier* when **T-A tracks the full model** (matched prefix >= 20.0 and agreement >= 82.5%, the thresholds of Amendment 3) at fewer bytes than any tracking coarse-code arm so far (604 MB), **or** T-B exceeds the 3-bit `act` 50% arm's agreement (86.2%) by at least 5 percentage points at equal bytes. Otherwise it does not.
- **Limits:** emulated reconstruction; no packed mixed-width store or kernel; decode cost of a coded tier is not modelled (SpAx reports 52-64 ms of GPU time per token for its trellis); one text, 12 prompts, 32 steps, one run per arm.

## Amendment 6 (tiered-plug frontier search; written after T-A's result was known, before the arms below were run; 2026-11-05)
Disclosure: Amendment 5's arm T-A (`k1` 10% at 4 bits, `k2` 30% at 3 bits, about 523 MB per token) had already been run and **tracked** the full model (matched prefix 29.33, agreement 91.7%, 10 of 12 prompts identical) when this amendment was written; T-B and T-C were still running. This amendment fixes the next arms and a byte target before their data exist.
- **Byte target for the operator's 10+ tokens/s goal (an estimate, not a measurement):** the fold-only decode takes 39.1 ms per token and the optimized selected fetch added time at an effective 4.0-4.9 GB/s (`SELFETCH_OPT_phi4mini.md`), so 100 ms per token allows about 61 ms of fetch, i.e. about 245-300 MB; **285 MB** is used as the threshold. It assumes a packed mixed-width store and a free unpack, which do not exist here.
- **Arms** (`k1` at 4 bits, `k2` at 3 bits, bytes = `(k1 + 0.75 k2)` x 1.61 GB): F1 (5%, 20%) = 20.0% = 322 MB; F2 (10%, 15%) = 21.25% = 342 MB; F3 (10%, 10%) = 17.5% = 282 MB; F4 (5%, 10%) = 12.5% = 201 MB; F5 (1%, 30%) = 23.5% = 378 MB, the control that is almost 3-bit only, to isolate what the 4-bit top band adds to T-A.
- **Rules (fixed now):** an arm *tracks* when its matched prefix >= 20.0 and agreement >= 82.5% (Amendment 3). The tiered plug is **byte-feasible for 10 tokens/s** if some arm that tracks moves <= 285 MB (F3 or F4). F5 is compared with T-A at the same band widths to attribute the gain to the 4-bit top band: the 4-bit band is credited if T-A exceeds F5 by at least 5 percentage points of agreement.
- **Limits:** as Amendment 5; the speed target is an estimate from a model of the fetch, not a measurement of a mixed-width kernel.

## Amendment 7 (does calibration lower the 4-bit floor? llama.cpp imatrix as the available proxy; written before any data; 2026-11-05)
`RESEARCH_BRIEF_2026-11-05.md` recommends testing a calibrated base, because calibrated 4-bit methods are reported (one unread preprint) at KL about 0.01 against the NF4 hole's 0.03-0.15. GPTQ, AWQ and HQQ cannot be run here (workspace Python-environment policy U-011). llama.cpp's importance-matrix quantization is installed and calibrated, so it measures the effect of calibration on 4.5-bit bases as a proxy, not a GLINT arm.
- **Method:** compute an importance matrix on the BF16 Phi-4-mini GGUF with `llama-imatrix` from corpus text that is **disjoint from the 2048 evaluation tokens** (the corpus after its first 20,000 bytes, 20 chunks of 2048 tokens); re-quantize the same BF16 file to Q4_0, IQ4_NL and Q4_K_M with that matrix; score each exactly as in `COMPETITOR_llamacpp_phi4mini.md` (`llama-perplexity --kl-divergence` on the first 2048-token chunk against the BF16 baseline logits). Compare with the same quantizations without a matrix (mean KL 4.84e-2, 2.69e-2, 2.09e-2 and same-top 93.06%, 95.70%, 95.80%).
- **Rule (fixed now):** calibration is **worth pursuing for the GLINT base** if the importance matrix lowers the mean KL of Q4_0 and of IQ4_NL by at least 30% each (relative to the no-matrix figures above); if it lowers them by less than 15% each it is not; in between is inconclusive.
- **Limits:** a proxy on a different codec and a different quantizer family; llama.cpp's matrix weights the quantization error per input channel, which is not what an NF4 hole with a data-free absmax does; one text, one chunk, one run.

### Amendment 7b (deviation, recorded before any imatrix data exists)
The calibration text of Amendment 7 cannot be built: `SCIENCE_JOURNAL.md` is only 24,256 bytes (about 6k tokens) and the evaluation uses its first 2048 tokens, leaving about 1k tokens after byte 20,000, not 20 chunks of 2048. The importance matrix will instead be computed from about 100 KB of markdown taken from `/home/workspace/docs/` (general project documentation, none of it the journal), 24 chunks of 512 tokens or fewer if the text runs out. The method, scoring and the 30% / 15% rule are unchanged. A calibration set from a different domain than the evaluation text is the common setting for imatrix quantization; it also makes any gain a conservative estimate of what a matched calibration would give.

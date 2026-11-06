# Calibrated GLINT hole (GPTQ-style rounding onto the NF4 parent grid), Phi-4-mini — spec Amendment 11, glint #92 (pre-registered; results at the end)

Written and committed **before** any result exists. Follows `CALIBRATION_PROXY_llamacpp.md`, which tested calibration on a different codec; this builds the real thing for GLINT's hole. Tool: `scripts/glint_gptq_hole.py`, run in the registered container environment `/opt/venv` (torch 2.5.1+cu121, transformers 4.57.6), no new packages. Rationale for not using a GPTQ library: they target a uniform 4-bit grid; GLINT's hole is the NF4 code, so the rounding is implemented on that grid directly.

Method: per decoder linear (128 of them), GPTQ with H = 2/n X^T X from 30 x 512 tokens of `calib_docs.txt` (workspace markdown, disjoint from the evaluation text), damp 0.01, columns left to right in blocks of 128, group scale = absmax of the updated weights of each 64-column group, layers in sequence over hole-only outputs. The plug is the nearest child cell *inside the chosen parent cell* for the original weight, so a weight whose calibrated hole is not its nearest cell saturates at the cell edge. The embedding/lm_head and norms are copied from the source pin unchanged.

Arms (evaluation text: first 2047 tokens of `SCIENCE_JOURNAL.md`, reference = BF16 logits `anchor_logits.bin`): (a) fold-only (parent only) on the existing NF4 pin = baseline; (b) fold-only on the calibrated pin; (c) hole + plug full on both pins.

**Decision rule (fixed now):**
- *Worth pursuing* if (b) mean KL <= 70% of (a) (a reduction of at least 30%, the bar used for the imatrix proxy) **and** (c) on the calibrated pin has PPL <= 1.01 x the existing pin's full PPL (5.4885, i.e. <= 5.543).
- *Calibration helps the base but costs the full arm* if (b) meets the 30% bar but (c) exceeds 1.01 x.
- *No help* if (b) is above 70% of (a).
One run, one calibration text, one seed; no noise estimate beyond that.


---
# RESULTS (2026-11-06; the rule above was committed before this run, commit 2997128)
Build: `scripts/glint_gptq_hole.py`, 30 x 512 calibration tokens (16,711 tokens in the file), 128 of 128 linears replaced, 462 s on the RTX 3060 in `/opt/venv`; the pin is at `/home/workspace/lakes/p7_glint/gptq_hole_20261106/pin`. Evaluation: first 2047 tokens of the journal against the BF16 logits, one run each. Raw logs next to the pin.

| Arm | mean KL vs BF16 | PPL (BF16 oracle 5.487) | top-1 agreement |
|---|---|---|---|
| (a) fold-only, existing NF4 pin | 7.90e-2 | 5.847 | 91.30% |
| **(b) fold-only, calibrated hole** | **3.03e-2** | **5.598** | **93.99%** |
| (c) hole + plug, existing pin | 8.7e-4 | 5.480 | 98.05% |
| **(c) hole + plug, calibrated pin** | **5.41e-2** | **5.742** | **92.82%** |

## Verdict under the pre-registered rule
- (b) / (a) = 0.384, a **61.6% reduction** in KL (bar: at least 30%): **met**. The calibrated base alone (PPL 5.598) is better than the data-free NF4 base by 0.25 PPL, a bigger gain than any llama.cpp proxy in `CALIBRATION_PROXY_llamacpp.md`.
- (c) on the calibrated pin: PPL 5.742 = 1.048 x the existing pin's 5.480 (bar: at most 1.01 x, 5.543): **not met**. The full reconstruction is worse than before by 4.8% PPL and its KL is 60 times higher (5.4e-2 against 8.7e-4).
- Outcome: **"calibration helps the base but costs the full arm."** This is the second branch of the rule.

## Reading
- *Measured:* a calibrated hole and the existing plug do not compose. The plug can only choose a child cell inside the parent cell the hole picked. GPTQ's error feedback sometimes picks a hole that is not the weight's nearest cell; the plug then saturates at that cell's edge, so hole + plug is worse than the nearest-cell pin. (The fraction of weights where the calibrated hole differs from the nearest cell was not counted in this run.)
- *Measured:* this matters for the fetch design, not only for fold-only. In any tracking arm the fetched channels use hole + plug and the rest use the hole alone, so a calibrated pin helps the unfetched 60-90% and hurts the fetched 10-40%.
- *Inference, untested:* the plug needs to be built to correct the calibrated hole, not to refine the nearest cell. Candidates: pick the plug against the GPTQ-updated weight; allow the plug to address the neighbouring parent cell; calibrate the hole with the full 256-level grid in the error model; or give the plug its own table. Which one keeps the 62% base gain and the 98% full-arm fidelity is the next experiment.

## What this does not show
One calibration text of 16.7k tokens, one run, 2047 evaluation tokens, one seed; the calibration and evaluation texts are different (workspace docs against the journal) but both are English technical prose. No rollout, tracking or speed measurement was made on the calibrated pin yet. Not reviewed by the Test Designer.

---
# Amendment 11b (written after the results above were known; the rule below is fixed BEFORE its runs)
The result above shows the plug saturating in the wrong cell. Test of the cheapest candidate fix: widen each parent row of the 16x16 child table by a factor F around the parent value (clipped to [-1, 1]) so a plug can reach past the cell the calibrated hole picked, then re-choose every plug as the nearest wide-row child of the original weight (`scripts/glint_wide_plug.py`; `ARM0_CELLS_FILE` loads the table in the harness). Holes and absmax are untouched, so fold-only is unchanged by construction (it uses the parent value only); only the full hole + plug arm is measured. Two factors, both reported: F = 1.5 and F = 2.0 (no best-of selection). The table is global, so the embedding plugs are re-encoded too.
**Rule:** the fix works if full hole + plug on the widened pin has PPL <= 1.01 x the existing pin's 5.480 (<= 5.535) with F = 1.5 or F = 2.0; it partly works if PPL is below the 5.742 of the unwidened calibrated pin but above 5.535; it fails otherwise. The existing (nearest-cell) pin evaluated with the same widened table is the control for the resolution cost of widening alone.

---
# RESULTS of Amendment 11b (2026-11-06)
**Correction first.** The first widened-table evaluations ran through `--glint` (the resident H-TILE path), which decodes with the built-in table in `cuda/glint_htile.cu`, not with `ARM0_CELLS_FILE`; those numbers (PPL 5.675 / 5.604, controls 5.570 / 5.612) are **invalid** and are not used. The valid runs use `--glint-demand` with `ARM0_DEMAND_RULE=act ARM0_DEMAND_F=1`, whose dequantization reads the custom table (the log prints "custom child table"). On this path the existing pin scores 5.4885 (the earlier 5.480 is the `--glint` path; the rule's bar is recomputed on the demand path: 1.01 x 5.4885 = 5.543).

| pin, table | PPL | top-1 | mean KL vs BF16 |
|---|---|---|---|
| existing (nearest-cell) pin, built-in table | 5.4885 | 98.19% | 9.5e-4 |
| existing pin re-encoded, F = 1.5 (control) | 5.4855 | 97.85% | 1.6e-3 |
| existing pin re-encoded, F = 2.0 (control) | 5.4938 | 97.61% | 2.5e-3 |
| calibrated pin, built-in table | 5.7468 | 93.21% | 5.4e-2 |
| calibrated pin, F = 1.5 | 5.7155 | 93.94% | 4.8e-2 |
| calibrated pin, F = 2.0 | 5.7172 | 93.60% | 4.6e-2 |

- *Measured:* widening costs little by itself (controls: PPL +0.0 to +0.005, KL 1.6e-3 to 2.5e-3), so a wide table is cheap for a pin whose holes are nearest cells.
- **Verdict under the rule:** the calibrated pin's full arm is 5.716-5.717 against the bar of 5.543: **not met**. It is below the unwidened 5.747, so the literal outcome is "partly works", but the improvement is 0.03 PPL (0.5%) and the full arm stays far from the existing pin (KL 4.6e-2 against 9.5e-4). Practically: the wide table does **not** fix the composition problem.
- *Measured (diagnostic on one layer, `scripts/glint_gptq_diag.py`):* 23.8% of the calibrated holes are not the weight's nearest cell; hole-only weight RMSE rises from 0.0358 (nearest) to 0.0502 while output KL falls, as GPTQ intends. With F = 2.0 the plugs bring the weight RMSE to 0.0086 (existing pin 0.0025).
- *Inference, not tested:* the full arm on the calibrated pin is worse than the calibrated hole alone (KL 4.6e-2 against 3.0e-2) because the plug removes part of the weight error that GPTQ had arranged to cancel in later columns: a partial correction of the first columns leaves the compensation in the later ones unmatched. If so, a calibrated hole and a fetched plug are in tension for any plug that does not itself go through the same error-feedback rounding. A fix would need the plug (or the hole + plug pair) chosen by GPTQ on the fine 256-level grid, which gives the full arm back but leaves the hole near its nearest cell (little base gain), or a second stored base. Which trade-off suits the tiered design is the question of Amendment 11c.

---
# Amendment 11c (fixed before its runs): does the calibrated hole help the tiered design?
The tiered arms use the hole alone for most channels and hole + plug for the fetched ones, so what matters is the mixture, not either end. Test: the rollout study of `TIERED_PLUG_phi4mini.md` (12 prompts x 32 steps, reference = the existing pin's full model decode, `decode_rollout_20261104/full.txt`) with the **calibrated pin (built-in table)** for T-A (10% + 30% 3-bit), F3 (10% + 10% 3-bit, 282 MB) and 4-bit `act` 25% (403 MB), against the existing-pin results (T-A 91.7%, F3 58.9%, act 25% 72.7%). Note the reference is the existing pin's own full model, so a calibrated pin that is closer to BF16 but different from that reference is penalized; the BF16-anchor agreement for the reference is 91.7%, so differences under about 8 points are noise.
**Rule:** helps the tier if all three calibrated arms score at least 5 points above their existing-pin counterparts; hurts if all three score at least 5 points below; otherwise no clear change. One run per arm.

---
# RESULTS of Amendment 11c (2026-11-06)
Rollout study (12 prompts x 32 steps), reference = the existing pin's full-model decode (`decode_rollout_20261104/full.txt`), as pre-registered; one run per arm; scorer `scripts/rollout_score.sh`.

| arm | existing pin (agreement) | calibrated pin (agreement) | change |
|---|---|---|---|
| T-A (10% + 30% 3-bit) | 91.7% | 52.9% | -38.8 |
| F3 (10% + 10% 3-bit) | 58.9% | 53.1% | -5.8 |
| 4-bit `act` 25% | 72.7% | 59.1% | -13.6 |

**Verdict under the rule: hurts** (all three at least 5 points below their existing-pin counterparts). The reference favours the existing pin (it is that pin's own full model), so the following comparisons, which were NOT pre-registered and are post hoc, use the better ground truth, the **BF16 anchor decode** (`decode_rollout_20261104/anchor.txt`; the noise floor is the existing full model scoring 91.7% against it).

| arm | existing pin vs BF16 anchor | calibrated pin vs BF16 anchor |
|---|---|---|
| base only (no plug) | 21.4% (prefix 3.7) | **46.1%** (prefix 9.6) |
| 4-bit `act` 25% | 68.8% | 54.4% |
| F3 | 54.2% | 47.7% |
| T-A | **83.6%** (prefix 24.1) | 47.9% (prefix 10.2) |
| full (hole + plug, all channels) | 91.7% | 56.0% |

- *Measured (the positive):* the calibrated base alone more than doubles generation agreement with BF16 at zero plug bytes: **46.1% against 21.4%** (KL 3.0e-2 against 7.9e-2). It is the best zero-byte arm measured.
- *Measured (the problem):* adding the plug does not add to it. On the calibrated pin every plug arm stays at 47-56%, within a few points of the base alone, and even the full arm drops to 56.0% against the existing pin's 91.7%. The existing pin with a plug (T-A 83.6%) is far better than any calibrated arm.
- Consequence: **a calibrated base and the in-cell plug do not compose, and the combination is worse than the data-free pin plus plug.** Quality-first ruling: keep the existing pin as the tiered design's base. The calibrated base is a result to build on, not a replacement yet.
- *Inference, untested:* GPTQ spreads each weight's rounding error over the later columns, so restoring some channels to their original values (what a plug does) un-cancels that compensation in the channels left at the hole. A plug (or a hole + plug pair) produced by the same error-feedback rounding, a residual plug stored relative to the calibrated hole with its own scale, or calibrating the base with the fetched fraction known are the candidates; none was run.

## What this does not show
One text, one calibration text, 12 x 32 rollouts, one run per arm; the BF16 comparisons are post hoc. The calibration set was 16.7k tokens (30 x 512), damp 0.01, one seed. Not reviewed by the Test Designer.

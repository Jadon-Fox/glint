# GLINT open questions, with context (2026-11-06)

Each entry: **the question**, *why it matters*, what is **known** (with where it lives), who or what can **answer** it, and what the answer would **change**. Numbers are Phi-4-mini on an RTX 3060, one text, unreviewed draft PRs (#95-#105) unless stated. Shorthand is defined at the end. Operator rulings in force: quality comes first; prefetch has no fixed method (any signal from the model's causal flow that works); a card stays open while work on it is open.

## A. Questions the existing record may already answer (read before building)
**A1. Does the in-multiply plug scheme (issue #25's Google Doc) need the channel list before a layer starts?**
*Why:* it targets the bottleneck I measured (a fetch that cannot hide behind compute). The Doc describes loading a plug tile, consuming it, evicting it and keeping the accumulator. If the tile list is fixed in advance it needs a predictor; if it can be decided mid-GEMM it may fit the per-linear chain.
**Known:** only the issue summary is read (#25). The Doc is marked NOT RUN / NOT VERIFIED and says the current GLINT 4+4 Gaussian code is lossy. **Answer:** read the Doc (Google Docs MCP) and query the NotebookLM notebooks. **Changes:** whether the speed route is "predict early" or "restructure the GEMM".

**A2. Has this exact premise (the model does not need all of itself to begin answering) been shown to track at speed?**
*Why:* if yes, copy its mechanism; if no, the claim is new.
**Known:** web research brief `RESEARCH_BRIEF_2026-11-05.md` (DecDEC OSDI'25, SpAx arXiv 2610.02598, HOBBIT, TEAL, Deja Vu, PowerInfer); SpAx is closest (per-token |x| tiers, no resident base, +10% PPL). Not checked against the project's own notebooks (@mve, @llmve-docs, @hsve). **Answer:** `nlm query notebook`. **Changes:** how much to borrow vs invent.

**A3. What did earlier generations measure on streaming vs plug?**
*Why:* you said a check on "what gets streamed and when" was only half done and then superseded.
**Known:** my streamed-plug arm reproduces the resident run byte for byte (`FOLD_STREAM_phi4mini.md`); budget knob saves 1.3 GB. Earlier pins/tests live in the glint repo history and Project 6 cards P7-0..P7-12. **Answer:** read `docs/p7_mode3_fold/STATUS_2026-11-04.md` and the ledger entry (#87). **Changes:** whether anything from before conflicts with the current ranking.

**A4. Is "tracks the full model" defined anywhere in the project?**
*Why:* I used token agreement on a 12-prompt x 32-step rollout with bar 82.5% and prefix >= 20; that bar is mine (set in `DEMAND_MAP_SPEC_2026-11-04.md` amendments). The BF16 anchor against the full model scores 91.7% by numerical noise alone.
**Answer:** operator or the spec. **Changes:** every "tracks / does not track" verdict.

## B. Design questions (the operator decides; context given)
**B1. What is the stored "base" format, and does the training ban matter?**
*Why:* "base" here means the number format of the stored low-resolution layer (the hole), not the model. The hole equals NF4; the orchestrator bans NF4 as a training base (it trains INT8).
**Known:** rulings 2026-10-29 (INT4 stored / INT8 compute) and 2026-11-04 (all Ampere-compatible sizes); INT4 base built and measured (PR #98); tensor-core rates (INT4 210 TOPS, INT8 102, BF16 52) in `TENSOR_FORMATS_rtx3060.md`. **Status:** treated as answered; loader (#100) starts inference-only, format-agnostic.

**B2. Quality first: what is the quality bar for each stage?**
*Why:* speed is ranked only after quality is established. Without a stated bar I cannot say when a stage is "good enough" to move on.
**Known:** fold-only (NF4) PPL 5.847, KL 7.9e-2; calibrated hole 5.598, 3.0e-2; hole + plug 5.480, 8.7e-4; BF16 oracle 5.487. **Answer:** operator picks, e.g. "full path within 1% PPL, base within X". **Changes:** what counts as a pass.

**B3. How should prefetch signals be found, given no fixed method?**
*Why:* two rules failed (previous token: T-A 51.0%; previous layer: 49.0%, bar 82.5%). The remaining space is large (attention outputs, residual-stream change, token identity, learned predictors).
**Known:** current-|x| selection is the ceiling (act 50% = 99.0%). **Proposal:** score candidate signals offline by how much of the current-|x| top-k they recover per layer and linear (no rollout needed), then roll out only the best. **Changes:** replaces one-rule-at-a-time experiments with a ranked search.

**B4. Is the end picture (20B model, 32 GB RAM + GPU) subject to the same per-linear dependency?**
*Why:* at 20B and PCIe Gen3 x16 (11.6 GB/s measured best) the byte budget per token is much tighter; the design may need the early-prefetch route.
**Known:** 523 MB/token at 3.8B costs ~77 ms plus corrections ~32 ms. **Answer:** a byte-budget calculation for 20B (cheap); then a target.

## C. Measurement questions (experiments or checks)
**C1. Does a calibrated hole compose with the plug?** (in progress)
*Known:* GPTQ rounding on the NF4 grid cut fold-only KL by 61.6% (7.90e-2 to 3.03e-2, PPL 5.847 to 5.598) but hole + plug got worse (PPL 5.480 to 5.742, KL 8.7e-4 to 5.4e-2) because the plug can only choose a child cell inside the parent the hole picked (`GPTQ_HOLE_phi4mini.md`). Amendment 11b tests widened child tables (F = 1.5, 2.0), rule fixed before the runs, pins building now. *Changes:* whether the best base (calibrated) and the best full path (plug) can be had together.

**C2. How often do the same channels get picked from one token to the next?**
*Why:* if overlap is high, a resident cache of recent channels removes most fetches; this is a different lever from prediction. **Known:** not measured. **Cost:** cheap (log top-k per linear per token).

**C3. Which layers and linears need the plug most?**
*Why:* every arm so far uses one fraction everywhere; a per-layer allocation might keep tracking at fewer bytes. **Known:** the earlier `ARM0_SELFETCH_HI_LAYERS` option exists and was only used on a few layers. **Cost:** a sweep.

**C4. Does tracking hold on a second text and on the other models?**
*Why:* all decode/rollout results are Phi-4-mini on one text. **Known:** fold/plug arms (no decode) exist for Qwen3-4B, DeepSeek-1.5B, SmolLM2 (`GLINT_ARMS_4_MODELS.md`). Card #85 is open for this.

**C5. Why does the real decode path take ~164 ms when the benchmark implies ~115 ms?**
*Known:* ablation: base+selection 41 ms, gathers ~100 ms, corrections ~32 ms (`E2E_DECODE_TIER_phi4mini.md`); fused benchmark 77 ms for the same bytes. Unexplained: ~49 ms. **Answer:** profile with ncu (nsys lacks an importer here) or fuse the kernel.

**C6. Is the decode path's run-to-run variation a race or greedy near-ties?**
*Known:* `DECODE_CHECK` varies in the old build too (top-1 equal 105-107 of 112). Not investigated. *Changes:* whether any decode-path result needs repeated runs.

**C7. Does the induction/copy probe matter at all?**
*Known:* all eight arms scored 87.2-89.2% including base-only; it does not separate arms. Natural-text retrieval (needle in a document) is untested.

## D. Board and process questions
**D1. Does "DONE" mean executed or accepted?** Ruling: executed and documented, never accepted; reopened cards (#92, #79, #85, #86, #100, #25, #21, #22, #14, #231, #335, P7-4/11/12) carry context in the board comment on #79. Remaining question: should unreviewed draft PRs (#77-#105) be reviewed in a batch, and by whom?

**D2. The stray branch** `fix/p7-tier-gather-overlap` on GitHub has one orphan commit (the rebased copy is `fix/p7-tier-overlap-stacked`). Delete it?

**D3. Should the stop-hook goal be rephrased?** The earlier goal ("finished boards and a rough plan executed") could not be met without reviews, so it produced repeated reports. A goal like "reach the next quality milestone" would be checkable.

## Shorthand
*hole* = 4-bit parent cell (NF4). *plug* = child 4-bit sub-quantile inside a hole cell (16x16 table). *fold-only* = hole alone. *act N%* = plug fetched for the top N% input channels by the current |x|. *T-A* = 10% of channels at 4-bit + next 30% at 3-bit (~523 MB/token). *tracks* = agreement >= 82.5% and prefix >= 20 of 32 vs the full 4-bit model. *M6* = routing with all mechanisms active (target: tracks and 10+ tokens/s; currently ~6).

## Update 2026-11-06 (after the Drive review "Project 7 — Architecture re-evaluation and external research")
Review document: Google Drive file `1EC3iTTpbRW2Bycyoo_5KH0nxk2Kgag4c4WivwZYCRoM` (written by another assistant, read-only audit; cites RRQ arXiv 2608.04048, NeuroPrefetcher 2608.22643, DIP MLSys 2025, SpAx 2610.02598).
- **A4 answered.** The recorded quality gates are in `docs/p7_mode3_fold/GATE_RULES.md`: top-1 agreement with the HF oracle >= 95.0% for every arm, PPL within 2.0% of BF16 (PASS; 2-5% INCONCLUSIVE; >5% FAIL), transport bit-identical, KL reported (the demand-map spec adds an agent-proposed KL <= 2x the full-plug arm). My "tracks" bar (rollout agreement >= 82.5%, prefix >= 20) is **not** a recorded gate; it was set in my spec amendments as a decision aid. Against the recorded gates: full hole+plug passes (PPL 5.4885, 98.19%); fold-only NF4 fails (PPL +6.5%, 91.3%); calibrated fold-only is INCONCLUSIVE on PPL (+2.0%) and fails top-1 (93.99%); the tier arms (T-A etc.) have **not been measured on the teacher-forced gate at all**, only by rollout.
- **The review's section 3 table uses the invalid widened-table runs** (the `--glint` path ignores `ARM0_CELLS_FILE`); its "widening sacrifices resolution" conclusion does not hold. Valid controls: existing pin with F=1.5 / 2.0 tables PPL 5.4855 / 5.4938 (`GPTQ_HOLE_phi4mini.md`).
- **New questions raised:** (1) residual plug relative to the calibrated hole (RRQ/AQLM) versus the in-cell child plug: does an independently scaled residual fix the composition problem, or does the GPTQ compensation mismatch remain? (2) Prediction versus residency: separate selection recall, cache hit rate and new bytes fetched; the demand map already shows block recall of the previous token's set of 35-66%. (3) One early-layer signal predicting many downstream selections (NeuroPrefetcher) versus per-linear. (4) The 20B byte budget: T-A scaled unchanged moves ~3.25 GB per token (~280 ms at 11.6 GB/s); at 10 tokens/s the link allows ~1.16 GB per token. (5) Measure T-A / F3 / act arms on the recorded teacher-forced gate.

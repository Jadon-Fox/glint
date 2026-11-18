# G-fold applied to GLINT plug selection (2026-11-15)

Status: DESIGNED, rules written before any run. Nothing here is measured yet except where a source is cited.
Card: GLINT-MOJO-S1 (#117). M6 (#92) is not accepted and this document does not claim it.

## 1. The operator's statements this rests on

- S66: "you are pulling yourself into the future. You're pulling yourself to your location. Through the math." The efficiency is "a relational efficiency metric. Not miles per gallon."
- S103: guided "not by what could happen, but what you desire to happen".
- S78: the forward pass ends a round; that boundary is where G-fold acts.
- 2026-11-14/15: the plug should "add a piece of itself to finish the calculation", with no decode; count in steps and causal change, not wall time; "we can't have any stumbles".
- Presuppositional axes P2 (constraints differentiate outcomes, ordered relative to telos), P3/P4 (a cross-domain mapping is an isomorphism of real functions and must be tested by interaction), Generate -> Constrain -> Select.

Everything below is the agent's working-out of those statements. It is inference until measured.

## 2. The isomorphism, stated as functions (P3/P4)

Published lossless convexification (Acikmese-Blackmore; `llmve-methodology/methods/GFOLD_LCXV_STOICH/README.md`): a control set with a hole, {0} or rho_min <= |u| <= rho_max, is lifted with a slack variable; because the cost taxes the slack, the optimum of the convex problem sits on the original set. It is solved from the current state to a fixed target and re-solved each round.

| G-fold | GLINT decode | Why it is the same function |
|---|---|---|
| Target state fixed in advance | The output the full-plug model produces at this token | Both are the end the present step is chosen by |
| Control u at each step | Which plug rows are added at each (layer, linear) | The only thing chosen per step |
| Dynamics linear in the control | `y = F x + sum_c u_c D_c x_c` inside one linear (F fold, D_c the plug delta column of input channel c) | Exactly linear in u before the next nonlinearity (2026-10-27 scratchpad identity `A_{k+1} = A_k + X_k (F_k + D_k)`) |
| Control set with a hole | u_c is 0 (row absent) or 1 (row present); with tiers: absent, 3-bit, 4-bit | A row cannot be "a little fetched"; this is the lean hole |
| Slack lift, tight at optimum | Relax u_c to [0,1]; the relaxed problem is a linear program whose optimum is 0/1 for every row except at most one per constraint | Fractional-knapsack property: the lift loses nothing but one boundary row |
| Fuel is taxed | Bytes moved and ring VRAM are the constraints, not the objective | Operator delta: the objective is relational (error relative to the target), not bytes |
| Costate, propagated back from the target | Weight g_j: how much an error at linear j moves the final output | This is the "pull" of the target on the present step |
| Glide-slope / time constraints | Bytes that can land before linear j is computed (lead time), ring capacity | Linear constraints on cumulative control |
| Re-solve each round from the observed state | Skeleton projects ahead; the real pass re-selects from its true activations and fetches only the misses | Receding horizon |

Deltas (where the mapping is NOT an identity):
1. The objective is error relative to the full model, not fuel. Minimum bytes is not the goal; the byte and VRAM limits are constraints.
2. Linearity holds inside a linear and inside a layer up to the nonlinearity. Across layers the model is not `x' = Ax + Bu`, so the published tightness theorem does not transfer across layers (the method doc already says this for attention). Across layers the weights g_j are an approximation to be measured, not a theorem.
3. Error energies are added as if uncorrelated across channels and linears. Recorded support: single-layer excess KL summed to 3.16e-2 against 2.91e-2 together, additive within 9% (DEMAND_MAP_phi4mini.md). Not proven per channel.

## 3. The rule that falls out

Minimise `sum_j g_j sum_c (1 - u_jc) |x_jc|^2 ||D_jc||^2` subject to `sum_j sum_c b_j u_jc <= B` (bytes per token) and the per-step lead-time and ring limits. b_j is the bytes of one plug row of linear j (half the output width). With one multiplier lambda on the byte limit, the optimum is:

    fetch row c of linear j  iff  g_j |x_jc|^2 ||D_jc||^2 / b_j >= lambda

If the delta column norms are near-uniform within a linear, `||D_jc||^2 = R_j d_j^2` (R_j output width, d_j the RMS delta element) and `b_j = R_j / 2`, so the output width cancels:

    fetch iff  |x_jc| >= lambda' / (d_j sqrt(g_j))

One price for the whole model, one threshold per linear, both static except lambda. Consequences:
- The fraction is no longer a setting. Each linear, each layer and each token takes what its activations are worth at the current price. Non-uniform per-layer allocation (operator ruling 2026-11-14) is the solution of the program, not a hand policy.
- lambda is the single control. It can be set to hold bytes per token at the link's budget, or set per step so the rows that must land before linear j is computed always can. A row that cannot land in time is priced out before it is requested, instead of stalling the real pass. That is the "no stumbles" rule as a constraint.
- Tiers are the same rule with two prices: rows above lambda_hi at 4-bit, rows between lambda_lo and lambda_hi at 3-bit.

What the present decoder does instead: the same fraction of channels in every linear, chosen by |x| rank within the linear. That is the price rule only if every linear had the same d_j, g_j and activation scale, which they do not (layer 3 `down_proj` input: max 2800 against rms 0.68, activation map).

Recorded evidence that bears on it:
- Column-norm weighting inside one linear did not help on the INT4 residual (decdec vs decdec_norm within noise, INT4_BASE_MEASURED). So the within-linear part of the rule may add nothing; the untested part is the cross-linear and cross-layer price.
- Hand per-layer policies (end layers at higher fractions) did not track (MOJO_GLINT_DECODE, nine arms). Those moved whole layers; none equalised a marginal value.
- The last two layers are the most sensitive (DEMAND_MAP), which is a first reading of g_j.

## 4. Hypotheses, written before data

H-G1 (price beats rank at equal bytes). At equal mean plug bytes per token, selection by the one-price rule with g_j = 1 gives a lower teacher-forced mean KL to the full-plug arm than the uniform-fraction rule, on the 2047 positions.
- Test: byte budgets matched to uniform 10%, 25% and 40% (163, 404, 646 MB per token); KL measured against the Mojo full-plug arm with the same kernels.
- falsifyIf: the price rule's KL is not lower by at least 10% relative at two of the three budgets. A null is reported as a null and the uniform rule stays.

H-G2 (costate weights help). With g_j set from the measured single-layer sensitivities, KL at equal bytes is lower than with g_j = 1.
- falsifyIf: not lower by at least 10% relative at two of three budgets.

H-G3 (tracking follows). The best rule from G1/G2, with skeleton and exact-miss fixup, tracks free-running at fewer bytes than the uniform rule.
- Test: the 12 prompts x 32 steps, each arm run 3 times, plus a second prompt set of the same size so the result is not one sample. Bars as recorded and flagged agent-written: prefix >= 20.0, agreement >= 82.5%.
- falsifyIf: no price-rule arm below 566 MB per token (the uniform 35% arm) passes both bars in all 3 runs on both prompt sets.

H-G4 (lead time). With lambda set per step from the bytes that can land before each linear is computed, the real pass waits zero steps for prefetched rows.
- Test: per-linear counters of rows needed, rows already landed and rows fetched late.
- falsifyIf: late rows above 1% of needed rows at the chosen budget.

Each run is a few minutes on the 3060 through `scripts/p7_mojo_select_retest.sh`; nothing here is multi-hour.

## 5. Build order (each step ends in a check)

1. Offline: d_j for the 128 linears and the spread of delta column norms, from the pin (CPU only). Output `lakes/p7_glint/gfold_price_20261115/plug_delta_stats.tsv`. Decides whether column norms can be dropped.
2. Decoder: a threshold mode for the device-side selector (threshold given, count returned) beside the existing rank mode, with per-linear caps so staging buffers stay bounded. Check: with thresholds back-computed from a uniform run, outputs equal that run bit for bit.
3. A calibration pass that finds lambda for a byte budget (bisection on a 128-position prefix). Check: measured mean bytes within 2% of the budget.
4. H-G1, then H-G2, then H-G4 counters, then H-G3.
5. Separately, the additive plug (`w = fold + r * scale`, no table decode at apply time): checked offline on weights against the child-nibble plug at equal bits before any kernel work.

## 6. Results

### Step 1, offline pin statistics (2026-11-15, CPU only, Phi-4-mini `glint_pin_v1`, 128 linears)
Source: `lakes/p7_glint/gfold_price_20261115/plug_delta_stats.tsv` (script kept beside it in the run log).

| Linear | delta RMS d_j (min / mean / max over 32 layers) | column-norm spread within a linear (cv mean / max) |
|---|---|---|
| qkv_proj | 0.00206 / 0.00331 / 0.00404 | 2.8% / 14.7% |
| o_proj | 0.00180 / 0.00263 / 0.00323 | 11.2% / 30.5% |
| gate_up_proj | 0.00296 / 0.00304 / 0.00357 | 1.8% / 7.9% |
| down_proj | 0.00231 / 0.00299 / 0.00313 | 4.2% / 10.8% |

Reading: inside a linear the delta column norms are close to uniform, so the per-channel norm can be dropped and one threshold per linear is enough (agrees with the recorded decdec vs decdec_norm null). Across linears d_j spans 2.2x. Whatever the price rule changes will therefore come mostly from the differences in activation scale between linears and from g_j, not from d_j. Price table with g_j = 1: `price_table_g1.txt` in the same folder.

### Step 2, price mode in the decoder
`select_topk` takes a key threshold; the percent argument becomes the staging cap. Check: price unset, 35%, 128 positions reproduces the earlier run exactly (PPL 74.34960059115652, 120/128, 566,013,868 B/token). `lakes/p7_glint/gfold_price_20261115/identity/`.

### H-G1 result: NULL (falsified as tested), 2026-11-15
2047 teacher-forced positions, mode `fast+attn+dsel+kmaj+vgat+skel+ksplit+fix+kgemv+ahead`, one run each, cap 60% for price arms. Deviation from the pre-registration: the logs give top-1 and PPL against the oracle, not KL against the full-plug arm, so the comparison is on those. `lakes/p7_glint/gfold_price_20261115/gate2047/`.

| Arm | MB/token | top-1 | PPL | tok/s |
|---|---|---|---|---|
| uniform 10% | 162.7 | 1957 (95.60%) | 5.5132 | 16.74 |
| price 1.0e-3 | 171.0 | 1949 (95.21%) | 5.5073 | 19.34 |
| uniform 25% | 404.5 | 1979 (96.68%) | 5.4764 | 11.07 |
| price 0.6e-3 | 391.1 | 1958 (95.65%) | 5.4856 | 12.87 |
| uniform 40% | 646.2 | 1999 (97.66%) | 5.4761 | 8.93 |
| price 0.32e-3 | 654.9 | 1986 (97.02%) | 5.4769 | 8.79 |
| price 1.5e-3 | 62.5 | 1946 (95.07%) | 5.5508 | 26.16 |

The one-price rule with g_j = 1 is not better than the uniform fraction at any of the three budgets; it is worse on top-1 at all three (by 0.4, 1.0 and 0.6 points). The uniform rule stays the baseline.

Reading (inference): the objective in section 3 counted ABSOLUTE error energy. The uniform-fraction rule ranks inside each linear, which normalises away each linear's activation scale, and it wins. Every linear's output feeds a residual stream that RMSNorm rescales, so an error matters relative to the signal where it lands, not in absolute size. That is the operator's S66 distinction: the absolute-energy objective is the "miles per gallon" reading, and the measurement rejects it. The costate weight g_j is therefore not optional; with g_j = 1 the program optimises the wrong quantity.

### H-G2r (replaces H-G2's g_j source; written before its data)
Relational price: fetch channel c of linear j iff |x_jc| >= tau * mean_c|x_j| for this token, with one tau for the whole model (g_j proportional to 1 / (d_j * mean|x_j|)^2, taken per token from the activation itself). Unlike the uniform fraction, the count then varies by linear and by token with the shape of the activation distribution: a peaked input takes few rows, a flat one takes many.
- Test: 2047 positions, tau chosen to match 163, 404 and 646 MB per token, against the uniform arms above.
- falsifyIf: top-1 is not higher than the uniform arm at two of the three budgets (one run each; a difference under 5 positions of 2047 counts as no difference).

### H-G2r result: NULL by its own rule, 2026-11-15
Same mode and positions, one run each, cap 60%. Unchanged-behaviour check repeated after the kernel change (identical). `lakes/p7_glint/gfold_price_20261115/gate2047_tau/`.

| Arm | MB/token | top-1 | PPL | tok/s |
|---|---|---|---|---|
| uniform 10% | 162.7 | 1957 (95.60%) | 5.5132 | 16.74 |
| relational tau 2.10 | 165.2 | 1956 (95.55%) | 5.5012 | 19.44 |
| uniform 25% | 404.5 | 1979 (96.68%) | 5.4764 | 11.07 |
| relational tau 1.40 | 395.0 | 1969 (96.19%) | 5.4806 | 13.05 |
| uniform 40% | 646.2 | 1999 (97.66%) | 5.4761 | 8.93 |
| relational tau 0.96 | 645.7 | 1998 (97.61%) | 5.4698 | 9.75 |

Top-1 is a tie at two budgets (1 position each, inside the 5-position band) and lower at the middle one (10 positions, at 2.3% fewer bytes). Not higher at two of three, so H-G2r is falsified as written. The relational rule recovers what the absolute price lost (95.21 -> 95.55, 95.65 -> 96.19, 97.02 -> 97.61), which supports the reading that relative, not absolute, error is what matters, but it does not beat ranking inside each linear.

Not pre-registered, so only an observation: the relational arms ran 9-16% faster at equal bytes. The uniform arms ran first in the batch and ambient load was not controlled, so this needs alternating repeats before it means anything.

Conclusion for selection: on the teacher-forced gate, three rules (rank per linear, absolute price, relational price) land within about one point of each other at equal bytes. The choice of rule among these is not where the remaining gain is. Free-running tracking was not tested for the price rules.

### Additive plug, offline weight-space check, 2026-11-15 (CPU, all 128 linears, against the BF16 source)
Operator idea: the plug should be a piece that is simply added, with no decode. Script `scripts/glint_additive_plug_offline.py`; output `lakes/p7_glint/gfold_price_20261115/additive_plug_offline.tsv`.

| Form (4 bits per weight each) | Apply needs | Mean RMS error vs BF16 | vs child nibble |
|---|---|---|---|
| fold only | - | 3.002e-3 | 14.2x |
| child nibble (the pin today) | hole + 16x16 delta table + absmax | 2.118e-4 | 1.00x |
| additive, scale per 64-block | plug + block scale (no hole, no table) | 3.766e-4 | 1.78x (1.61-1.79 over linears) |
| additive, scaled by absmax only | plug + absmax (no hole, no table, no extra bytes) | 5.367e-4 | 2.53x (2.50-2.79) |

Reading: a uniform 4-bit additive piece costs 1.8x to 2.5x the weight error of the child nibble, because the residual's range depends on which cell the hole chose and a uniform grid cannot use that. In weight error that is roughly what the 3-bit child plug gives. Not measured: what it does to model outputs, and how much time the decode costs today. The correction kernel (`apply_corr_ksc`) already uses a precomputed 16x16 delta table, so its "decode" is one table lookup plus reading the hole and the scale; the saving from an additive form would be memory streams read per element (three down to two), which has to be timed before it is worth a build.

### H-G4 lead-time test, 2026-11-15: lead is NOT the bottleneck
Skeleton lead over the real pass made adjustable (`GLINT_AHEAD_LAG`, 0..3 layers; ring of 4). Uniform 40%, 512 positions, three alternating repeats. `lakes/p7_glint/gfold_price_20261115/lag/`.

| Lead (layers) | ms/token, 3 repeats |
|---|---|
| 0 | 101.04, 98.55, 98.63 |
| 1 | 98.41, 98.41, 98.58 |
| 2 | 98.60, 98.43, 98.54 |
| 3 | 104.79, 98.48, 98.80 |

Outputs identical at every lead (PPL 37.305394711155124, 485/512 in all 12 runs). Time per token does not depend on the lead (98.4-98.8 ms; two slower runs in the first round). So the real pass is not waiting for rows that arrive late: with one GPU the skeleton, the gathers and the real pass share the same device, and the token takes the sum of their work whatever the order. More lead does not buy speed here.

### Where the time goes: cost per byte moved
From the seven 2047-position runs above (one run each), time per token against bytes per token is a straight line:

| Arms | Slope | Equivalent rate | Intercept (zero bytes) |
|---|---|---|---|
| relational / price (62.5 to 645.7 MB) | 0.110 ms per MB | 9.1 GB/s | 32.5 ms |
| uniform (162.7 to 646.2 MB) | 0.108 ms per MB | 9.3 GB/s | 43.6 ms |

The measured link ceiling is 11.2-11.6 GB/s (0.086-0.089 ms per MB). So about four fifths of the marginal cost of every extra byte is the link itself, and gather plus decode plus apply together add about one fifth on top. Decode is therefore not the main per-byte cost; removing it entirely would save at most about 20% of the byte-dependent time. The fixed part (two fold passes, attention, logits) is 32.5-43.6 ms.

Consequence: at a given selection, speed is set by bytes that cross the link per token. The levers that remain are (a) not re-sending rows that are already on the device (the reuse probe says a 4-token window covers about 83% of bytes at 35% and 90% at 45%, as an upper bound), and (b) needing fewer rows for the same tracking. Selection stays the model's own choice; residency only avoids moving the same row twice.

### Reuse of rows already on the device, 2026-11-15
Whole-token, channel-addressed staging (`...+kgemv`, no `ahead`), uniform 40%, 512 positions, two alternating repeats. `GLINT_REUSE_TOKENS=W` treats a row used within the last W tokens as still resident. `lakes/p7_glint/gfold_price_20261115/reuse/`.

| W | ms/token | Outputs |
|---|---|---|
| 0 (every selected row re-sent) | 95.70, 96.42 | reference (PPL 37.305394711155124, 485/512) |
| 1 | 74.70, 73.93 | identical |
| 4 | 74.12, 73.48 | identical |
| 99999 | invalid | PPL 1.6e14: the window made the comparison negative so no row was ever fetched (test bug, discarded) |

Reuse of last token's rows saves about 22 ms per token with identical outputs (13.4-13.5 tok/s at 40%). W = 4 saves under 1 ms more, although the reuse-distance probe says it skips far more bytes; the remaining fetch time is therefore not bytes on the link. Not established what it is (the fetch kernel still scans every selected row, one barrier per row). This arm keeps rows in the 1.6 GB whole-token staging, so it is the speed a cache could reach, not a bounded-VRAM result.

### Ephemeral plug (operator, 2026-11-15)
`GLINT_EPHEMERAL=1` with `ahead`: no skeleton stream and no staging. The real pass selects its true channels and the correction kernel reads the selected plug rows directly from pinned host memory. Uniform 40%, 512 positions, two alternating repeats per version, each against the staged path in the same batch. `lakes/p7_glint/gfold_price_20261115/eph*/`.

| Path | ms/token | Outputs |
|---|---|---|
| staged + skeleton (`ahead`) | 110.6-111.5 (109.1-109.3 with a memory sampler) | reference (PPL 37.305394711155124, 485/512) |
| ephemeral v1: one byte per thread from host | 241.3, 242.6 | identical |
| ephemeral v2: 16-byte vector reads (`apply_corr_kse`) | 127.2, 126.8 | identical |
| ephemeral v3: v2 on a second stream beside the fold GEMV | 121.6, 121.1, 121.7 | identical |

Peak card memory (nvidia-smi, 1 s samples, idle 586 MiB before both): staged 4524 MiB, ephemeral v3 with no staging allocated 4334 MiB (190 MiB less; 3748 MiB attributable).

Reading: the ephemeral plug is exact (same outputs as true same-token selection), needs one fold pass instead of two, has no skeleton, no misses and no plug bytes stored on the device, and is 9% slower than the staged path at 40%. The byte-read shape was most of the first gap (241 -> 127 ms); overlap with the fold GEMV added 5.5 ms. Skeleton scratch buffers are still allocated in this build, so 190 MiB is a lower bound on the saving. The machine carried more background load today (staged 98.5 ms on 2026-11-15 early morning, about 111 ms now); only same-batch comparisons are used.

### Free-running results, 2026-11-15 (12 prompts x 32 greedy tokens, reference = Mojo full-plug arm; bars agent-written: prefix >= 20.0, agreement >= 82.5%)
`lakes/p7_glint/gfold_price_20261115/eph_roll/`. Machine under more background load than on 2026-11-14 (staged 40% was 10.65 tok/s then, 8.98 now); compare within this batch only.

| Arm | MB/token | prefix | agreement | tok/s | tracks |
|---|---|---|---|---|---|
| staged + skeleton, uniform 40% | 646 | 24.08 | 89.6% | 8.98 | yes |
| ephemeral, uniform 35% (2 runs) | 566 | 23.83 | 82.0% | 9.22, 9.19 | no (0.5 point short) |
| ephemeral, uniform 40% (2 runs) | 646 | 24.08 | 89.6% | 8.25, 8.26 | yes |
| ephemeral, uniform 45% (2 runs) | 727 | 25.33 | 84.6% | 7.47, 7.49 | yes |
| staged, relational tau 0.96 | 642 | 15.67 | 57.3% | 8.65 | **no** |
| ephemeral, relational tau 0.96 | 642 | 15.67 | 57.3% | 8.20 | **no** |

Ephemeral gate, uniform 40%, 2047 positions: PPL 5.476149214861787, top-1 1999/2047 (97.66%), 8.19 tok/s: the same top-1 and PPL as the staged uniform 40% run.

Findings:
1. The ephemeral plug generates exactly what the staged path generates at every fraction (same prefix and agreement to the digit, same gate result). It is the same model output with no skeleton, no staging and one fold pass, at 8% lower speed.
2. The relational price rule FAILS free-running: 57.3% agreement against 89.6% for the uniform fraction at the same bytes, although the two were within one position of 2047 on the teacher-forced gate. So H-G3 is falsified for the relational rule, and the teacher-forced gate did not predict tracking here. What the uniform rule guarantees and the threshold rule does not is a fixed share of every linear at every token; that is the candidate explanation and it is not tested.
3. No arm in this batch reaches 10 tok/s on today's machine state.

### Transient-staging ephemeral plug, 2026-11-15/16
`GLINT_EPHEMERAL=2` with `ahead`: no skeleton. The real pass selects its true channels; the rows for ONE linear are fetched to a slot on the fetch stream while that linear's fold GEMV runs, applied, and the slot is overwritten by the next layer (one slot per linear type, about 50 MB in all, against the 4-layer ring of about 200 MB). Uniform 40%, 512 positions, alternating runs, peak card memory by nvidia-smi at 1 s (idle 608-614 MiB before each run). `lakes/p7_glint/gfold_price_20261115/eph5/`, `eph6/`, `eph7/`.

| Path | ms/token | Peak card MiB | Outputs |
|---|---|---|---|
| staged + skeleton (`ahead`) | 110.5, 110.8, 111.4, 111.5 | 4546 | reference (PPL 37.305394711155124, 485/512) |
| ephemeral, direct host read (`=1`) | 122.0 | 4352 | identical |
| ephemeral, transient, 4-slot ring | 108.0, 107.8 | not sampled | identical |
| ephemeral, transient, 1 slot per linear | 108.0, 108.3 | 4402 | identical |

Discarded: a first one-slot run gave 65 ms/token with WRONG outputs (PPL 43.1, 404/512). The fetch kernel marks a row with stamp + 1 and consecutive layers share the slot, so the next layer's stamp equalled the mark and its rows were not fetched. Stamps now step by 2 in that path; the table above is after the fix. A batched direct-read kernel (32 channels per step) was slower (157 ms) and was reverted.

Reading: on one GPU the skeleton is not needed. True selection with one fold pass and a one-linear transient slot is 2.5% faster than prediction with two fold passes, with identical outputs, 144 MiB less peak memory, no misses and no late fetches. The direct-read form holds nothing of the plug on the device and is 50 MiB lower still, at 10% lower speed. Skeleton scratch buffers are still allocated in both ephemeral builds.

## 7. What this does not claim

No measurement. No statement that the tightness theorem holds across layers. No M6 acceptance. The free-running bars remain agent-written and unratified.

## Second prompt set, floor share, transient gate (2026-11-16)

Data: `/home/workspace/lakes/p7_glint/gfold_price_20261115/roll2/`. One batch, one build (identity check passed: 35%, 128
positions, PPL 74.34960059115652). New switches, default off: `GLINT_FLOOR_PCT` (guaranteed share of each linear's largest
channels; the relational threshold decides only what goes above it, under the cap) and `GLINT_ROLL_OFFSET` (shifts the 12
rollout prompts inside the evaluation text; 64 gives 12 prompts that do not overlap the first 12). Set B is scored against
its own full-plug run (`B_ref100`). The re-run full-plug reference for set A reproduces the 2026-11-09 tokens exactly.

Transient ephemeral plug (`GLINT_EPHEMERAL=2`), free-running, rule prefix >= 20.0 and agreement >= 82.5% (agent-written):

| plug share | set A prefix / agree | set B prefix / agree | pooled 24 prompts | ms/token |
|---|---|---|---|---|
| 35% | 23.83 / 82.0% no | 17.00 / 61.7% no | 20.42 / 71.9% no | 97.2 |
| 40% | 24.08 / 89.6% yes | 18.42 / 63.0% no | 21.25 / 76.3% no | 107.7 |
| 45% | 25.33 / 84.6% yes | 23.58 / 74.7% no | 24.46 / 79.7% no | 118.9 |

- Set A results equal the staged path token for token (same numbers as `eph_roll`). The transient path on the 2047 gate
  gives 1999/2047, PPL 5.476149214861787, the same as staged.
- **The tracking result does not generalise.** No share from 35 to 45% passes on set B, and none passes pooled. Every earlier
  "tracks at 40%" statement rests on set A alone and is withdrawn as a general claim.
- Per-prompt matched prefixes are two-valued: a prompt either matches all 32 tokens or diverges early and stays diverged
  (set A at 40%: 32 8 32 32 32 7 32 19 26 32 5 32; set B at 40%: 5 10 29 32 4 9 32 32 32 4 0 32). They are not monotone in
  the share (set A prompt 0: 1, 32, 1 at 35/40/45%). One changed token decides a prompt, so a 12-prompt mean moves by
  several points on one flip, and agreement after the first divergence compares different contexts.

Floor share with the relational threshold (set A only, cap 60%, floor 25%; byte counter does not count the ephemeral path,
so time per token stands in for bytes):

| arm | prefix / agree | ms/token | nearest uniform arm by time |
|---|---|---|---|
| floor 25% + tau 1.10 | 26.50 / 82.8% yes | 102.8 | between 35% (97.2) and 40% (107.7) |
| floor 25% + tau 1.30 | 19.42 / 63.0% no | 88.7 | below 35% |
| pure threshold, 40% bytes (earlier) | - / 57.3% no | - | 40% |

The floor moves the threshold rule from 57.3% to 82.8% on set A at fewer bytes than uniform 40%, with the longest mean
prefix measured. Given the set B result this is one sample of 12 prompts and is not a ranking.

Open: the byte counter in ephemeral mode; whether first divergences sit at near-tie positions (next section).

## Margin at divergence and margin-triggered re-run (pre-registered 2026-11-16, before any run)

Referent (S103, "guided by what you desire to happen"; S78, the forward pass ends a round): the round's outcome is the chosen
token. The quantity available at the end of the round, without any label, is the margin: bf16 logit of the chosen token
minus the largest other logit. Isomorphism under test: in G-fold the control is spent where the constraint is about to
bind; here plug bytes are spent on the tokens whose choice is about to flip. Delta: G-fold knows the constraint in advance,
the margin is known only after the cheap pass, so the spend is a second pass on that token (same position, key/value cache
row overwritten). Agent inference, not an operator statement.

Switches (default off): `GLINT_MARGIN=1` records the margin per generated token; `GLINT_ESC_M100` (threshold x 100) with
`GLINT_ESC_PCT` re-runs a token at the larger share when its margin is below the threshold (transient ephemeral path).

- **H-M1 (measurement).** At the first divergence of each diverging prompt (contexts still identical), the arm's own margin
  is small. Pass: a threshold m exists such that >= 90% of first divergences (pooled sets A and B, 35% and 40%) have
  margin < m while <= 20% of all generated positions before divergence have margin < m. Fail otherwise; a fail means the
  margin does not locate the flips and re-run by margin is dropped.
- **H-M2 (mechanism, run only if H-M1 passes).** With m from H-M1 and re-run at 100%, base 35%: pooled 24-prompt tracking
  passes the rule (prefix >= 20.0, agreement >= 82.5%) on each set separately, at a mean time per token below the uniform
  45% arm (118.9 ms). Null: either set fails, or the time is not below 118.9 ms.
- Identity: with `GLINT_MARGIN=1` and no re-run, generated tokens must equal the roll2 runs token for token.

### Results (2026-11-16). Data: `/home/workspace/lakes/p7_glint/gfold_price_20261115/margin/`

Identity: build check 35%/128 positions PPL 74.34960059115652; with `GLINT_MARGIN=1` the four rollouts and both full-plug
references reproduce the roll2 tokens exactly. Margins are bf16 logit gaps, so they come in steps of 0.125 to 0.25.

**H-M1 PASS.** 25 first divergences (sets A and B, 35% and 40%), arm's own margin x100: 0 (7), 12 (2), 25 (13), 50 (3).
All are below 0.75; 13.6% of the 1000 generated positions before any divergence are below 0.75. m = 0.75.

**H-M2 NULL.** Base 35%, re-run when margin < 0.75:

| arm | set A prefix / agree | set B prefix / agree | ms/token | tokens re-run |
|---|---|---|---|---|
| no re-run | 23.83 / 82.0% no | 17.00 / 61.7% no | 97.2 | 0 |
| re-run at 100% (pre-registered) | 26.75 / 83.6% yes | 22.50 / 70.8% no | 109.2 / 108.4 | 58 / 54 of 1140 |
| re-run at 60% (exploratory) | 27.58 / 86.2% yes | 24.42 / 77.1% no | 104.6 / 104.6 | 52 / 56 of 1140 |

Time is below the uniform 45% arm (118.9 ms) but set B fails agreement, so the rule is not met. The divergences left after a
100% re-run have a re-run margin of 0 to 0.25: the full plug on that token is itself tied, and the earlier context was
built at 35%.

**Reference margin at the first divergence** (full-plug run's own margin at the position where an arm first differs):

| arm | set A | set B |
|---|---|---|
| 35% | 0, 25, 12, 25 | 0, 25, 0, 0, 0, 25, 0, 0, 25 |
| 40% | 25, 25, 25, 25, 0 | 0, 25, 0, 25, 0, 0, 0 |
| 35% + re-run 100% | 25, 0, 50 | 0, 0, 0, 0, 0 |
| 35% + re-run 60% | 25, 0, 50 | 0, 0, 0, 0, 25 |

(x100.) Every first divergence in every arm sits where the reference's top two logits differ by 0.50 or less, and most are
exact bf16 ties (0) that the argmax breaks by lowest token index. 9.8% of reference positions have margin <= 0.25, 14.2%
<= 0.50. In 24 prompts no arm departed from the reference at a position where the reference was decided by more than
0.50. Set B scores worse than set A because more of its prompts reach an exact tie.

Consequence (agent reading, not a ruling): the free-running rule as written scores the outcome of reference ties. A measure
of stumbles needs the reference's logit for the token the arm chose at the divergence (not yet recorded: this run shows
the reference was tied between its top two, not that the arm chose one of those two).

### Reference score of the token each arm chose at its first divergence (2026-11-16)

Data: `/home/workspace/lakes/p7_glint/gfold_price_20261115/margin/alt/`. New switch `GLINT_ALT_FILE` (triplets prompt, step,
token): the run prints its own bf16 logit gap between its choice and the listed token. Both full-plug references were
re-run with the 25 distinct (prompt, step, token) first divergences of the four arms (35%, 40%, 35% + re-run 100%,
35% + re-run 60%); their generated tokens are identical to the earlier reference runs.

| reference gap to the arm's token | count of 25 |
|---|---|
| 0 (exact bf16 tie, argmax broke it by token index) | 12 |
| 0.12 | 1 |
| 0.25 | 10 |
| 0.50 | 2 |
| above 0.50 | 0 |

Every token on which an arm first left the reference is one the reference itself scored within 0.50 logits of its own
choice (probability ratio at most e^0.5 = 1.65), and 12 of 25 were exact ties. No arm, including plain 35% at 97.2
ms/token, chose a token the reference scored clearly below its choice. Limits: only the first divergence per prompt can
be scored this way (afterwards the contexts differ); 24 prompts; one model.

### Teacher-forced gate at lower shares, transient ephemeral path (2026-11-16)

2047 positions, gate top-1 >= 95.0% and PPL within 2% of 5.486957. Data: `margin/gate_tr_p35.log`, `margin/gate_tr_p30.log`.

| share | top-1 | PPL | gate | speed |
|---|---|---|---|---|
| 40% (roll2) | 1999/2047 = 97.66% | 5.476149 | pass | 109.0 ms/token, 9.17 tok/s |
| 35% | 1988/2047 = 97.12% | 5.479107 | pass | 98.4 ms/token, 10.17 tok/s |
| 30% | 1981/2047 = 96.78% | 5.483165 | pass | 87.1 ms/token, 11.48 tok/s |

Top-1 falls with the share (1999, 1988, 1981). 30% has not been run free-running.

### Lower shares free-running and speed repeats (2026-11-16)

Data: `margin/{A,B}_m30.log`, `{A,B}_m25.log`, `{A,B}_ref100alt2.log`, `gate_tr_p35_r2/r3.log`, `gate_tr_p30_r2.log`. Same build.

| share | set A prefix / agree | set B prefix / agree | first divergences (24 prompts) | reference gap to the arm's token | ms/token free-running |
|---|---|---|---|---|---|
| 30% | 20.58 / 66.1% no | 17.42 / 56.2% no | 15 | 0 (8), 0.25 (5), 0.37 (1), 0.50 (1) | 86.3 / 86.5 |
| 25% | 15.08 / 50.3% no | 19.42 / 61.7% no | 16 | 0 (6), 0.12 (1), 0.25 (7), 0.37 (1), 0.50 (1) | 75.7 / 75.4 |

The 20 distinct (prompt, step, token) first divergences of the 30% and 25% arms were all scored by the reference within
0.50 logits of its own choice; 8 are exact ties. So down to 25% no arm has first left the reference on a token the
reference clearly rejected, while the agreement number falls to 50 to 66%. The 25% arm has not been run on the 2047 gate.

Speed repeats, 2047-position gate, transient ephemeral: 35% = 98.37, 97.98, 98.27 ms/token (10.17, 10.21, 10.18 tok/s),
identical outputs each time (1988/2047, PPL 5.47910741239947); 30% = 87.09, 87.32 ms/token (11.48, 11.45 tok/s), identical
outputs (1981/2047, PPL 5.4831653173713635).

### Where decided tokens start to change: 25%, 20%, 15% (2026-11-16)

Data: `margin/{A,B}_m20.log`, `{A,B}_m15.log`, `{A,B}_ref100alt3.log`, `gate_tr_p25/p20/p15.log`. Transient ephemeral, same build.

| share | gate top-1 | gate PPL | gate | ms/token (gate) | first divergences (24 prompts) | reference gap to the arm's token | above 0.50 |
|---|---|---|---|---|---|---|---|
| 40% | 1999/2047 = 97.66% | 5.476149 | pass | 109.0 | 12 | all <= 0.50 | 0 |
| 35% | 1988/2047 = 97.12% | 5.479107 | pass | 98.2 | 13 | all <= 0.50 | 0 |
| 30% | 1981/2047 = 96.78% | 5.483165 | pass | 87.2 | 15 | all <= 0.50 | 0 |
| 25% | 1979/2047 = 96.68% | 5.476430 | pass | 76.3 | 16 | all <= 0.50 | 0 |
| 20% | 1970/2047 = 96.24% | 5.494052 | pass | 65.2 | 15 | 0 (6), 0.25 (6), 0.37 (1), 0.50 (1), 1.00 (1) | 1 |
| 15% | 1962/2047 = 95.85% | 5.490655 | pass | 54.1 | 17 | 0 (5), 0.25 (5), 0.37 (1), 0.50 (2), 1.00 (4) | 4 |

- The first departure onto a token the reference scores more than 0.50 below its choice appears at 20% (1 of 24 prompts)
  and grows at 15% (4 of 24). From 25% up there is none in this sample.
- The teacher-forced gate passes at every share down to 15%, so the gate does not see this edge; the reference-gap
  measure does. The old agreement number (50 to 90%) does not see it either: it moves with tie-breaks at every share.
- 25% is the smallest share with no such departure: 76.3 ms/token (13.1 tok/s) on the gate, 75.4 to 75.7 free-running.

## Attribution test: what does what, and why (pre-registered 2026-11-18, before any run)

Operator direction (2026-11-18, on the M6 rule question): "turn this in to a test what does what and why". This section
replaces the open choice between the agreement rule and the reference-gap rule with a test. It rules on neither; M6
stays the operator's.

Why the old measure cannot answer it: free-running agreement stops comparing like with like after the first departure,
because the contexts differ. Only the first departure per prompt could be scored, about 15 of 768 positions per arm.

**Design.** Every arm is fed the full-plug reference's own generated tokens (`GLINT_FORCE_FILE`), so at all 12 x 32
generated positions of both prompt sets (768 positions) the arm and the reference see the same context. At each position
the arm's argmax is compared with the reference token. Every disagreement (a *flip*) is scored by the reference's own
logit gap between its token and the arm's token (`GLINT_ALT_FILE` on a reference re-run).
- *Tie flip*: reference gap <= 0.50. The reference itself barely prefers its token.
- *Decided flip*: reference gap > 0.50. The arm changed a token the reference was sure of. This is the stumble.

**What (inputs varied).**
1. Plug share s = 15, 20, 25, 30, 35, 40%, all classes at s.
2. At s = 15%: one linear class at a time gets the full plug (`GLINT_FULL_CLASS` = 1 qkv, 2 o, 4 gate+up, 8 down).

**Does what (outputs).** Tie flips T(s) and decided flips D(s) out of 768 positions; time per token as the stand-in for bytes.

**Why (attribution).**
- Tie: P(flip | reference margin <= 0.50) against P(flip | reference margin > 0.50), from the reference margins already
  recorded (`margin/{A,B}_ref100m.log`).
- Class: the reduction in D(15) per class and per added ms/token. This is the G-fold reading: spend control where it
  removes the most constraint violation per unit spent. That mapping is agent inference, not an operator statement.

**Rules, written before the runs.**
- H-A1. D(s) = 0 for s >= 25%, and D(15) > 0. Falsified if D(s) >= 2 at any s >= 25%, or D(15) = 0.
- H-A2. Flips concentrate on reference ties: P(flip | margin <= 0.50) >= 5 x P(flip | margin > 0.50) at every share.
  Falsified at any share where the ratio is below 5.
- H-A3 (exploratory, no prediction about which class). A class *carries* decided flips at 15% if its full plug cuts D(15)
  by at least half while no other single class does. If two or more do, or none does, the report says so.
- Sanity. Forced 35% must reproduce the free-running 35% tokens at every position before that prompt's first departure.
  The build identity check must give PPL 74.34960059115652.

## G-fold rebuilt to our intent: coast and burn (2026-11-18)

Operator direction (2026-11-18, restated in their terms): the math is ours to shape. Fidelity is not optimized; it is
the landing standard, fixed like G-fold's landing conditions. The slack variable is "completely dependent upon how much
we try to change something at one period of time". Burn "full tilt" until the point where it enters the plug. Configure
G-fold to that intent, then build everything else to hold the straight lines.

Formulation (agent derivation from that direction):
- Time step = layer l = 0..31; one token's forward pass is one flight.
- State = residual stream h_l. Each layer adds into it, so a correction at layer l propagates forward by addition. This is
  the linear part that holds the straight lines. Norms, SwiGLU and attention softmax bend them and are measured, not fought.
- Slack = Gamma_l, the change the plug may make at layer l (bytes moved at that layer). Fuel = sum over l of Gamma_l.
- Coast = the fold alone. Burn = full plug.
- Landing standard. (1) Position: the token lands in the accepted zone (the reference's choice, or a token the reference
  scores within the landing margin; 0.50 logits measured so far). The number is the operator's to set. (2) Velocity: the
  error the token leaves in the key/value cache stays under a bound (agent reading of "speed at which it hits the ground").
- Structure: linear in control with chosen levels, so the optimum is coast or burn per layer (straight lines between
  levels). One price compares each layer's landing effect per byte. That effect is the costate.
- Limit: a layer's plug corrects only that layer's own error, so the burn window is not automatically the last layers. It
  is wherever the landing is most sensitive per byte, and that has to be measured.

Today's runs burn the same share in every layer for the whole flight, which is the opposite of this plan.

Switch `GLINT_BURN=a-b`: layers a..b-1 get the full plug in every class; all other layers keep the percent share. Percent
0 now selects nothing (before this change the selector kept the largest channel), so 0 is a true coast.

**Burn-window sweep (pre-registered before the run).** Forced reference context, both prompt sets, base share 0 (coast).
Windows of 8 layers at starts 0, 4, 8, ..., 24 (7 windows), plus coast-only. Output per window: decided flips D, tie flips,
ms/token. Costate estimate per window = (D(coast) - D(window)) / (ms(window) - ms(coast)).
- H-B1. The costate is not uniform: the best window removes at least 2x the decided flips per ms of the worst window.
  Falsified if all windows are within 2x of one another. A uniform costate would mean the uniform share was already right.
- H-B2. A flight plan that burns only the top windows by costate, at the bytes of uniform 25%, has fewer decided flips
  than uniform 25%. Run only after H-B1 passes.

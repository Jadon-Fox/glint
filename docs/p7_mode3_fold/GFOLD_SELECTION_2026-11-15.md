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

### Attribution test results (2026-11-18)

Data: `/home/workspace/lakes/p7_glint/gfold_price_20261115/attr/`. Build identity passed (PPL 74.34960059115652). Sanity
passed: forced 35% reproduces free-running 35% through each prompt's first departure on both sets, and both reference
scoring runs reproduce the reference tokens. Forced context, 384 positions per set, transient ephemeral plug.

| arm | set A flips (tie / decided) | set B flips (tie / decided) | ms/token A / B |
|---|---|---|---|
| 15% | 19 / 3 | 14 / 5 | 49.3 / 50.0 |
| 20% | 14 / 0 | 12 / 1 | 56.5 / 58.7 |
| 25% | 13 / 0 | 14 / 0 | 65.2 / 67.6 |
| 30% | 10 / 0 | 15 / 0 | 77.6 / 76.7 |
| 35% | 9 / 0 | 16 / 0 | 93.0 / 81.4 |
| 40% | 8 / 0 | 13 / 0 | 96.7 / 90.4 |
| 15% + full qkv | 16 / 1 | 18 / 2 | 69.2 / 79.6 |
| 15% + full o | 20 / 3 | 15 / 3 | 64.9 / 68.1 |
| 15% + full gate+up | 14 / 0 | 13 / 0 | 133.3 / 140.4 |
| 15% + full down | 15 / 3 | 14 / 4 | 94.2 / 94.8 |

Decided flips had reference gaps of 0.75 to 3.50 logits (most 1.00).

- **H-A1 PASS.** D = 0 at 25, 30, 35, 40% on both sets; D(15) = 3 and 5. (20% has D = 0 on A and 1 on B.)
- **H-A2 PASS.** P(flip | reference tie) / P(flip | reference decided) = 41.4 (A) and 15.7 (B) at 15%, 67.4 at 20% B,
  infinite elsewhere (no decided flips). A reference tie (margin <= 0.50, 14% of positions) is where 16 to 39% of flips
  happen at every share; tie flips do not fall to zero even at 40%. They are the reference's own coin flips.
- **H-A3: two classes carry the decided flips.** Full gate+up removes all of them (3 to 0, 5 to 0) and full qkv removes
  two thirds and three fifths (3 to 1, 5 to 2). o and down remove almost none. Per added ms, qkv removes about 0.10
  decided flips per ms on both sets and gate+up 0.036 (A) and 0.055 (B): qkv is the more efficient burn per byte,
  gate+up the more complete.
- **Full-class burns lose to a small uniform increase.** Uniform 20% (D = 0 / 1) costs 56.5 / 58.7 ms and uniform 25%
  (D = 0 / 0) costs 65.2 / 67.6 ms. Both beat 15% + full qkv (69 / 80 ms) and 15% + full gate+up (133 / 140 ms).
  The first extra channels in every linear buy more landing than the last channels of one class. That is diminishing
  returns within a linear, the condition under which intermediate burn levels pay (operator, "quarter burn").
- Reading for the planner: the burn should go to qkv and gate+up, at partial levels, not to o and down. The burn-window
  sweep gives the layer axis.

## Overlap baseline on the transient path (pre-registered 2026-11-18, before any run)

Step 1 of the prefetch rebuild. Switch `GLINT_TIME_ABL` (timing only, outputs invalid): 1 = compute only (the transient
path skips its host-to-slot plug copy), 2 = copy only (per token, only the selection and the host-to-slot copy for every
linear, back to back on the gather stream). Method as in MOJO_GLINT_DECODE "Streamed decode: copy, compute and overlap":
overlap = copy-only + compute-only - full, 3 repeats each, median, 35% and 25%, 128 positions.

Correction to my 2026-11-18 chat statement: the transient path's gather (`gather_rows_fix`) is already a persistent
kernel of GB = 16 blocks reading 16-byte vectors, not one block per row (that was the older skeleton gather). What it
cannot do is start early: it selects on the layer's true activation, so each linear's copy starts only when that
activation exists, and the correction waits for it.

- H-O1. Overlap is small: overlap < 25% of the shorter side at both shares. This would mean fetch and compute mostly take
  turns, and the speed-up available from true prefetch is about min(copy, compute) per token. Falsified if overlap >= 25%.
- Also reported: copy-only bandwidth = selected bytes / copy-only time, against 11.2 GB/s measured for the copy engine.

**Burn sweep, first attempt: invalid (2026-11-18).** Every window ran at coast speed (17.7 to 19.1 ms/token) because the
plug kernels were guarded by the layer's base share (`nh[l] > 0`), which is 0 when coasting, so `GLINT_BURN` never
reached them. Data kept at `burn_invalid_guard_bug/` for the record; the coast-only arms are valid (fold alone, forced
context, rollout positions 0 to 94: 18.9 / 17.8 ms/token). Fix: each guard now tests its own linear's selected count
`nc[l * 4 + c] > 0`, which equals the old test whenever no window or class mask is set. Sweep re-run follows.

## Transport primitives microbenchmark (2026-11-18)

`mojo/p7_h2d_bench.mojo`, no model, one run, log `/home/workspace/lakes/p7_glint/prims_20261118/h2d_bench.log`. Pinned
host buffer 256 MiB, scattered offsets, second of two repeats reported.

Copy engine (`enqueue_copy`, one copy per chunk):

| chunk | 1536 B | 2560 B | 8192 B | 64 KiB | 1 MiB | 16 MiB |
|---|---|---|---|---|---|---|
| GB/s | 1.03 | 1.56 | 3.81 | 9.16 | 11.58 | 11.83 |
| us per copy | 1.50 | 1.64 | 2.15 | 7.15 | 90.6 | 1419 |

Zero-copy gather kernel (persistent, 128 threads per block, 16-byte vectors), GB/s by row width and block count:

| row | 1 | 2 | 4 | 8 | 16 | 28 | 56 |
|---|---|---|---|---|---|---|---|
| 1536 B (o, down) | 1.44 | 2.79 | 5.30 | **8.38** | 8.25 | 8.17 | 6.30 |
| 2560 B (qkv) | 2.32 | 4.51 | 8.23 | **9.79** | 7.76 | 6.60 | 6.17 |
| 8192 B (gate+up) | 6.37 | 11.18 | **11.69** | 11.18 | 10.13 | 7.55 | 7.45 |

Contention: a VRAM-bound read of 1 GiB takes 3.15 ms alone (341 GB/s, 95% of the 360 GB/s rating), and 3.22 to 3.52 ms
while a zero-copy gather of 100 MB runs on another stream (2 to 56 blocks), and 3.37 ms while the copy engine moves 100 MB
in 1 MiB chunks.

Readings:
- **The GPU's memory is not the limit.** A plain streaming kernel reaches 341 GB/s. The decoder's coast path moves about
  3.04 GB per token in 17.7 to 19.1 ms (159 to 172 GB/s), so its kernels run at about half the card's memory speed.
- **The gather does not starve compute.** The VRAM kernel slows by at most 12% with any gather running beside it. My
  2026-11-18 chat claim that fetch and compute "take turns because they share cores" is not supported at the hardware
  level; any serialization in the decoder comes from its dependencies (each linear's copy waits for its activation, and
  the correction waits for the copy). The overlap baseline measures that next.
- **The gather is mis-sized.** The decoder uses 16 blocks for every class. Best is 8 for 1536 and 2560 B rows and 4 for
  8192 B rows. Byte-weighted (qkv 15.6%, o 9.4%, gate+up 50%, down 25%) that is about 10.0 GB/s against 9.0 GB/s at 16
  blocks; 9.0 matches the 0.11 ms/MB measured in the decoder. More blocks is slower.
- **The copy engine cannot move single rows.** About 1.5 us per copy caps it at 1 to 4 GB/s for plug rows; it needs runs
  of 64 KiB or more (9.2 GB/s) and 1 MiB for the link rate.

### Checked against the Antigravity synthesized design (2026-11-18)

Source (another harness's design, read as data): `/home/jfox/.gemini/antigravity/brain/0014c506-.../scratch/arena-glint-p6/synthesized/`
and `forge/agent_routing/templates/redesign/glint_p6_mojo_gfold_first_principles.md`.

| claim in that design | measurement | status |
|---|---|---|
| A persistent worker of 2 blocks (7.1% of SMs) services prefetch | 2 blocks give 2.8 / 4.5 / 11.2 GB/s for 1536 / 2560 / 8192 B rows | holds for gate+up only; too slow for o, down, qkv unless misses are tiny |
| The inline 16-block gather steals 57% of SMs from compute | compute slows at most 12% beside 16 or 56 gather blocks | not supported |
| Copy engine at 11.28 GB/s with 0 SMs | true for 1 MiB+ chunks; 1.0 to 3.8 GB/s for row-sized copies | holds only with host packing into large runs |
| W = 8 recency covers 95 to 98% of bytes, so a 548 MiB pool cuts link traffic to 20 to 60 MB per token | the 95% is the share of selected bytes last used within 8 tokens (2026-11-14 age probe, unbounded staging). A 548 MiB pool holds 34% of the 1.61 GB plug, about one token's selection at 25 to 35% | untested at that pool size; the window's distinct rows are not measured and are likely far above 548 MiB |
| Interior shares beat 0/100% burns | attribution test: uniform 20 to 25% beats full-class burns (65 vs 133 ms) | holds on the class axis |
| o and down anchored near 14 to 15%, never 0 | attribution: full o or down at 15% removes almost no decided flips | the floor is a design choice, not measured; testable with `GLINT_CLASS_PCT` |

### Burn-window sweep results (2026-11-18, guard fix 293dffc)

Data `burn/`. Forced context, base share 0 (true coast), full plug in layers a..a+7. Scored from the reference's stored
margins (`glint_attr_score.py quick`; 0 disagreements with the full method on 307 attribution flips), so no reference
re-runs. Build identity passed.

| arm | set A decided / tie | set B decided / tie | ms/token A / B |
|---|---|---|---|
| coast (fold alone) | 11 / 30 | 13 / 29 | 17.6 / 16.8 |
| burn 0-7 | 4 / 24 | 8 / 22 | 66.4 / 69.0 |
| burn 4-11 | 9 / 22 | 9 / 34 | 65.2 / 67.5 |
| burn 8-15 | 8 / 27 | 13 / 32 | 65.6 / 65.0 |
| burn 12-19 | 8 / 26 | 9 / 36 | 63.5 / 68.8 |
| burn 16-23 | 9 / 27 | 10 / 30 | 65.1 / 66.1 |
| burn 20-27 | 9 / 26 | 13 / 30 | 66.9 / 66.6 |
| burn 24-31 | 8 / 22 | 13 / 32 | 68.7 / 64.9 |
| uniform 25% (attribution) | 0 / 13 | 0 / 14 | 65.2 / 67.6 |

- **H-B1 PASS (costate not uniform).** Best window per added ms: layers 0-7 (A 0.143, B 0.096 decided flips removed per
  ms); worst windows remove 2 per 48 ms on A and none on B. Ratio >= 3.4, above the 2x rule. The first layers matter most.
- **H-B2 not run: the coast/burn plan is refuted.** Every 8-layer window costs about what uniform 25% costs (same bytes)
  and leaves 4 to 13 decided flips where uniform 25% leaves none. The best window removes only 7 of 11 (A) and 5 of 13 (B)
  coast flips. Burning full plug in a few layers loses to a thin share in every layer, on the layer axis as on the class
  axis (attribution: full gate+up at 133 ms vs uniform 25% at 65 ms).
- So the costate is not concentrated anywhere. Each linear's first channels buy the most landing, in every layer and
  class: diminishing returns inside every linear. That is the interior (partial-level) case. The Antigravity design's
  interior-share thesis holds; my coast/burn flight plan does not.
- Coast alone: 11 / 13 decided flips at 17.6 / 16.8 ms (57 / 60 tok/s). The fold is close but not on the pad.

### Overlap baseline results (2026-11-18)

Data `overlap/`. Transient path, 128 positions, 3 repeats, medians (all runs listed in `summary.txt`). Full runs reproduce
the identity PPL at 35%.

| share | full | compute only (copy skipped) | copy only (selection + copy, back to back) | coast (fold alone, forced rollout) |
|---|---|---|---|---|
| 35% | 89.0 (87.7-90.8) | 75.0 (73.9-75.1) | 172.9 (170.6-173.5) | 17-18 |
| 25% | 72.4 (67.1-74.3) | 58.0 (56.6-58.1) | 166.5 (163.1-175.4) | 17-18 |

- **H-O1 FALSIFIED, and the pre-registered formula does not apply.** "Copy only" turned out to include the 128
  selections per token run back to back, and it barely changes with bytes (172.9 vs 166.5 ms for 566 vs about 404 MB),
  so it is dominated by per-linear fixed cost, not transfer. The formula copy + compute - full is not meaningful here.
- **The copy is not the cost.** Adding the plug copy to compute-only costs 14.0 ms at 35% and 14.4 ms at 25%. Moving those
  bytes at the measured 9 GB/s alone would take about 63 and 45 ms, so at least 78% (35%) and 68% (25%) of the transfer is
  already hidden behind other work.
- **The cost is the selection-and-correction machinery.** Compute-only at 35% is 75 ms against about 18 ms for the fold
  alone: about 57 ms per token comes from per-linear selection, the correction kernels and the event round trips between
  streams (each linear's correction waits for its selection and copy). At 25% it is about 40 ms. The operator's statement
  "it can't be PCIe" (2026-11-15) is borne out; my 2026-11-15 reading that the link was about 4/5 of marginal byte cost
  was a fit of time against bytes that could not separate the two, because the selection work grows with the share too.
- Next: decompose those 57 ms with two more timing modes (`GLINT_TIME_ABL=3` selection only, `4` no correction).

### Per-class shares and gather block counts in the decoder (2026-11-18)

Data `/home/workspace/lakes/p7_glint/prims_20261118/` (`summary2.txt`). New build `/tmp/p7_sel2`; identity passed on both
the staged and the transient path (PPL 74.34960059115652, 120/128).

Gather blocks per class, transient path, 25%, 128 positions, 3 interleaved repeats: 16 blocks 68.8 / 67.9 / 71.6 ms,
8/8/4/8 blocks 76.5 / 76.3 / 74.7 ms, identical outputs. **Fewer blocks is slower inside the decoder**, against the
microbenchmark. Each linear moves only 1 to 10 MB, so the copy is latency-bound per linear, not throughput-bound; the
microbenchmark measured 96 MB bulk transfers. Keep 16. (The class runs below used 8/8/4/8, so their times are
inflated by about 5 to 8 ms; compare them with the uniform 25% row of the same batch.)

Per-class shares (q, o, gate+up, down percent), forced context, scored from stored reference margins:

| shares q/o/gu/d | share of plug bytes | set A decided / tie | set B decided / tie | ms/token A / B |
|---|---|---|---|---|
| 25/25/25/25 | 25.0% | 0 / 13 | 0 / 14 | 75.5 / 76.5 |
| 25/0/25/0 | 16.4% | 6 / 23 | 7 / 30 | 55.5 / 54.7 |
| 30/0/30/0 | 19.7% | 6 / 24 | 9 / 29 | 63.9 / 61.8 |
| 35/0/25/0 | 18.0% | 7 / 22 | 7 / 26 | 57.6 / 58.6 |
| 25/15/25/15 | 21.6% | 1 / 19 | 2 / 13 | 68.7 / 72.3 |
| 30/10/30/10 | 23.1% | 0 / 20 | 2 / 19 | 73.7 / 69.1 |
| 20/10/30/10 | 21.6% | 0 / 20 | 2 / 16 | 71.2 / 67.2 |

(byte shares: qkv 15.6%, o 9.4%, gate+up 50%, down 25% of the plug.)
- o and down cannot coast: removing them adds 6 to 9 decided flips even with qkv and gate+up raised. The full-o and
  full-down burns removed almost nothing because they started from 15%, where those classes were already enough.
- No tested class split beats uniform 25% at equal landing: every split that saves bytes leaves decided flips on at
  least one set. At this resolution the per-byte costate is close to balanced across classes, so the uniform interior
  share is already near the G-fold optimum on the class axis. Antigravity's o/down floor of about 15% is consistent
  with this; its claim that o and down can sit far below qkv and gate+up is not supported at 25% total.

### Decomposition of the per-linear machinery (2026-11-18)

Data `prims_20261118/decomp/`. Build `/tmp/p7_sel3`, identity passed (35%, PPL 74.34960059115652). Transient path, 128
positions, 3 repeats, medians (runs listed in `summary.txt`).

| mode | 35% | 25% |
|---|---|---|
| full | 96.0 | 72.4 |
| copy skipped (`ABL=1`) | 83.2 | 63.8 |
| correction skipped (`ABL=4`) | 85.4 | 65.4 |
| selection only, back to back (`ABL=3`) | 4.7 | 4.8 |
| fold alone (share 0) | 19.1 | 19.1 |

- Selection is cheap: 128 selections cost 4.7 ms per token.
- Removing the copy saves 12.8 / 8.6 ms; removing the corrections saves 10.6 / 7.0 ms. Neither alone comes close to the
  fold-alone 19.1 ms: both modes still sit 45 to 66 ms above it. So most of the cost is common to both, i.e. neither the
  bytes nor the correction arithmetic. Hypothesis (next test): the per-linear cross-stream event chain (compute records,
  gather stream waits, selects, copies, records; compute waits) leaves the GPU idle between dependent steps, 128 times
  per token. Tests: `GLINT_TIME_ABL=5` (no copy and no correction; the chain remains) and `GLINT_ONE_STREAM=1` (all plug
  work on the compute stream; the waits become no-ops).
- Nsight Systems in the container (2022.4) cannot import its own trace with the available 2025 importer, so no timeline.

### Correction: ABL=1 and ABL=5 still read the plug (2026-11-18)

Found by the lean-decoder build review. In `p7_glint_select_decode.mojo` the copy-skip condition sat on an `if` whose
`else` branch runs `apply_corr_kse`, the retired `GLINT_EPHEMERAL=1` kernel that reads plug rows straight from pinned host
memory. So `GLINT_TIME_ABL=1` ("compute only") and `=5` did not skip the plug transfer: they replaced the copy with a
host-reading correction (racing the normal one on the same buffer under ABL=1). **Withdrawn:** the overlap-baseline
reading "the copy costs only about 14 ms; the selection-and-correction machinery is the cost", the decomposition row
"copy skipped", and my statement to the operator that "it can't be PCIe" was borne out. Those rows measured a different
transfer path, not a path without transfer. Still valid: full, ABL=3 (selection only, 4.7 ms), ABL=4 (no correction),
fold alone.

One-stream test (`/tmp/p7_sel4`, before the fix, ABL=5 rows invalid for the same reason): all plug work on the compute
stream is slower, 103.9 / 101.7 ms against 96.1 / 96.7 (35%) and 83.9 against 76.3 (25%), identical outputs. So the
cross-stream hand-offs are not the cost; the second stream buys some overlap.

Fix: the lean decoder (`mojo/p7_glint_lean.mojo`, 21 kernels, 41 s build, bit-identical on five checks) now skips the
copy under ABL=1 and 5 without any host read. Corrected decomposition follows.

### Corrected decomposition (2026-11-18, lean decoder with the fix)

Data `prims_20261118/decomp2/`. Build `/tmp/p7_lean2` in 38 s; identity passed (35%, PPL 74.34960059115652, 120/128).
Transient path, 128 positions, 3 repeats, medians.

| mode | 35% | 25% |
|---|---|---|
| full | 92.8 | 70.9 |
| correction skipped (copy remains) | 79.0 | 62.3 |
| copy skipped (corrections remain, no host read) | 31.9 | 27.8 |
| neither (selection and stream hand-offs remain) | 18.4 | 18.2 |
| fold alone | 17.6 | 17.6 |
| one stream, full | 98.3 | 77.9 |
| one stream, neither | 22.2 | 22.2 |

- **The plug copy is the cost.** Copy alone adds 61 ms at 35% and 44 ms at 25% over the selection-only run: about 566 and
  404 MB in those times, 9.3 and 9.2 GB/s, the zero-copy gather's measured rate. Corrections add 14 / 10 ms. Selection and
  the 256 cross-stream hand-offs per token add 0.6 to 0.8 ms in total.
- **The copy is on the critical path, not overlapped.** Full is about copy-only plus corrections (92.8 against 79.0 +
  13.5). Each linear's copy can start only once its activation exists, and its correction waits for it.
- So my first reading (2026-11-15: the link carries most of the marginal cost, about 9 GB/s) stands, and the operator's
  statement "it can't be PCIe" is not supported by this measurement. The withdrawn reading in between came from the
  ABL=1 bug.
- What this sets: compute without the copy is 31.9 / 27.8 ms. A prefetch that moves the same bytes while compute runs
  would give about max(copy, compute) = 61 / 44 ms per token (16 / 23 tok/s) at the same bytes; fewer fresh bytes per
  token (reuse across tokens) lowers the copy side further. The two levers are overlap (predict and prefetch ahead) and
  reuse (do not re-fetch rows already in VRAM).

### Oracle prefetch: the overlap ceiling (2026-11-18)

`GLINT_ORACLE=1` in the lean decoder (build `/tmp/p7_lean3`, 41 s). Each token runs twice: pass A is the normal path and
records every linear's true selection; pass B recomputes the token with all 128 plug copies queued at its start into
per-linear staging slots (1.61 GB, study only), and only pass B is timed. Perfect prediction, full overlap. Outputs
identical to the normal path at 35% and 25% (all 128 positions). 128 positions, 3 repeats.

| share | normal | oracle pass B | copy alone (decomposition) | fold alone |
|---|---|---|---|---|
| 35% | 99.4 / 99.7 / 100.5 | 75.1 / 75.4 / 76.1 | 79.0 | 17.6 |
| 25% | 76.4 / 76.4 / 76.8 | 56.9 / 57.5 / 57.9 | 62.3 | 17.6 |

- Perfect prefetch saves about 25%: 13.3 tok/s at 35%, 17.4 tok/s at 25% (normal 10.0 / 13.1 under today's load).
- Pass B is about the copy alone: the link is the floor. Overlap hides the compute behind the copy; it cannot shorten
  the copy. Prediction alone therefore caps at these numbers, and only fewer link bytes per token go further.
- Next: cross-token reuse inside the oracle (`GLINT_ORACLE_REUSE=W`: a row fetched into its slot within the last W tokens
  is not fetched again).

### Oracle prefetch with cross-token reuse (2026-11-18)

Build `/tmp/p7_lean4`. Same oracle pass B, plus `GLINT_ORACLE_REUSE=W`: rows fetched into their linear's slot within the
last W tokens are not fetched again. Outputs identical to normal at 35% and 25% (W = 8 checked position by position; all
runs give the identity PPL). 128 positions, 2 repeats. The machine was less loaded than in the previous batch (normal 81 /
63 ms against 100 / 76), so compare within this table only.

| share | normal | W=0 | W=1 | W=2 | W=4 | W=8 |
|---|---|---|---|---|---|---|
| 35% | 81.3 / 80.9 | 62.4 / 60.5 | 43.5 / 44.8 | 37.6 / 37.9 | 32.7 / 33.2 | 30.7 / 31.3 |
| 25% | 63.0 / 62.5 | 48.2 / 46.6 | 38.6 / 38.1 | 35.8 / 34.4 | 30.4 / 30.3 | 27.7 / 27.4 |

- With perfect prediction and an 8-token reuse window: 31 ms at 35% (32 tok/s) and 27.5 ms at 25% (36 tok/s), against
  81 / 63 ms for today's path. That sits close to compute without any copy (31.9 / 27.8 ms in the decomposition), so at
  W = 8 the link is no longer the limit.
- Reuse does most of the work: W = 4 alone cuts pass B from 61 to 33 ms at 35%. Reuse does not need prediction (a row
  already in VRAM is valid whatever the predictor says), so a non-predictive path with per-linear slots and a reuse
  window is the next build. Its expected time is compute plus only the fresh rows on the critical path.
- Cost not yet bounded: these slots hold the whole plug (1.61 GB of VRAM). The operator's goal is no extra VRAM, so the
  rows actually resident within a window W have to be counted and fitted into a bounded pool before this is a design.

### Cross-token reuse in the real decode path, and the VRAM it needs (2026-11-18)

`GLINT_REUSE=W` in the lean decoder (build `/tmp/p7_lean5`, 52 s): no oracle, no prediction. Each linear keeps its own
channel-addressed slot; a selected row fetched within the last W tokens is not fetched again. Outputs identical to the
normal path (all runs give the identity PPL; W = 16 checked position by position). 128 positions, 2 repeats (the machine
load moved between repeats, so read the two columns separately). `GLINT_REUSE_STAT=1` samples every 16 tokens the plug
bytes whose rows were selected within the last W tokens: the VRAM a bounded pool would need to hold that window.

| share | W | ms/token r1 / r2 | rows resident for the window (MB) |
|---|---|---|---|
| 35% | 0 (normal) | 83.4 / 97.2 | 566 (one token's selection, transient) |
| 35% | 1 | 61.6 / 69.8 | 881 |
| 35% | 2 | 47.2 / 57.3 | 1085 |
| 35% | 4 | 38.6 / 46.1 | 1317 |
| 35% | 8 | 34.4 / 41.0 | 1496 |
| 35% | 16 | 40.5 / 40.6 | 1573 |
| 25% | 0 (normal) | 75.9 / 76.4 | 404 |
| 25% | 1 | 58.8 / 59.4 | 660 |
| 25% | 2 | 51.8 / 51.7 | 846 |
| 25% | 4 | 42.7 / 43.1 | 1094 |
| 25% | 8 | 36.6 / 36.8 | 1348 |
| 25% | 16 | 34.5 / 32.7 | 1512 |

- Reuse works in the real path without prediction: at 25%, W = 8 gives 36.7 ms (27 tok/s) and W = 16 gives 33.6 ms
  (30 tok/s), against 76 ms; at 35%, W = 8 gives 34 to 41 ms against 83 to 97 ms.
- **It costs VRAM nearly in proportion.** The selected rows wander across most of the plug within a few tokens: holding
  the last 8 tokens' rows takes 1.35 to 1.50 GB of the 1.61 GB plug. At W = 16 it is effectively the whole plug in VRAM,
  which is the 8-bit-equivalent footprint GLINT exists to avoid. Even W = 1 needs 660 / 881 MB (the current and previous
  token's rows) for a 22 to 26% speed-up.
- So on this model the speed-for-VRAM trade is roughly linear, and the operator's "without taking up more VRAM" goal and
  this lever conflict. A bounded pool of B MB would get about the speed of the window whose resident size is B (LRU is
  assumed to behave like the window; not measured). Choosing B is the operator's call.

## Minimum plug share per layer (2026-11-18)

Operator direction (2026-11-18): "the goal is make the vram operating part as small as possible this may inform us that
there is a bare min amount needed per layer and peer head etc. or a percentage per". Switch `GLINT_LAYER_PCT` (32
percents, every class) in the lean decoder, build `/tmp/p7_lean6`, identity passed. Forced context, both sets, scored
from stored reference margins. One 8-layer band lowered at a time, every other layer at 25%.

| band lowered | to 15%: decided A / B | to 5%: decided A / B |
|---|---|---|
| none (uniform 25%) | 0 / 0 | |
| layers 0-7 | 0 / 0 | 2 / 2 |
| layers 8-15 | 0 / 0 | 1 / 2 |
| layers 16-23 | 2 / 1 | 2 / 1 |
| layers 24-31 | 3 / 3 | 4 / 2 |

- The minimum is not uniform across depth. Layers 0-15 land at 15% (and not at 5%); layers 16-31 do not land at 15%.
- At the margin the later layers need the larger share. The burn sweep's "layers 0-7 matter most" measured a different
  thing (full plug in one window, every other layer at 0) and does not carry over to small changes around 25%.
- Each band at 15% saves 40 MB per token (10% of the 402 MB at uniform 25%). Combined profiles follow.

Combined profiles (layers 0-7 / 8-15 / 16-23 / 24-31), same method:

| profile | plug MB per token | decided A / B |
|---|---|---|
| 25 / 25 / 25 / 25 | 403 | 0 / 0 |
| 15 / 15 / 25 / 25 | 322 | 1 / 4 |
| 10 / 10 / 30 / 30 | 322 | 4 / 2 |
| 10 / 15 / 25 / 25 | 302 | 3 / 2 |
| 15 / 15 / 20 / 25 | 302 | 1 / 3 |
| 15 / 15 / 25 / 30 | 342 | 2 / 3 |

- Bands that land alone at 15% fail together. The residual each layer leaves accumulates through depth into one landing
  budget per token, so the minimum is not a per-layer floor that can be summed: it is a total, shared across layers.
  This is the G-fold structure (one constraint, a shared price), measured: cutting one band uses most of the slack.
- Moving bytes from early to late layers does not buy the slack back (10/10/30/30 and 15/15/25/30 fail).
- So far the smallest landing operating point is one band at 15% with the rest at 25%: 362 MB per token, 10% below
  uniform 25%. The smallest uniform share that lands is being pinned next (21 to 23%).

Uniform shares, same method (data `layer_min3/`):

| uniform share | plug MB per token | decided A / B |
|---|---|---|
| 20% | 322 | 0 / 1 |
| 21% | 338 | 0 / 1 |
| 22% | 354 | 0 / 2 |
| 23% | 370 | 0 / 0 |
| 25% | 403 | 0 / 0 |
| 25% with one of layers 0-7 or 8-15 at 15% | 362 | 0 / 0 |

- The smallest landing operating point on these 24 prompts is about 362 to 370 MB of plug per token (about 23% of the
  plug). Below that, set B starts to show a decided flip at single positions, and not monotonically (22% shows 2, 21%
  shows 1), so the edge is resolved only to about 2 percentage points by 24 prompts.
- The VRAM side of the plug is already small on the transient path: one channel-addressed slot per class (50 MB), and only
  the selected rows of one linear are in use at a time (largest: gate+up at 23%, about 5.8 MB). The 362 to 370 MB is link
  traffic per token, not resident VRAM.

## Partial plug: how much of the correction the top bits carry (2026-11-18)

Operator idea (2026-11-18, "quarter burn": burn points at any level, with the opposite dimension calibrating the on-off
point). `scripts/glint_plug_prefix_offline.py`, layers 0, 8, 16, 24, 31, all four linears, against the BF16 source. The
pin's cell table is ordered by child index in every hole row, so a b-bit prefix of the plug selects a contiguous range of
cells; "table" uses the mean of that range, "fit" the data-weighted mean (what a table built for prefixes could reach).
Data `prims_20261118/prefix/prefix_rms.tsv`.

| plug bits | bytes per row | share of the fold-to-full weight error removed (table / fit) |
|---|---|---|
| 1 | 1/4 | 50.7% / 53.3% |
| 2 | 1/2 | 78.2% / 80.1% |
| 3 | 3/4 | 92.6% / 93.9% |
| 4 | 1 | 100% / 100.8% |

- Each bit roughly halves the remaining error, uniformly across layers and classes. Returns per byte are concave: 2 bits
  carry 78% of the correction for half the bytes. That is the condition under which intermediate levels pay.
- The current table already works as a nested code (table within 2 points of fit), so no new table is needed to test it.
- Landing test (pre-registered): `GLINT_PLUG_BITS=b` coarsens the table so the low 4-b bits change nothing (same bytes
  move; landing only). Rule: a b-bit arm is better than 4-bit 23% if it has 0 decided flips on both sets at a byte budget
  (share x b/4) below 23%. Arms: 2 bits at 23 / 34 / 46 / 60%, 3 bits at 31%, 1 bit at 46%.

### Partial-plug landing results (2026-11-18): null

Data `prims_20261118/plug_bits/`. Build `/tmp/p7_lean7`, identity passed. Forced context, scored from stored margins.
Bytes are those a real b-bit layout would move (share x b/4); the emulation moves full rows, so its ms/token is not the
b-bit speed.

| arm | bytes (4-bit-share terms) | decided A / B |
|---|---|---|
| 4 bits, 23% (reference) | 23% | 0 / 0 |
| 2 bits, 23% | 11.5% | 5 / 3 |
| 2 bits, 34% | 17% | 3 / 2 |
| 2 bits, 46% | 23% | 3 / 2 |
| 2 bits, 60% | 30% | 3 / 1 |
| 3 bits, 31% | 23% | 1 / 0 |
| 1 bit, 46% | 11.5% | 10 / 11 |

- **Null by the pre-registered rule.** No partial plug lands at fewer bytes than 4 bits at 23%. 2 bits fail even at 60%
  of channels (30% bytes); 3 bits at the same bytes come within one decided flip.
- Reading: the decided flips are driven by the leftover error in the channels that matter most (largest |x|). A 2-bit plug
  leaves 22% of their correction undone however many extra channels it covers. Concave returns in weight space did not
  carry over to landing, because the extra channels are the ones that matter least.
- Untested and consistent with this: two levels at once (4 bits for the top channels, 2 bits for the next tier), which
  is the interior-level G-fold structure. It needs a two-tier selection and cannot be emulated with one table.

### Two-tier plug: full plug on the top channels, prefix on the next tier (2026-11-18)

`GLINT_TIER2_PCT=p2` with `GLINT_PLUG_BITS=b` in the lean decoder (build `/tmp/p7_lean8`, identity passed): the top
`percent` channels of each linear get the full 4-bit correction and the next channels up to p2 the b-bit prefix. Applied
as corr(prefix table, top p2) + corr(full minus prefix, top percent), exact because the correction is linear in channels
and the top set lies inside the wider set. Bytes are those of a real two-tier layout (top rows at 4 bits, tier-2 rows at
b bits). Forced context, scored from stored margins. Data `prims_20261118/plug_tiers/`.

| split | plug MB per token | decided A / B | tie flips A / B |
|---|---|---|---|
| 15% + next 12% at 2 bits | 338 | 0 / 0 | 13 / 12 |
| 12% + next 22% at 2 bits | 371 | 0 / 0 | 7 / 14 |
| 15% + next 16% at 2 bits | 371 | 0 / 0 | 14 / 15 |
| 18% + next 10% at 2 bits | 371 | 0 / 0 | 11 / 12 |
| 20% + next 6% at 2 bits | 371 | 1 / 0 | 12 / 10 |
| 15% + next 11% at 3 bits | 375 | 1 / 0 | 11 / 13 |
| full plug 23% (reference) | 370 | 0 / 0 | 16 / 13 |

- **The two-tier plug lands at fewer bytes than any single-level plug:** 338 MB (15% + 12% at 2 bits), 9% below full plug
  at 23% and 16% below the 25% used before. Partial levels pay as an interior level under a full burn, not instead of it
  (the operator's "quarter burn" with the full burn calibrating the on-off point).
- Speed is not measured here: the emulation moves full rows for both tiers. A real two-tier layout (a separate plane
  holding the top 2 bits of each row) is needed for the byte saving to reach the link.
- Lower splits are being searched next.

Lower splits (data `plug_tiers2/`):

| split | plug MB per token | decided A / B |
|---|---|---|
| 10% + next 16% at 2 bits | 290 | 0 / 0 |
| 12% + next 12% at 2 bits | 290 | 0 / 2 |
| 12% + next 16% at 2 bits | 322 | 0 / 0 |
| 10% + next 20% at 2 bits | 322 | 0 / 0 |
| 15% + next 10% at 2 bits | 322 | 1 / 0 |
| 8% + next 24% at 2 bits | 322 | 1 / 1 |

- 10% full + 16% at 2 bits lands at 290 MB per token: 22% below the full-plug minimum (370 MB) and 28% below 25%.
- The edge is not monotone (12% + 12% fails on set B at the same bytes), so 24 prompts resolve the split only to a few
  points. Searching lower next.

Lowest splits (data `plug_tiers3/`), all failing:

| split | plug MB per token | decided A / B |
|---|---|---|
| 8% + next 16% at 2 bits | 258 | 1 / 3 |
| 10% + next 12% at 2 bits | 258 | 1 / 2 |
| 10% + next 14% at 2 bits | 274 | 0 / 1 |
| 8% + next 20% at 2 bits | 290 | 1 / 0 |
| 6% + next 24% at 2 bits | 290 | 1 / 0 |
| 9% + next 18% at 2 bits | 290 | 1 / 1 |

- At 290 MB only one of five splits lands (10% + 16%), so that pass is at the edge and may be sample luck. At 322 MB two
  of four land (12% + 16%, 10% + 20%). The two-tier minimum on these 24 prompts is 290 to 322 MB per token, against 370 MB
  for the best single-level plug: a 13 to 22% cut in link bytes for the same landing standard.
- Resolving the edge further needs more prompts (a third set and its full-plug reference), not finer splits.

### Real two-tier layout on the link (2026-11-18)

`GLINT_TIER2_REAL=1` (commit 9aac869; kernels in `mojo/p7_plug_planes.mojo`): at load each linear's plug is split into a
pinned HI plane (top 2 bits) and LO plane (low 2 bits); the wide set fetches HI rows, the top set LO rows, and
`apply_corr_ksc_planes` keeps `apply_corr_ksc`'s arithmetic and order. Data `prims_20261118/tier_real/`.

- Identity: normal path PPL 74.34960059115652; real against emulation identical at every position (12/28 and 10/26,
  teacher-forced 128) and on the forced rollout (10/26, all 24 roll and margin lines).
- Link bytes are now real: 290 MB per token at 10/26 (measured 291.7 MB), 322 MB at 12/28 (measured 324.0 MB).

| arm, teacher-forced 128, 3 interleaved repeats | median ms/token | link MB per token |
|---|---|---|
| normal 25% | 76.6 | 403 |
| normal 23% | 76.1 | 371 |
| real two-tier 12/28 | 89.8 | 322 |
| real two-tier 10/26 | 83.3 | 290 |
| emulated two-tier 12/28 | 97.7 | 564 |

- **Slower despite fewer bytes.** Against normal 23%, 10/26 moves 80 MB less (about 8.7 ms at 9.2 GB/s) yet takes 7 ms
  longer, so the two-tier path carries about 16 ms of other cost per token. Host enqueue time is 10 to 13 ms per token on
  every tier arm against 2 ms on normal arms (measured). The second selection and the second correction per linear are
  the likely cause (not profiled). One selection kernel producing both thresholds and one fused two-plane correction would
  remove a serial step and a launch per linear.
- The real planes do save link time where bytes differ at equal machinery: real 12/28 is 7.9 ms faster than emulated
  12/28 for 242 MB less traffic.
- Cost: about 1.6 GB more pinned host memory (the two planes next to the original plug); no VRAM change.

### Fused two-tier path (2026-11-18)

`GLINT_TIER2_FUSED` (commit 28aa051, kernels in `mojo/p7_plug_planes.mojo`), on top of `GLINT_TIER2_REAL`. Data and
gates `prims_20261118/tier_fused/` (`summary.txt`). Modes: `sel` one selection launch for both sets; `x` also one gather
launch and one correction launch with the unfused arithmetic (bit-identical); `1` one pass over the wide list with top-set
membership by key threshold (summation order changes).

- Gates: normal path identity passed; `sel` and `x` identical to the unfused real path at every position (12/28, 10/26);
  `x` rollouts identical on both sets (0 decided). `1` drifts by fp32 order (max per-linear difference 3.6e-5) and shows
  1 decided flip on set B at (3, 7): fails the landing gate.
- Speed, teacher-forced 128, 3 interleaved repeats on a quiet host (medians, ms/token): normal 23% 58.17; real unfused
  10/26 69.09; `sel` 68.32; `x` 65.65; `1` 62.63. Host enqueue falls from 8.8 to 1.6 ms/token with the fused launches.
- Still slower than normal 23%. Nsight Systems kernel totals per token (normal 23% / mode 1): gather 37.1 / 31.3 ms (the
  byte saving), selection 5.7 / 8.6, correction 7.1 / 14.5. With corrections off (timing only) mode 1 beats normal
  (47.9 vs 50.8), so the remaining gap is the plane correction, about 1.7x the per-channel cost of the normal one; cause
  not found (a branch-free variant gave the same time).

### Two-tier correction cost found and removed (2026-11-19)

Branch `perf/glint-tier2-corr`, kernels in `mojo/p7_plug_planes.mojo`. Data `lakes/p7_glint/prims_20261119/`
(`corr_var/nsys2/` kernel traces, `e2e/` end-to-end and landing). Nsight Compute could not read counters (the driver restricts
GPU performance counters to admin users; a host setting left to the operator), so costs come from Nsight Systems kernel
durations. The desktop shares this GPU (about 39% utilization with no GLINT process), so whole-token times are noisy and every
comparison below is interleaved.

- **Cause.** Probes of the per-row tier correction, each run beside the exact kernel so selections stay identical (an
  ablation that changes values changes the selected sets and invalidates the timing; the first probe round did this and was
  discarded): base loads 5.32 ms/token, + table read 6.17, + HI-plane read without table 9.15, HI + table 10.96, full 14.05.
  The 2-bit plane reads in VRAM were the cost, not the tables (a combined table and shared-memory tables both changed nothing).
- **Fix.** The planes exist to save link bytes. `GLINT_TIER2_FUSED=m`: the gather pulls the same plane rows and writes each
  channel's row into the 4-bit stage slot as `(HI << 2) | LO` (LO = 0 for wide-only channels); the correction reads it like the
  full-plug kernel and picks `d_pre` or `d_top = d_pre + d_dif` per channel. Bit-identical to mode `1`.
- **Row blocking** (`GLINT_CORR_R`, per class qkv,o,gate+up,down): one thread owns R rows with vector loads and shared-memory
  tables; best 2,2,4,2. **Fast selection** (`GLINT_SEL_FAST=1`): keys in registers, both counts in one packed int, warp-shuffle
  reduction, one barrier per bisection round. Both are bit-identical (position lines and PPL identical) and apply to the full
  plug too.
- Kernel totals per token (32 steps): full plug 23% gather / selection / correction 37.4 / 5.7 / 5.3 with the new kernels
  (40.3 / 6.4 / 7.5 before); two-tier 10/26 `m` 32.3 / 4.5 / 8.8 (31.6 / 8.6 / 14.4 for mode `1`).

| arm, teacher-forced 128, 5 interleaved repeats | ms/token min / median / max | link MB per token |
|---|---|---|
| full plug 23%, old kernels | 59.4 / 61.0 / 62.6 | 371 |
| full plug 23%, new kernels (identical outputs) | 57.1 / 59.3 / 62.8 | 371 |
| two-tier 10/26 `x` (exact unfused arithmetic) | 66.0 / 66.3 / 70.7 | 290 |
| two-tier 10/26 `m` | 54.9 / 58.7 / 66.1 | 290 |
| two-tier 12/28 `m` | 59.4 / 60.4 / 61.8 | 322 |

- Two-tier is no longer slower: 10/26 `m` matches the improved full plug (median 0.6 ms faster, within the spread) and beats
  the old full-plug path by 2.3 ms. It is not clearly faster; per-token time stays dominated by the link-bound gather.
- **Landing headroom.** Mode `1`'s decided flip (set B (3, 7)) sits where the full plug at 23% also keeps the reference token by
  only 0.12 logits; on set A (7, 28) the full plug at 23% and 25% and every two-tier config win only on an exact tie. At minimum
  share every arm lands with almost no headroom, so summation order decides that flip. Two-tier 12/28 `m` lands on both sets
  with 0 decided flips and at least 0.25 logits of headroom at every decided position (`e2e/roll_{A,B}_tm12.log`), more than the
  full plug at 23% (A 0.00, B 0.12), for 13% fewer link bytes and about 1 ms per token more than the improved full plug.

#### Landing with margin (proposed, not locked)

The operator's G-fold form elsewhere (`mercs-gfold-buying-vectors` README) lands at `x(T_land) >= Cost + delta`, and drops
trajectories that reach the floor only on paper. Mapped here: the floor is "the reference token stays top-1 at a decided
position"; delta is the arm's own margin there. A tie win (margin 0) holds only for one summation order, which is what mode `1`'s
flip showed. Gate: `scripts/glint_attr_score.py headroom 25 REF ARM...` (delta = 0.25 logits, two bf16 logit steps; the measured
order perturbation moved one margin by 0.125). Delta is the operator's to set; 0.25 is a proposal.

| arm | link MB/token | set A | set B |
|---|---|---|---|
| full plug 23% | 371 | FAIL, tie at (7, 28) | FAIL, 0.12 at (3, 7) |
| full plug 25% | 403 | FAIL, tie at (7, 28) | PASS |
| two-tier 10/26 `x` | 290 | FAIL | FAIL |
| two-tier 12/28 `m` | 322 | PASS, min 0.25 | PASS, min 0.25 |
| two-tier 12/28 emulated (exact unfused order) | 322 | PASS, min 0.37 | PASS, min 0.25 |

Two-tier 12/28 passes under both summation orders and beats the full plug at 25% on 20% fewer bytes: covering more channels
coarsely closes the landing gap at (7, 28) where more exact bits on fewer channels do not.

### Fold GEMV load-ahead and the critical path (2026-11-19)

Data `lakes/p7_glint/prims_20261119/{gemv_u,e2e2,ovl}/`.

- `GLINT_GEMV_U=4` (`glint_gemv_km_u`): each thread loads the hole words of 4 channels before using any, so 4 loads are in flight
  instead of 1. Same accumulation order, bit-identical (128 teacher-forced positions, PPL 72.24476008256185 both). Fold GEMV
  kernel time 11.4 -> 8.1 ms/token (about 160 -> 224 GB/s on 1.8 GB of fold and scales per token).
- End to end it barely shows (full plug 23%, 3 interleaved repeats: old kernels 58.9 / 59.0 / 59.2, all new kernels 57.1 / 57.9 /
  60.2 ms/token). The schedule explains it: a linear's selection and gather wait for its input, the fold GEMV runs beside the
  gather, and the next linear's input exists only after this linear's correction. Per linear the critical path is selection,
  gather, correction; the fold is hidden under the gather, and the link idles during every selection and correction.
- Null: splitting each gather into two halves with an event so the first half's correction chunks overlap the second half's
  transfer (same chunks, bit-identical) was slower: full plug 65 to 68 vs 57 to 62 ms/token, two-tier 12/28 69 to 78 vs 62 to 69.
  Likely the correction blocks take the SMs the second gather's blocks need; Mojo exposes no stream priority to test that. Code
  removed; numbers kept.
- What is left on the critical path is serial by data dependence. Removing it needs the selection for linear i+1 before its
  exact input exists: a prediction (the fold-only input is the cheap look-ahead), which changes selections and must pass the
  landing gate.

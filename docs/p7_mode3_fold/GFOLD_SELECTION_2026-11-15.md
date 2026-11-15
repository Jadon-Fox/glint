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
| relational / price (62.5 to 645.7 MB) | 0.110 ms per MB | 9.1 GB/s | about 31 ms |
| uniform (162.7 to 646.2 MB) | 0.108 ms per MB | 9.3 GB/s | about 42 ms |

The measured link ceiling is 11.2-11.6 GB/s (0.086-0.089 ms per MB). So about four fifths of the marginal cost of every extra byte is the link itself, and gather plus decode plus apply together add about one fifth on top. Decode is therefore not the main per-byte cost; removing it entirely would save at most about 20% of the byte-dependent time. The fixed part (two fold passes, attention, logits) is 31-42 ms.

Consequence: at a given selection, speed is set by bytes that cross the link per token. The levers that remain are (a) not re-sending rows that are already on the device (the reuse probe says a 4-token window covers about 83% of bytes at 35% and 90% at 45%, as an upper bound), and (b) needing fewer rows for the same tracking. Selection stays the model's own choice; residency only avoids moving the same row twice.

## 7. What this does not claim

No measurement. No statement that the tightness theorem holds across layers. No M6 acceptance. The free-running bars remain agent-written and unratified.

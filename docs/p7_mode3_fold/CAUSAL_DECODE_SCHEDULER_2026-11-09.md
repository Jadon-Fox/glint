# R2 causal demand and R3/R4 scheduling hypotheses — 2026-11-09

**Scope:** GitHub Project #6 #111/#112/#113. **Status:** DESIGNED, NOT MEASURED. Not a replacement for MLIR/CUDA/Mojo acceptance tests.

## Actual within-token dependency graph

The compiler knows the operation graph in advance; it does **not** know which corrections will matter numerically before sufficient activations exist.

| Consumer | Canonical input available when | Earliest known structural dependency | First possible current-input selection | Latest safe correction deadline |
|---|---|---|---|---|
| Layer L QKV | post input RMSNorm | before layer L, exact known tensor/shape | after input norm | before final Q/K/V, RoPE and canonical KV write |
| Layer L attention output | completed Q and cached K/V | position/context length and layout static | after Q/K | before attention output is consumed by o_proj |
| Layer L o_proj | attention output (after QKV/softmax) | attention projection is known | after attention output | before residual add and next norm |
| Layer L MLP gate/up | post-attention normalized residual | layer/shape known | after post-attention norm | before SwiGLU |
| Layer L down_proj | canonical SwiGLU activation | MLP projection known | after activation | before residual update and next-layer input |
| Layer L+1 QKV and MLP | corresponding L+1 normalized inputs | operation graph known | after corresponding L+1 inputs materialize | before their respective numerical consumer deadlines |

Early residuals, attention outputs, token identity, prior tokens and prior-layer signals are **predictors** for later selections, never guaranteed current-input oracle data. Mismatch caused by a coarse preview propagates into later activation-dependent demand. For every selected unit record identity, needed precision and arithmetic consumer, not just top-k overlap.

## Hypothesis / falsifyIf / outcome contract

### H-R2-1: Causal earlier signals rank future correction demand

**Test:** at each module/layer, timestamp when each signal is available relative to the earliest prefetch start; record current-input oracle top-k solely *offline*. Compare static hot, prior token, prior layer, residual input and attention-output candidates at equal transferred bytes and held-out documents. Measure top-k recall, weighted output-effect coverage, lead time, cache misses and induced free-running quality.

**falsifyIf:** predictive signals offer no useful lead time at required byte budget, or their missed critical corrections cause unacceptable fidelity despite high top-k recall. Do not invent a universal recall pass cutoff; acceptance follows the existing numerical gates.

### H-R3-1: Transport grouping can improve speed without changing arithmetic

**Test:** exact same selected correction sets and order with (A) current CUDA-selected gather, (B) contiguous group transfer, and (C) static lookahead prefetch. Keep correction precision and canonical inputs fixed; record transaction sizes, fresh bytes, unused overfetch, H2D time, compute stalls, overlap and peak VRAM.

**falsifyIf:** grouping increases fresh bytes/working-set pressure enough that end-to-end decoding is not faster under the same numerical path. Synthetic copy speed alone is inconclusive.

### H-R3-2: Graduated broad-to-narrow staging is a net benefit

**Test:** early broad candidate set using only causal signals, copy in bounded GPU staging, then narrow when later activations become available. Separate predicted-demand confidence, cache reuse, deadline misses and extra transferred bytes. Compare a policy that never speculates.

**falsifyIf:** the early signal does not arrive sufficiently before the latest correction deadline, speculative overfetch saturates PCIe, or temporary memory pressure increases stalls. No nominally free VRAM assumption.

### H-R4-1: Linear correction accumulation can reduce ephemeral storage

**Test:** same canonical input X, compare direct full-cell GEMV vs fold-plus-delta; calculate exact and f32/BF16 errors, downstream effects and acceptable tolerance. Retain the output accumulator but free/reuse delta storage only after its last consumer and completion event. Verify codebook reconstruction, scale-block and rounding.

**falsifyIf:** correction reassociation violates numerical gates, a nonlinear consumer already committed before correction, or a buffer is reused before its last consumer. Recompute is mandatory if delaying correction across an incompatible dependency.

### H-R4-2: MLIR scheduling transformation improves measured decode

**Test:** after a correct DeviceContext/Mojo scheduler is established, choose one explicit MLIR-level dependency/tiling transformation; measure before/after same-pin, same-token, same-oracle decode including H2D bytes, stalls, occupancy, memory peaks, per-token time and fidelity. Log compiler hashes and hardware. Keep CUDA as reference.

**falsifyIf:** transformation introduces unsupported hardware dependence (Ampere RTX 3060 sm_86), numerical mismatch or no end-to-end gain; mark supported/falsified/inconclusive separately.

## Required instrumentation before claiming success

Per token, layer and linear: source activation availability, prediction time, prefetch start/end, transferred bytes, buffer slot, last consumer, actual selection, correction completion, canonical KV/residual update, arithmetic deadline, and numerical error. Time the same event types under CUDA and Mojo, or explicitly mark a metric unavailable.

**Read-only planning outputs only at this stage.** R2, R3, and R4 are not complete or accepted. #92 M6 remains separately controlled by pre-registered free-running tracking and >=10 measured tok/s in the integrated all-active mode.

# Project 7 — 20B deployment arithmetic, preliminary feasibility bounds (2026-11-09)

**GitHub Project #6:** R6 #115. **Status:** ANALYTICAL ONLY; no 20B deployment was executed. Nothing below is an operator-approved hardware target or model-format decision.

## Known planning inputs

- Long-range scenario: approximately 20 billion parameters, constrained 12-GiB-class GPU, approximately 32 GB system RAM; reference workstation RTX 3060 12 GiB, NVIDIA driver 595.99.02.
- Previously measured best PCIe host-to-device data rate in Project 7 notes approximately 11.6 GB/s for the tested path. Actual negotiated link, workloads and reusable caches must be measured at deployment time.
- Phi-4 tiered T-A example: approximately 523 MB fresh plug data per generated token before reuse, not an accepted full-system 20B plan.
- M6 Phi-4-mini milestone requires at least 10 measured decode tok/s and independent free-running tracking under #92. This rate is not automatically ratified for 20B.

## Ideal storage floor

For 20 billion weights, excluding all metadata and additional state:

| Stored precision | Weight bytes | Comment |
|---|---:|---|
| 4-bit | 10.0 GB (decimal) | no scales / codebooks |
| 8-bit | 20.0 GB | no scales |
| BF16 | 40.0 GB | cannot be entirely resident in 32 GB RAM |

At one F32 scale per 64 weights, block-64 metadata alone is (20B / 64) * 4 = 1.25 GB decimal. A 4-bit fold plus these scales is approximately 11.25 GB, before embeddings, KV, buffers, correction staging and fragmentation. A 12 GiB GPU is approximately 12.88 GB decimal, leaving only ~1.63 GB for all other allocations in that simplified example. Different scale codecs, residency strategies and model architecture will change this substantially.

## Ideal transfer ceiling

Bandwidth upper bound for 10 tokens/s on 11.6 GB/s: 11.6/10 = **1.16 GB fresh transfer per generated token** assuming no other bus traffic or inefficiency. Scaling the existing T-A selection fraction linearly to 20B yields approximately **3.25 GB/token** in the earlier planning estimate; even ideal PCIe transfer of 3.25/11.6 ~0.28 s/token would cap at ~3.6 tok/s without reuse, compression or another placement policy. This is a bound on that transfer strategy, not proof that Project 7 is impossible.

## Critical unmeasured variables

- Real effective fresh bytes per token after candidate overlap, static hot sets, caches and upper-layer staging.
- Exact source/host/disk split; full BF16 source cannot be silently assumed to fit host RAM.
- KV-cache capacity as context increases; head-count and dtype; beam/batch effects.
- GPU active weights, embeddings, temp accumulators, runtime workspace, staging buffers and fragmentation peaks.
- Actual PCIe link speed, pinned host-buffer residency, transfer sizes and overlap; device-local copies are not PCIe transport.
- Computational throughput after direct codebook/correction arithmetic and the cost of source rehydration.
- Numerical fidelity and free-running robustness at independent model families and prompts.

## Hypothesis / falsifyIf

**H-R6-1:** a native progressive representation plus grouped transfer and cache-aware scheduling can keep fresh transfer demand low enough for useful 20B decode without full precision GPU or host residency.

**falsifyIf:** validated fresh bytes/token exceed sustained link budget under the tested target throughput after accounting for realistic misses, or the full model cannot fit the measured combined GPU/host/disk capacity and latency envelope, or corrected output fidelity fails the independent approved gates.

## Implementation consequence

Do not scale the current fixed-fraction T-A traffic blindly. Evaluate native representation and compression jointly with a selective resident cache, grouped prefetch, and possibly partial folded-base residency. Always report actual total peak VRAM/host memory, PCIe bytes, and final decode speed. This document is a constraint model; it is not a completed R6 test.

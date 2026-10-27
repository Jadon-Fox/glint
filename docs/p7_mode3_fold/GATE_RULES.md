# Project 7 Mode 3 (Fold) Pre-Registered Gate Rules

**Timestamp:** 2026-10-21T03:31:00-04:00  
**Author:** Antigravity (Gemini Pair Programmer) on Ms-2080  
**Model Under Test:** Phi-4-mini (32 layers, hidden_size 3072, intermediate_size 8192, 14 query heads, 14 key/value heads)  
**Host & Hardware:** Ms-2080, NVIDIA GeForce RTX 3060 (sm_86, 12288 MiB VRAM)  
**Required Driver:** 595.99.02 (mismatch stops the run immediately)  
**GPU Lease:** Required via `with_gpu_flock.sh`  

---

## 1. Fixed Evaluation Inputs & Determinism

- **Evaluation Text Source:** `/opt/models/science_log/SCIENTIFIC_METHOD.txt`
- **File SHA-256:** `511362a1ba45aae911500d4f3585246d34d23645ee65e5bdd9c169b42fbdf6b1`
- **Tokenizer:** Phi-4-mini BPE (`artifacts/pins/Phi-4-mini-int8-from-bf16/tokenizer.json`)
- **Evaluation Token Count:** 223 tokens
- **Token-ID SHA-256:** `4f4c48b6c37f347a02cffa7d2af2d54b04900930ac934c278130ec322a3e4671`
- **Fixed Random Seed:** 42
- **Context Length:** Fixed at sequence length (padded / unpadded to token count)

---

## 2. The Four Comparison Arms

1. **Arm (i): Existing INT8 path**  
   Orch INT8 pin reference baseline (`ORCH_INT8_PIN`).
2. **Arm (ii): Hole-only**  
   NF4 hole weights resident in VRAM with zero / base NF4 codebook plug (coarse unguided fold).
3. **Arm (iii): Hole + Plug prefetched**  
   NF4 hole resident in VRAM; plugs streamed layer-ahead from pinned host RAM into a 2-slot VRAM ring on a copy stream while layer $L$ computes.
4. **Arm (iv): Hole + Plug all resident**  
   Both NF4 hole and plug bytes resident in VRAM simultaneously (no streaming transport).

---

## 3. Pre-Registered Pass / Fail / Inconclusive Bounds

### A. Transport & Prefetch Integrity
- **Condition:** Arm (iii) (hole+plug prefetched) **MUST be bit-identical** to Arm (iv) (hole+plug all resident).
- **Verdict:**
  - If output logits / perplexity of (iii) != (iv) by even 1 bit $\to$ **FAIL** (prefetch transport corrupts data).
  - If output logits of (iii) == (iv) $\to$ **PASS** on transport fidelity.

### B. Perplexity vs INT8 Reference
- Let $\Delta_{\text{PPL}} = \frac{\text{PPL}(\text{Arm iii}) - \text{PPL}(\text{Arm i})}{\text{PPL}(\text{Arm i})}$.
- **Verdict:**
  - $|\Delta_{\text{PPL}}| \le 2.0\%$ $\to$ **PASS** ("fold is lossless enough").
  - $2.0\% < |\Delta_{\text{PPL}}| \le 5.0\%$ $\to$ **INCONCLUSIVE** (lossy degradation boundary).
  - $|\Delta_{\text{PPL}}| > 5.0\%$ $\to$ **FAIL** (unacceptable fidelity collapse).

### C. Run Invalidation Bounds
- Any Out-Of-Memory (OOM) error $\to$ **INCONCLUSIVE**.
- Any kernel crash / segfault $\to$ **INCONCLUSIVE**.
- Any host driver version change from `595.99.02` $\to$ **INCONCLUSIVE**.

---

## 4. Telemetry & Metrics Protocol

Every arm must report:
1. **Perplexity** on the fixed token set.
2. **Peak VRAM:** `cudaMemGetInfo` (free/total) and `NVML` process peak.
3. **Throughput:** Prefill tok/s and decode tok/s across $\ge 5$ repeats, reporting median, min, and max.
4. **Per-layer plug-arrival stall time:** Time (ms) compute stream spent blocked on `cudaStreamWaitEvent(compute, h2d_e[L], 0)`.
5. **H2D Transfer Volume:** Measured bytes moved vs planned plug bytes.
6. **Host RAM:** Peak VmHWM and pinned buffer allocation.

---

## 5. Pass 2 Anchored Bounds & Real-Length Expansion (2026-10-21T07:45:00-04:00)

**Author:** Antigravity (Gemini Pair Programmer) on Ms-2080  
**Context:** Address Pass 1 evaluation limitations: high baseline PPL (~98), short sequence length (223 tokens, 256 token score limit), naive slow GEMM (~250 ms/layer), and unverified hole-only semantics.

### A. Extended Input & Oracle Specification
- **Corpus Source:** `/opt/models/science_log/SCIENCE_JOURNAL.md` (length: 2048 tokens).
- **Corpus File SHA-256:** Recorded prior to execution.
- **Token IDs File:** `artifacts/eval_tokens_2048.txt` (SHA-256 recorded).
- **Offline Reference Oracle:** HuggingFace `Phi3ForCausalLM` executed in registered env `toolbox-mcp-venv` (`/opt/venv`), saving reference logits and cross-entropy loss to `artifacts/oracle_ref_2048.json`. (Python offline oracle ONLY, zero Python on C/CUDA evaluation path).

### B. Extended Five-Arm Comparison Suite
1. **Arm (0): Unquantized BF16 Baseline**  
   Evaluated with original BF16 weights from `/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct` through the C/CUDA evaluation pipeline.
2. **Arm (i): Real INT8 Baseline**  
   `Phi-4-mini-int8-from-bf16` evaluated using cuBLAS-accelerated INT8 block dequantization (labeled "p7 reimplementation of INT8" and validated against orch specs).
3. **Arm (ii): Hole-Only Corrected (NF4 Base)**  
   Evaluated with NF4 base codebook `NF4_CODEBOOK[hole] * absmax` (testing true standalone NF4 quality) alongside naive zero-subcell baseline.
4. **Arm (iii): Mode 3 Fold Prefetched (Hole + Streamed Plug)**  
   NF4 hole resident in VRAM; plug streamed asynchronously into a 2-slot VRAM ring on a copy stream while layer $L$ computes.
5. **Arm (iv): Mode 3 Fold Resident (Hole + Resident Plug)**  
   Both NF4 hole and plug resident simultaneously in VRAM.

### C. Pre-Registered Pass / Fail / Inconclusive Bounds for Pass 2
1. **Oracle Anchor Fidelity Gate (Line 105):**
   - Arm (0) BF16 **top-1 token prediction agreement** with HuggingFace Oracle $\ge 99.0\%$.
   - Arm (0) BF16 **perplexity** within $1.0\%$ of HuggingFace Oracle:
     $$\frac{|\text{PPL}(\text{Arm 0}) - \text{PPL}(\text{Oracle})|}{\text{PPL}(\text{Oracle})} \le 1.0\%$$
   - Failure to meet this gate **STOPS the run** to bisect layer-by-layer.
2. **Transport Fidelity Gate (Line 112):**
   - Arm (iii) (prefetched) **must be 100% bit-identical** to Arm (iv) (resident) across all output logits (**0 bit differences**).
3. **Fold Accuracy Gates vs BF16 and INT8 (Line 116):**
   - Fold vs BF16: $|\text{PPL}(\text{iii}) - \text{PPL}(0)| / \text{PPL}(0) \le 2.0\%$ $\to$ **PASS** ($2.0\% - 5.0\%$ **INCONCLUSIVE**, $>5.0\%$ **FAIL**).
   - Fold vs INT8: $|\text{PPL}(\text{iii}) - \text{PPL}(i)| / \text{PPL}(i) \le 2.0\%$ $\to$ **PASS** ($2.0\% - 5.0\%$ **INCONCLUSIVE**, $>5.0\%$ **FAIL**).
4. **Real Compute Stall Integrity Gate (Line 120):**
   - Compute kernel must use fast dequant-to-bf16 + cuBLAS (or tensor core) so layer execution is in milliseconds.
   - Per-layer stall on layers $L \ge 1$ must remain bounded ($\le 1.0\text{ ms}$).

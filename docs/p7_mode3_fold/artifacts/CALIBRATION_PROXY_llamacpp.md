# Does calibration lower the 4-bit floor? llama.cpp importance-matrix proxy, Phi-4-mini (2026-11-05) — spec Amendments 7 and 7b

Tests the research brief's recommendation 3 (`RESEARCH_BRIEF_2026-11-05.md`) with the one calibrated quantizer available here: GPTQ, AWQ and HQQ need Python environments that workspace policy U-011 blocks, so llama.cpp's importance-matrix quantization is used as a **proxy, not a GLINT arm**. The rule was fixed before the data (spec Amendment 7); one deviation was recorded before any imatrix data existed (Amendment 7b: the journal is only 24 KB, so the calibration text could not be disjoint from the evaluation tokens in the planned way, and about 63 KB of workspace markdown from a different domain was used instead).

Method: `llama-imatrix` on the BF16 Phi-4-mini GGUF with that text (30 chunks of 512 tokens), the same BF16 file re-quantized to Q4_0, IQ4_NL and Q4_K_M with the matrix, each scored as in `COMPETITOR_llamacpp_phi4mini.md` (`llama-perplexity --kl-divergence`, first 2048-token chunk of `SCIENCE_JOURNAL.md`, against the BF16 baseline logits; llama.cpp commit `5ce06fc`).

| quantization | mean KL, no matrix | mean KL, imatrix | change | same top-1 (no matrix / imatrix) | PPL / PPL(BF16) with matrix |
|---|---|---|---|---|---|
| Q4_0 | 4.84e-2 | 2.57e-2 | **-46.8%** | 93.06% / 95.31% | 1.0083 |
| IQ4_NL | 2.69e-2 | 1.88e-2 | **-30.1%** | 95.70% / 95.60% | 1.0116 |
| Q4_K_M | 2.09e-2 | 1.90e-2 | -9.2% | 95.80% / 95.89% | 1.0163 |
Raw table: `CALIBRATION_PROXY_llamacpp.tsv`.

## Verdict under the pre-registered rule
*Worth pursuing for the GLINT base* required the matrix to lower the mean KL of Q4_0 **and** IQ4_NL by at least 30% each. **Met**: -46.8% and -30.1%. IQ4_NL clears the bar by 0.1 point (30.06%), so on one text and one run it is a borderline pass, not a margin; Q4_0 passes comfortably. Q4_K_M, whose two-level scales already adapt to the data, gains only 9%.

## Reading
- *Measured:* for the simple 4.5-bit formats (a per-block scale with a fixed codebook, like GLINT's hole), calibration removes between a third and a half of the KL, even with a calibration text from a different domain than the evaluation text.
- *Inference, not tested on GLINT:* applying a comparable reduction to the NF4 hole's fold-only KL (7.9e-2 on Phi-4-mini, 2.9e-2 on Qwen3, 1.5e-1 on the two small models) would give roughly 4e-2 to 5.5e-2 on Phi-4-mini, still above what calibrated IQ4_NL reaches (1.9e-2) at the same 4.5 bits. A calibrated base would shrink what the streamed residual has to correct but would not by itself replace it.
- The proxy changes how weights are rounded using activation statistics; GLINT's hole is a data-free absmax NF4 code, so a real test needs an error-compensating rounding of the hole (GPTQ-style) in C/CUDA or a Python environment the policy allows.

## What this does not show
One text, one run, three formats, an importance matrix from a different domain than the evaluation text; no GLINT arm was calibrated. Not reviewed by the Test Designer.

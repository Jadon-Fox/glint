# Tensor-core throughput per number format, RTX 3060 under driver 595.99.02 (2026-11-04)

Question from the operator: use every Ampere-compatible size, and look at the "GPU unlock AI firmware upgrade" for specifics. Nothing on this machine documents that upgrade (searched the repo, PR #75, the memory notes and earlier sessions; PR #75 records only the versions and says no VBIOS reading existed before the update to compare with). So the specifics are measured: `tests/bench_tensor_formats.cu` times cuBLAS GEMMs (M = N = K = 4096) and raw `mma.sync` loops (no memory traffic) for each format. RTX 3060, 28 SMs, 1807 MHz as reported by the runtime, driver 595.99.02, VBIOS 94.04.71.40.09, one run. Raw output: `TENSOR_FORMATS_rtx3060.tsv`.

| Format (accumulate) | raw `mma.sync` | cuBLAS 4096^3 | per SM per clock (raw) |
|---|---|---|---|
| INT4 (int32) | **209.8 TOPS** | not exposed by cuBLAS | 4146 |
| INT8 (int32) | **101.6 TOPS** | 84.2 TOPS | 2008 |
| FP16 (fp16) | 52.4 TFLOPS | 46.4 TFLOPS | 1036 |
| FP16 (**fp32**) | **52.0 TFLOPS** | 47.7 TFLOPS | 1028 |
| BF16 (**fp32**) | **51.6 TFLOPS** | 46.6 TFLOPS | 1020 |
| TF32 (fp32) | **25.9 TFLOPS** | 23.4 TFLOPS | 512 |
| FP32 (CUDA cores, SGEMM) | - | 7.2 TFLOPS | - |

## What the numbers say
- **INT4 : INT8 : FP16 = 4 : 2 : 1 at the same accumulate speed**, and FP16 with an FP32 accumulator runs at the same rate as FP16 with an FP16 accumulator. TF32 is half of FP16. All six tensor formats are present and run at a clean multiple of 1024 flops per SM per clock (INT4 4096, INT8 2048, FP16 and BF16 1024, TF32 512).
- **The FP32-accumulate formats (FP16, BF16) and TF32 run at twice the rate I expected for a stock GeForce card.** Third-party spec listings give about 25.5 TFLOPS dense for this card's FP16/BF16 tensor path with FP32 accumulate, and list GeForce Ampere FP32-accumulate at half the FP16-accumulate rate ([Flopper RTX 3060 12GB](https://flopper.io/gpu/nvidia-geforce-rtx-3060-12gb), [Wccftech RTX 30-series spec table](https://wccftech.com/review/nvidia-geforce-rtx-3060-ti-founders-edition-graphics-card-review/2/), [NVIDIA forum thread on 3090 peak](https://forums.developer.nvidia.com/t/rtx-3090-peak-performance/198049)). Measured here: 52 TFLOPS, i.e. full rate. TF32 measures 25.9 against about 13 expected from the same listings. I did not reach an NVIDIA document for the 3060 itself, so the stock figures are from those listings and from my recollection of the GA10x whitepaper, not verified here.
- **Reading, as an inference:** the FP32-accumulate and TF32 throttle that GeForce parts normally have is not present on this card in its current state. That is consistent with the firmware change you describe, but this repository has no measurement from before the change, so the numbers show the state and cannot attribute the cause.
- Related correction found while checking: the card's PCIe link reports **Gen3 x16** (max 3, current 3), not Gen4 as the Oct-21 report states; that matches the 11.3-11.7 GB/s measured for the plug stream.

## What it means for GLINT (inference, not tested here)
- **Prefill is compute-bound and can use the tensor cores at their full rate in any format.** BF16 or FP16 at 52 TFLOPS needs no accumulate workaround; INT8 gives 102 TOPS, twice that; INT4 gives 210 TOPS.
- **A 4-bit hole can feed a tensor core without a lossy step only through INT8 or BF16:** Ampere's integer `mma` needs both operands in the same width (s4 x s4 or s8 x s8), so a 4-bit weight with 8-bit activations means widening the hole to s8 in registers (INT8 rate, 102 TOPS) rather than a native mixed s4 x s8 instruction. A native s4 path needs 4-bit activations, which is a different (lossier) arm.
- **Decode is memory-bound**, so the format that matters there is bytes per weight, not tensor rate: the current decode GEMV reads about 100 GB/s of a possible ~360.

## Not shown
One run, one driver state, 4096-cubed GEMMs and a register-only `mma` loop; no sparse (2:4) rates, no FP8 (not on Ampere). No comparison to a stock-firmware 3060.

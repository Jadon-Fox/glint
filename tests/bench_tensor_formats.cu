/* tests/bench_tensor_formats.cu — what this GPU's tensor cores deliver per number format.
 * Part 1: cuBLAS GemmEx, M = N = K = 4096, FP32 / TF32 / FP16 (16-bit and 32-bit accumulate) / BF16 / INT8.
 * Part 2: raw mma.sync loops (no memory traffic) for INT4, INT8, FP16 (16- and 32-bit accumulate), BF16, TF32.
 * Prints TSV: part, format, accumulate, TFLOPS_or_TOPS. Compare with the card's published rates to see whether a driver/firmware
 * state changes them (a throttle lifted shows as a value above the published number). usage: bench_tensor_formats */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define CB(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d cublas %d\n", __FILE__, __LINE__, (int)s_); } } while (0)

template <typename F> static double time_ms(F fn, int reps) {
    for (int i = 0; i < 3; i++) fn();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a)); for (int i = 0; i < reps; i++) fn(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms / reps;
}

#define MMA_KERNEL(NAME, ASM, ACC_T, ACC_C, NA, NB)                                                              \
__global__ void NAME(ACC_T *out, int iters) {                                                                    \
    unsigned a0 = threadIdx.x * 2654435761u, a1 = a0 ^ 0x9e3779b9u, a2 = a0 + 7u, a3 = a0 ^ 0x1234567u, b0 = a0 ^ 0x4321u, b1 = a0 + 99u; \
    ACC_T c[4][4];                                                                                                \
    for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) c[j][k] = 0;                                         \
    for (int i = 0; i < iters; i++) {                                                                            \
        _Pragma("unroll") for (int j = 0; j < 4; j++) asm volatile(ASM : ACC_C(c[j]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); \
    }                                                                                                            \
    ACC_T s = 0; for (int j = 0; j < 4; j++) for (int k = 0; k < NA; k++) s += c[j][k];                           \
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;                                                              \
}
#define ACC4_I(c) "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
#define ACC4_F(c) "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
#define ACC2_H(c) "+r"(c[0]), "+r"(c[1])
MMA_KERNEL(k_s4,  "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", int,   ACC4_I, 4, 2)
MMA_KERNEL(k_s8,  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", int,   ACC4_I, 4, 2)
MMA_KERNEL(k_f16f32, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", float, ACC4_F, 4, 2)
MMA_KERNEL(k_bf16, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", float, ACC4_F, 4, 2)
MMA_KERNEL(k_tf32, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};", float, ACC4_F, 4, 2)
/* FP16 with FP16 accumulate: two accumulator registers */
__global__ void k_f16f16(unsigned *out, int iters) {
    unsigned a0 = threadIdx.x * 2654435761u, a1 = a0 ^ 0x9e3779b9u, a2 = a0 + 7u, a3 = a0 ^ 0x1234567u, b0 = a0 ^ 0x4321u, b1 = a0 + 99u;
    unsigned c[4][2]; for (int j = 0; j < 4; j++) { c[j][0] = 0; c[j][1] = 0; }
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : ACC2_H(c[j]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    unsigned s = 0; for (int j = 0; j < 4; j++) s ^= c[j][0] ^ c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device\t%s\tSMs\t%d\tclock_MHz\t%d\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    printf("part\tformat\taccumulate\tTFLOPS_or_TOPS\n");
    const int M = 4096, N = 4096, K = 4096; const double flops = 2.0 * M * N * K;
    void *A, *B, *C; CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4)); CK(cudaMalloc(&C, (size_t)M * N * 4));
    CK(cudaMemset(A, 0x3c, (size_t)M * K * 4)); CK(cudaMemset(B, 0x3c, (size_t)K * N * 4)); CK(cudaMemset(C, 0, (size_t)M * N * 4));
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    float fa = 1.f, fb = 0.f; __half ha = __float2half(1.f), hb = __float2half(0.f); int ia = 1, ib = 0; const int reps = 20;
    auto rep = [&](const char *fmt, const char *acc, double ms) { printf("cublas\t%s\t%s\t%.2f\n", fmt, acc, flops / (ms * 1e-3) / 1e12); fflush(stdout); };
    rep("FP32", "fp32", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &fa, A, CUDA_R_32F, M, B, CUDA_R_32F, K, &fb, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); }, reps));
    rep("TF32", "fp32", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &fa, A, CUDA_R_32F, M, B, CUDA_R_32F, K, &fb, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
    rep("FP16", "fp16", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &ha, A, CUDA_R_16F, M, B, CUDA_R_16F, K, &hb, C, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
    rep("FP16", "fp32", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &fa, A, CUDA_R_16F, M, B, CUDA_R_16F, K, &fb, C, CUDA_R_16F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
    rep("BF16", "fp32", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &fa, A, CUDA_R_16BF, M, B, CUDA_R_16BF, K, &fb, C, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
    rep("INT8", "int32", time_ms([&] { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &ia, A, CUDA_R_8I, K, B, CUDA_R_8I, K, &ib, C, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, reps));
    const int blocks = p.multiProcessorCount * 8, threads = 128, iters = 20000; void *out; CK(cudaMalloc(&out, (size_t)blocks * threads * 4));
    double mmas = (double)blocks * (threads / 32) * iters * 4;
    auto run = [&](const char *fmt, const char *acc, double opsper, void (*launch)(void *, int, int, int)) {
        double ms = time_ms([&] { launch(out, blocks, threads, iters); }, 3);
        printf("mma\t%s\t%s\t%.2f\n", fmt, acc, mmas * opsper / (ms * 1e-3) / 1e12); fflush(stdout);
    };
    run("INT4", "int32", 16384, [](void *o, int b, int t, int i) { k_s4<<<b, t>>>((int *)o, i); });
    run("INT8", "int32", 8192, [](void *o, int b, int t, int i) { k_s8<<<b, t>>>((int *)o, i); });
    run("FP16", "fp32", 4096, [](void *o, int b, int t, int i) { k_f16f32<<<b, t>>>((float *)o, i); });
    run("FP16", "fp16", 4096, [](void *o, int b, int t, int i) { k_f16f16<<<b, t>>>((unsigned *)o, i); });
    run("BF16", "fp32", 4096, [](void *o, int b, int t, int i) { k_bf16<<<b, t>>>((float *)o, i); });
    run("TF32", "fp32", 2048, [](void *o, int b, int t, int i) { k_tf32<<<b, t>>>((float *)o, i); });
    CK(cudaGetLastError());
    return 0;
}

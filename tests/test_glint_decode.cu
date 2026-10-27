#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
#include "../core/cells.h"
#include "../core/glint_htile.cu"

int main() {
    int M = 1, N = 5120, K = 3072, B = 64;
    uint8_t *d_h, *d_p; float *d_am, *d_y; __nv_bfloat16 *d_x;
    cudaMalloc(&d_h, (size_t)N * K / 2);
    cudaMalloc(&d_p, (size_t)N * K / 2);
    cudaMalloc(&d_am, (size_t)N * K / B * sizeof(float));
    cudaMalloc(&d_x, (size_t)M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_y, (size_t)M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_glint_htile_s3_f32(d_h, d_p, d_am, d_x, d_y, M, N, K, B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("glint htile M=1 N=%d K=%d took %f ms\n", N, K, ms);
    return 0;
}

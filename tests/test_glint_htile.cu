/* Test glint_htile against CPU reference across standard and partial-K shapes.
 * Tests Glint Issue #7 fixes: K-alignment and partial-K bounds safety.
 */
#include "../include/glint_cells.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" int launch_glint_htile_s3_f32(
    const uint8_t *d_hole, const uint8_t *d_plug, const float *d_absmax,
    const __nv_bfloat16 *d_x, float *d_y,
    int M, int N, int K, int blocksize);

static float cpu_bf16_round(float v) {
    __nv_bfloat16 b = __float2bfloat16(v);
    return __bfloat162float(b);
}

static void cpu_reference(
    const float *x, const uint8_t *hole, const uint8_t *plug, const float *absmax,
    float *y_ref, int M, int N, int K, int blocksize)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int64_t idx = (int64_t)n * K + k;
                uint8_t hb = hole[idx >> 1];
                uint8_t pb = plug[idx >> 1];
                uint8_t h = (k & 1) ? ((hb >> 4) & 15) : (hb & 15);
                uint8_t p = (k & 1) ? ((pb >> 4) & 15) : (pb & 15);
                float cell = GLINT_CELLS[h][p];
                float am = absmax[idx / blocksize];
                float xv = cpu_bf16_round(x[(int64_t)m * K + k]);
                sum += cell * am * xv;
            }
            y_ref[(int64_t)m * N + n] = sum;
        }
    }
}

static int run_case(int M, int N, int K, int blocksize, const char *desc) {
    printf("Testing case [%s]: M=%d N=%d K=%d blocksize=%d ... ", desc, M, N, K, blocksize);
    fflush(stdout);

    int64_t num_w = (int64_t)N * K;
    int64_t num_x = (int64_t)M * K;
    int64_t num_y = (int64_t)M * N;
    int64_t num_packed = (num_w + 1) / 2;
    int64_t num_am = (num_w + blocksize - 1) / blocksize;

    float *h_x = (float *)malloc(num_x * sizeof(float));
    __nv_bfloat16 *h_x_bf16 = (__nv_bfloat16 *)malloc(num_x * sizeof(__nv_bfloat16));
    uint8_t *h_hole = (uint8_t *)malloc(num_packed * sizeof(uint8_t));
    uint8_t *h_plug = (uint8_t *)malloc(num_packed * sizeof(uint8_t));
    float *h_am = (float *)malloc(num_am * sizeof(float));
    float *h_y_gpu = (float *)malloc(num_y * sizeof(float));
    float *h_y_cpu = (float *)malloc(num_y * sizeof(float));

    for (int64_t i = 0; i < num_x; ++i) {
        float val = ((float)((i * 17 + 3) % 100) - 50.0f) * 0.01f;
        h_x[i] = val;
        h_x_bf16[i] = __float2bfloat16(val);
    }
    for (int64_t i = 0; i < num_packed; ++i) {
        h_hole[i] = (uint8_t)((i * 7 + 1) % 256);
        h_plug[i] = (uint8_t)((i * 13 + 5) % 256);
    }
    for (int64_t i = 0; i < num_am; ++i) {
        h_am[i] = 0.5f + (float)(i % 10) * 0.1f;
    }

    cpu_reference(h_x, h_hole, h_plug, h_am, h_y_cpu, M, N, K, blocksize);

    uint8_t *d_hole = NULL, *d_plug = NULL;
    float *d_am = NULL, *d_y = NULL;
    __nv_bfloat16 *d_x = NULL;

    cudaMalloc(&d_hole, num_packed * sizeof(uint8_t));
    cudaMalloc(&d_plug, num_packed * sizeof(uint8_t));
    cudaMalloc(&d_am, num_am * sizeof(float));
    cudaMalloc(&d_x, num_x * sizeof(__nv_bfloat16));
    cudaMalloc(&d_y, num_y * sizeof(float));

    cudaMemcpy(d_hole, h_hole, num_packed * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plug, h_plug, num_packed * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_am, h_am, num_am * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_bf16, num_x * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, num_y * sizeof(float));

    int rc = launch_glint_htile_s3_f32(d_hole, d_plug, d_am, d_x, d_y, M, N, K, blocksize);
    if (rc != 0) {
        printf("FAILED (launch returned %d)\n", rc);
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_y_gpu, d_y, num_y * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int64_t i = 0; i < num_y; ++i) {
        float diff = fabsf(h_y_gpu[i] - h_y_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    cudaFree(d_hole);
    cudaFree(d_plug);
    cudaFree(d_am);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_x);
    free(h_x_bf16);
    free(h_hole);
    free(h_plug);
    free(h_am);
    free(h_y_gpu);
    free(h_y_cpu);

    printf("max_diff = %.6e ", max_diff);
    if (max_diff < 1e-3f) {
        printf("[PASS]\n");
        return 0;
    } else {
        printf("[FAIL - diff too large]\n");
        return 2;
    }
}

int main(void) {
    printf("=== GLINT H-TILE S3 & GLINT #7 CPU REFERENCE VERIFICATION ===\n");
    int failed = 0;

    /* Standard aligned shapes */
    failed += run_case(16, 16, 64, 64, "1 tile M16 N16 K64");
    failed += run_case(16, 16, 128, 64, "2 tiles M16 N16 K128");
    failed += run_case(32, 32, 192, 64, "multi-block multi-tile M32 N32 K192");

    /* Partial-K and unaligned K shapes (glint #7) */
    failed += run_case(16, 16, 32, 64, "partial K=32 (< TK 64)");
    failed += run_case(16, 16, 48, 64, "partial K=48 (unaligned K%32 != 0)");
    failed += run_case(16, 16, 40, 64, "partial K=40 (unaligned K%32 != 0)");
    failed += run_case(16, 16, 70, 64, "tile crossing K=70 (tail of 6)");
    failed += run_case(16, 16, 100, 64, "tile crossing K=100 (tail of 36)");
    failed += run_case(7, 23, 64, 64, "unaligned M=7 N=23 K=64");

    /* Production scale */
    failed += run_case(4, 3072, 3072, 64, "Phi-4-mini hidden dim M=4 N=3072 K=3072");

    if (failed == 0) {
        printf("ALL %d TESTS PASSED GREEN.\n", 10);
        return 0;
    } else {
        printf("%d TESTS FAILED.\n", failed);
        return 1;
    }
}

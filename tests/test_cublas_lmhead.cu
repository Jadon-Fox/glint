#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 2;
    int K = 4;
    int V = 3;

    // Row-major X [2, 4]:
    // [1, 2, 3, 4]
    // [5, 6, 7, 8]
    __nv_bfloat16 h_X[2 * 4] = {
        __float2bfloat16(1), __float2bfloat16(2), __float2bfloat16(3), __float2bfloat16(4),
        __float2bfloat16(5), __float2bfloat16(6), __float2bfloat16(7), __float2bfloat16(8)
    };

    // Row-major W [3, 4]:
    // [1, 0, 0, 0]
    // [0, 1, 0, 0]
    // [1, 1, 1, 1]
    __nv_bfloat16 h_W[3 * 4] = {
        __float2bfloat16(1), __float2bfloat16(0), __float2bfloat16(0), __float2bfloat16(0),
        __float2bfloat16(0), __float2bfloat16(1), __float2bfloat16(0), __float2bfloat16(0),
        __float2bfloat16(1), __float2bfloat16(1), __float2bfloat16(1), __float2bfloat16(1)
    };

    // Expected X @ W^T [2, 3]:
    // Row 0: [1*1, 2*1, 1+2+3+4] = [1, 2, 10]
    // Row 1: [5*1, 6*1, 5+6+7+8] = [5, 6, 26]

    __nv_bfloat16 *d_X, *d_W;
    float *d_C;
    cudaMalloc(&d_X, sizeof(h_X));
    cudaMalloc(&d_W, sizeof(h_W));
    cudaMalloc(&d_C, M * V * sizeof(float));

    cudaMemcpy(d_X, h_X, sizeof(h_X), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(h_W), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    // C is row-major [M, V], which in memory is M rows of V floats, i.e. leading dimension V!
    // As column-major, it is [V, M] with leading dimension V!
    // W is row-major [V, K], i.e. as col-major it is [K, V] with leading dimension K!
    // To get W @ X^T, we transpose W (CUBLAS_OP_T) from [K, V] to [V, K]!
    // X is row-major [M, K], i.e. as col-major it is [K, M] with leading dimension K.
    // So B is X [K, M], A is W [K, V].
    // C [V, M] = A^T [V, K] @ B [K, M]!
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        V, M, K,
        &alpha,
        d_W, CUDA_R_16BF, K,
        d_X, CUDA_R_16BF, K,
        &beta,
        d_C, CUDA_R_32F, V,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    printf("cublas status=%d\n", (int)st);
    float h_C[2 * 3];
    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);

    printf("Row 0: %f, %f, %f (expect 1, 2, 10)\n", h_C[0], h_C[1], h_C[2]);
    printf("Row 1: %f, %f, %f (expect 5, 6, 26)\n", h_C[3], h_C[4], h_C[5]);

    cublasDestroy(handle);
    return 0;
}

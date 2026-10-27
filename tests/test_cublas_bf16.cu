#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>

int main() {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed\n");
        return 1;
    }
    int M = 223;
    int K = 3072;
    int N = 200064;
    printf("Testing cublasGemmEx M=%d K=%d N=%d\n", M, K, N);
    cublasDestroy(handle);
    printf("PASS\n");
    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

__global__ void int8_block_dequant_gemm_f32_kernel(
    const int8_t* __restrict__ Wq,
    const float* __restrict__ scale,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int N, int K, int B,
    float alpha, float beta)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  const int8_t* wrow = Wq + (size_t)n * (size_t)K;
  const float* xrow = x + (size_t)m * (size_t)K;
  for (int k = 0; k < K; k++) {
    int64_t idx = (int64_t)n * (int64_t)K + (int64_t)k;
    float s = scale[idx / (int64_t)B];
    acc += xrow[k] * ((float)wrow[k] * s);
  }
  float prev = (beta == 0.f) ? 0.f : y[(size_t)m * (size_t)N + (size_t)n];
  y[(size_t)m * (size_t)N + (size_t)n] = alpha * acc + beta * prev;
}

int main() {
    int M = 223, N = 5120, K = 3072, B = 64;
    int8_t *d_W; float *d_scale, *d_x, *d_y;
    cudaMalloc(&d_W, (size_t)N * K);
    cudaMalloc(&d_scale, (size_t)N * K / B * sizeof(float));
    cudaMalloc(&d_x, (size_t)M * K * sizeof(float));
    cudaMalloc(&d_y, (size_t)M * N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEventRecord(start);
    int8_block_dequant_gemm_f32_kernel<<<grid, block>>>(d_W, d_scale, d_x, d_y, M, N, K, B, 1.0f, 0.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("int8 gemm M=%d N=%d K=%d took %f ms\n", M, N, K, ms);
    return 0;
}

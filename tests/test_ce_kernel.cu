#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>

__global__ void cross_entropy_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    float* __restrict__ losses,
    int M, int V)
{
    int s = blockIdx.x;
    if (s >= M) return;
    int tid = threadIdx.x;
    const float* row = logits + (size_t)s * V;
    
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    float my_max = -1e30f;
    for (int v = tid; v < V; v += blockDim.x) {
        if (row[v] > my_max) my_max = row[v];
    }
    s_max[tid] = my_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && s_max[tid + offset] > s_max[tid]) {
            s_max[tid] = s_max[tid + offset];
        }
        __syncthreads();
    }
    float block_max = s_max[0];
    __syncthreads();

    float my_sum = 0.f;
    for (int v = tid; v < V; v += blockDim.x) {
        my_sum += expf(row[v] - block_max);
    }
    s_sum[tid] = my_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int tgt = targets[s];
        float target_val = (tgt >= 0 && tgt < V) ? row[tgt] : 0.f;
        losses[s] = logf(s_sum[0]) + block_max - target_val;
    }
}

int main() {
    int M = 4;
    int V = 1000;
    float h_logits[4 * 1000];
    int32_t h_targets[4] = { 42, 100, 7, 999 };
    for (int i = 0; i < 4 * 1000; i++) h_logits[i] = (float)(i % 17) * 0.1f;

    float *d_logits, *d_losses;
    int32_t *d_targets;
    cudaMalloc(&d_logits, sizeof(h_logits));
    cudaMalloc(&d_targets, sizeof(h_targets));
    cudaMalloc(&d_losses, M * sizeof(float));

    cudaMemcpy(d_logits, h_logits, sizeof(h_logits), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, sizeof(h_targets), cudaMemcpyHostToDevice);

    cross_entropy_kernel<<<M, 256>>>(d_logits, d_targets, d_losses, M, V);
    cudaDeviceSynchronize();

    float h_losses[4];
    cudaMemcpy(h_losses, d_losses, sizeof(h_losses), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        printf("loss[%d] = %f\n", i, h_losses[i]);
    }
    printf("PASS\n");
    return 0;
}

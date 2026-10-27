#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>

#define HIDDEN_DIM 3072
#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define ROTARY_DIM 96
#define INTERMEDIATE_DIM 8192

__global__ void rms_norm_2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int M, int D, float eps)
{
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    const float* x_row = x + (size_t)m * D;
    float* y_row = y + (size_tm * D);

    __shared__ float s_sum[256];
    float thread_sumsq = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        thread_sumsq += val * val;
    }
    s_sum[tid] = thread_sumsq;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_sum[tid] += s_sum[tid + offset];
        __syncthreads();
    }
    float scale = rsqrtf(s_sum[0] / (float)D + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float w = weight ? weight[i] : 1.f;
        y_row[i] = x_row[i] * scale * w;
    }
}

__global__ void split_qkv_rope_kernel(
    const float* __restrict__ qkv, // [M, 5120]
    float* __restrict__ q,         // [M, 24, 128]
    float* __restrict__ k,         // [M, 8, 128]
    float* __restrict__ v,         // [M, 8, 128]
    int M, int start_pos)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = M * (NUM_Q_HEADS + NUM_KV_HEADS + NUM_KV_HEADS) * (HEAD_DIM / 2); // 40 heads * 64 pairs
    if (idx >= total_threads) return;

    int pair = idx % (HEAD_DIM / 2); // 0..63
    int head = (idx / (HEAD_DIM / 2)) % (NUM_Q_HEADS + 2 * NUM_KV_HEADS); // 0..39
    int m = idx / ((HEAD_DIM / 2) * (NUM_Q_HEADS + 2 * NUM_KV_HEADS)); // 0..M-1

    int pos = start_pos + m;
    const float* qkv_row = qkv + (size_t)m * 5120;

    if (head < NUM_Q_HEADS) {
        // Query head
        int h = head;
        int in_base = h * HEAD_DIM;
        float d0 = qkv_row[in_base + pair];
        float d1 = qkv_row[in_base + pair + (ROTARY_DIM / 2)];
        if (pair < (ROTARY_DIM / 2)) {
            // RoPE pair
            float inv_freq = powf(10000.0f, -((float)pair) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            float q0_rot = d0 * c - d1 * sn;
            float q1_rot = d1 * c + d0 * sn;
            q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + pair] = q0_rot;
            q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + pair + (ROTARY_DIM / 2)] = q1_rot;
        } else {
            // Pass through
            q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + pair] = qkv_row[in_base + pair];
            q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + pair + (HEAD_DIM / 2)] = qkv_row[in_base + pair + (HEAD_DIM / 2)];
        }
    } else if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
        // Key head
        int h = head - NUM_Q_HEADS;
        int in_base = NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM;
        float d0 = qkv_row[in_base + pair];
        float d1 = qkv_row[in_base + pair + (ROTARY_DIM / 2)];
        if (pair < (ROTARY_DIM / 2)) {
            float inv_freq = powf(10000.0f, -((float)pair) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            float k0_rot = d0 * c - d1 * sn;
            float k1_rot = d1 * c + d0 * sn;
            k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair] = k0_rot;
            k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair + (ROTARY_DIM / 2)] = k1_rot;
        } else {
            k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair] = qkv_row[in_base + pair];
            k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair + (HEAD_DIM / 2)] = qkv_row[in_base + pair + (HEAD_DIM / 2)];
        }
    } else {
        // Value head: no RoPE
        int h = head - (NUM_Q_HEADS + NUM_KV_HEADS);
        int in_base = (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM + h * HEAD_DIM;
        v[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair] = qkv_row[in_base + pair];
        v[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + pair + (HEAD_DIM / 2)] = qkv_row[in_base + pair + (HEAD_DIM / 2)];
    }
}

int main() {
    printf("Compiling and checking syntax...\n");
    return 0;
}

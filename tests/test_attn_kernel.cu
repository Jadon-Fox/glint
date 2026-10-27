#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128

__global__ void causal_gqa_attention_kernel(
    const float* __restrict__ q, // [M, 24, 128]
    const float* __restrict__ k, // [M, 8, 128]
    const float* __restrict__ v, // [M, 8, 128]
    float* __restrict__ out,     // [M, 24, 128]
    int M)
{
    int hq = blockIdx.x; // 0..23
    if (hq >= NUM_Q_HEADS) return;
    int hkv = hq / 3; // GQA 3:1 mapping
    int tq = threadIdx.x; // query token

    if (tq >= M) return;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    const float* q_vec = q + ((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM;

    float scores[256];
    float max_s = -1e30f;

    for (int tk = 0; tk <= tq; tk++) {
        const float* k_vec = k + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float dot = 0.f;
        for (int d = 0; d < HEAD_DIM; d++) {
            dot += q_vec[d] * k_vec[d];
        }
        float s = dot * scale;
        scores[tk] = s;
        if (s > max_s) max_s = s;
    }

    float sum_exp = 0.f;
    for (int tk = 0; tk <= tq; tk++) {
        float e = expf(scores[tk] - max_s);
        scores[tk] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-20f);

    float* out_vec = out + ((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.f;
        for (int tk = 0; tk <= tq; tk++) {
            const float* v_vec = v + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
            acc += scores[tk] * inv_sum * v_vec[d];
        }
        out_vec[d] = acc;
    }
}

int main() {
    int M = 223;
    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, (size_t)M * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_k, (size_t)M * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_v, (size_t)M * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_out, (size_t)M * NUM_Q_HEADS * HEAD_DIM * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256>>>(d_q, d_k, d_v, d_out, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("causal gqa attention M=%d took %f ms\n", M, ms);
    return 0;
}

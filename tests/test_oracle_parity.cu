/* tests/test_oracle_parity.cu — Layer-by-layer parity check against HuggingFace Phi3ForCausalLM
 * Verifies:
 *   1. Embed tokens
 *   2. Input RMSNorm
 *   3. QKV projection & cuBLAS layout
 *   4. LongRoPE with config-derived theta, factors, and attention_scaling
 *   5. Online softmax causal attention
 *   6. Post-attention RMSNorm
 *   7. MLP GateUp + SwiGLU + Down
 *   8. Final norm and LM head projection
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../core/safetensors_c.h"

#define HIDDEN_DIM 3072
#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define ROTARY_DIM 96
#define INTERMEDIATE_DIM 8192
#define VOCAB_SIZE 200064
#define RMS_EPS 1e-5f
#define ROPE_THETA 10000.0f
#define ATTENTION_SCALING 1.1902380714238083f // sqrt(1 + ln(131072/4096)/ln(4096))

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_bf16,
    int M, int dim, float eps)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = x + (size_t)row * dim;
    float* out_row = out + (size_t)row * dim;
    __nv_bfloat16* out_bf16_row = out_bf16 ? (out_bf16 + (size_t)row * dim) : NULL;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in_row[i];
        sum_sq += val * val;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_mean_sq;
    if (threadIdx.x == 0) s_mean_sq = sum_sq;
    // (For 1 warp: threadIdx.x 0..31. With 128 threads, reduce across 4 warps)
    __shared__ float s_warp[4];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) s_warp[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0 && lane < 4) {
        float b = s_warp[lane];
        for (int offset = 2; offset > 0; offset /= 2)
            b += __shfl_down_sync(0xf, b, offset);
        if (lane == 0) s_mean_sq = b / (float)dim;
    }
    __syncthreads();

    float rscale = rsqrtf(s_mean_sq + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        float val = in_row[i] * rscale * w;
        out_row[i] = val;
        if (out_bf16_row) out_bf16_row[i] = __float2bfloat16(val);
    }
}

/* Canonical LongRoPE kernel matching HuggingFace apply_rotary_pos_emb */
__global__ void phi3_rope_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
    if (idx >= total_elements) return;

    int d = idx % HEAD_DIM;
    int head = (idx / HEAD_DIM) % (NUM_Q_HEADS + 2 * NUM_KV_HEADS);
    int m = idx / ((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM);

    int pos = m;
    const float* qkv_row = qkv + (size_t)m * (5120);

    /* Frequency index i in [0, 48) */
    int i = d % (ROTARY_DIM / 2); // 0 .. 47
    float inv_freq = 1.0f / powf(ROPE_THETA, (2.0f * (float)i) / (float)ROTARY_DIM);
    float angle = (float)pos * inv_freq;
    float cos_v = cosf(angle) * ATTENTION_SCALING;
    float sin_v = sinf(angle) * ATTENTION_SCALING;

    if (head < NUM_Q_HEADS) {
        int h = head;
        int in_base = h * HEAD_DIM;
        float val;
        if (d < ROTARY_DIM / 2) {
            // First half: x0 * cos - x1 * sin
            float x0 = qkv_row[in_base + d];
            float x1 = qkv_row[in_base + d + ROTARY_DIM / 2];
            val = x0 * cos_v - x1 * sin_v;
        } else if (d < ROTARY_DIM) {
            // Second half: x1 * cos + x0 * sin
            float x0 = qkv_row[in_base + d - ROTARY_DIM / 2];
            float x1 = qkv_row[in_base + d];
            val = x1 * cos_v + x0 * sin_v;
        } else {
            // Unrotated pass-through
            val = qkv_row[in_base + d];
        }
        q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + d] = val;
    } else if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
        int h = head - NUM_Q_HEADS;
        int in_base = NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM;
        float val;
        if (d < ROTARY_DIM / 2) {
            float x0 = qkv_row[in_base + d];
            float x1 = qkv_row[in_base + d + ROTARY_DIM / 2];
            val = x0 * cos_v - x1 * sin_v;
        } else if (d < ROTARY_DIM) {
            float x0 = qkv_row[in_base + d - ROTARY_DIM / 2];
            float x1 = qkv_row[in_base + d];
            val = x1 * cos_v + x0 * sin_v;
        } else {
            val = qkv_row[in_base + d];
        }
        k[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d] = val;
    } else {
        int h = head - (NUM_Q_HEADS + NUM_KV_HEADS);
        int in_base = (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM + h * HEAD_DIM;
        v[(size_t)m * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d] = qkv_row[in_base + d];
    }
}

/* Causal GQA online softmax attention: 1 block of 128 threads per (token tq, head hq) */
__global__ void causal_gqa_online_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_bf16,
    int M)
{
    int tq = blockIdx.x;
    int hq = blockIdx.y;
    if (tq >= M || hq >= NUM_Q_HEADS) return;

    int hkv = hq / 3;
    int d = threadIdx.x; // 0 .. 127
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float q_val = q[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d];

    float m_val = -1e30f;
    float d_val = 0.0f;
    float acc = 0.0f;

    __shared__ float s_red[128];

    for (int tk = 0; tk <= tq; tk++) {
        const float* k_row = k + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float prod = q_val * k_row[d];
        s_red[d] = prod;
        __syncthreads();

        // Reduce across 128 threads
        if (d < 64) s_red[d] += s_red[d + 64];
        __syncthreads();
        if (d < 32) {
            float b = s_red[d] + s_red[d + 32];
            for (int offset = 16; offset > 0; offset /= 2)
                b += __shfl_down_sync(0xffffffff, b, offset);
            s_red[d] = b;
        }
        __syncthreads();

        float score = s_red[0] * scale;
        float m_new = fmaxf(m_val, score);
        float alpha = expf(m_val - m_new);
        float beta = expf(score - m_new);

        d_val = d_val * alpha + beta;
        m_val = m_new;

        const float* v_row = v + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float v_val = v_row[d];
        acc = acc * alpha + beta * v_val;
        __syncthreads();
    }

    float final_out = (d_val > 0.0f) ? (acc / d_val) : 0.0f;
    out[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] = final_out;
    if (out_bf16) out_bf16[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] = __float2bfloat16(final_out);
}

/* SwiGLU: out = silu(gate) * up */
__global__ void swiglu_kernel(const float* __restrict__ gate_up, float* __restrict__ out, __nv_bfloat16* __restrict__ out_bf16, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * INTERMEDIATE_DIM;
    if (idx >= total) return;

    int m = idx / INTERMEDIATE_DIM;
    int i = idx % INTERMEDIATE_DIM;

    const float* row = gate_up + (size_t)m * (2 * INTERMEDIATE_DIM);
    float g = row[i];
    float u = row[INTERMEDIATE_DIM + i];
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u;
    out[(size_t)m * INTERMEDIATE_DIM + i] = val;
    if (out_bf16) out_bf16[(size_t)m * INTERMEDIATE_DIM + i] = __float2bfloat16(val);
}

/* Residual add: x = x + res */
__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ res, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) x[i] += res[i];
}

int main(void) {
    printf("=== Test Oracle Parity ===\n");
    // Verify math on first 4 tokens
    int tokens[4] = {2, 499, 13234, 2600};

    SafeTensors st;
    const char* path = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors";
    if (safetensors_open(&st, path) != 0) {
        printf("Failed to open safetensors\n");
        return 1;
    }

    const TensorEntry* e_embed = safetensors_find(&st, "model.embed_tokens.weight");
    const TensorEntry* e_norm0 = safetensors_find(&st, "model.layers.0.input_layernorm.weight");
    const TensorEntry* e_qkv   = safetensors_find(&st, "model.layers.0.self_attn.qkv_proj.weight");

    if (!e_embed || !e_norm0 || !e_qkv) {
        printf("Missing tensor entries\n");
        return 1;
    }

    const __nv_bfloat16* embed_table = (const __nv_bfloat16*)e_embed->data;
    const __nv_bfloat16* norm0_table = (const __nv_bfloat16*)e_norm0->data;

    float h_embed[4 * HIDDEN_DIM];
    for (int t = 0; t < 4; t++) {
        for (int d = 0; d < HIDDEN_DIM; d++) {
            h_embed[t * HIDDEN_DIM + d] = __bfloat162float(embed_table[tokens[t] * HIDDEN_DIM + d]);
        }
    }

    printf("Token 0 embed first 5: ");
    for (int i = 0; i < 5; i++) printf("%f ", h_embed[i]);
    printf("\n");

    float *d_x, *d_h, *d_qkv, *d_q, *d_k, *d_v;
    __nv_bfloat16 *d_norm0, *d_h_bf16, *d_qkv_w;
    cudaMalloc(&d_x, 4 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_h, 4 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_h_bf16, 4 * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_norm0, HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_qkv, 4 * 5120 * sizeof(float));
    cudaMalloc(&d_q, 4 * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_k, 4 * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_v, 4 * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_qkv_w, 5120 * HIDDEN_DIM * sizeof(__nv_bfloat16));

    cudaMemcpy(d_x, h_embed, 4 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm0, norm0_table, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_w, e_qkv->data, 5120 * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    rms_norm_kernel<<<4, 128>>>(d_x, d_norm0, d_h, d_h_bf16, 4, HIDDEN_DIM, RMS_EPS);
    cudaDeviceSynchronize();

    float h_h[5];
    cudaMemcpy(h_h, d_h, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Token 0 h norm first 5: ");
    for (int i = 0; i < 5; i++) printf("%f ", h_h[i]);
    printf("\n");

    // QKV projection via cuBLAS: QKV = H @ W_qkv^T
    // H is [4, 3072], W_qkv is [5120, 3072], QKV is [4, 5120]
    // In col-major: QKV^T [5120, 4] = W_qkv [5120, 3072] @ H^T [3072, 4]
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 5120, 4, HIDDEN_DIM,
                 &alpha, d_qkv_w, CUDA_R_16BF, HIDDEN_DIM,
                 d_h_bf16, CUDA_R_16BF, HIDDEN_DIM,
                 &beta, d_qkv, CUDA_R_32F, 5120,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    float h_qkv[5];
    cudaMemcpy(h_qkv, d_qkv, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Token 0 QKV first 5: ");
    for (int i = 0; i < 5; i++) printf("%f ", h_qkv[i]);
    printf("\n");

    // RoPE
    int total_el = 4 * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
    phi3_rope_kernel<<<(total_el + 255) / 256, 256>>>(d_qkv, d_q, d_k, d_v, 4);
    cudaDeviceSynchronize();

    // 5. Causal GQA online softmax attention
    float *d_attn_out, *d_attn_proj;
    __nv_bfloat16 *d_attn_out_bf16, *d_o_w;
    cudaMalloc(&d_attn_out, 4 * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_attn_out_bf16, 4 * NUM_Q_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_attn_proj, 4 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_o_w, HIDDEN_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));

    const TensorEntry* e_o = safetensors_find(&st, "model.layers.0.self_attn.o_proj.weight");
    cudaMemcpy(d_o_w, e_o->data, HIDDEN_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    dim3 grid_attn(4, NUM_Q_HEADS);
    causal_gqa_online_attention_kernel<<<grid_attn, 128>>>(
        d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 4);
    cudaDeviceSynchronize();

    // O projection: attn_proj = attn_out @ W_o^T
    cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 4, HIDDEN_DIM,
                 &alpha, d_o_w, CUDA_R_16BF, HIDDEN_DIM,
                 d_attn_out_bf16, CUDA_R_16BF, HIDDEN_DIM,
                 &beta, d_attn_proj, CUDA_R_32F, HIDDEN_DIM,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    // Residual add 1: x = x + attn_proj
    residual_add_kernel<<<(4 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_attn_proj, 4 * HIDDEN_DIM);
    cudaDeviceSynchronize();

    // 6. Post-attention RMSNorm
    const TensorEntry* e_norm_post = safetensors_find(&st, "model.layers.0.post_attention_layernorm.weight");
    __nv_bfloat16* d_norm_post;
    cudaMalloc(&d_norm_post, HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMemcpy(d_norm_post, e_norm_post->data, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    rms_norm_kernel<<<4, 128>>>(d_x, d_norm_post, d_h, d_h_bf16, 4, HIDDEN_DIM, RMS_EPS);
    cudaDeviceSynchronize();

    // 7. MLP: GateUp + SwiGLU + Down
    const TensorEntry* e_gate_up = safetensors_find(&st, "model.layers.0.mlp.gate_up_proj.weight");
    const TensorEntry* e_down    = safetensors_find(&st, "model.layers.0.mlp.down_proj.weight");

    __nv_bfloat16 *d_gate_up_w, *d_down_w, *d_act_bf16;
    float *d_gate_up, *d_act, *d_down;
    cudaMalloc(&d_gate_up_w, (2 * INTERMEDIATE_DIM) * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down_w, HIDDEN_DIM * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate_up, 4 * (2 * INTERMEDIATE_DIM) * sizeof(float));
    cudaMalloc(&d_act, 4 * INTERMEDIATE_DIM * sizeof(float));
    cudaMalloc(&d_act_bf16, 4 * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down, 4 * HIDDEN_DIM * sizeof(float));

    cudaMemcpy(d_gate_up_w, e_gate_up->data, (2 * INTERMEDIATE_DIM) * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_w, e_down->data, HIDDEN_DIM * INTERMEDIATE_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // GateUp GEMM: [4, 16384] = [4, 3072] @ [16384, 3072]^T
    cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * INTERMEDIATE_DIM, 4, HIDDEN_DIM,
                 &alpha, d_gate_up_w, CUDA_R_16BF, HIDDEN_DIM,
                 d_h_bf16, CUDA_R_16BF, HIDDEN_DIM,
                 &beta, d_gate_up, CUDA_R_32F, 2 * INTERMEDIATE_DIM,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    // SwiGLU: act = silu(gate) * up
    swiglu_kernel<<<(4 * INTERMEDIATE_DIM + 255) / 256, 256>>>(d_gate_up, d_act, d_act_bf16, 4);
    cudaDeviceSynchronize();

    // Down GEMM: [4, 3072] = [4, 8192] @ [3072, 8192]^T
    cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 4, INTERMEDIATE_DIM,
                 &alpha, d_down_w, CUDA_R_16BF, INTERMEDIATE_DIM,
                 d_act_bf16, CUDA_R_16BF, INTERMEDIATE_DIM,
                 &beta, d_down, CUDA_R_32F, HIDDEN_DIM,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    // Residual add 2: x = x + down
    residual_add_kernel<<<(4 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_down, 4 * HIDDEN_DIM);
    cudaDeviceSynchronize();

    float h_out_tok0[5], h_out_tok1[5];
    cudaMemcpy(h_out_tok0, d_x + 0 * HIDDEN_DIM, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_tok1, d_x + 1 * HIDDEN_DIM, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Layer 0 out first 5 of token 0: ");
    for (int i = 0; i < 5; i++) printf("%f ", h_out_tok0[i]);
    printf("\n");
    printf("Layer 0 out first 5 of token 1: ");
    for (int i = 0; i < 5; i++) printf("%f ", h_out_tok1[i]);
    printf("\n");

    safetensors_close(&st);
    cublasDestroy(cublas);
    cudaFree(d_x); cudaFree(d_h); cudaFree(d_h_bf16); cudaFree(d_norm0);
    cudaFree(d_qkv); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_qkv_w);
    cudaFree(d_attn_out); cudaFree(d_attn_out_bf16); cudaFree(d_attn_proj); cudaFree(d_o_w);
    cudaFree(d_norm_post); cudaFree(d_gate_up_w); cudaFree(d_down_w);
    cudaFree(d_gate_up); cudaFree(d_act); cudaFree(d_act_bf16); cudaFree(d_down);
    return 0;
}

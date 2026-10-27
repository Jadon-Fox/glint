/* tests/test_arm0_bf16_full.cu — Full 32-Layer Arm (0) BF16 Evaluation on 2048 Tokens
 * Loads unquantized BF16 weights directly from model-00001-of-00002.safetensors and model-00002-of-00002.safetensors.
 * Executes on RTX 3060 via Tensor Core cuBLAS and online softmax causal attention.
 * Verifies Gate 1:
 *   - Top-1 argmax token agreement >= 99.0% vs HuggingFace Oracle
 *   - Perplexity within 1.0% of HuggingFace Oracle (5.486957)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "../core/safetensors_c.h"

#define N_LAYERS 32
#define HIDDEN_DIM 3072
#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define ROTARY_DIM 96
#define INTERMEDIATE_DIM 8192
#define VOCAB_SIZE 200064
#define RMS_EPS 1e-5f
#define ROPE_THETA 10000.0f
#define ATTENTION_SCALING 1.1902380714238083f

struct LayerWeightsBF16 {
    const __nv_bfloat16 *input_norm;
    const __nv_bfloat16 *post_norm;
    const __nv_bfloat16 *qkv_w;      // [5120, 3072]
    const __nv_bfloat16 *o_w;        // [3072, 3072]
    const __nv_bfloat16 *gate_up_w;  // [16384, 3072]
    const __nv_bfloat16 *down_w;     // [3072, 8192]

    // Device pointers
    __nv_bfloat16 *d_input_norm;
    __nv_bfloat16 *d_post_norm;
    __nv_bfloat16 *d_qkv_w;
    __nv_bfloat16 *d_o_w;
    __nv_bfloat16 *d_gate_up_w;
    __nv_bfloat16 *d_down_w;
};

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
    for (int offset = 16; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_mean_sq;
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
    const float* qkv_row = qkv + (size_t)m * 5120;

    int i = d % (ROTARY_DIM / 2);
    float inv_freq = 1.0f / powf(ROPE_THETA, (2.0f * (float)i) / (float)ROTARY_DIM);
    float angle = (float)pos * inv_freq;
    float cos_v = cosf(angle) * ATTENTION_SCALING;
    float sin_v = sinf(angle) * ATTENTION_SCALING;

    if (head < NUM_Q_HEADS) {
        int h = head;
        int in_base = h * HEAD_DIM;
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
    int d = threadIdx.x;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float q_val = q[((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM + d] * scale;

    float m_val = -1e30f;
    float d_val = 0.0f;
    float acc = 0.0f;

    __shared__ float s_red[128];

    for (int tk = 0; tk <= tq; tk++) {
        const float* k_row = k + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float prod = q_val * k_row[d];
        s_red[d] = prod;
        __syncthreads();

        if (d < 64) s_red[d] += s_red[d + 64];
        __syncthreads();
        if (d < 32) {
            float b = s_red[d] + s_red[d + 32];
            for (int offset = 16; offset > 0; offset /= 2)
                b += __shfl_down_sync(0xffffffff, b, offset);
            s_red[d] = b;
        }
        __syncthreads();

        float score = s_red[0];
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

__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ res, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float sum = x[i] + res[i];
        x[i] = __bfloat162float(__float2bfloat16(sum));
    }
}

__global__ void compute_ce_and_argmax_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    float* __restrict__ losses,
    int32_t* __restrict__ top1,
    int M, int vocab_size)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float* l_row = logits + (size_t)row * vocab_size;
    int target = targets[row];

    float max_val = -1e30f;
    int max_idx = 200064;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        // Cast to bfloat16 float to match HuggingFace's bfloat16 logits representation
        float val = __bfloat162float(__float2bfloat16(l_row[v]));
        if (val > max_val || (val == max_val && v < max_idx)) {
            max_val = val;
            max_idx = v;
        }
    }

    __shared__ float s_max[256];
    __shared__ int s_idx[256];
    s_max[threadIdx.x] = max_val;
    s_idx[threadIdx.x] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float other_val = s_max[threadIdx.x + stride];
            int other_idx = s_idx[threadIdx.x + stride];
            if (other_val > s_max[threadIdx.x] || (other_val == s_max[threadIdx.x] && other_idx < s_idx[threadIdx.x])) {
                s_max[threadIdx.x] = other_val;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }
    float block_max = s_max[0];
    if (threadIdx.x == 0 && top1) {
        top1[row] = s_idx[0];
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        sum_exp += expf(l_row[v] - block_max);
    }

    __shared__ float s_sum[256];
    s_sum[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float target_l = (target >= 0 && target < vocab_size) ? l_row[target] : 0.0f;
        float log_sum_exp = block_max + logf(fmaxf(s_sum[0], 1e-20f));
        losses[row] = log_sum_exp - target_l;
    }
}

static const TensorEntry* find_tensor(const SafeTensors *st1, const SafeTensors *st2, const char *name) {
    const TensorEntry *e = safetensors_find(st1, name);
    if (!e) e = safetensors_find(st2, name);
    return e;
}

int main(void) {
    printf("=========================================================\n");
    printf("Arm (0) Full 32-Layer BF16 Evaluator (2048 Tokens)\n");
    printf("=========================================================\n");

    // 1. Read token IDs
    FILE* ft = fopen("/home/workspace/p7-scratch/artifacts/eval_tokens_2048.txt", "r");
    if (!ft) { printf("Cannot open eval_tokens_2048.txt\n"); return 1; }
    int32_t tokens[2048];
    for (int i = 0; i < 2048; i++) {
        if (fscanf(ft, "%d", &tokens[i]) != 1) { printf("Error reading token %d\n", i); return 1; }
    }
    fclose(ft);
    printf("Read 2048 tokens successfully.\n");

    // Read oracle reference top-1 predictions
    FILE* fo = fopen("/home/workspace/p7-scratch/artifacts/oracle_top1_2048.bin", "rb");
    if (!fo) { printf("Cannot open oracle_top1_2048.bin\n"); return 1; }
    uint32_t oracle_top1[2047];
    if (fread(oracle_top1, sizeof(uint32_t), 2047, fo) != 2047) {
        printf("Error reading oracle_top1_2048.bin\n"); return 1;
    }
    fclose(fo);
    printf("Read 2047 oracle reference top-1 tokens.\n");

    // 2. Open safetensors shards
    SafeTensors st1, st2;
    const char* p1 = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors";
    const char* p2 = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00002-of-00002.safetensors";
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0) {
        printf("Cannot open safetensors shards\n"); return 1;
    }

    const TensorEntry* e_embed = find_tensor(&st1, &st2, "model.embed_tokens.weight");
    const TensorEntry* e_final_norm = find_tensor(&st1, &st2, "model.norm.weight");
    if (!e_embed || !e_final_norm) {
        printf("Missing embed or final norm\n"); return 1;
    }

    const __nv_bfloat16* embed_table = (const __nv_bfloat16*)e_embed->data;
    __nv_bfloat16 *d_embed_tokens, *d_final_norm;
    cudaMalloc(&d_embed_tokens, (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_final_norm, HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMemcpy(d_embed_tokens, embed_table, (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_final_norm, e_final_norm->data, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Initial embed tokens on host -> copy to d_x
    float* h_x = (float*)malloc(2048 * HIDDEN_DIM * sizeof(float));
    for (int t = 0; t < 2048; t++) {
        for (int d = 0; d < HIDDEN_DIM; d++) {
            h_x[t * HIDDEN_DIM + d] = __bfloat162float(embed_table[tokens[t] * HIDDEN_DIM + d]);
        }
    }

    LayerWeightsBF16 layers[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        char name[256];
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        layers[l].input_norm = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        layers[l].post_norm = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.qkv_proj.weight", l);
        layers[l].qkv_w = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        layers[l].o_w = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_up_proj.weight", l);
        layers[l].gate_up_w = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        layers[l].down_w = (const __nv_bfloat16*)find_tensor(&st1, &st2, name)->data;

        // Allocate device weights
        cudaMalloc(&layers[l].d_input_norm, HIDDEN_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&layers[l].d_post_norm, HIDDEN_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&layers[l].d_qkv_w, 5120 * HIDDEN_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&layers[l].d_o_w, HIDDEN_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&layers[l].d_gate_up_w, 2 * INTERMEDIATE_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&layers[l].d_down_w, HIDDEN_DIM * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));

        cudaMemcpy(layers[l].d_input_norm, layers[l].input_norm, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_post_norm, layers[l].post_norm, HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_qkv_w, layers[l].qkv_w, 5120 * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_o_w, layers[l].o_w, HIDDEN_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_gate_up_w, layers[l].gate_up_w, 2 * INTERMEDIATE_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_down_w, layers[l].down_w, HIDDEN_DIM * INTERMEDIATE_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    printf("Uploaded all 32 layers of BF16 weights to VRAM.\n");

    // Allocate activations
    float *d_x, *d_h, *d_qkv, *d_q, *d_k, *d_v, *d_attn_out, *d_attn_proj;
    float *d_gate_up, *d_act, *d_down;
    __nv_bfloat16 *d_h_bf16, *d_attn_out_bf16, *d_act_bf16, *d_x_final_bf16;

    cudaMalloc(&d_x, 2048 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_h, 2048 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_h_bf16, 2048 * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_qkv, 2048 * 5120 * sizeof(float));
    cudaMalloc(&d_q, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_k, 2048 * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_v, 2048 * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_attn_out, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_attn_out_bf16, 2048 * NUM_Q_HEADS * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_attn_proj, 2048 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_gate_up, 2048 * (2 * INTERMEDIATE_DIM) * sizeof(float));
    cudaMalloc(&d_act, 2048 * INTERMEDIATE_DIM * sizeof(float));
    cudaMalloc(&d_act_bf16, 2048 * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down, 2048 * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_x_final_bf16, 2048 * HIDDEN_DIM * sizeof(__nv_bfloat16));

    // Target tokens & output buffers
    int32_t *d_targets, *d_top1;
    float *d_losses, *d_logits_chunk;
    cudaMalloc(&d_targets, 2047 * sizeof(int32_t));
    cudaMalloc(&d_top1, 2047 * sizeof(int32_t));
    cudaMalloc(&d_losses, 2047 * sizeof(float));
    cudaMalloc(&d_logits_chunk, 256 * VOCAB_SIZE * sizeof(float));

    int32_t h_targets[2047];
    for (int i = 0; i < 2047; i++) h_targets[i] = tokens[i + 1];
    cudaMemcpy(d_targets, h_targets, 2047 * sizeof(int32_t), cudaMemcpyHostToDevice);

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    float alpha = 1.0f, beta = 0.0f;

    printf("Starting forward pass over 32 layers...\n");
    clock_t t0 = clock();

    cudaMemcpy(d_x, h_x, 2048 * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);

    for (int l = 0; l < N_LAYERS; l++) {
        // 1. Input RMSNorm
        rms_norm_kernel<<<2048, 128>>>(d_x, layers[l].d_input_norm, d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);

        // 2. QKV GEMM
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 5120, 2048, HIDDEN_DIM,
                     &alpha, layers[l].d_qkv_w, CUDA_R_16BF, HIDDEN_DIM,
                     d_h_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_qkv, CUDA_R_32F, 5120,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        // 3. RoPE
        int total_el = 2048 * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
        phi3_rope_kernel<<<(total_el + 255) / 256, 256>>>(d_qkv, d_q, d_k, d_v, 2048);

        // 4. Causal GQA online softmax attention
        dim3 grid_attn(2048, NUM_Q_HEADS);
        causal_gqa_online_attention_kernel<<<grid_attn, 128>>>(
            d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 2048);

        // 5. O projection
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 2048, HIDDEN_DIM,
                     &alpha, layers[l].d_o_w, CUDA_R_16BF, HIDDEN_DIM,
                     d_attn_out_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_attn_proj, CUDA_R_32F, HIDDEN_DIM,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        // 6. Residual add 1
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_attn_proj, 2048 * HIDDEN_DIM);

        // 7. Post-attention RMSNorm
        rms_norm_kernel<<<2048, 128>>>(d_x, layers[l].d_post_norm, d_h, d_h_bf16, 2048, HIDDEN_DIM, RMS_EPS);

        // 8. GateUp GEMM
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * INTERMEDIATE_DIM, 2048, HIDDEN_DIM,
                     &alpha, layers[l].d_gate_up_w, CUDA_R_16BF, HIDDEN_DIM,
                     d_h_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_gate_up, CUDA_R_32F, 2 * INTERMEDIATE_DIM,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        // 9. SwiGLU
        swiglu_kernel<<<(2048 * INTERMEDIATE_DIM + 255) / 256, 256>>>(d_gate_up, d_act, d_act_bf16, 2048);

        // 10. Down GEMM
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, 2048, INTERMEDIATE_DIM,
                     &alpha, layers[l].d_down_w, CUDA_R_16BF, INTERMEDIATE_DIM,
                     d_act_bf16, CUDA_R_16BF, INTERMEDIATE_DIM,
                     &beta, d_down, CUDA_R_32F, HIDDEN_DIM,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        // 11. Residual add 2
        residual_add_kernel<<<(2048 * HIDDEN_DIM + 255) / 256, 256>>>(d_x, d_down, 2048 * HIDDEN_DIM);
    }

    // Final RMSNorm
    rms_norm_kernel<<<2048, 128>>>(d_x, d_final_norm, d_h, d_x_final_bf16, 2048, HIDDEN_DIM, RMS_EPS);
    cudaDeviceSynchronize();

    // Dump first 10 tokens of final hidden state
    {
        __nv_bfloat16 h_dump[10 * HIDDEN_DIM];
        cudaMemcpy(h_dump, d_x_final_bf16, 10 * HIDDEN_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        FILE* fp = fopen("/home/workspace/p7-scratch/artifacts/cuda_final_h_first10.bin", "wb");
        if (fp) { fwrite(h_dump, sizeof(__nv_bfloat16), 10 * HIDDEN_DIM, fp); fclose(fp); }
    }

    // Chunked LM Head GEMM and loss calculation
    int chunk_size = 256;
    for (int offset = 0; offset < 2047; offset += chunk_size) {
        int cur_chunk = (offset + chunk_size <= 2047) ? chunk_size : (2047 - offset);
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, cur_chunk, HIDDEN_DIM,
                     &alpha, d_embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                     d_x_final_bf16 + offset * HIDDEN_DIM, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_logits_chunk, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        compute_ce_and_argmax_kernel<<<cur_chunk, 256>>>(
            d_logits_chunk, d_targets + offset, d_losses + offset, d_top1 + offset, cur_chunk, VOCAB_SIZE);
    }
    cudaDeviceSynchronize();

    clock_t t1 = clock();
    double total_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Collect metrics
    float h_losses[2047];
    int32_t h_top1[2047];
    cudaMemcpy(h_losses, d_losses, 2047 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top1, d_top1, 2047 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    double total_ce = 0.0;
    int top1_matches = 0;
    int printed = 0;
    for (int i = 0; i < 2047; i++) {
        total_ce += h_losses[i];
        if (h_top1[i] == (int32_t)oracle_top1[i]) {
            top1_matches++;
        } else if (printed < 10) {
            printf("Mismatch at pos %d: our_top1=%d oracle_top1=%d loss=%.4f\n",
                   i, h_top1[i], oracle_top1[i], h_losses[i]);
            printed++;
        }
    }

    double mean_ce = total_ce / 2047.0;
    double ppl = exp(mean_ce);
    double top1_acc = (double)top1_matches / 2047.0 * 100.0;

    double oracle_ppl = 5.486957;
    double ppl_delta_pct = fabs(ppl - oracle_ppl) / oracle_ppl * 100.0;

    printf("\n=== Arm (0) BF16 Evaluation Results ===\n");
    printf("Forward Pass Time:      %.3f s (%.1f tok/s)\n", total_sec, 2048.0 / total_sec);
    printf("Mean Cross Entropy:     %.6f\n", mean_ce);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", ppl_delta_pct);
    printf("Top-1 Matches:          %d / 2047\n", top1_matches);
    printf("Top-1 Agreement:        %.2f %%\n", top1_acc);

    if (top1_acc >= 99.0 && ppl_delta_pct <= 1.0) {
        printf("VERDICT: PASS (Gate 1 Anchor Verified)\n");
    } else {
        printf("VERDICT: FAIL (Gate 1 Anchor Mismatch)\n");
    }

    safetensors_close(&st1);
    safetensors_close(&st2);
    cublasDestroy(cublas);
    free(h_x);
    return 0;
}

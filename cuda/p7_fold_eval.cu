/* core/p7_fold_eval.cu — Standalone 4-Arm C/CUDA Evaluator for Project 7 Mode 3 (Fold)
 * Evaluates Phi-4-mini on RTX 3060 across:
 *   Arm (i):   Existing INT8 path (int8_block_dequant_gemm_f32)
 *   Arm (ii):  Hole-only (glint_htile_s3_f32, plug=0)
 *   Arm (iii): Hole + plug prefetched (glint_htile_s3_f32 with 2-slot VRAM ring streamer)
 *   Arm (iv):  Hole + plug all resident (glint_htile_s3_f32 with resident plugs)
 *
 * Pre-registered Pass Gate:
 *   1. Arm (iii) must be bit-identical to Arm (iv) (0 bit errors).
 *   2. Arm (iii) perplexity within 2.0% of Arm (i) PASSES as lossless fold.
 *   3. Driver version must be 595.99.02.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "cells.h"
#include "safetensors_c.h"
#include "p7_plug_streamer.h"

#define N_LAYERS 32
#define HIDDEN_DIM 3072
#define NUM_Q_HEADS 24
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define ROTARY_DIM 96
#define INTERMEDIATE_DIM 8192
#define VOCAB_SIZE 200064
#define BLOCKSIZE 64
#define RMS_EPS 1e-5f

/* NVML prototypes for dynamic loading */
typedef void* nvmlDevice_t;
typedef struct {
    uint64_t total;
    uint64_t free;
    uint64_t used;
} nvmlMemory_t;
typedef int (*nvmlInit_f)(void);
typedef int (*nvmlShutdown_f)(void);
typedef int (*nvmlDeviceGetHandleByIndex_f)(unsigned int, nvmlDevice_t*);
typedef int (*nvmlDeviceGetMemoryInfo_f)(nvmlDevice_t, nvmlMemory_t*);
typedef int (*nvmlSystemGetDriverVersion_f)(char*, unsigned int);

static nvmlInit_f p_nvmlInit = NULL;
static nvmlShutdown_f p_nvmlShutdown = NULL;
static nvmlDeviceGetHandleByIndex_f p_nvmlDeviceGetHandleByIndex = NULL;
static nvmlDeviceGetMemoryInfo_f p_nvmlDeviceGetMemoryInfo = NULL;
static nvmlSystemGetDriverVersion_f p_nvmlSystemGetDriverVersion = NULL;
static void* g_nvml_lib = NULL;
static nvmlDevice_t g_nvml_dev = NULL;

static int init_nvml() {
    g_nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!g_nvml_lib) g_nvml_lib = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!g_nvml_lib) return -1;
    p_nvmlInit = (nvmlInit_f)dlsym(g_nvml_lib, "nvmlInit");
    p_nvmlShutdown = (nvmlShutdown_f)dlsym(g_nvml_lib, "nvmlShutdown");
    p_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_f)dlsym(g_nvml_lib, "nvmlDeviceGetHandleByIndex");
    p_nvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_f)dlsym(g_nvml_lib, "nvmlDeviceGetMemoryInfo");
    p_nvmlSystemGetDriverVersion = (nvmlSystemGetDriverVersion_f)dlsym(g_nvml_lib, "nvmlSystemGetDriverVersion");
    if (!p_nvmlInit || !p_nvmlDeviceGetHandleByIndex || !p_nvmlDeviceGetMemoryInfo || !p_nvmlSystemGetDriverVersion) {
        return -2;
    }
    if (p_nvmlInit() != 0) return -3;
    if (p_nvmlDeviceGetHandleByIndex(0, &g_nvml_dev) != 0) return -4;
    return 0;
}

static void close_nvml() {
    if (p_nvmlShutdown) p_nvmlShutdown();
    if (g_nvml_lib) dlclose(g_nvml_lib);
}

static uint64_t get_host_peak_ram_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t vmhwm = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%lu", &vmhwm);
            break;
        }
    }
    fclose(f);
    return vmhwm;
}

static double get_vram_used_mib() {
    if (p_nvmlDeviceGetMemoryInfo && g_nvml_dev) {
        nvmlMemory_t mem;
        if (p_nvmlDeviceGetMemoryInfo(g_nvml_dev, &mem) == 0) {
            return (double)mem.used / (1024.0 * 1024.0);
        }
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return (double)(total_b - free_b) / (1024.0 * 1024.0);
}

/* Include glint_htile.cu implementation directly */
#include "glint_htile.cu"

/* INT8 block dequant GEMM kernel */
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

/* Embedding lookup kernel */
__global__ void embed_lookup_kernel(
    const int32_t* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embed_table,
    float* __restrict__ out_f32,
    __nv_bfloat16* __restrict__ out_bf16,
    int M, int D)
{
    int m = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || tid >= D) return;

    int32_t tok = tokens[m];
    if (tok < 0 || tok >= VOCAB_SIZE) tok = 0;

    __nv_bfloat16 val = embed_table[(size_t)tok * D + tid];
    out_f32[(size_t)m * D + tid] = __bfloat162float(val);
    if (out_bf16) out_bf16[(size_t)m * D + tid] = val;
}

/* Row-wise RMSNorm kernel */
__global__ void rms_norm_2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    __nv_bfloat16* __restrict__ y_bf16,
    int M, int D, float eps)
{
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    const float* x_row = x + (size_t)m * D;
    float* y_row = y + (size_t)m * D;
    __nv_bfloat16* y_bf16_row = y_bf16 ? (y_bf16 + (size_t)m * D) : NULL;

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
        float out_v = x_row[i] * scale * w;
        y_row[i] = out_v;
        if (y_bf16_row) y_bf16_row[i] = __float2bfloat16(out_v);
    }
}

/* Convert float to BF16 kernel */
__global__ void float_to_bf16_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = __float2bfloat16(in[i]);
}

/* Split QKV and apply NeoX partial RoPE deterministically */
__global__ void split_qkv_rope_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int M, int start_pos)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = M * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
    if (idx >= total_threads) return;

    int d = idx % HEAD_DIM;
    int head = (idx / HEAD_DIM) % (NUM_Q_HEADS + 2 * NUM_KV_HEADS);
    int m = idx / ((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM);

    int pos = start_pos + m;
    const float* qkv_row = qkv + (size_t)m * 5120;

    if (head < NUM_Q_HEADS) {
        int h = head;
        int in_base = h * HEAD_DIM;
        float val;
        if (d < (ROTARY_DIM / 2)) {
            float inv_freq = powf(10000.0f, -((float)d) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float d0 = qkv_row[in_base + d];
            float d1 = qkv_row[in_base + d + (ROTARY_DIM / 2)];
            val = d0 * cosf(angle) - d1 * sinf(angle);
        } else if (d < ROTARY_DIM) {
            int p = d - (ROTARY_DIM / 2);
            float inv_freq = powf(10000.0f, -((float)p) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float d0 = qkv_row[in_base + p];
            float d1 = qkv_row[in_base + d];
            val = d1 * cosf(angle) + d0 * sinf(angle);
        } else {
            val = qkv_row[in_base + d];
        }
        q[(size_t)m * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + d] = val;
    } else if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
        int h = head - NUM_Q_HEADS;
        int in_base = NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM;
        float val;
        if (d < (ROTARY_DIM / 2)) {
            float inv_freq = powf(10000.0f, -((float)d) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float d0 = qkv_row[in_base + d];
            float d1 = qkv_row[in_base + d + (ROTARY_DIM / 2)];
            val = d0 * cosf(angle) - d1 * sinf(angle);
        } else if (d < ROTARY_DIM) {
            int p = d - (ROTARY_DIM / 2);
            float inv_freq = powf(10000.0f, -((float)p) / ((float)(ROTARY_DIM / 2)));
            float angle = (float)pos * inv_freq;
            float d0 = qkv_row[in_base + p];
            float d1 = qkv_row[in_base + d];
            val = d1 * cosf(angle) + d0 * sinf(angle);
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

/* Causal GQA Attention kernel for M <= 256 */
__global__ void causal_gqa_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    __nv_bfloat16* __restrict__ out_bf16,
    int M)
{
    int hq = blockIdx.x;
    if (hq >= NUM_Q_HEADS) return;
    int hkv = hq / 3;
    int tq = threadIdx.x;
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
    __nv_bfloat16* out_bf16_vec = out_bf16 ? (out_bf16 + ((size_t)tq * NUM_Q_HEADS + hq) * HEAD_DIM) : NULL;

    for (int d = 0; d < HEAD_DIM; d++) {
        float acc = 0.f;
        for (int tk = 0; tk <= tq; tk++) {
            const float* v_vec = v + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
            acc += scores[tk] * inv_sum * v_vec[d];
        }
        out_vec[d] = acc;
        if (out_bf16_vec) out_bf16_vec[d] = __float2bfloat16(acc);
    }
}

/* SwiGLU activation: out = silu(gate) * up */
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

/* Residual add kernel: x = x + res */
__global__ void residual_add_kernel(float* __restrict__ x, const float* __restrict__ res, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) x[i] += res[i];
}

/* Cross entropy loss kernel from logits */
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
        losses[s] = logf(fmaxf(s_sum[0], 1e-20f)) + block_max - target_val;
    }
}

/* Weight structures */
typedef struct {
    int8_t *d_qkv_w;
    float *d_qkv_scale;
    int8_t *d_o_w;
    float *d_o_scale;
    int8_t *d_gate_up_w;
    float *d_gate_up_scale;
    int8_t *d_down_w;
    float *d_down_scale;
} LayerInt8Weights;

typedef struct {
    uint8_t *d_qkv_hole;
    float *d_qkv_absmax;
    uint8_t *d_o_hole;
    float *d_o_absmax;
    uint8_t *d_gate_up_hole;
    float *d_gate_up_absmax;
    uint8_t *d_down_hole;
    float *d_down_absmax;
    /* Device plug pointers (for Arm iv resident) */
    uint8_t *d_qkv_plug;
    uint8_t *d_o_plug;
    uint8_t *d_gate_up_plug;
    uint8_t *d_down_plug;
} LayerNestedWeights;

typedef struct {
    float *d_input_norm[N_LAYERS];
    float *d_post_norm[N_LAYERS];
    float *d_final_norm;
    __nv_bfloat16 *d_embed_tokens;
} SharedWeights;

typedef struct {
    double ppl;
    double prefill_tok_s[5];
    double decode_tok_s[5];
    double median_prefill;
    double min_prefill;
    double max_prefill;
    double median_decode;
    double min_decode;
    double max_decode;
    double peak_vram_mib;
    double peak_host_ram_mib;
    double total_stall_ms;
    double per_layer_stall_ms[N_LAYERS];
    uint64_t h2d_bytes;
} ArmMetrics;

static int comp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char** argv) {
    printf("====================================================================\n");
    printf("Project 7 Mode 3 (Fold) End-to-End Evaluator on RTX 3060 (Phi-4-mini)\n");
    printf("====================================================================\n");

    /* 1. Verify Driver Version */
    if (init_nvml() != 0) {
        fprintf(stderr, "WARNING: Could not init NVML directly; checking nvidia-smi...\n");
    }
    char driver_version[64] = "unknown";
    if (p_nvmlSystemGetDriverVersion) {
        p_nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));
    } else {
        FILE* p = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader", "r");
        if (p) {
            if (fgets(driver_version, sizeof(driver_version), p)) {
                strtok(driver_version, "\r\n ");
            }
            pclose(p);
        }
    }
    printf("Detected GPU Driver: %s\n", driver_version);
    if (strncmp(driver_version, "595.99.02", 9) != 0) {
        fprintf(stderr, "HARD_BLOCK: Driver version %s does not match expected 595.99.02\n", driver_version);
        return 2;
    }
    printf("DRIVER_CHECK: 595.99.02 PASS\n");

    /* 2. Load Evaluation Tokens */
    const char* token_path = "/home/workspace/p7-scratch/artifacts/eval_tokens.txt";
    FILE* tf = fopen(token_path, "r");
    if (!tf) {
        fprintf(stderr, "HARD_BLOCK: cannot open %s\n", token_path);
        return 3;
    }
    int tokens[1024];
    int n_tokens = 0;
    while (fscanf(tf, "%d", &tokens[n_tokens]) == 1 && n_tokens < 1024) {
        n_tokens++;
    }
    fclose(tf);
    printf("Loaded %d eval tokens from %s\n", n_tokens, token_path);
    if (n_tokens != 223) {
        fprintf(stderr, "HARD_BLOCK: expected 223 tokens, got %d\n", n_tokens);
        return 4;
    }

    int32_t *d_tokens, *d_targets;
    cudaMalloc(&d_tokens, n_tokens * sizeof(int32_t));
    cudaMalloc(&d_targets, (n_tokens - 1) * sizeof(int32_t));
    cudaMemcpy(d_tokens, tokens, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, tokens + 1, (n_tokens - 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    /* 3. Open Safetensors Pins */
    SafeTensors st_nested, st_int8;
    const char* nested_pin_path = "/home/workspace/p7-scratch/artifacts/pins/Phi-4-mini-nested-nf8/model.safetensors";
    const char* int8_pin_path = "/home/workspace/training_orchestrator/artifacts/pins/Phi-4-mini-int8-from-bf16/model.safetensors";

    if (safetensors_open(&st_nested, nested_pin_path) != 0) {
        fprintf(stderr, "HARD_BLOCK: failed to open nested-nf8 pin %s\n", nested_pin_path);
        return 5;
    }
    if (safetensors_open(&st_int8, int8_pin_path) != 0) {
        fprintf(stderr, "HARD_BLOCK: failed to open int8 pin %s\n", int8_pin_path);
        return 6;
    }
    printf("Pins loaded: Nested-NF8 (%d tensors), INT8 (%d tensors)\n", st_nested.n_entries, st_int8.n_entries);

    /* 4. Load Shared Weights */
    SharedWeights shared;
    const TensorEntry* e_embed = safetensors_find(&st_nested, "model.embed_tokens.weight");
    if (!e_embed) { fprintf(stderr, "HARD_BLOCK: missing embed_tokens\n"); return 7; }
    cudaMalloc(&shared.d_embed_tokens, e_embed->end - e_embed->begin);
    cudaMemcpy(shared.d_embed_tokens, e_embed->data, e_embed->end - e_embed->begin, cudaMemcpyHostToDevice);

    const TensorEntry* e_norm = safetensors_find(&st_nested, "model.norm.weight");
    if (!e_norm) { fprintf(stderr, "HARD_BLOCK: missing model.norm.weight\n"); return 8; }
    cudaMalloc(&shared.d_final_norm, HIDDEN_DIM * sizeof(float));
    float h_norm_f32[HIDDEN_DIM];
    const __nv_bfloat16* bf_norm = (const __nv_bfloat16*)e_norm->data;
    for (int i = 0; i < HIDDEN_DIM; i++) h_norm_f32[i] = __bfloat162float(bf_norm[i]);
    cudaMemcpy(shared.d_final_norm, h_norm_f32, sizeof(h_norm_f32), cudaMemcpyHostToDevice);

    for (int l = 0; l < N_LAYERS; l++) {
        char k_in[128], k_post[128];
        snprintf(k_in, sizeof(k_in), "model.layers.%d.input_layernorm.weight", l);
        snprintf(k_post, sizeof(k_post), "model.layers.%d.post_attention_layernorm.weight", l);
        const TensorEntry* e_in = safetensors_find(&st_nested, k_in);
        const TensorEntry* e_post = safetensors_find(&st_nested, k_post);
        if (!e_in || !e_post) { fprintf(stderr, "HARD_BLOCK: missing layernorms layer %d\n", l); return 9; }

        cudaMalloc(&shared.d_input_norm[l], HIDDEN_DIM * sizeof(float));
        cudaMalloc(&shared.d_post_norm[l], HIDDEN_DIM * sizeof(float));
        const __nv_bfloat16* bf_in = (const __nv_bfloat16*)e_in->data;
        const __nv_bfloat16* bf_post = (const __nv_bfloat16*)e_post->data;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h_norm_f32[i] = __bfloat162float(bf_in[i]);
        }
        cudaMemcpy(shared.d_input_norm[l], h_norm_f32, sizeof(h_norm_f32), cudaMemcpyHostToDevice);
        for (int i = 0; i < HIDDEN_DIM; i++) {
            h_norm_f32[i] = __bfloat162float(bf_post[i]);
        }
        cudaMemcpy(shared.d_post_norm[l], h_norm_f32, sizeof(h_norm_f32), cudaMemcpyHostToDevice);
    }
    printf("Shared weights uploaded to VRAM.\n");

    /* Allocate Activations & Scratch Buffers */
    float *d_x, *d_h, *d_qkv, *d_q, *d_k, *d_v, *d_attn_out, *d_attn_proj;
    float *d_gate_up, *d_act, *d_down, *d_losses, *d_logits;
    __nv_bfloat16 *d_x_bf16, *d_h_bf16, *d_attn_out_bf16, *d_act_bf16, *d_x_final_bf16;

    int max_M = 256;
    cudaMalloc(&d_x, max_M * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_h, max_M * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_x_bf16, max_M * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_h_bf16, max_M * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_x_final_bf16, max_M * HIDDEN_DIM * sizeof(__nv_bfloat16));

    cudaMalloc(&d_qkv, max_M * 5120 * sizeof(float));
    cudaMalloc(&d_q, max_M * NUM_Q_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_k, max_M * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    cudaMalloc(&d_v, max_M * NUM_KV_HEADS * HEAD_DIM * sizeof(float));

    cudaMalloc(&d_attn_out, max_M * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_attn_out_bf16, max_M * HIDDEN_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_attn_proj, max_M * HIDDEN_DIM * sizeof(float));

    cudaMalloc(&d_gate_up, max_M * 16384 * sizeof(float));
    cudaMalloc(&d_act, max_M * INTERMEDIATE_DIM * sizeof(float));
    cudaMalloc(&d_act_bf16, max_M * INTERMEDIATE_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down, max_M * HIDDEN_DIM * sizeof(float));

    cudaMalloc(&d_losses, max_M * sizeof(float));
    cudaMalloc(&d_logits, max_M * VOCAB_SIZE * sizeof(float));

    /* cuBLAS initialization */
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    /* Streams for prefetch */
    cudaStream_t compute_stream, copy_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);
    cublasSetStream(cublas, compute_stream);

    /* Prepare Arms */
    ArmMetrics metrics[5];
    memset(metrics, 0, sizeof(metrics));
    float *arm3_final_logits = (float*)malloc((n_tokens - 1) * VOCAB_SIZE * sizeof(float));
    float *arm4_final_logits = (float*)malloc((n_tokens - 1) * VOCAB_SIZE * sizeof(float));

    /* -------------------------------------------------------------
     * ARM (i): Existing INT8 Path
     * ------------------------------------------------------------- */
    printf("\n=== Running Arm (i): Existing INT8 Path ===\n");
    LayerInt8Weights arm1_w[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        char k_w[128], k_s[128];
        snprintf(k_w, sizeof(k_w), "model.layers.%d.self_attn.qkv_proj.weight", l);
        snprintf(k_s, sizeof(k_s), "model.layers.%d.self_attn.qkv_proj.weight.int8_scale", l);
        const TensorEntry *ew = safetensors_find(&st_int8, k_w);
        const TensorEntry *es = safetensors_find(&st_int8, k_s);
        cudaMalloc(&arm1_w[l].d_qkv_w, ew->end - ew->begin);
        cudaMalloc(&arm1_w[l].d_qkv_scale, es->end - es->begin);
        cudaMemcpy(arm1_w[l].d_qkv_w, ew->data, ew->end - ew->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(arm1_w[l].d_qkv_scale, es->data, es->end - es->begin, cudaMemcpyHostToDevice);

        snprintf(k_w, sizeof(k_w), "model.layers.%d.self_attn.o_proj.weight", l);
        snprintf(k_s, sizeof(k_s), "model.layers.%d.self_attn.o_proj.weight.int8_scale", l);
        ew = safetensors_find(&st_int8, k_w); es = safetensors_find(&st_int8, k_s);
        cudaMalloc(&arm1_w[l].d_o_w, ew->end - ew->begin);
        cudaMalloc(&arm1_w[l].d_o_scale, es->end - es->begin);
        cudaMemcpy(arm1_w[l].d_o_w, ew->data, ew->end - ew->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(arm1_w[l].d_o_scale, es->data, es->end - es->begin, cudaMemcpyHostToDevice);

        snprintf(k_w, sizeof(k_w), "model.layers.%d.mlp.gate_up_proj.weight", l);
        snprintf(k_s, sizeof(k_s), "model.layers.%d.mlp.gate_up_proj.weight.int8_scale", l);
        ew = safetensors_find(&st_int8, k_w); es = safetensors_find(&st_int8, k_s);
        cudaMalloc(&arm1_w[l].d_gate_up_w, ew->end - ew->begin);
        cudaMalloc(&arm1_w[l].d_gate_up_scale, es->end - es->begin);
        cudaMemcpy(arm1_w[l].d_gate_up_w, ew->data, ew->end - ew->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(arm1_w[l].d_gate_up_scale, es->data, es->end - es->begin, cudaMemcpyHostToDevice);

        snprintf(k_w, sizeof(k_w), "model.layers.%d.mlp.down_proj.weight", l);
        snprintf(k_s, sizeof(k_s), "model.layers.%d.mlp.down_proj.weight.int8_scale", l);
        ew = safetensors_find(&st_int8, k_w); es = safetensors_find(&st_int8, k_s);
        cudaMalloc(&arm1_w[l].d_down_w, ew->end - ew->begin);
        cudaMalloc(&arm1_w[l].d_down_scale, es->end - es->begin);
        cudaMemcpy(arm1_w[l].d_down_w, ew->data, ew->end - ew->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(arm1_w[l].d_down_scale, es->data, es->end - es->begin, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    metrics[1].peak_vram_mib = get_vram_used_mib();
    metrics[1].peak_host_ram_mib = (double)get_host_peak_ram_kb() / 1024.0;

    for (int rep = 0; rep < 5; rep++) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, compute_stream);

        /* Forward Pass INT8 (Prefill M=223) */
        embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, n_tokens), 256, 0, compute_stream>>>(
            d_tokens, shared.d_embed_tokens, d_x, NULL, n_tokens, HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_input_norm[l], d_h, NULL, n_tokens, HIDDEN_DIM, RMS_EPS);

            /* QKV GEMM */
            dim3 blk(16, 16);
            dim3 grid_qkv((5120 + 15) / 16, (n_tokens + 15) / 16);
            int8_block_dequant_gemm_f32_kernel<<<grid_qkv, blk, 0, compute_stream>>>(
                arm1_w[l].d_qkv_w, arm1_w[l].d_qkv_scale, d_h, d_qkv, n_tokens, 5120, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);

            int total_split = n_tokens * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
            split_qkv_rope_kernel<<<(total_split + 255) / 256, 256, 0, compute_stream>>>(
                d_qkv, d_q, d_k, d_v, n_tokens, 0);

            causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(
                d_q, d_k, d_v, d_attn_out, NULL, n_tokens);

            /* O proj GEMM */
            dim3 grid_o((HIDDEN_DIM + 15) / 16, (n_tokens + 15) / 16);
            int8_block_dequant_gemm_f32_kernel<<<grid_o, blk, 0, compute_stream>>>(
                arm1_w[l].d_o_w, arm1_w[l].d_o_scale, d_attn_out, d_attn_proj, n_tokens, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_attn_proj, n_tokens * HIDDEN_DIM);

            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_post_norm[l], d_h, NULL, n_tokens, HIDDEN_DIM, RMS_EPS);

            /* Gate Up GEMM */
            dim3 grid_gu((16384 + 15) / 16, (n_tokens + 15) / 16);
            int8_block_dequant_gemm_f32_kernel<<<grid_gu, blk, 0, compute_stream>>>(
                arm1_w[l].d_gate_up_w, arm1_w[l].d_gate_up_scale, d_h, d_gate_up, n_tokens, 16384, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);

            swiglu_kernel<<<(n_tokens * INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_gate_up, d_act, NULL, n_tokens);

            /* Down GEMM */
            dim3 grid_down((HIDDEN_DIM + 15) / 16, (n_tokens + 15) / 16);
            int8_block_dequant_gemm_f32_kernel<<<grid_down, blk, 0, compute_stream>>>(
                arm1_w[l].d_down_w, arm1_w[l].d_down_scale, d_act, d_down, n_tokens, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE, 1.0f, 0.0f);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_down, n_tokens * HIDDEN_DIM);
        }

        /* Final Norm */
        rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
            d_x, shared.d_final_norm, d_h, d_x_final_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

        /* LM Head GEMM: logits = X @ W_embed^T */
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, n_tokens - 1, HIDDEN_DIM,
                     &alpha, shared.d_embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                     d_x_final_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        cross_entropy_kernel<<<n_tokens - 1, 256, 0, compute_stream>>>(
            d_logits, d_targets, d_losses, n_tokens - 1, VOCAB_SIZE);

        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);

        float prefill_ms = 0;
        cudaEventElapsedTime(&prefill_ms, ev0, ev1);
        metrics[1].prefill_tok_s[rep] = (double)n_tokens / (prefill_ms / 1000.0);

        float h_losses[256];
        cudaMemcpy(h_losses, d_losses, (n_tokens - 1) * sizeof(float), cudaMemcpyDeviceToHost);
        double total_loss = 0;
        for (int i = 0; i < n_tokens - 1; i++) total_loss += h_losses[i];
        double mean_ce = total_loss / (n_tokens - 1);
        metrics[1].ppl = exp(mean_ce);

        /* Decode 16 steps */
        cudaEventRecord(ev0, compute_stream);
        for (int step = 0; step < 16; step++) {
            embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, 1), 256, 0, compute_stream>>>(
                d_tokens + n_tokens - 1, shared.d_embed_tokens, d_x, NULL, 1, HIDDEN_DIM);
            for (int l = 0; l < N_LAYERS; l++) {
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_input_norm[l], d_h, NULL, 1, HIDDEN_DIM, RMS_EPS);
                dim3 blk(16, 16);
                dim3 grid_qkv((5120 + 15) / 16, 1);
                int8_block_dequant_gemm_f32_kernel<<<grid_qkv, blk, 0, compute_stream>>>(
                    arm1_w[l].d_qkv_w, arm1_w[l].d_qkv_scale, d_h, d_qkv, 1, 5120, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);
                split_qkv_rope_kernel<<<(40 * 128 + 255) / 256, 256, 0, compute_stream>>>(d_qkv, d_q, d_k, d_v, 1, n_tokens + step);
                causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(d_q, d_k, d_v, d_attn_out, NULL, 1);
                dim3 grid_o((HIDDEN_DIM + 15) / 16, 1);
                int8_block_dequant_gemm_f32_kernel<<<grid_o, blk, 0, compute_stream>>>(
                    arm1_w[l].d_o_w, arm1_w[l].d_o_scale, d_attn_out, d_attn_proj, 1, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_attn_proj, HIDDEN_DIM);
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_post_norm[l], d_h, NULL, 1, HIDDEN_DIM, RMS_EPS);
                dim3 grid_gu((16384 + 15) / 16, 1);
                int8_block_dequant_gemm_f32_kernel<<<grid_gu, blk, 0, compute_stream>>>(
                    arm1_w[l].d_gate_up_w, arm1_w[l].d_gate_up_scale, d_h, d_gate_up, 1, 16384, HIDDEN_DIM, BLOCKSIZE, 1.0f, 0.0f);
                swiglu_kernel<<<(INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(d_gate_up, d_act, NULL, 1);
                dim3 grid_down((HIDDEN_DIM + 15) / 16, 1);
                int8_block_dequant_gemm_f32_kernel<<<grid_down, blk, 0, compute_stream>>>(
                    arm1_w[l].d_down_w, arm1_w[l].d_down_scale, d_act, d_down, 1, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE, 1.0f, 0.0f);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_down, HIDDEN_DIM);
            }
            rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_final_norm, d_h, d_x_final_bf16, 1, HIDDEN_DIM, RMS_EPS);
        }
        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);
        float decode_ms = 0;
        cudaEventElapsedTime(&decode_ms, ev0, ev1);
        metrics[1].decode_tok_s[rep] = 16.0 / (decode_ms / 1000.0);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        printf("  Arm (i) Rep %d: Prefill %6.1f tok/s, Decode %6.1f tok/s, PPL: %.4f\n",
               rep + 1, metrics[1].prefill_tok_s[rep], metrics[1].decode_tok_s[rep], metrics[1].ppl);
    }

    /* Free Arm 1 INT8 weights to test VRAM of subsequent arms honestly */
    for (int l = 0; l < N_LAYERS; l++) {
        cudaFree(arm1_w[l].d_qkv_w); cudaFree(arm1_w[l].d_qkv_scale);
        cudaFree(arm1_w[l].d_o_w); cudaFree(arm1_w[l].d_o_scale);
        cudaFree(arm1_w[l].d_gate_up_w); cudaFree(arm1_w[l].d_gate_up_scale);
        cudaFree(arm1_w[l].d_down_w); cudaFree(arm1_w[l].d_down_scale);
    }

    /* -------------------------------------------------------------
     * Load Nested-NF8 / Glint Pin Weights
     * ------------------------------------------------------------- */
    LayerNestedWeights nested_w[N_LAYERS];
    uint64_t layer_plug_bytes[N_LAYERS];
    uint8_t *host_pinned_plugs[N_LAYERS];
    uint64_t total_plug_bytes_per_layer = (7864320 + 4718592 + 25165824 + 12582912);

    for (int l = 0; l < N_LAYERS; l++) {
        layer_plug_bytes[l] = total_plug_bytes_per_layer;
        cudaHostAlloc((void**)&host_pinned_plugs[l], total_plug_bytes_per_layer, cudaHostAllocDefault);

        char k_h[128], k_am[128], k_p[128];
        /* QKV */
        snprintf(k_h, sizeof(k_h), "model.layers.%d.self_attn.qkv_proj.weight", l);
        snprintf(k_am, sizeof(k_am), "model.layers.%d.self_attn.qkv_proj.weight.absmax", l);
        snprintf(k_p, sizeof(k_p), "model.layers.%d.self_attn.qkv_proj.weight.glint_plug", l);
        const TensorEntry *eh = safetensors_find(&st_nested, k_h);
        const TensorEntry *eam = safetensors_find(&st_nested, k_am);
        const TensorEntry *ep = safetensors_find(&st_nested, k_p);
        cudaMalloc(&nested_w[l].d_qkv_hole, eh->end - eh->begin);
        cudaMalloc(&nested_w[l].d_qkv_absmax, eam->end - eam->begin);
        cudaMemcpy(nested_w[l].d_qkv_hole, eh->data, eh->end - eh->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(nested_w[l].d_qkv_absmax, eam->data, eam->end - eam->begin, cudaMemcpyHostToDevice);
        memcpy(host_pinned_plugs[l] + 0, ep->data, ep->end - ep->begin);

        /* O proj */
        snprintf(k_h, sizeof(k_h), "model.layers.%d.self_attn.o_proj.weight", l);
        snprintf(k_am, sizeof(k_am), "model.layers.%d.self_attn.o_proj.weight.absmax", l);
        snprintf(k_p, sizeof(k_p), "model.layers.%d.self_attn.o_proj.weight.glint_plug", l);
        eh = safetensors_find(&st_nested, k_h); eam = safetensors_find(&st_nested, k_am); ep = safetensors_find(&st_nested, k_p);
        cudaMalloc(&nested_w[l].d_o_hole, eh->end - eh->begin);
        cudaMalloc(&nested_w[l].d_o_absmax, eam->end - eam->begin);
        cudaMemcpy(nested_w[l].d_o_hole, eh->data, eh->end - eh->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(nested_w[l].d_o_absmax, eam->data, eam->end - eam->begin, cudaMemcpyHostToDevice);
        memcpy(host_pinned_plugs[l] + 7864320, ep->data, ep->end - ep->begin);

        /* Gate Up */
        snprintf(k_h, sizeof(k_h), "model.layers.%d.mlp.gate_up_proj.weight", l);
        snprintf(k_am, sizeof(k_am), "model.layers.%d.mlp.gate_up_proj.weight.absmax", l);
        snprintf(k_p, sizeof(k_p), "model.layers.%d.mlp.gate_up_proj.weight.glint_plug", l);
        eh = safetensors_find(&st_nested, k_h); eam = safetensors_find(&st_nested, k_am); ep = safetensors_find(&st_nested, k_p);
        cudaMalloc(&nested_w[l].d_gate_up_hole, eh->end - eh->begin);
        cudaMalloc(&nested_w[l].d_gate_up_absmax, eam->end - eam->begin);
        cudaMemcpy(nested_w[l].d_gate_up_hole, eh->data, eh->end - eh->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(nested_w[l].d_gate_up_absmax, eam->data, eam->end - eam->begin, cudaMemcpyHostToDevice);
        memcpy(host_pinned_plugs[l] + 7864320 + 4718592, ep->data, ep->end - ep->begin);

        /* Down */
        snprintf(k_h, sizeof(k_h), "model.layers.%d.mlp.down_proj.weight", l);
        snprintf(k_am, sizeof(k_am), "model.layers.%d.mlp.down_proj.weight.absmax", l);
        snprintf(k_p, sizeof(k_p), "model.layers.%d.mlp.down_proj.weight.glint_plug", l);
        eh = safetensors_find(&st_nested, k_h); eam = safetensors_find(&st_nested, k_am); ep = safetensors_find(&st_nested, k_p);
        cudaMalloc(&nested_w[l].d_down_hole, eh->end - eh->begin);
        cudaMalloc(&nested_w[l].d_down_absmax, eam->end - eam->begin);
        cudaMemcpy(nested_w[l].d_down_hole, eh->data, eh->end - eh->begin, cudaMemcpyHostToDevice);
        cudaMemcpy(nested_w[l].d_down_absmax, eam->data, eam->end - eam->begin, cudaMemcpyHostToDevice);
        memcpy(host_pinned_plugs[l] + 7864320 + 4718592 + 25165824, ep->data, ep->end - ep->begin);
    }
    printf("Hole weights and host pinned plug data prepared for 32 layers.\n");

    /* -------------------------------------------------------------
     * ARM (ii): Hole-Only (Plug = 0)
     * ------------------------------------------------------------- */
    printf("\n=== Running Arm (ii): Hole-Only (Plug = 0) ===\n");
    uint8_t *d_zero_plug = NULL;
    cudaMalloc(&d_zero_plug, total_plug_bytes_per_layer);
    cudaMemset(d_zero_plug, 0, total_plug_bytes_per_layer);
    cudaDeviceSynchronize();
    metrics[2].peak_vram_mib = get_vram_used_mib();
    metrics[2].peak_host_ram_mib = (double)get_host_peak_ram_kb() / 1024.0;

    for (int rep = 0; rep < 5; rep++) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, compute_stream);

        embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, n_tokens), 256, 0, compute_stream>>>(
            d_tokens, shared.d_embed_tokens, d_x, d_x_bf16, n_tokens, HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_input_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, d_zero_plug + 0, nested_w[l].d_qkv_absmax,
                                      d_h_bf16, d_qkv, n_tokens, 5120, HIDDEN_DIM, BLOCKSIZE);

            int total_split = n_tokens * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
            split_qkv_rope_kernel<<<(total_split + 255) / 256, 256, 0, compute_stream>>>(
                d_qkv, d_q, d_k, d_v, n_tokens, 0);

            causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(
                d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_o_hole, d_zero_plug + 7864320, nested_w[l].d_o_absmax,
                                      d_attn_out_bf16, d_attn_proj, n_tokens, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_attn_proj, n_tokens * HIDDEN_DIM);

            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_post_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, d_zero_plug + 7864320 + 4718592, nested_w[l].d_gate_up_absmax,
                                      d_h_bf16, d_gate_up, n_tokens, 16384, HIDDEN_DIM, BLOCKSIZE);

            swiglu_kernel<<<(n_tokens * INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_gate_up, d_act, d_act_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_down_hole, d_zero_plug + 7864320 + 4718592 + 25165824, nested_w[l].d_down_absmax,
                                      d_act_bf16, d_down, n_tokens, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_down, n_tokens * HIDDEN_DIM);
        }

        rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
            d_x, shared.d_final_norm, d_h, d_x_final_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, n_tokens - 1, HIDDEN_DIM,
                     &alpha, shared.d_embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                     d_x_final_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        cross_entropy_kernel<<<n_tokens - 1, 256, 0, compute_stream>>>(
            d_logits, d_targets, d_losses, n_tokens - 1, VOCAB_SIZE);

        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);

        float prefill_ms = 0;
        cudaEventElapsedTime(&prefill_ms, ev0, ev1);
        metrics[2].prefill_tok_s[rep] = (double)n_tokens / (prefill_ms / 1000.0);

        float h_losses[256];
        cudaMemcpy(h_losses, d_losses, (n_tokens - 1) * sizeof(float), cudaMemcpyDeviceToHost);
        double total_loss = 0;
        for (int i = 0; i < n_tokens - 1; i++) total_loss += h_losses[i];
        double mean_ce = total_loss / (n_tokens - 1);
        metrics[2].ppl = exp(mean_ce);

        /* Decode 16 steps */
        cudaEventRecord(ev0, compute_stream);
        for (int step = 0; step < 16; step++) {
            embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, 1), 256, 0, compute_stream>>>(
                d_tokens + n_tokens - 1, shared.d_embed_tokens, d_x, d_x_bf16, 1, HIDDEN_DIM);
            for (int l = 0; l < N_LAYERS; l++) {
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_input_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, d_zero_plug + 0, nested_w[l].d_qkv_absmax,
                                          d_h_bf16, d_qkv, 1, 5120, HIDDEN_DIM, BLOCKSIZE);
                split_qkv_rope_kernel<<<(40 * 128 + 255) / 256, 256, 0, compute_stream>>>(d_qkv, d_q, d_k, d_v, 1, n_tokens + step);
                causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_o_hole, d_zero_plug + 7864320, nested_w[l].d_o_absmax,
                                          d_attn_out_bf16, d_attn_proj, 1, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_attn_proj, HIDDEN_DIM);
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_post_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, d_zero_plug + 7864320 + 4718592, nested_w[l].d_gate_up_absmax,
                                          d_h_bf16, d_gate_up, 1, 16384, HIDDEN_DIM, BLOCKSIZE);
                swiglu_kernel<<<(INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(d_gate_up, d_act, d_act_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_down_hole, d_zero_plug + 7864320 + 4718592 + 25165824, nested_w[l].d_down_absmax,
                                          d_act_bf16, d_down, 1, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_down, HIDDEN_DIM);
            }
            rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_final_norm, d_h, d_x_final_bf16, 1, HIDDEN_DIM, RMS_EPS);
        }
        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);
        float decode_ms = 0;
        cudaEventElapsedTime(&decode_ms, ev0, ev1);
        metrics[2].decode_tok_s[rep] = 16.0 / (decode_ms / 1000.0);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        printf("  Arm (ii) Rep %d: Prefill %6.1f tok/s, Decode %6.1f tok/s, PPL: %.4f\n",
               rep + 1, metrics[2].prefill_tok_s[rep], metrics[2].decode_tok_s[rep], metrics[2].ppl);
    }
    cudaFree(d_zero_plug);

    /* -------------------------------------------------------------
     * ARM (iii): Hole + Plug Prefetched (2-Slot VRAM Ring Streamer)
     * ------------------------------------------------------------- */
    printf("\n=== Running Arm (iii): Hole + Plug Prefetched (2-Slot Ring) ===\n");
    P7PlugStreamer *streamer = NULL;
    if (p7_plug_streamer_create(&streamer, N_LAYERS, layer_plug_bytes, copy_stream, compute_stream) != 0) {
        fprintf(stderr, "HARD_BLOCK: failed to create plug streamer\n");
        return 10;
    }
    for (int l = 0; l < N_LAYERS; l++) {
        p7_plug_streamer_set_host_data(streamer, l, host_pinned_plugs[l], total_plug_bytes_per_layer);
    }
    cudaDeviceSynchronize();
    metrics[3].peak_vram_mib = get_vram_used_mib();
    metrics[3].peak_host_ram_mib = (double)get_host_peak_ram_kb() / 1024.0;

    for (int rep = 0; rep < 5; rep++) {
        p7_plug_streamer_reset_stats(streamer);
        p7_plug_streamer_start(streamer);

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, compute_stream);

        embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, n_tokens), 256, 0, compute_stream>>>(
            d_tokens, shared.d_embed_tokens, d_x, d_x_bf16, n_tokens, HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            const uint8_t *d_cur_plug = NULL;
            p7_plug_streamer_acquire(streamer, l, &d_cur_plug);

            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_input_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, d_cur_plug + 0, nested_w[l].d_qkv_absmax,
                                      d_h_bf16, d_qkv, n_tokens, 5120, HIDDEN_DIM, BLOCKSIZE);

            int total_split = n_tokens * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
            split_qkv_rope_kernel<<<(total_split + 255) / 256, 256, 0, compute_stream>>>(
                d_qkv, d_q, d_k, d_v, n_tokens, 0);

            causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(
                d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_o_hole, d_cur_plug + 7864320, nested_w[l].d_o_absmax,
                                      d_attn_out_bf16, d_attn_proj, n_tokens, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_attn_proj, n_tokens * HIDDEN_DIM);

            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_post_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, d_cur_plug + 7864320 + 4718592, nested_w[l].d_gate_up_absmax,
                                      d_h_bf16, d_gate_up, n_tokens, 16384, HIDDEN_DIM, BLOCKSIZE);

            swiglu_kernel<<<(n_tokens * INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_gate_up, d_act, d_act_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_down_hole, d_cur_plug + 7864320 + 4718592 + 25165824, nested_w[l].d_down_absmax,
                                      d_act_bf16, d_down, n_tokens, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_down, n_tokens * HIDDEN_DIM);

            p7_plug_streamer_release(streamer, l);
        }

        rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
            d_x, shared.d_final_norm, d_h, d_x_final_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, n_tokens - 1, HIDDEN_DIM,
                     &alpha, shared.d_embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                     d_x_final_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        cross_entropy_kernel<<<n_tokens - 1, 256, 0, compute_stream>>>(
            d_logits, d_targets, d_losses, n_tokens - 1, VOCAB_SIZE);

        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);

        float prefill_ms = 0;
        cudaEventElapsedTime(&prefill_ms, ev0, ev1);
        metrics[3].prefill_tok_s[rep] = (double)n_tokens / (prefill_ms / 1000.0);

        float h_losses[256];
        cudaMemcpy(h_losses, d_losses, (n_tokens - 1) * sizeof(float), cudaMemcpyDeviceToHost);
        double total_loss = 0;
        for (int i = 0; i < n_tokens - 1; i++) total_loss += h_losses[i];
        double mean_ce = total_loss / (n_tokens - 1);
        metrics[3].ppl = exp(mean_ce);

        if (rep == 0) {
            cudaStreamSynchronize(compute_stream);
            cudaMemcpy(arm3_final_logits, d_logits, (n_tokens - 1) * VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            P7StreamerStats s_stats;
            p7_plug_streamer_get_stats(streamer, &s_stats);
            metrics[3].total_stall_ms = s_stats.total_stall_ms;
            metrics[3].h2d_bytes = s_stats.total_h2d_bytes;
            for (int l = 0; l < N_LAYERS; l++) metrics[3].per_layer_stall_ms[l] = s_stats.per_layer_stall_ms[l];
        }

        /* Decode 16 steps with streaming */
        cudaEventRecord(ev0, compute_stream);
        for (int step = 0; step < 16; step++) {
            p7_plug_streamer_reset_stats(streamer);
            p7_plug_streamer_start(streamer);
            embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, 1), 256, 0, compute_stream>>>(
                d_tokens + n_tokens - 1, shared.d_embed_tokens, d_x, d_x_bf16, 1, HIDDEN_DIM);
            for (int l = 0; l < N_LAYERS; l++) {
                const uint8_t *d_cur_plug = NULL;
                p7_plug_streamer_acquire(streamer, l, &d_cur_plug);

                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_input_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, d_cur_plug + 0, nested_w[l].d_qkv_absmax,
                                          d_h_bf16, d_qkv, 1, 5120, HIDDEN_DIM, BLOCKSIZE);
                split_qkv_rope_kernel<<<(40 * 128 + 255) / 256, 256, 0, compute_stream>>>(d_qkv, d_q, d_k, d_v, 1, n_tokens + step);
                causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_o_hole, d_cur_plug + 7864320, nested_w[l].d_o_absmax,
                                          d_attn_out_bf16, d_attn_proj, 1, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_attn_proj, HIDDEN_DIM);
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_post_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, d_cur_plug + 7864320 + 4718592, nested_w[l].d_gate_up_absmax,
                                          d_h_bf16, d_gate_up, 1, 16384, HIDDEN_DIM, BLOCKSIZE);
                swiglu_kernel<<<(INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(d_gate_up, d_act, d_act_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_down_hole, d_cur_plug + 7864320 + 4718592 + 25165824, nested_w[l].d_down_absmax,
                                          d_act_bf16, d_down, 1, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_down, HIDDEN_DIM);

                p7_plug_streamer_release(streamer, l);
            }
            rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_final_norm, d_h, d_x_final_bf16, 1, HIDDEN_DIM, RMS_EPS);
        }
        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);
        float decode_ms = 0;
        cudaEventElapsedTime(&decode_ms, ev0, ev1);
        metrics[3].decode_tok_s[rep] = 16.0 / (decode_ms / 1000.0);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        printf("  Arm (iii) Rep %d: Prefill %6.1f tok/s, Decode %6.1f tok/s, PPL: %.4f\n",
               rep + 1, metrics[3].prefill_tok_s[rep], metrics[3].decode_tok_s[rep], metrics[3].ppl);
    }
    p7_plug_streamer_destroy(streamer);

    /* -------------------------------------------------------------
     * ARM (iv): Hole + Plug All Resident (No Prefetch)
     * ------------------------------------------------------------- */
    printf("\n=== Running Arm (iv): Hole + Plug All Resident ===\n");
    for (int l = 0; l < N_LAYERS; l++) {
        uint8_t *d_all_plug = NULL;
        cudaMalloc(&d_all_plug, total_plug_bytes_per_layer);
        cudaMemcpy(d_all_plug, host_pinned_plugs[l], total_plug_bytes_per_layer, cudaMemcpyHostToDevice);
        nested_w[l].d_qkv_plug = d_all_plug + 0;
        nested_w[l].d_o_plug = d_all_plug + 7864320;
        nested_w[l].d_gate_up_plug = d_all_plug + 7864320 + 4718592;
        nested_w[l].d_down_plug = d_all_plug + 7864320 + 4718592 + 25165824;
    }
    cudaDeviceSynchronize();
    metrics[4].peak_vram_mib = get_vram_used_mib();
    metrics[4].peak_host_ram_mib = (double)get_host_peak_ram_kb() / 1024.0;

    for (int rep = 0; rep < 5; rep++) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0, compute_stream);

        embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, n_tokens), 256, 0, compute_stream>>>(
            d_tokens, shared.d_embed_tokens, d_x, d_x_bf16, n_tokens, HIDDEN_DIM);

        for (int l = 0; l < N_LAYERS; l++) {
            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_input_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, nested_w[l].d_qkv_plug, nested_w[l].d_qkv_absmax,
                                      d_h_bf16, d_qkv, n_tokens, 5120, HIDDEN_DIM, BLOCKSIZE);

            int total_split = n_tokens * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
            split_qkv_rope_kernel<<<(total_split + 255) / 256, 256, 0, compute_stream>>>(
                d_qkv, d_q, d_k, d_v, n_tokens, 0);

            causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(
                d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_o_hole, nested_w[l].d_o_plug, nested_w[l].d_o_absmax,
                                      d_attn_out_bf16, d_attn_proj, n_tokens, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_attn_proj, n_tokens * HIDDEN_DIM);

            rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
                d_x, shared.d_post_norm[l], d_h, d_h_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

            launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, nested_w[l].d_gate_up_plug, nested_w[l].d_gate_up_absmax,
                                      d_h_bf16, d_gate_up, n_tokens, 16384, HIDDEN_DIM, BLOCKSIZE);

            swiglu_kernel<<<(n_tokens * INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_gate_up, d_act, d_act_bf16, n_tokens);

            launch_glint_htile_s3_f32(nested_w[l].d_down_hole, nested_w[l].d_down_plug, nested_w[l].d_down_absmax,
                                      d_act_bf16, d_down, n_tokens, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);

            residual_add_kernel<<<(n_tokens * HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(
                d_x, d_down, n_tokens * HIDDEN_DIM);
        }

        rms_norm_2d_kernel<<<n_tokens, 256, 0, compute_stream>>>(
            d_x, shared.d_final_norm, d_h, d_x_final_bf16, n_tokens, HIDDEN_DIM, RMS_EPS);

        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, n_tokens - 1, HIDDEN_DIM,
                     &alpha, shared.d_embed_tokens, CUDA_R_16BF, HIDDEN_DIM,
                     d_x_final_bf16, CUDA_R_16BF, HIDDEN_DIM,
                     &beta, d_logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

        cross_entropy_kernel<<<n_tokens - 1, 256, 0, compute_stream>>>(
            d_logits, d_targets, d_losses, n_tokens - 1, VOCAB_SIZE);

        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);

        float prefill_ms = 0;
        cudaEventElapsedTime(&prefill_ms, ev0, ev1);
        metrics[4].prefill_tok_s[rep] = (double)n_tokens / (prefill_ms / 1000.0);

        float h_losses[256];
        cudaMemcpy(h_losses, d_losses, (n_tokens - 1) * sizeof(float), cudaMemcpyDeviceToHost);
        double total_loss = 0;
        for (int i = 0; i < n_tokens - 1; i++) total_loss += h_losses[i];
        double mean_ce = total_loss / (n_tokens - 1);
        metrics[4].ppl = exp(mean_ce);

        if (rep == 0) {
            cudaStreamSynchronize(compute_stream);
            cudaMemcpy(arm4_final_logits, d_logits, (n_tokens - 1) * VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        }

        /* Decode 16 steps */
        cudaEventRecord(ev0, compute_stream);
        for (int step = 0; step < 16; step++) {
            embed_lookup_kernel<<<dim3((HIDDEN_DIM + 255) / 256, 1), 256, 0, compute_stream>>>(
                d_tokens + n_tokens - 1, shared.d_embed_tokens, d_x, d_x_bf16, 1, HIDDEN_DIM);
            for (int l = 0; l < N_LAYERS; l++) {
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_input_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_qkv_hole, nested_w[l].d_qkv_plug, nested_w[l].d_qkv_absmax,
                                          d_h_bf16, d_qkv, 1, 5120, HIDDEN_DIM, BLOCKSIZE);
                split_qkv_rope_kernel<<<(40 * 128 + 255) / 256, 256, 0, compute_stream>>>(d_qkv, d_q, d_k, d_v, 1, n_tokens + step);
                causal_gqa_attention_kernel<<<NUM_Q_HEADS, 256, 0, compute_stream>>>(d_q, d_k, d_v, d_attn_out, d_attn_out_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_o_hole, nested_w[l].d_o_plug, nested_w[l].d_o_absmax,
                                          d_attn_out_bf16, d_attn_proj, 1, HIDDEN_DIM, HIDDEN_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_attn_proj, HIDDEN_DIM);
                rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_post_norm[l], d_h, d_h_bf16, 1, HIDDEN_DIM, RMS_EPS);
                launch_glint_htile_s3_f32(nested_w[l].d_gate_up_hole, nested_w[l].d_gate_up_plug, nested_w[l].d_gate_up_absmax,
                                          d_h_bf16, d_gate_up, 1, 16384, HIDDEN_DIM, BLOCKSIZE);
                swiglu_kernel<<<(INTERMEDIATE_DIM + 255) / 256, 256, 0, compute_stream>>>(d_gate_up, d_act, d_act_bf16, 1);
                launch_glint_htile_s3_f32(nested_w[l].d_down_hole, nested_w[l].d_down_plug, nested_w[l].d_down_absmax,
                                          d_act_bf16, d_down, 1, HIDDEN_DIM, INTERMEDIATE_DIM, BLOCKSIZE);
                residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256, 0, compute_stream>>>(d_x, d_down, HIDDEN_DIM);
            }
            rms_norm_2d_kernel<<<1, 256, 0, compute_stream>>>(d_x, shared.d_final_norm, d_h, d_x_final_bf16, 1, HIDDEN_DIM, RMS_EPS);
        }
        cudaEventRecord(ev1, compute_stream);
        cudaEventSynchronize(ev1);
        float decode_ms = 0;
        cudaEventElapsedTime(&decode_ms, ev0, ev1);
        metrics[4].decode_tok_s[rep] = 16.0 / (decode_ms / 1000.0);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        printf("  Arm (iv) Rep %d: Prefill %6.1f tok/s, Decode %6.1f tok/s, PPL: %.4f\n",
               rep + 1, metrics[4].prefill_tok_s[rep], metrics[4].decode_tok_s[rep], metrics[4].ppl);
    }

    /* -------------------------------------------------------------
     * Verification & Pre-Registered Bounds Checks
     * ------------------------------------------------------------- */
    printf("\n====================================================================\n");
    printf("VERIFICATION & GATE EVALUATION\n");
    printf("====================================================================\n");

    /* 1. Bit-identity check between Arm (iii) and Arm (iv) */
    size_t total_logits_tested = (size_t)(n_tokens - 1) * VOCAB_SIZE;
    size_t bit_diffs = 0;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < total_logits_tested; i++) {
        uint32_t b3 = *(uint32_t*)&arm3_final_logits[i];
        uint32_t b4 = *(uint32_t*)&arm4_final_logits[i];
        if (b3 != b4) {
            bit_diffs++;
            float diff = fabsf(arm3_final_logits[i] - arm4_final_logits[i]);
            if (diff > max_abs_diff) max_abs_diff = diff;
        }
    }
    printf("1. Arm (iii) vs Arm (iv) Bit-Identity Check:\n");
    printf("   Logits compared: %lu\n", (unsigned long)total_logits_tested);
    printf("   Bit mismatches:  %lu\n", (unsigned long)bit_diffs);
    printf("   Max abs diff:    %e\n", max_abs_diff);
    int bit_identical_pass = (bit_diffs == 0);
    printf("   Verdict: %s\n", bit_identical_pass ? "PASS (0 bit differences)" : "FAIL (prefetch corruption)");

    /* 2. Compute Summary Statistics for all arms */
    for (int a = 1; a <= 4; a++) {
        double p_sort[5], d_sort[5];
        memcpy(p_sort, metrics[a].prefill_tok_s, sizeof(p_sort));
        memcpy(d_sort, metrics[a].decode_tok_s, sizeof(d_sort));
        qsort(p_sort, 5, sizeof(double), comp_double);
        qsort(d_sort, 5, sizeof(double), comp_double);
        metrics[a].min_prefill = p_sort[0];
        metrics[a].median_prefill = p_sort[2];
        metrics[a].max_prefill = p_sort[4];
        metrics[a].min_decode = d_sort[0];
        metrics[a].median_decode = d_sort[2];
        metrics[a].max_decode = d_sort[4];
    }

    /* 3. Perplexity Delta Check */
    double ppl_int8 = metrics[1].ppl;
    double ppl_hole = metrics[2].ppl;
    double ppl_stream = metrics[3].ppl;
    double ppl_resident = metrics[4].ppl;
    double delta_ppl = fabs(ppl_stream - ppl_int8) / ppl_int8 * 100.0;

    printf("\n2. Perplexity Comparison vs INT8 Reference:\n");
    printf("   Arm (i)   INT8 Reference:      %.4f\n", ppl_int8);
    printf("   Arm (ii)  Hole-only:           %.4f\n", ppl_hole);
    printf("   Arm (iii) Hole+Plug Stream:    %.4f\n", ppl_stream);
    printf("   Arm (iv)  Hole+Plug Resident:  %.4f\n", ppl_resident);
    printf("   Delta PPL (|iii - i| / i):     %.3f%%\n", delta_ppl);

    const char* ppl_verdict = "FAIL";
    if (delta_ppl <= 2.0) {
        ppl_verdict = "PASS (within 2.0% threshold)";
    } else if (delta_ppl <= 5.0) {
        ppl_verdict = "INCONCLUSIVE (2.0% - 5.0%)";
    } else {
        ppl_verdict = "FAIL (> 5.0%)";
    }
    printf("   Perplexity Verdict: %s\n", ppl_verdict);

    /* 4. Streamer & Transport Statistics */
    printf("\n3. Arm (iii) Streamer Transport & Memory Metrics:\n");
    printf("   Total H2D bytes transferred: %lu bytes (Plan: 1,610,612,736 bytes per pass)\n", metrics[3].h2d_bytes);
    printf("   Total compute stall:         %.2f ms across %d layers\n", metrics[3].total_stall_ms, N_LAYERS);
    printf("   Per-layer stall (first 8):   ");
    for (int l = 0; l < 8; l++) printf("L%d: %.2f ms  ", l, metrics[3].per_layer_stall_ms[l]);
    printf("\n");

    /* 5. VRAM Savings */
    printf("\n4. Peak VRAM Footprint:\n");
    printf("   Arm (i)   INT8:             %.1f MiB\n", metrics[1].peak_vram_mib);
    printf("   Arm (ii)  Hole-only:        %.1f MiB\n", metrics[2].peak_vram_mib);
    printf("   Arm (iii) Prefetched:       %.1f MiB\n", metrics[3].peak_vram_mib);
    printf("   Arm (iv)  Resident:         %.1f MiB\n", metrics[4].peak_vram_mib);
    printf("   VRAM Saved (iv - iii):      %.1f MiB (~%.2f GiB)\n",
           metrics[4].peak_vram_mib - metrics[3].peak_vram_mib,
           (metrics[4].peak_vram_mib - metrics[3].peak_vram_mib) / 1024.0);

    /* Write JSON results */
    FILE* jf = fopen("/home/workspace/p7-scratch/artifacts/EVAL_METRICS.json", "w");
    if (jf) {
        fprintf(jf, "{\n");
        fprintf(jf, "  \"driver\": \"%s\",\n", driver_version);
        fprintf(jf, "  \"n_tokens\": %d,\n", n_tokens);
        fprintf(jf, "  \"bit_diffs_iii_vs_iv\": %lu,\n", (unsigned long)bit_diffs);
        fprintf(jf, "  \"bit_identical_pass\": %s,\n", bit_identical_pass ? "true" : "false");
        fprintf(jf, "  \"delta_ppl_percent\": %.4f,\n", delta_ppl);
        fprintf(jf, "  \"ppl_verdict\": \"%s\",\n", ppl_verdict);
        fprintf(jf, "  \"arms\": {\n");
        for (int a = 1; a <= 4; a++) {
            const char* names[] = {"", "int8", "hole_only", "prefetched", "resident"};
            fprintf(jf, "    \"%s\": {\n", names[a]);
            fprintf(jf, "      \"ppl\": %.6f,\n", metrics[a].ppl);
            fprintf(jf, "      \"prefill_tok_s_median\": %.2f,\n", metrics[a].median_prefill);
            fprintf(jf, "      \"prefill_tok_s_min\": %.2f,\n", metrics[a].min_prefill);
            fprintf(jf, "      \"prefill_tok_s_max\": %.2f,\n", metrics[a].max_prefill);
            fprintf(jf, "      \"decode_tok_s_median\": %.2f,\n", metrics[a].median_decode);
            fprintf(jf, "      \"decode_tok_s_min\": %.2f,\n", metrics[a].min_decode);
            fprintf(jf, "      \"decode_tok_s_max\": %.2f,\n", metrics[a].max_decode);
            fprintf(jf, "      \"peak_vram_mib\": %.1f,\n", metrics[a].peak_vram_mib);
            fprintf(jf, "      \"peak_host_ram_mib\": %.1f\n", metrics[a].peak_host_ram_mib);
            fprintf(jf, "    }%s\n", a < 4 ? "," : "");
        }
        fprintf(jf, "  },\n");
        fprintf(jf, "  \"streamer\": {\n");
        fprintf(jf, "    \"total_stall_ms\": %.3f,\n", metrics[3].total_stall_ms);
        fprintf(jf, "    \"h2d_bytes\": %lu\n", metrics[3].h2d_bytes);
        fprintf(jf, "  }\n");
        fprintf(jf, "}\n");
        fclose(jf);
        printf("\nSaved full machine-readable results to artifacts/EVAL_METRICS.json\n");
    }

    /* Cleanup */
    free(arm3_final_logits);
    free(arm4_final_logits);
    for (int l = 0; l < N_LAYERS; l++) {
        cudaFreeHost(host_pinned_plugs[l]);
        cudaFree(nested_w[l].d_qkv_hole); cudaFree(nested_w[l].d_qkv_absmax);
        cudaFree(nested_w[l].d_o_hole); cudaFree(nested_w[l].d_o_absmax);
        cudaFree(nested_w[l].d_gate_up_hole); cudaFree(nested_w[l].d_gate_up_absmax);
        cudaFree(nested_w[l].d_down_hole); cudaFree(nested_w[l].d_down_absmax);
        /* If Arm iv allocated device plugs */
        if (nested_w[l].d_qkv_plug) cudaFree(nested_w[l].d_qkv_plug);
        cudaFree(shared.d_input_norm[l]); cudaFree(shared.d_post_norm[l]);
    }
    cudaFree(shared.d_embed_tokens); cudaFree(shared.d_final_norm);
    cudaFree(d_tokens); cudaFree(d_targets);
    cudaFree(d_x); cudaFree(d_h); cudaFree(d_x_bf16); cudaFree(d_h_bf16); cudaFree(d_x_final_bf16);
    cudaFree(d_qkv); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_attn_out); cudaFree(d_attn_out_bf16); cudaFree(d_attn_proj);
    cudaFree(d_gate_up); cudaFree(d_act); cudaFree(d_act_bf16); cudaFree(d_down);
    cudaFree(d_losses); cudaFree(d_logits);
    safetensors_close(&st_nested);
    safetensors_close(&st_int8);
    cublasDestroy(cublas);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(copy_stream);
    close_nvml();

    printf("\nExecution completed successfully.\n");
    return (bit_identical_pass && delta_ppl <= 2.0) ? 0 : 1;
}

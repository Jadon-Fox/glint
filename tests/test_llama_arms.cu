/* tests/test_llama_arms.cu — GLINT arms for Qwen3 / Qwen2 (DeepSeek-R1-Distill) / Llama (SmolLM2) layouts on the CUDA harness (glint #85): BF16 anchor, GLINT hole + plug, GLINT fold only; dimensions from config.json,
 * 2047 tokens, scored against the HF oracle in ARTIFACTS (eval_tokens_2048.txt, oracle_top1_2048.bin, oracle_ref_2048.json).
 * Same method as test_arm0_bf16_full.cu (bf16 rounding where HF rounds; cuBLAS GEMMs on bf16 weights), with the Qwen3 differences:
 * separate q/k/v and gate/up projections, per-head QK RMSNorm before RoPE, full-width RoPE (theta 1e6), GQA 32:8, eps 1e-6, tied
 * embedding (vocab 151936), three shards. Linears come from the BF16 model (Q3_MODE=bf16) or from a glint_pin_v1 (hole + plug + absmax)
 * decoded to bf16 one matrix at a time: Q3_MODE=full uses GLINT_CELLS[hole][plug] * absmax, Q3_MODE=fold uses PARENT[hole] * absmax.
 * Embedding and norms always come from the BF16 model. ARM0_LOGITS_OUT / ARM0_LOGITS_REF write / compare logits (KL(ref || arm)).
 * usage: test_q3_arms BF16_MODEL_DIR GLINT_PIN_DIR ARTIFACTS_DIR   (env Q3_MODE=bf16|full|fold)
 */
#include "arm0_forward.h"
#include "nf4.h"
#include "../include/glint_cells.h"
#include <algorithm>
#include <string>
#include <vector>

#define Q3_T 2048
#define Q3_NT 2047
struct Cfg { int H, I, L, NQ, NKV, HD, V; float eps; double theta; int qknorm, bias, tie; } C;
static const char *LNAME[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
static __constant__ float c_cells[16][16];
static __constant__ float c_parent[16];

enum { L_Q, L_K, L_V, L_O, L_G, L_U, L_D, N_LIN };
static int LN[N_LIN], LK[N_LIN];   /* n_out, k_in per linear, set from the config */

__global__ void k_q3_qkprep(const float *q, const float *k, const float *v, const __nv_bfloat16 *qn, const __nv_bfloat16 *kn,
                            const float *cos_t, const float *sin_t, float *qo, float *ko, float *vo, int NQ, int NKV, int HD, float eps, int qknorm) {
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x, QD = NQ * HD, KD = NKV * HD, half = HD / 2;
    if (hh >= NQ + NKV) { vo[(size_t)t * KD + (hh - NQ - NKV) * HD + d] = bf16r(v[(size_t)t * KD + (hh - NQ - NKV) * HD + d]); return; }
    const bool isq = hh < NQ;
    const float *src = isq ? q + (size_t)t * QD + hh * HD : k + (size_t)t * KD + (hh - NQ) * HD;
    __shared__ float sr[128], sn[128];
    const float x = bf16r(src[d]);
    float nv = x;
    if (qknorm) {
        const __nv_bfloat16 *w = isq ? qn : kn;
        sr[d] = x * x; __syncthreads();
        for (int st = HD / 2; st > 0; st >>= 1) { if (d < st) sr[d] += sr[d + st]; __syncthreads(); }
        const float rs = rsqrtf(sr[0] / (float)HD + eps);
        nv = bf16r(bf16r(x * rs) * __bfloat162float(w[d]));
    }
    sn[d] = nv; __syncthreads();
    const int i = d % half; const float cs = cos_t[(size_t)t * half + i], sn_ = sin_t[(size_t)t * half + i];
    float val;
    if (d < half) val = bf16r(bf16r(sn[d] * cs) - bf16r(sn[d + half] * sn_));
    else          val = bf16r(bf16r(sn[d] * cs) + bf16r(sn[d - half] * sn_));
    if (isq) qo[(size_t)t * QD + hh * HD + d] = val; else ko[(size_t)t * KD + (hh - NQ) * HD + d] = val;
}
__global__ void k_q3_bias(float *y, const __nv_bfloat16 *b, int N) { /* y[T][N] += bias[N] */
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t)Q3_T * N) return;
    y[i] += __bfloat162float(b[i % N]);
}

__global__ void k_q3_attn(const float *q, const float *k, const float *v, float *out, __nv_bfloat16 *out_bf16, int NQ, int NKV, int HD) { /* grid (T, NQ), block HD */
    const int tq = blockIdx.x, hq = blockIdx.y, hkv = hq / (NQ / NKV), d = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)HD);
    const float q_val = q[((size_t)tq * NQ + hq) * HD + d] * scale;
    float m_val = -1e30f, d_val = 0.0f, acc = 0.0f;
    __shared__ float s_red[128];
    for (int tk = 0; tk <= tq; tk++) {
        const float *k_row = k + ((size_t)tk * NKV + hkv) * HD;
        s_red[d] = q_val * k_row[d]; __syncthreads();
        for (int st = HD / 2; st > 0; st >>= 1) { if (d < st) s_red[d] += s_red[d + st]; __syncthreads(); }
        const float score = s_red[0];
        const float m_new = fmaxf(m_val, score), alpha = expf(m_val - m_new), beta = expf(score - m_new);
        d_val = d_val * alpha + beta; m_val = m_new;
        acc = acc * alpha + bf16r(beta) * v[((size_t)tk * NKV + hkv) * HD + d];
        __syncthreads();
    }
    const float fo = (d_val > 0.0f) ? (acc / d_val) : 0.0f;
    out[((size_t)tq * NQ + hq) * HD + d] = fo;
    out_bf16[((size_t)tq * NQ + hq) * HD + d] = __float2bfloat16(fo);
}

__global__ void k_q3_swiglu(const float *g, const float *u, float *out, __nv_bfloat16 *out_bf16, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    const float gg = bf16r(g[i]), uu = bf16r(u[i]);
    const float sg = bf16r(gg / (1.0f + expf(-gg)));
    const float val = bf16r(sg * uu);
    out[i] = val; out_bf16[i] = __float2bfloat16(val);
}

__global__ void k_q3_decode(const uint8_t *hole, const uint8_t *plug, const float *am, __nv_bfloat16 *w, size_t n, int fold) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const uint8_t hb = hole[i >> 1];
    const int h = (i & 1) ? (hb >> 4) : (hb & 15);
    float v;
    if (fold) v = c_parent[h] * am[i >> 6];
    else { const uint8_t pb = plug[i >> 1]; const int p = (i & 1) ? (pb >> 4) : (pb & 15); v = c_cells[h][p] * am[i >> 6]; }
    w[i] = __float2bfloat16(v);
}

struct Shards { std::vector<SafeTensors> st; };
static const TensorEntry *findS(const Shards &sh, const char *name, const char *dtype, size_t bytes) {
    for (auto &t : sh.st) {
        const TensorEntry *e = safetensors_find(&t, name);
        if (e) { if (strcmp(e->dtype, dtype) || e->end - e->begin != bytes) setup_fail("%s: want %s %zu bytes, have %s %llu", name, dtype, bytes, e->dtype, (unsigned long long)(e->end - e->begin)); return e; }
    }
    setup_fail("missing tensor %s", name); return NULL;
}
static bool hasS(const Shards &sh, const char *name) { for (auto &t : sh.st) if (safetensors_find(&t, name)) return true; return false; }
static void openS(Shards &sh, const char *dir) {
    char p[1100]; snprintf(p, sizeof(p), "%s/model.safetensors", dir); SafeTensors t;
    if (safetensors_open(&t, p) == 0) { sh.st.push_back(t); return; }
    for (int n = 2; n <= 16; n++) {
        snprintf(p, sizeof(p), "%s/model-00001-of-%05d.safetensors", dir, n); if (safetensors_open(&t, p) != 0) continue;
        sh.st.push_back(t);
        for (int s2 = 2; s2 <= n; s2++) { snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, s2, n); if (safetensors_open(&t, p) != 0) setup_fail("cannot open %s", p); sh.st.push_back(t); }
        return;
    }
    setup_fail("no safetensors in %s", dir);
}
static __nv_bfloat16 *up_bf16(const Shards &sh, const std::string &name, size_t numel) {
    __nv_bfloat16 *d; CK(cudaMalloc(&d, numel * 2)); CK(cudaMemcpy(d, findS(sh, name.c_str(), "BF16", numel * 2)->data, numel * 2, cudaMemcpyHostToDevice)); return d;
}
static double jnum(const std::string &js, const char *key, double dflt) {
    size_t p = js.find(std::string("\"") + key + "\""); if (p == std::string::npos) return dflt;
    p = js.find(':', p); while (js[++p] == ' ') {} if (js.compare(p, 4, "null") == 0) return dflt; return atof(js.c_str() + p);
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s BF16_MODEL_DIR GLINT_PIN_DIR ARTIFACTS_DIR   (env Q3_MODE=bf16|full|fold)\n", argv[0]); return 2; }
    const char *mode_s = getenv("Q3_MODE") ? getenv("Q3_MODE") : "bf16";
    const int mode = !strcmp(mode_s, "bf16") ? 0 : !strcmp(mode_s, "full") ? 1 : !strcmp(mode_s, "fold") ? 2 : -1;
    if (mode < 0) setup_fail("Q3_MODE must be bf16|full|fold");
    { /* model config */
        char p[1100]; snprintf(p, sizeof(p), "%s/config.json", argv[1]); FILE *f = fopen(p, "r"); if (!f) setup_fail("cannot read %s", p);
        std::string js; char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) js.append(b, n); fclose(f);
        C.H = (int)jnum(js, "hidden_size", 0); C.I = (int)jnum(js, "intermediate_size", 0); C.L = (int)jnum(js, "num_hidden_layers", 0);
        C.NQ = (int)jnum(js, "num_attention_heads", 0); C.NKV = (int)jnum(js, "num_key_value_heads", C.NQ); C.V = (int)jnum(js, "vocab_size", 0);
        C.HD = (int)jnum(js, "head_dim", C.NQ ? (double)C.H / C.NQ : 0); C.eps = (float)jnum(js, "rms_norm_eps", 1e-6); C.theta = jnum(js, "rope_theta", 10000.0);
        C.qknorm = js.find("Qwen3ForCausalLM") != std::string::npos; C.bias = js.find("Qwen2ForCausalLM") != std::string::npos;
        C.tie = js.find("\"tie_word_embeddings\": true") != std::string::npos || js.find("\"tie_word_embeddings\":true") != std::string::npos;
        if (jnum(js, "head_dim", -1) > 0 && C.qknorm == 0 && js.find("Qwen3") == std::string::npos) {}
        if (C.HD > 128 || C.HD < 2 || (C.HD & (C.HD - 1)) || C.H % 64 || C.I % 64 || (C.NQ * C.HD) % 64) setup_fail("unsupported dims H=%d I=%d HD=%d", C.H, C.I, C.HD);
        printf("config: H=%d I=%d L=%d NQ=%d NKV=%d HD=%d V=%d eps=%g theta=%g qknorm=%d bias=%d tie=%d\n", C.H, C.I, C.L, C.NQ, C.NKV, C.HD, C.V, C.eps, C.theta, C.qknorm, C.bias, C.tie);
    }
    const int QD = C.NQ * C.HD, KD = C.NKV * C.HD, H = C.H, I = C.I, L = C.L, V = C.V, HD = C.HD, half = C.HD / 2;
    LN[L_Q] = QD; LN[L_K] = KD; LN[L_V] = KD; LN[L_O] = H; LN[L_G] = I; LN[L_U] = I; LN[L_D] = H;
    LK[L_Q] = H; LK[L_K] = H; LK[L_V] = H; LK[L_O] = QD; LK[L_G] = H; LK[L_U] = H; LK[L_D] = I;
    Shards mst, gst; openS(mst, argv[1]); if (mode) openS(gst, argv[2]);
    std::vector<int32_t> tok; { char p[1100]; snprintf(p, sizeof(p), "%s/eval_tokens_2048.txt", argv[3]); FILE *f = fopen(p, "r"); if (!f) setup_fail("cannot read %s", p); int v; while (fscanf(f, "%d", &v) == 1) tok.push_back(v); fclose(f); }
    if (tok.size() != Q3_T) setup_fail("eval tokens: %zu", tok.size());
    std::vector<uint32_t> oracle(Q3_NT); { char p[1100]; snprintf(p, sizeof(p), "%s/oracle_top1_2048.bin", argv[3]); FILE *f = fopen(p, "rb"); if (!f || fread(oracle.data(), 4, Q3_NT, f) != Q3_NT) setup_fail("cannot read %s", p); fclose(f); }
    double oracle_ppl = 0; { char p[1100]; snprintf(p, sizeof(p), "%s/oracle_ref_2048.json", argv[3]); FILE *f = fopen(p, "r"); std::string js; char b[4096]; size_t n; while (f && (n = fread(b, 1, sizeof b, f)) > 0) js.append(b, n); if (f) fclose(f);
        oracle_ppl = jnum(js, "ppl", 0); if (oracle_ppl <= 0) setup_fail("no ppl in oracle json"); }
    printf("arms: mode=%s oracle PPL %.6f\n", mode_s, oracle_ppl);
    CK(cudaMemcpyToSymbol(c_cells, GLINT_CELLS, sizeof(float) * 256)); CK(cudaMemcpyToSymbol(c_parent, NF4_CODE, sizeof(float) * 16));

    const TensorEntry *emb_e = findS(mst, "model.embed_tokens.weight", "BF16", (size_t)V * H * 2);
    __nv_bfloat16 *d_head = C.tie ? up_bf16(mst, "model.embed_tokens.weight", (size_t)V * H) : up_bf16(mst, "lm_head.weight", (size_t)V * H);
    __nv_bfloat16 *d_fnorm = up_bf16(mst, "model.norm.weight", H);
    std::vector<__nv_bfloat16 *> in_n(L), post_n(L), qn(L, nullptr), kn(L, nullptr), bq(L, nullptr), bk(L, nullptr), bv(L, nullptr);
    std::vector<__nv_bfloat16 *> Wbf(L * N_LIN, nullptr); std::vector<uint8_t *> Hh(L * N_LIN, nullptr), Pp(L * N_LIN, nullptr); std::vector<float *> Am(L * N_LIN, nullptr);
    for (int l = 0; l < L; l++) {
        std::string pre = "model.layers." + std::to_string(l) + ".";
        in_n[l] = up_bf16(mst, pre + "input_layernorm.weight", H); post_n[l] = up_bf16(mst, pre + "post_attention_layernorm.weight", H);
        if (C.qknorm) { qn[l] = up_bf16(mst, pre + "self_attn.q_norm.weight", HD); kn[l] = up_bf16(mst, pre + "self_attn.k_norm.weight", HD); }
        if (C.bias) { bq[l] = up_bf16(mst, pre + "self_attn.q_proj.bias", QD); bk[l] = up_bf16(mst, pre + "self_attn.k_proj.bias", KD); bv[l] = up_bf16(mst, pre + "self_attn.v_proj.bias", KD); }
        for (int j = 0; j < N_LIN; j++) {
            const size_t n = (size_t)LN[j] * LK[j]; std::string nm = pre + LNAME[j] + ".weight"; const int idx = l * N_LIN + j;
            if (mode == 0) { Wbf[idx] = up_bf16(mst, nm, n); continue; }
            const size_t nb = (n + 1) / 2, ab = (n + 63) / 64 * 4;
            CK(cudaMalloc(&Hh[idx], nb)); CK(cudaMemcpy(Hh[idx], findS(gst, nm.c_str(), "U8", nb)->data, nb, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&Am[idx], ab)); CK(cudaMemcpy(Am[idx], findS(gst, (nm + ".absmax").c_str(), "F32", ab)->data, ab, cudaMemcpyHostToDevice));
            if (mode == 1) { CK(cudaMalloc(&Pp[idx], nb)); CK(cudaMemcpy(Pp[idx], findS(gst, (nm + ".glint_plug").c_str(), "U8", nb)->data, nb, cudaMemcpyHostToDevice)); }
        }
    }
    size_t maxn = 0; for (int j = 0; j < N_LIN; j++) maxn = std::max(maxn, (size_t)LN[j] * LK[j]);
    __nv_bfloat16 *d_wscratch = nullptr; if (mode) CK(cudaMalloc(&d_wscratch, maxn * 2));
    printf("Weights uploaded (%s).\n", mode == 0 ? "BF16 anchor" : mode == 1 ? "GLINT hole + plug" : "GLINT fold only (parent decode)");

    std::vector<float> cth((size_t)Q3_T * half), sth((size_t)Q3_T * half);
    for (int pos = 0; pos < Q3_T; pos++) for (int i = 0; i < half; i++) {
        double inv = 1.0 / pow(C.theta, (2.0 * i) / (double)HD), a = (double)pos * inv; float c = (float)cos(a), s_ = (float)sin(a);
        uint32_t u; memcpy(&u, &c, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&c, &u, 4);
        memcpy(&u, &s_, 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; memcpy(&s_, &u, 4); cth[(size_t)pos * half + i] = c; sth[(size_t)pos * half + i] = s_; }
    float *d_cos, *d_sin; CK(cudaMalloc(&d_cos, cth.size() * 4)); CK(cudaMalloc(&d_sin, sth.size() * 4));
    CK(cudaMemcpy(d_cos, cth.data(), cth.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_sin, sth.data(), sth.size() * 4, cudaMemcpyHostToDevice));

    float *x, *hf, *qb, *kb, *vb, *qo, *ko, *vo, *attn, *proj, *gb, *ub, *act, *down; __nv_bfloat16 *hbf, *attnb, *actb, *xfb;
    CK(cudaMalloc(&x, (size_t)Q3_T * H * 4)); CK(cudaMalloc(&hf, (size_t)Q3_T * H * 4)); CK(cudaMalloc(&hbf, (size_t)Q3_T * H * 2));
    CK(cudaMalloc(&qb, (size_t)Q3_T * QD * 4)); CK(cudaMalloc(&kb, (size_t)Q3_T * KD * 4)); CK(cudaMalloc(&vb, (size_t)Q3_T * KD * 4));
    CK(cudaMalloc(&qo, (size_t)Q3_T * QD * 4)); CK(cudaMalloc(&ko, (size_t)Q3_T * KD * 4)); CK(cudaMalloc(&vo, (size_t)Q3_T * KD * 4));
    CK(cudaMalloc(&attn, (size_t)Q3_T * QD * 4)); CK(cudaMalloc(&attnb, (size_t)Q3_T * QD * 2)); CK(cudaMalloc(&proj, (size_t)Q3_T * H * 4));
    CK(cudaMalloc(&gb, (size_t)Q3_T * I * 4)); CK(cudaMalloc(&ub, (size_t)Q3_T * I * 4)); CK(cudaMalloc(&act, (size_t)Q3_T * I * 4));
    CK(cudaMalloc(&actb, (size_t)Q3_T * I * 2)); CK(cudaMalloc(&down, (size_t)Q3_T * H * 4)); CK(cudaMalloc(&xfb, (size_t)Q3_T * H * 2));
    int32_t *d_tgt, *d_top1; float *d_loss, *d_lg; CK(cudaMalloc(&d_tgt, Q3_NT * 4)); CK(cudaMalloc(&d_top1, Q3_NT * 4)); CK(cudaMalloc(&d_loss, Q3_NT * 4)); CK(cudaMalloc(&d_lg, (size_t)256 * V * 4));
    CK(cudaMemcpy(d_tgt, tok.data() + 1, Q3_NT * 4, cudaMemcpyHostToDevice));
    const char *lo = getenv("ARM0_LOGITS_OUT"), *lr = getenv("ARM0_LOGITS_REF"); FILE *fo = lo ? fopen(lo, "wb") : NULL, *fr = lr ? fopen(lr, "rb") : NULL;
    if ((lo && !fo) || (lr && !fr)) setup_fail("cannot open logits files");
    float *d_ref = NULL, *d_kl = NULL, *hbuf = NULL; int32_t *d_rt1 = NULL, *d_qt1 = NULL;
    if (fo || fr) hbuf = (float *)malloc((size_t)256 * V * 4);
    if (fr) { CK(cudaMalloc(&d_ref, (size_t)256 * V * 4)); CK(cudaMalloc(&d_kl, Q3_NT * 4)); CK(cudaMalloc(&d_rt1, Q3_NT * 4)); CK(cudaMalloc(&d_qt1, Q3_NT * 4)); }
    cublasHandle_t cb; CKB(cublasCreate(&cb)); const float one = 1.f, zero = 0.f;

    auto linear = [&](int l, int j, const __nv_bfloat16 *xin, float *yout) {
        const int idx = l * N_LIN + j; const size_t n = (size_t)LN[j] * LK[j]; const __nv_bfloat16 *W = Wbf[idx];
        if (mode) { k_q3_decode<<<(unsigned)((n + 255) / 256), 256>>>(Hh[idx], Pp[idx], Am[idx], d_wscratch, n, mode == 2); W = d_wscratch; }
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, LN[j], Q3_T, LK[j], &one, W, CUDA_R_16BF, LK[j], xin, CUDA_R_16BF, LK[j], &zero, yout, CUDA_R_32F, LN[j], CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        if (C.bias && j <= L_V) { const __nv_bfloat16 *bb = j == L_Q ? bq[l] : j == L_K ? bk[l] : bv[l]; k_q3_bias<<<(unsigned)(((size_t)Q3_T * LN[j] + 255) / 256), 256>>>(yout, bb, LN[j]); }
    };

    std::vector<float> hx((size_t)Q3_T * H); const __nv_bfloat16 *emb = (const __nv_bfloat16 *)emb_e->data;
    for (int t = 0; t < Q3_T; t++) for (int d = 0; d < H; d++) hx[(size_t)t * H + d] = __bfloat162float(emb[(size_t)tok[t] * H + d]);
    CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    printf("Starting forward pass over %d layers...\n", L); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0));
    for (int l = 0; l < L; l++) {
        rms_norm_kernel<<<Q3_T, 128>>>(x, in_n[l], hf, hbf, Q3_T, H, C.eps);
        linear(l, L_Q, hbf, qb); linear(l, L_K, hbf, kb); linear(l, L_V, hbf, vb);
        k_q3_qkprep<<<dim3(Q3_T, C.NQ + 2 * C.NKV), HD>>>(qb, kb, vb, qn[l], kn[l], d_cos, d_sin, qo, ko, vo, C.NQ, C.NKV, HD, C.eps, C.qknorm);
        k_q3_attn<<<dim3(Q3_T, C.NQ), HD>>>(qo, ko, vo, attn, attnb, C.NQ, C.NKV, HD);
        linear(l, L_O, attnb, proj);
        residual_add_kernel<<<(Q3_T * H + 255) / 256, 256>>>(x, proj, Q3_T * H);
        rms_norm_kernel<<<Q3_T, 128>>>(x, post_n[l], hf, hbf, Q3_T, H, C.eps);
        linear(l, L_G, hbf, gb); linear(l, L_U, hbf, ub);
        k_q3_swiglu<<<(unsigned)(((size_t)Q3_T * I + 255) / 256), 256>>>(gb, ub, act, actb, (size_t)Q3_T * I);
        linear(l, L_D, actb, down);
        residual_add_kernel<<<(Q3_T * H + 255) / 256, 256>>>(x, down, Q3_T * H);
        CK(cudaGetLastError());
    }
    rms_norm_kernel<<<Q3_T, 128>>>(x, d_fnorm, hf, xfb, Q3_T, H, C.eps);
    for (int off = 0; off < Q3_NT; off += 256) {
        const int cur = off + 256 <= Q3_NT ? 256 : Q3_NT - off;
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, V, cur, H, &one, d_head, CUDA_R_16BF, H, xfb + (size_t)off * H, CUDA_R_16BF, H, &zero, d_lg, CUDA_R_32F, V, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        compute_ce_and_argmax_kernel<<<cur, 256>>>(d_lg, d_tgt + off, d_loss + off, d_top1 + off, cur, V);
        if (fo) { CK(cudaMemcpy(hbuf, d_lg, (size_t)cur * V * 4, cudaMemcpyDeviceToHost)); if (fwrite(hbuf, 4, (size_t)cur * V, fo) != (size_t)cur * V) setup_fail("short write"); }
        if (fr) { if (fread(hbuf, 4, (size_t)cur * V, fr) != (size_t)cur * V) setup_fail("short read"); CK(cudaMemcpy(d_ref, hbuf, (size_t)cur * V * 4, cudaMemcpyHostToDevice));
                  kl_and_argmax_kernel<<<cur, 256>>>(d_ref, d_lg, d_kl + off, d_rt1 + off, d_qt1 + off, cur, V); }
        CK(cudaGetLastError());
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    std::vector<float> loss(Q3_NT); std::vector<int32_t> t1(Q3_NT); std::vector<float> kl(Q3_NT);
    CK(cudaMemcpy(loss.data(), d_loss, Q3_NT * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(t1.data(), d_top1, Q3_NT * 4, cudaMemcpyDeviceToHost));
    double tot = 0; int match = 0; for (int i = 0; i < Q3_NT; i++) { tot += loss[i]; match += ((uint32_t)t1[i] == oracle[i]); }
    const double ce = tot / Q3_NT, ppl = exp(ce);
    printf("\n=== %s ===\nForward Pass Time:      %.3f s\nMean Cross Entropy:     %.6f\nMeasured Perplexity:    %.6f\nHuggingFace Oracle PPL: %.6f\nPPL Delta vs Oracle:    %.4f %%\nTop-1 Matches:          %d / %d\nTop-1 Agreement:        %.2f %%\n",
           mode_s, ms / 1000.0, ce, ppl, oracle_ppl, fabs(ppl - oracle_ppl) / oracle_ppl * 100.0, match, Q3_NT, 100.0 * match / Q3_NT);
    if (fr) { CK(cudaMemcpy(kl.data(), d_kl, Q3_NT * 4, cudaMemcpyDeviceToHost)); std::vector<float> s2 = kl; std::sort(s2.begin(), s2.end()); double ks = 0; for (float v : kl) ks += v;
              printf("KL vs reference: mean %.6e p99 %.6e max %.6e (%d tokens)\n", ks / Q3_NT, s2[(size_t)(0.99 * (Q3_NT - 1))], s2[Q3_NT - 1], Q3_NT); }
    printf("VERDICT: REPORT ONLY (%s)\n", mode_s);
    if (fo) fclose(fo); if (fr) fclose(fr);
    return 0;
}

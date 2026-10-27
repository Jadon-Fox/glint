// fwd30.cu (job #22). Extension of job20/layer0.cu (the copy; layer0.cu itself is not modified and its code is not merged into p7fold2.cu).
// Phi-4-mini FP32 forward through layers 0..N-1 (default 30) in C++/CUDA, with capture of the four layer inputs named in
// rules section 2 and optional dump of the residual stream after every layer.
//   fwd30 --st1 SHARD1 --st2 SHARD2 --sets "name:ids.i32:seqlens.txt;name2:..." --out-prefix P
//         [--variant real|rope128|gqamod|gateup_swap] [--variant-layer L (default -1 = every layer)]
//         [--capture-layers 3,16,29] [--capture-kinds qkv,o,gu,down] [--resid-prefix R] [--nlayers 30] [--src PATH]
// Captures are appended per sequence to <P>_<set>_L<layer>_<kind>.f32 (token-major, row = one token, FP32). No arm uses them
// in this job; they are for G-f parity and the C rank log only.
//   qkv  = post-input_layernorm hidden state (q/k/v input, n=3072)
//   o    = attention output before o_proj (n=3072)
//   gu   = post-post_attention_layernorm hidden state (gate/up input, n=3072)
//   down = act(gate)*up (down input, n=8192)
// Every sequence is its own sequence; positions restart at 0. Weights are read by byte range one layer at a time.
// Variants (negative controls): rope128 = RoPE over all 128 dims; gqamod = kv_head = q_head % 8; gateup_swap = gate and up
// halves swapped. The variant applies to every layer unless --variant-layer L restricts it to layer L.
// Build: nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o job22/fwd30 job22/src/fwd30.cu -lcublas -lcusolver -lcrypto
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <sys/resource.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                                       \
  do {                                                                                      \
    cudaError_t e_ = (call);                                                                \
    if (e_ != cudaSuccess) {                                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                              \
    }                                                                                       \
  } while (0)
#define CUBLAS_OK(call)                                                                 \
  do {                                                                                  \
    cublasStatus_t s_ = (call);                                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      fprintf(stderr, "cuBLAS error %s:%d status %d\n", __FILE__, __LINE__, (int)s_);   \
      exit(2);                                                                          \
    }                                                                                   \
  } while (0)

static const int kHidden = 3072, kInter = 8192;
static const int kNQ = 24, kNKV = 8, kHD = 128, kRot = 96;
static const int kQKV = (kNQ + 2 * kNKV) * kHD;  // 5120
static const int kVocab = 200064;
static const float kEps = 1e-5f;
static const int kOrigMax = 4096, kMaxPos = 131072;
// rope_scaling.short_factor is 48 x 1.0 (job22/layout_check.json: short_factor_all_ones)

struct Tensor {
  int file;
  long a, b;
};
struct Shard {
  FILE* f = nullptr;
  long data0 = 0;
  std::string hdr;
};
static Shard g_sh[2];

static std::vector<char> read_range(FILE* f, long off, size_t n) {
  std::vector<char> b(n);
  if (fseek(f, off, SEEK_SET) != 0 || fread(b.data(), 1, n, f) != n) {
    fprintf(stderr, "short read at %ld len %zu\n", off, n);
    exit(1);
  }
  return b;
}

static Tensor find_tensor(const std::string& name) {
  for (int s = 0; s < 2; ++s) {
    const std::string& hdr = g_sh[s].hdr;
    size_t p = hdr.find("\"" + name + "\"");
    if (p == std::string::npos) continue;
    size_t q = hdr.find("\"data_offsets\"", p);
    q = hdr.find('[', q);
    long a, b;
    if (sscanf(hdr.c_str() + q, "[%ld,%ld]", &a, &b) != 2) {
      fprintf(stderr, "bad data_offsets for %s\n", name.c_str());
      exit(1);
    }
    return {s, a, b};
  }
  fprintf(stderr, "tensor %s not in either header\n", name.c_str());
  exit(1);
}

static std::vector<char> load_tensor(const std::string& name, size_t expect_bytes) {
  Tensor t = find_tensor(name);
  if ((size_t)(t.b - t.a) != expect_bytes) {
    fprintf(stderr, "tensor %s: %ld bytes, expected %zu\n", name.c_str(), t.b - t.a, expect_bytes);
    exit(1);
  }
  return read_range(g_sh[t.file].f, g_sh[t.file].data0 + t.a, expect_bytes);
}

static float bf16_to_f32(uint16_t u) {
  uint32_t v = (uint32_t)u << 16;
  float f;
  memcpy(&f, &v, 4);
  return f;
}

__global__ void bf16_to_f32_kernel(const uint16_t* in, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __uint_as_float(((uint32_t)in[i]) << 16);
}

// one block per token; FP32 sum of squares (mean over 3072), x * rsqrt(var+eps), then w * that
__global__ void rmsnorm_kernel(const float* x, const float* w, float* out) {
  int t = blockIdx.x;
  const float* xr = x + (size_t)t * kHidden;
  __shared__ float red[256];
  float s = 0.f;
  for (int i = threadIdx.x; i < kHidden; i += blockDim.x) s += xr[i] * xr[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float var = red[0] / (float)kHidden;
  float r = rsqrtf(var + kEps);
  for (int i = threadIdx.x; i < kHidden; i += blockDim.x) out[(size_t)t * kHidden + i] = w[i] * (xr[i] * r);
}

__global__ void rope_kernel(float* qkv, const float* cosv, const float* sinv, int T, int rot) {
  int t = blockIdx.x;
  int h = blockIdx.y;  // 0..31: 24 q heads then 8 k heads
  int i = threadIdx.x;  // 0..rot/2-1
  float* p = qkv + (size_t)t * kQKV + h * kHD;
  float x1 = p[i], x2 = p[i + rot / 2];
  const float* c = cosv + (size_t)t * rot;
  const float* s = sinv + (size_t)t * rot;
  p[i] = x1 * c[i] + (-x2) * s[i];
  p[i + rot / 2] = x2 * c[i + rot / 2] + x1 * s[i + rot / 2];
}

__global__ void attn_kernel(const float* qkv, float* out, int T, float scale, int gqa_mod) {
  int t = blockIdx.x, h = blockIdx.y;
  int kvh = gqa_mod ? h % kNKV : h / (kNQ / kNKV);
  extern __shared__ float sc[];
  __shared__ float qs[kHD];
  __shared__ float red[128];
  const float* q = qkv + (size_t)t * kQKV + h * kHD;
  qs[threadIdx.x] = q[threadIdx.x];
  __syncthreads();
  const int kOff = kNQ * kHD, vOff = (kNQ + kNKV) * kHD;
  for (int s = threadIdx.x; s <= t; s += blockDim.x) {
    const float* k = qkv + (size_t)s * kQKV + kOff + kvh * kHD;
    float a = 0.f;
    for (int d = 0; d < kHD; ++d) a += qs[d] * k[d];
    sc[s] = a * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (int s = threadIdx.x; s <= t; s += blockDim.x) m = fmaxf(m, sc[s]);
  red[threadIdx.x] = m;
  __syncthreads();
  for (int o = 64; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int s = threadIdx.x; s <= t; s += blockDim.x) {
    float e = expf(sc[s] - m);
    sc[s] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int o = 64; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  sum = red[0];
  __syncthreads();
  float acc = 0.f;
  for (int s = 0; s <= t; ++s) acc += sc[s] * qkv[(size_t)s * kQKV + vOff + kvh * kHD + threadIdx.x];
  out[(size_t)t * kHidden + h * kHD + threadIdx.x] = acc / sum;
}

__global__ void add_kernel(float* h, const float* d, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) h[i] += d[i];
}

// gu [T, 16384] row = [first half 8192 | second half 8192]; real: gate = first half (chunk(2,-1)[0]), act = silu(gate) * up
__global__ void act_kernel(const float* gu, float* act, int T, int swap) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)T * kInter) return;
  long t = i / kInter, j = i % kInter;
  float a = gu[t * 2 * kInter + j], b = gu[t * 2 * kInter + kInter + j];
  float gate = swap ? b : a, up = swap ? a : b;
  act[i] = up * (gate / (1.f + expf(-gate)));
}

static void sgemm_nt(cublasHandle_t h, const float* dA, const float* dW, float* dC, int T, int outd, int ind) {
  // row-major C[T,outd] = A[T,ind] * W[outd,ind]^T
  const float one = 1.f, zero = 0.f;
  CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, outd, T, ind, &one, dW, ind, dA, ind, &zero, dC, outd));
}

static std::map<std::string, std::string> parse_args(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  return a;
}
static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::string t;
  for (char ch : s + std::string(1, c)) {
    if (ch == c) {
      if (!t.empty()) out.push_back(t);
      t.clear();
    } else {
      t += ch;
    }
  }
  return out;
}
static std::string sha256_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return "unreadable";
  SHA256_CTX c;
  SHA256_Init(&c);
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) SHA256_Update(&c, buf, n);
  fclose(f);
  unsigned char md[32];
  SHA256_Final(md, &c);
  char hex[65];
  for (int i = 0; i < 32; ++i) sprintf(hex + 2 * i, "%02x", md[i]);
  return std::string(hex, 64);
}

struct SetInfo {
  std::string name;
  std::vector<int32_t> ids;
  std::vector<int> lens;
  size_t off = 0;  // token offset into the concatenated hidden buffer
};

static void build_rope(int rot, int maxL, std::vector<float>* cosv, std::vector<float>* sinv, double* attn_out) {
  double factor = (double)kMaxPos / kOrigMax;
  double attn_d = factor <= 1.0 ? 1.0 : std::sqrt(1.0 + std::log(factor) / std::log((double)kOrigMax));
  float attn = (float)attn_d;
  *attn_out = attn_d;
  std::vector<float> invf(rot / 2);
  for (int i = 0; i < rot / 2; ++i) {
    float e = (float)(2 * i) / (float)rot;
    float pw = powf(10000.0f, e);
    invf[i] = 1.0f / (1.0f * pw);  // short_factor[i] == 1.0 for i < 48; 1.0 beyond (rope128 control only)
  }
  cosv->assign((size_t)maxL * rot, 0.f);
  sinv->assign((size_t)maxL * rot, 0.f);
  for (int t = 0; t < maxL; ++t)
    for (int i = 0; i < rot / 2; ++i) {
      float ang = (float)t * invf[i];
      float c = cosf(ang) * attn, s = sinf(ang) * attn;
      (*cosv)[(size_t)t * rot + i] = (*cosv)[(size_t)t * rot + i + rot / 2] = c;
      (*sinv)[(size_t)t * rot + i] = (*sinv)[(size_t)t * rot + i + rot / 2] = s;
    }
}

int main(int argc, char** argv) {
  auto a = parse_args(argc, argv);
  for (const char* k : {"--st1", "--st2", "--sets", "--out-prefix"})
    if (!a.count(k)) {
      fprintf(stderr, "usage: fwd30 --st1 F --st2 F --sets name:ids:seqlens;... --out-prefix P [--variant V] [--variant-layer L] [--capture-layers 3,16,29] [--capture-kinds qkv,o,gu,down] [--resid-prefix R] [--nlayers 30] [--src PATH]\n");
      return 1;
    }
  std::string variant = a.count("--variant") ? a["--variant"] : "real";
  if (variant != "real" && variant != "rope128" && variant != "gqamod" && variant != "gateup_swap") {
    fprintf(stderr, "unknown variant %s\n", variant.c_str());
    return 1;
  }
  int vlayer = a.count("--variant-layer") ? atoi(a["--variant-layer"].c_str()) : -1;
  int nlayers = a.count("--nlayers") ? atoi(a["--nlayers"].c_str()) : 30;
  std::set<int> caplayers;
  if (a.count("--capture-layers")) for (auto& s : split(a["--capture-layers"], ',')) caplayers.insert(atoi(s.c_str()));
  std::set<std::string> kinds;
  for (auto& s : split(a.count("--capture-kinds") ? a["--capture-kinds"] : "qkv,o,gu,down", ',')) {
    if (s != "qkv" && s != "o" && s != "gu" && s != "down") { fprintf(stderr, "bad kind %s\n", s.c_str()); return 1; }
    kinds.insert(s);
  }
  const std::string outp = a["--out-prefix"];
  const bool dump_resid = a.count("--resid-prefix") > 0;

  for (int s = 0; s < 2; ++s) {
    g_sh[s].f = fopen(a[s == 0 ? "--st1" : "--st2"].c_str(), "rb");
    if (!g_sh[s].f) { fprintf(stderr, "cannot open shard %d\n", s + 1); return 1; }
    uint64_t hlen;
    if (fread(&hlen, 8, 1, g_sh[s].f) != 1) return 1;
    std::vector<char> hb = read_range(g_sh[s].f, 8, hlen);
    g_sh[s].hdr.assign(hb.begin(), hb.end());
    g_sh[s].data0 = 8 + (long)hlen;
  }

  std::vector<SetInfo> sets;
  size_t Ttot = 0;
  int maxL = 0;
  for (auto& spec : split(a["--sets"], ';')) {
    auto p = split(spec, ':');
    if (p.size() != 3) { fprintf(stderr, "bad set spec %s\n", spec.c_str()); return 1; }
    SetInfo si;
    si.name = p[0];
    FILE* fi = fopen(p[1].c_str(), "rb");
    if (!fi) { fprintf(stderr, "cannot open ids %s\n", p[1].c_str()); return 1; }
    fseek(fi, 0, SEEK_END);
    long n = ftell(fi) / 4;
    fseek(fi, 0, SEEK_SET);
    si.ids.resize(n);
    if (fread(si.ids.data(), 4, n, fi) != (size_t)n) return 1;
    fclose(fi);
    FILE* fl = fopen(p[2].c_str(), "r");
    if (!fl) { fprintf(stderr, "cannot open seqlens %s\n", p[2].c_str()); return 1; }
    char buf[1 << 16];
    size_t bl = fread(buf, 1, sizeof buf - 1, fl);
    buf[bl] = 0;
    fclose(fl);
    for (auto& s : split(buf, ',')) si.lens.push_back(atoi(s.c_str()));
    long sum = 0;
    for (int L : si.lens) { sum += L; if (L > maxL) maxL = L; }
    if (sum != n) { fprintf(stderr, "set %s: seqlens sum %ld != tokens %ld\n", si.name.c_str(), sum, n); return 1; }
    si.off = Ttot;
    Ttot += n;
    sets.push_back(si);
  }
  if (maxL > kOrigMax) { fprintf(stderr, "T=%d > original_max_position_embeddings; long_factor branch not implemented\n", maxL); return 1; }

  // embeddings by byte range (only the needed rows), straight into the device hidden buffer
  CUDA_OK(cudaFree(0));
  float* dH;
  CUDA_OK(cudaMalloc(&dH, Ttot * kHidden * sizeof(float)));
  {
    Tensor emb = find_tensor("model.embed_tokens.weight");
    if ((size_t)(emb.b - emb.a) != (size_t)kVocab * kHidden * 2) { fprintf(stderr, "embed size\n"); return 1; }
    std::vector<float> row(kHidden);
    for (auto& si : sets)
      for (size_t t = 0; t < si.ids.size(); ++t) {
        int id = si.ids[t];
        if (id < 0 || id >= kVocab) { fprintf(stderr, "id out of range %d\n", id); return 1; }
        std::vector<char> rb = read_range(g_sh[emb.file].f, g_sh[emb.file].data0 + emb.a + (long)id * kHidden * 2, kHidden * 2);
        for (int i = 0; i < kHidden; ++i) row[i] = bf16_to_f32(((const uint16_t*)rb.data())[i]);
        CUDA_OK(cudaMemcpy(dH + (si.off + t) * kHidden, row.data(), kHidden * 4, cudaMemcpyHostToDevice));
      }
  }

  std::vector<float> cos96, sin96, cos128, sin128;
  double attn_d = 0;
  build_rope(96, maxL, &cos96, &sin96, &attn_d);
  build_rope(128, maxL, &cos128, &sin128, &attn_d);
  float *dCos96, *dSin96, *dCos128, *dSin128;
  CUDA_OK(cudaMalloc(&dCos96, cos96.size() * 4)); CUDA_OK(cudaMalloc(&dSin96, sin96.size() * 4));
  CUDA_OK(cudaMalloc(&dCos128, cos128.size() * 4)); CUDA_OK(cudaMalloc(&dSin128, sin128.size() * 4));
  CUDA_OK(cudaMemcpy(dCos96, cos96.data(), cos96.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dSin96, sin96.data(), sin96.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dCos128, cos128.data(), cos128.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dSin128, sin128.data(), sin128.size() * 4, cudaMemcpyHostToDevice));

  cublasHandle_t hb;
  CUBLAS_OK(cublasCreate(&hb));
  CUBLAS_OK(cublasSetMathMode(hb, CUBLAS_PEDANTIC_MATH));
  float *dWqkv, *dWo, *dWgu, *dWd, *dLn1, *dLn2, *dXn, *dQKV, *dAtt, *dO, *dGU, *dAct, *dD;
  uint16_t* dStage;  // BF16 staging for the largest tensor (gate_up)
  CUDA_OK(cudaMalloc(&dStage, (size_t)2 * kInter * kHidden * 2));
  CUDA_OK(cudaMalloc(&dWqkv, (size_t)kQKV * kHidden * 4));
  CUDA_OK(cudaMalloc(&dWo, (size_t)kHidden * kHidden * 4));
  CUDA_OK(cudaMalloc(&dWgu, (size_t)2 * kInter * kHidden * 4));
  CUDA_OK(cudaMalloc(&dWd, (size_t)kHidden * kInter * 4));
  CUDA_OK(cudaMalloc(&dLn1, kHidden * 4));
  CUDA_OK(cudaMalloc(&dLn2, kHidden * 4));
  CUDA_OK(cudaMalloc(&dXn, (size_t)maxL * kHidden * 4));
  CUDA_OK(cudaMalloc(&dQKV, (size_t)maxL * kQKV * 4));
  CUDA_OK(cudaMalloc(&dAtt, (size_t)maxL * kHidden * 4));
  CUDA_OK(cudaMalloc(&dO, (size_t)maxL * kHidden * 4));
  CUDA_OK(cudaMalloc(&dGU, (size_t)maxL * 2 * kInter * 4));
  CUDA_OK(cudaMalloc(&dAct, (size_t)maxL * kInter * 4));
  CUDA_OK(cudaMalloc(&dD, (size_t)maxL * kHidden * 4));

  auto upload_w = [&](const std::string& name, size_t n_el, float* dst) {
    std::vector<char> b = load_tensor(name, n_el * 2);
    CUDA_OK(cudaMemcpy(dStage, b.data(), b.size(), cudaMemcpyHostToDevice));
    bf16_to_f32_kernel<<<(n_el + 255) / 256, 256>>>(dStage, dst, (long)n_el);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
  };

  std::vector<float> hostbuf;
  const float scale = 1.0f / sqrtf((float)kHD);
  for (int L = 0; L < nlayers; ++L) {
    char pre[96];
    snprintf(pre, sizeof pre, "model.layers.%d.", L);
    std::string P(pre);
    upload_w(P + "input_layernorm.weight", kHidden, dLn1);
    upload_w(P + "post_attention_layernorm.weight", kHidden, dLn2);
    upload_w(P + "self_attn.qkv_proj.weight", (size_t)kQKV * kHidden, dWqkv);
    upload_w(P + "self_attn.o_proj.weight", (size_t)kHidden * kHidden, dWo);
    upload_w(P + "mlp.gate_up_proj.weight", (size_t)2 * kInter * kHidden, dWgu);
    upload_w(P + "mlp.down_proj.weight", (size_t)kHidden * kInter, dWd);
    const bool vact = variant != "real" && (vlayer < 0 || vlayer == L);
    const int rot = (vact && variant == "rope128") ? 128 : kRot;
    const int gqa_mod = (vact && variant == "gqamod") ? 1 : 0;
    const int swap = (vact && variant == "gateup_swap") ? 1 : 0;
    const float* cs = rot == 128 ? dCos128 : dCos96;
    const float* sn = rot == 128 ? dSin128 : dSin96;
    const bool cap = caplayers.count(L) > 0;
    for (auto& si : sets) {
      FILE* cf[4] = {nullptr, nullptr, nullptr, nullptr};
      const char* kn[4] = {"qkv", "o", "gu", "down"};
      if (cap)
        for (int k = 0; k < 4; ++k)
          if (kinds.count(kn[k])) {
            char fn[512];
            snprintf(fn, sizeof fn, "%s_%s_L%d_%s.f32", outp.c_str(), si.name.c_str(), L, kn[k]);
            cf[k] = fopen(fn, "wb");
            if (!cf[k]) { fprintf(stderr, "cannot write %s\n", fn); return 1; }
          }
      size_t off = si.off;
      for (int T : si.lens) {
        float* h = dH + off * kHidden;
        rmsnorm_kernel<<<T, 256>>>(h, dLn1, dXn);
        CUDA_OK(cudaGetLastError());
        if (cf[0]) { hostbuf.resize((size_t)T * kHidden); CUDA_OK(cudaMemcpy(hostbuf.data(), dXn, hostbuf.size() * 4, cudaMemcpyDeviceToHost)); fwrite(hostbuf.data(), 4, hostbuf.size(), cf[0]); }
        sgemm_nt(hb, dXn, dWqkv, dQKV, T, kQKV, kHidden);
        rope_kernel<<<dim3(T, kNQ + kNKV), rot / 2>>>(dQKV, cs, sn, T, rot);
        CUDA_OK(cudaGetLastError());
        attn_kernel<<<dim3(T, kNQ), kHD, T * sizeof(float)>>>(dQKV, dAtt, T, scale, gqa_mod);
        CUDA_OK(cudaGetLastError());
        if (cf[1]) { hostbuf.resize((size_t)T * kHidden); CUDA_OK(cudaMemcpy(hostbuf.data(), dAtt, hostbuf.size() * 4, cudaMemcpyDeviceToHost)); fwrite(hostbuf.data(), 4, hostbuf.size(), cf[1]); }
        sgemm_nt(hb, dAtt, dWo, dO, T, kHidden, kHidden);
        add_kernel<<<((long)T * kHidden + 255) / 256, 256>>>(h, dO, (long)T * kHidden);
        rmsnorm_kernel<<<T, 256>>>(h, dLn2, dXn);
        CUDA_OK(cudaGetLastError());
        if (cf[2]) { hostbuf.resize((size_t)T * kHidden); CUDA_OK(cudaMemcpy(hostbuf.data(), dXn, hostbuf.size() * 4, cudaMemcpyDeviceToHost)); fwrite(hostbuf.data(), 4, hostbuf.size(), cf[2]); }
        sgemm_nt(hb, dXn, dWgu, dGU, T, 2 * kInter, kHidden);
        act_kernel<<<((long)T * kInter + 255) / 256, 256>>>(dGU, dAct, T, swap);
        CUDA_OK(cudaGetLastError());
        if (cf[3]) { hostbuf.resize((size_t)T * kInter); CUDA_OK(cudaMemcpy(hostbuf.data(), dAct, hostbuf.size() * 4, cudaMemcpyDeviceToHost)); fwrite(hostbuf.data(), 4, hostbuf.size(), cf[3]); }
        sgemm_nt(hb, dAct, dWd, dD, T, kHidden, kInter);
        add_kernel<<<((long)T * kHidden + 255) / 256, 256>>>(h, dD, (long)T * kHidden);
        CUDA_OK(cudaGetLastError());
        off += T;
      }
      for (int k = 0; k < 4; ++k) if (cf[k]) fclose(cf[k]);
    }
    CUDA_OK(cudaDeviceSynchronize());
    if (dump_resid) {
      for (auto& si : sets) {
        size_t n = si.ids.size() * (size_t)kHidden;
        hostbuf.resize(n);
        CUDA_OK(cudaMemcpy(hostbuf.data(), dH + si.off * kHidden, n * 4, cudaMemcpyDeviceToHost));
        char fn[512];
        snprintf(fn, sizeof fn, "%s_%s_L%d.f32", a["--resid-prefix"].c_str(), si.name.c_str(), L);
        FILE* f = fopen(fn, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", fn); return 1; }
        fwrite(hostbuf.data(), 4, n, f);
        fclose(f);
      }
    }
  }
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  printf("fwd30 variant=%s variant_layer=%d layers=%d tokens=%zu sets=%zu max_seq_len=%d max_position=%d attention_factor=%.9f branch=short_factor peak_rss_MB=%ld gpu_used_MiB=%zu src_sha256=%s\n",
         variant.c_str(), vlayer, nlayers, Ttot, sets.size(), maxL, maxL - 1, attn_d, ru.ru_maxrss / 1024, (tot - fr) >> 20,
         a.count("--src") ? sha256_file(a["--src"]).c_str() : "not-given");
  return 0;
}

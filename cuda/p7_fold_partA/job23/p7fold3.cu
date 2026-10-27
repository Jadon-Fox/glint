// p7fold3.cu -- P7 fold harness, round 2 (job #23). PENDING Test Designer review.
// Copy of job22/src/p7fold2.cu (sha256 caa34abd1a1bbc8aa260eef5dc4e5a7c8106a5044b71a473fefd586fff78e710) changed per rules
// sha256 03f97724... Amendment B: (B-1) A4 stored as U_k and U_k^T E, no floor, no inverse; (B-2) sparse byte slack logged;
// per-arm per-factor FP16 underflow/overflow counts; eigenvalues <= 0 count; extra rank-deficient / full-rank synthetic X in ga.
// The NF4 quantizer block (kBlock .. nf4_quantize_device) and the A0/A1/A3/A5 math are unchanged (see job23/diffs).
// Below: the job #22 header, kept for lineage.
// p7fold2.cu -- P7 fold harness, round 2 (job #22). PENDING Test Designer review.
// Copy of src/p7fold.cu (sha256 04bfd293c1e22b54a79661aa2a0782844f9b780e715a7f19274f37d20f4c36e1) extended for rules
// sha256 5bda0d59... (round-2 rules + Amendment A). NF4 quantizer path (kBlock, kCodebook, kCB, nearest_code, nf4_index,
// absmax_kernel, pack_kernel, dequant_kernel, nf4_quantize_device) is byte-identical to 04bfd293 (checked in job22/build.log).
// A2 is retired (removed). A0/A1/A3 keep the 04bfd293 math, generalized from 3072x3072 to any m x n (smaller-side Gram for A1).
// New: A4 (activation-weighted low rank), A5 (magnitude-only sparse), stored-bytes writer + independent decoders,
// per-projection row slices, 16,384-token calibration, rank(C)/floored-eigenvalue logging, G-a apparatus, C-rank mode.
// Modes:
//   p7fold3 parity <in.f64> <out_packed.bin> <out_absmax.f64>               (unchanged from 04bfd293)
//   p7fold3 ga1dump  --st1 F --st2 F --layers 3,16,29 --outdir D --out JSON  (NF4 fold of the 21 real matrices; Python cross-checks)
//   p7fold3 ga       --st1 F --st2 F --outdir D --out JSON [--seed S] [--layer 16] (G-a2..G-a4 on synthetic X only)
//   p7fold3 crank    --xcal F --seqlens F --n N --label S --out JSON          (rank/floored count of C = X_cal^T X_cal; logging only)
//   p7fold3 measure  ... (per-cell arms A0,A1,A4,A3,A5; NOT RUN in job #22 on real data; smoke-tested on synthetic weights+X only)
// Build: nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o job23/p7fold3 job23/src/p7fold3.cu -lcublas -lcusolver -lcrypto
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <openssl/sha.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
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
#define CUBLAS_OK(call)                                                             \
  do {                                                                              \
    cublasStatus_t s_ = (call);                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                              \
      fprintf(stderr, "cuBLAS error %s:%d status %d\n", __FILE__, __LINE__, (int)s_); \
      exit(2);                                                                      \
    }                                                                               \
  } while (0)
#define CUSOLVER_OK(call)                                                             \
  do {                                                                                \
    cusolverStatus_t s_ = (call);                                                     \
    if (s_ != CUSOLVER_STATUS_SUCCESS) {                                              \
      fprintf(stderr, "cuSOLVER error %s:%d status %d\n", __FILE__, __LINE__, (int)s_); \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

static const int kBlock = 64;

// Same literals as bf16-to-int8/nf4.py NF4_CODEBOOK.
static const double kCodebook[16] = {
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
};
__constant__ double kCB[16];

// ---------- device: NF4 (nearest_code strict '<' => ties to lower index; idx 7 when absmax == 0) ----------
__device__ __forceinline__ int nearest_code(double v) {
  int best = 0;
  double best_d = fabs(v - kCB[0]);
  for (int j = 1; j < 16; ++j) {
    double d = fabs(v - kCB[j]);
    if (d < best_d) {
      best_d = d;
      best = j;
    }
  }
  return best;
}

__device__ __forceinline__ int nf4_index(const double* w, const double* absmax, int i) {
  double scale = absmax[i / kBlock];
  return scale == 0.0 ? 7 : nearest_code(w[i] / scale);
}

__global__ void absmax_kernel(const double* w, int n, int nblocks, double* absmax) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  double m = 0.0;
  int end = min(n, (b + 1) * kBlock);
  for (int i = b * kBlock; i < end; ++i) {
    double a = fabs(w[i]);
    if (a > m) m = a;
  }
  absmax[b] = m;
}

// lo nibble = even element, hi nibble = odd element (nibble order 0 in nf4.py)
__global__ void pack_kernel(const double* w, const double* absmax, int n, int nbytes, uint8_t* packed) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nbytes) return;
  int lo = nf4_index(w, absmax, 2 * t);
  int hi = (2 * t + 1 < n) ? nf4_index(w, absmax, 2 * t + 1) : 0;
  packed[t] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
}

__global__ void dequant_kernel(const uint8_t* packed, const double* absmax, int n, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int byte = packed[i >> 1];
  int code = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
  out[i] = (float)(kCB[code] * absmax[i / kBlock]);
}

static void nf4_quantize_device(const double* dW, int n, double* dAbsmax, uint8_t* dPacked) {
  int nb = (n + kBlock - 1) / kBlock;
  int nbytes = (n + 1) / 2;
  absmax_kernel<<<(nb + 255) / 256, 256>>>(dW, n, nb, dAbsmax);
  CUDA_OK(cudaGetLastError());
  pack_kernel<<<(nbytes + 255) / 256, 256>>>(dW, dAbsmax, n, nbytes, dPacked);
  CUDA_OK(cudaGetLastError());
}

// ---------- helpers ----------
static std::string sha256_hex(const void* p, size_t n) {
  unsigned char md[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(p), n, md);
  char buf[65];
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) sprintf(buf + 2 * i, "%02x", md[i]);
  return std::string(buf, 64);
}

static std::vector<char> read_all(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz);
  if (sz && fread(buf.data(), 1, sz, f) != (size_t)sz) {
    fprintf(stderr, "short read %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
  return buf;
}

static std::map<std::string, std::string> parse_args(int argc, char** argv, int start) {
  std::map<std::string, std::string> a;
  for (int i = start; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) continue;
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      a[k] = argv[++i];
    } else {
      a[k] = "1";
    }
  }
  return a;
}

static const std::string& need(const std::map<std::string, std::string>& a, const char* k) {
  auto it = a.find(k);
  if (it == a.end()) {
    fprintf(stderr, "missing %s\n", k);
    exit(1);
  }
  return it->second;
}

static void gemm_nt(cublasHandle_t h, const float* dA, const float* dB, float* dC, int m, int nn, int k) {
  // Row-major C[m x nn] = A[m x k] * B[nn x k]^T, computed as column-major C^T = B * A^T.
  const float one = 1.f, zero = 0.f;
  CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nn, m, k, &one, dB, k, dA, k, &zero, dC, nn));
}

// ---------- parity mode ----------
static int run_parity(const std::string& in_path, const std::string& out_packed, const std::string& out_absmax) {
  std::vector<char> raw = read_all(in_path);
  int n = (int)(raw.size() / sizeof(double));
  std::vector<double> w(n);
  memcpy(w.data(), raw.data(), (size_t)n * sizeof(double));
  int nb = (n + kBlock - 1) / kBlock;
  int nbytes = (n + 1) / 2;
  CUDA_OK(cudaMemcpyToSymbol(kCB, kCodebook, sizeof(kCodebook)));
  double *dW, *dA;
  uint8_t* dP;
  CUDA_OK(cudaMalloc(&dW, n * sizeof(double)));
  CUDA_OK(cudaMalloc(&dA, nb * sizeof(double)));
  CUDA_OK(cudaMalloc(&dP, nbytes));
  CUDA_OK(cudaMemcpy(dW, w.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  nf4_quantize_device(dW, n, dA, dP);
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint8_t> P(nbytes);
  std::vector<double> A(nb);
  CUDA_OK(cudaMemcpy(P.data(), dP, nbytes, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(A.data(), dA, nb * sizeof(double), cudaMemcpyDeviceToHost));
  FILE* f = fopen(out_packed.c_str(), "wb");
  fwrite(P.data(), 1, nbytes, f);
  fclose(f);
  f = fopen(out_absmax.c_str(), "wb");
  fwrite(A.data(), sizeof(double), nb, f);
  fclose(f);
  printf("parity n=%d blocks=%d packed_bytes=%d device=%s\n", n, nb, nbytes, "cuda");
  return 0;
}

// =====================================================================================================
// job #22 additions start here (everything above is the 04bfd293 lineage: includes, NF4 path, helpers, gemm_nt, parity)
// =====================================================================================================
static const double kEpsD = 2.220446049250313e-16;
static cublasHandle_t g_hb = nullptr;
static cusolverDnHandle_t g_hs = nullptr;
static std::string g_srcs_json = "{}";  // sha256 of every built source, in every output record
static std::string g_bin_sha = "";

static void init_handles() {
  if (g_hb) return;
  CUBLAS_OK(cublasCreate(&g_hb));
  CUBLAS_OK(cublasSetMathMode(g_hb, CUBLAS_PEDANTIC_MATH));
  CUSOLVER_OK(cusolverDnCreate(&g_hs));
  CUDA_OK(cudaMemcpyToSymbol(kCB, kCodebook, sizeof(kCodebook)));
}

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t n_ = 0) { alloc(n_); }
  void alloc(size_t n_) {
    release();
    n = n_;
    if (n) CUDA_OK(cudaMalloc(&p, n * sizeof(T)));
  }
  void release() {
    if (p) cudaFree(p);
    p = nullptr;
  }
  ~Dev() { release(); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void up(const T* h, size_t cnt) { CUDA_OK(cudaMemcpy(p, h, cnt * sizeof(T), cudaMemcpyHostToDevice)); }
  void down(T* h, size_t cnt) const { CUDA_OK(cudaMemcpy(h, p, cnt * sizeof(T), cudaMemcpyDeviceToHost)); }
};

// ---------- tiny JSON writer ----------
static std::string jq(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o + "\"";
}
static std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[64];
  snprintf(b, sizeof b, "%.12g", v);
  return b;
}
struct JObj {
  std::vector<std::pair<std::string, std::string>> kv;
  JObj& n(const char* k, double v) { kv.push_back({k, jnum(v)}); return *this; }
  JObj& i(const char* k, long long v) { kv.push_back({k, std::to_string(v)}); return *this; }
  JObj& s(const char* k, const std::string& v) { kv.push_back({k, jq(v)}); return *this; }
  JObj& b(const char* k, bool v) { kv.push_back({k, v ? "true" : "false"}); return *this; }
  JObj& r(const char* k, const std::string& raw) { kv.push_back({k, raw}); return *this; }
  std::string dump() const {
    std::string o = "{";
    for (size_t t = 0; t < kv.size(); ++t) o += (t ? ", " : "") + jq(kv[t].first) + ": " + kv[t].second;
    return o + "}";
  }
};
static std::string jarr(const std::vector<std::string>& v, const char* sep = ",\n ") {
  std::string o = "[";
  for (size_t t = 0; t < v.size(); ++t) o += (t ? sep : "") + v[t];
  return o + "]";
}

static void write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
  if (!b.empty()) fwrite(b.data(), 1, b.size(), f);
  fclose(f);
}
static void write_text(const std::string& path, const std::string& s) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
  fwrite(s.data(), 1, s.size(), f);
  fclose(f);
}
static long long file_size(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 ? (long long)st.st_size : -1;
}
static std::string sha256_file(const std::string& p) {
  std::vector<char> b = read_all(p);
  return sha256_hex(b.data(), b.size());
}
static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::string t;
  for (char ch : s + std::string(1, c)) {
    if (ch == c) { if (!t.empty()) out.push_back(t); t.clear(); }
    else t += ch;
  }
  return out;
}
static void set_sources(const std::string& srcs) {  // comma-separated paths; also the running binary
  JObj o;
  for (auto& p : split(srcs, ',')) {
    std::string base = p.substr(p.rfind('/') == std::string::npos ? 0 : p.rfind('/') + 1);
    o.s(base.c_str(), sha256_file(p));
  }
  g_srcs_json = o.dump();
  g_bin_sha = sha256_file("/proc/self/exe");
}
static std::vector<float> read_f32_file(const std::string& path, size_t* count) {
  std::vector<char> raw = read_all(path);
  *count = raw.size() / sizeof(float);
  std::vector<float> v(*count);
  memcpy(v.data(), raw.data(), *count * sizeof(float));
  return v;
}

// ---------- half helpers ----------
struct H16Stats { long long overflow = 0, underflow = 0; };
static uint16_t f2h(float x) { __half h = __float2half(x); uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }
static uint16_t f2h_log(float x, H16Stats& st) {
  uint16_t u = f2h(x);
  if ((u & 0x7C00) == 0x7C00) st.overflow++;
  else if (x != 0.f && (u & 0x7C00) == 0) st.underflow++;  // stored as zero or subnormal
  return u;
}

// ---------- GEMM wrappers (row-major C[M,N] = op(A) op(B); op(A) is M x K, op(B) is K x N) ----------
// tA: A stored K x M; tB: B stored N x K. gemm_f(false,true) is exactly gemm_nt of 04bfd293.
static void gemm_d(const double* A, const double* B, double* C, int M, int N, int K, bool tA, bool tB, double beta = 0.0) {
  const double one = 1.0;
  CUBLAS_OK(cublasDgemm(g_hb, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tB ? K : N, A,
                        tA ? M : K, &beta, C, N));
}
static void gemm_f(const float* A, const float* B, float* C, int M, int N, int K, bool tA, bool tB) {
  const float one = 1.f, zero = 0.f;
  CUBLAS_OK(cublasSgemm(g_hb, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tB ? K : N, A,
                        tA ? M : K, &zero, C, N));
}
// host-side wrappers (upload, multiply, download)
static std::vector<double> hgemm_d(const std::vector<double>& A, const std::vector<double>& B, int M, int N, int K, bool tA, bool tB) {
  Dev<double> dA(A.size()), dB(B.size()), dC((size_t)M * N);
  dA.up(A.data(), A.size());
  dB.up(B.data(), B.size());
  gemm_d(dA.p, dB.p, dC.p, M, N, K, tA, tB);
  std::vector<double> C((size_t)M * N);
  dC.down(C.data(), C.size());
  return C;
}
static std::vector<float> hgemm_f(const std::vector<float>& A, const std::vector<float>& B, int M, int N, int K, bool tA, bool tB) {
  Dev<float> dA(A.size()), dB(B.size()), dC((size_t)M * N);
  dA.up(A.data(), A.size());
  dB.up(B.data(), B.size());
  gemm_f(dA.p, dB.p, dC.p, M, N, K, tA, tB);
  std::vector<float> C((size_t)M * N);
  dC.down(C.data(), C.size());
  return C;
}

// ---------- symmetric eigendecomposition (cuSOLVER syevd, FP64) ----------
// dA: n x n symmetric on device (overwritten). w ascending; Z[j*n + i] = component i of eigenvector j.
static void sym_eig_dev(double* dA, int n, std::vector<double>& w, std::vector<double>& Z) {
  Dev<double> dW(n);
  int lwork = 0;
  CUSOLVER_OK(cusolverDnDsyevd_bufferSize(g_hs, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dA, n, dW.p, &lwork));
  Dev<double> work(lwork);
  Dev<int> info(1);
  CUSOLVER_OK(cusolverDnDsyevd(g_hs, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dA, n, dW.p, work.p, lwork, info.p));
  int hi = 0;
  info.down(&hi, 1);
  if (hi != 0) { fprintf(stderr, "syevd info=%d\n", hi); exit(2); }
  Z.resize((size_t)n * n);
  w.resize(n);
  CUDA_OK(cudaMemcpy(Z.data(), dA, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToHost));
  dW.down(w.data(), n);
}

// ---------- SVD of an m x n matrix via the smaller-side Gram (same method as A1 in 04bfd293: syevd, V = M^T U / sigma) ----------
struct Svd {
  int m = 0, n = 0, r = 0, kmax = 0;
  std::vector<double> U, V;       // unit singular vectors, row-major m x kmax and n x kmax
  std::vector<double> sig;        // kmax
  std::vector<double> eig_desc;   // all r squared singular values, descending (clipped at 0)
  double tail_fro(int k) const {  // sqrt(sum of dropped sigma^2) for rank k
    double t = 0.0;
    for (int j = k; j < r; ++j) t += eig_desc[j];
    return sqrt(t);
  }
};
static Svd svd_topk(const std::vector<double>& M, int m, int n, int kmax) {
  Svd s;
  s.m = m; s.n = n; s.r = std::min(m, n); s.kmax = kmax = std::min(kmax, s.r);
  Dev<double> dM(M.size());
  dM.up(M.data(), M.size());
  const bool left = (m <= n);
  const int g = left ? m : n;
  std::vector<double> w, Z;
  {
    Dev<double> dG((size_t)g * g);
    if (left) gemm_d(dM.p, dM.p, dG.p, m, m, n, false, true);   // M M^T
    else gemm_d(dM.p, dM.p, dG.p, n, n, m, true, false);        // M^T M
    sym_eig_dev(dG.p, g, w, Z);
  }
  s.eig_desc.resize(s.r);
  for (int j = 0; j < s.r; ++j) s.eig_desc[j] = std::max(w[g - 1 - j], 0.0);
  s.sig.resize(kmax);
  for (int j = 0; j < kmax; ++j) s.sig[j] = sqrt(s.eig_desc[j]);
  std::vector<double> Wt((size_t)g * kmax);  // g x kmax, column j = top-j eigenvector
  for (int j = 0; j < kmax; ++j)
    for (int r = 0; r < g; ++r) Wt[(size_t)r * kmax + j] = Z[(size_t)(g - 1 - j) * g + r];
  Z.clear(); Z.shrink_to_fit();
  Dev<double> dW(Wt.size());
  dW.up(Wt.data(), Wt.size());
  std::vector<double> O;  // the other side, unnormalized = M^T U or M V
  if (left) {
    O.resize((size_t)n * kmax);
    Dev<double> dO(O.size());
    gemm_d(dM.p, dW.p, dO.p, n, kmax, m, true, false);
    dO.down(O.data(), O.size());
    s.U = Wt;
    s.V = O;
    for (int r = 0; r < n; ++r)
      for (int j = 0; j < kmax; ++j) s.V[(size_t)r * kmax + j] = s.sig[j] > 0.0 ? O[(size_t)r * kmax + j] / s.sig[j] : 0.0;
  } else {
    O.resize((size_t)m * kmax);
    Dev<double> dO(O.size());
    gemm_d(dM.p, dW.p, dO.p, m, kmax, n, false, false);
    dO.down(O.data(), O.size());
    s.V = Wt;
    s.U = O;
    for (int r = 0; r < m; ++r)
      for (int j = 0; j < kmax; ++j) s.U[(size_t)r * kmax + j] = s.sig[j] > 0.0 ? O[(size_t)r * kmax + j] / s.sig[j] : 0.0;
  }
  return s;
}

// ---------- budget arithmetic (rules section 2/3, A1 largest rank, A3/A5 entry count) ----------
static long long budget_R(int pct, int m, int n) { return (long long)pct * 2LL * m * n / 100; }  // floor(p*2*m*n), p = pct/100 exactly
static int rank_k(long long R, int m, int n) {
  long long k = (R - 4) / (2LL * (m + n));  // largest k with 2k(m+n)+4 <= R
  if (k < 0) k = 0;
  return (int)std::min<long long>(k, std::min(m, n));
}
static long long a1_bytes(int k, int m, int n) { return 2LL * k * (m + n) + 4; }
static long long n_entries(long long a1b) { return (a1b - 4) / 6; }  // floor((A1_bytes-4)/6)

// ---------- arm outputs ----------
struct H16Factor { std::string name; long long count = 0; H16Stats st; };
struct ArmOut {
  std::string arm;
  int k = -1;
  long long entries = -1;
  std::vector<uint8_t> bytes;   // exactly what is stored
  std::vector<float> res;       // FP32 residual used in W_hat (m x n)
  long long added() const { return (long long)bytes.size(); }
  std::vector<H16Factor> h16f;  // one per stored FP16 factor (index arrays are int32 and not listed)
  void set_factors(std::initializer_list<std::pair<const char*, long long>> l) {
    h16f.clear();
    for (auto& x : l) { H16Factor f; f.name = x.first; f.count = x.second; h16f.push_back(f); }
  }
  long long overflow_total() const { long long t = 0; for (auto& f : h16f) t += f.st.overflow; return t; }
  long long underflow_total() const { long long t = 0; for (auto& f : h16f) t += f.st.underflow; return t; }
  std::string h16_json() const {
    std::vector<std::string> v;
    for (auto& f : h16f)
      v.push_back(JObj().s("factor", f.name).i("fp16_elements", f.count).i("overflow", f.st.overflow)
                      .i("underflow_to_zero_or_subnormal", f.st.underflow).dump());
    return jarr(v, ", ");
  }
};
static void put_i32(std::vector<uint8_t>& b, int32_t v) { uint8_t t[4]; memcpy(t, &v, 4); b.insert(b.end(), t, t + 4); }
static void put_u16(std::vector<uint8_t>& b, uint16_t v) { uint8_t t[2]; memcpy(t, &v, 2); b.insert(b.end(), t, t + 2); }

// A1: stores U*Sigma (m x k, FP16, row-major) then V (n x k, FP16, row-major), 4 B header = k. residual = Us * V^T (FP32 GEMM).
static ArmOut build_A1(const Svd& s, int k) {
  ArmOut a; a.arm = "A1"; a.k = k;
  const int m = s.m, n = s.n, K = s.kmax;
  put_i32(a.bytes, k);
  a.set_factors({{"U_sigma", (long long)m * k}, {"V", (long long)n * k}});
  std::vector<float> Us((size_t)m * k), Vh((size_t)n * k);
  for (int r = 0; r < m; ++r)
    for (int j = 0; j < k; ++j) {
      uint16_t u = f2h_log((float)(s.U[(size_t)r * K + j] * s.sig[j]), a.h16f[0].st);
      put_u16(a.bytes, u);
      Us[(size_t)r * k + j] = h2f(u);
    }
  for (int r = 0; r < n; ++r)
    for (int j = 0; j < k; ++j) {
      uint16_t u = f2h_log((float)s.V[(size_t)r * K + j], a.h16f[1].st);
      put_u16(a.bytes, u);
      Vh[(size_t)r * k + j] = h2f(u);
    }
  a.res = hgemm_f(Us, Vh, m, n, k, false, true);
  return a;
}

// A4 context from C = X_cal^T X_cal (FP64, n x n): C = Q Lambda Q^T (syevd). Amendment B-1: Lambda_+ = max(raw eigenvalue, 0);
// no floor, no inverse. Raw eigenvalues are kept for logging.
struct A4Ctx {
  int n = 0;
  std::vector<double> Qt;    // n x n row-major, row j = eigenvector j (ascending eigenvalue)
  std::vector<double> lam;   // Lambda_+ = max(raw, 0)
  std::vector<double> w;     // raw eigenvalues ascending
  double lam_max = 0;
  int rank_c = 0;            // count of raw eigenvalues > n*eps*max(Lambda)  (tolerance n * 2.22e-16 * lam_max)
  int n_negative = 0;        // raw eigenvalue < 0
  int n_nonpos = 0;          // raw eigenvalue <= 0
};
static A4Ctx a4_ctx_from_dev(double* dC, int n) {
  A4Ctx c; c.n = n;
  sym_eig_dev(dC, n, c.w, c.Qt);
  c.lam_max = c.w[n - 1];
  c.lam.resize(n);
  const double tol = (double)n * kEpsD * c.lam_max;
  for (int j = 0; j < n; ++j) {
    if (c.w[j] < 0.0) c.n_negative++;
    if (c.w[j] <= 0.0) c.n_nonpos++;
    if (c.w[j] > tol) c.rank_c++;
    c.lam[j] = std::max(c.w[j], 0.0);
  }
  return c;
}
static A4Ctx a4_ctx_from_host(const std::vector<double>& C, int n) {
  Dev<double> dC(C.size());
  dC.up(C.data(), C.size());
  return a4_ctx_from_dev(dC.p, n);
}
// C += X^T X for X (rows x n) FP32 on host, accumulated in FP64 on device
static void gram_accumulate(double* dC, const float* Xh, int rows, int n, bool first) {
  Dev<double> dXd((size_t)rows * n);
  std::vector<double> blk((size_t)rows * n);
  for (size_t i = 0; i < blk.size(); ++i) blk[i] = Xh[i];
  dXd.up(blk.data(), blk.size());
  gemm_d(dXd.p, dXd.p, dC, n, n, rows, true, false, first ? 0.0 : 1.0);
}
static std::vector<double> gram_host(const std::vector<float>& X, int rows, int n) {
  Dev<double> dC((size_t)n * n);
  gram_accumulate(dC.p, X.data(), rows, n, true);
  std::vector<double> C((size_t)n * n);
  dC.down(C.data(), C.size());
  return C;
}
// M = E Q Lambda_+^{1/2}  (m x n)
static std::vector<double> a4_weighted(const std::vector<double>& E, int m, int n, const A4Ctx& c) {
  std::vector<double> M = hgemm_d(E, c.Qt, m, n, n, false, true);  // E * Q  (Qt row j = column j of Q)
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) M[(size_t)i * n + j] *= sqrt(c.lam[j]);
  return M;
}
// U_k (m x k, FP64): first k columns of the left singular vectors of M
static std::vector<double> a4_uk64(const Svd& s, int k) {
  std::vector<double> U((size_t)s.m * k);
  for (int r = 0; r < s.m; ++r)
    for (int j = 0; j < k; ++j) U[(size_t)r * k + j] = s.U[(size_t)r * s.kmax + j];
  return U;
}
// A4 storage (Amendment B-1): 4 B header k, U_k (m x k, FP16 row-major), then U_k^T E (k x n, FP16 row-major).
// U_k^T E is formed in FP64 from the FP64 U_k (not from the FP16-rounded U_k) and E (FP32 values widened to FP64), then rounded
// once to FP16. The residual used in W_hat is rebuilt from the stored FP16 factors (FP32 GEMM), which is what the decoder does.
static ArmOut build_A4(const Svd& s, const std::vector<float>& E, int k) {
  ArmOut a; a.arm = "A4"; a.k = k;
  const int m = s.m, n = s.n;
  std::vector<double> U = a4_uk64(s, k);
  std::vector<double> Ed(E.begin(), E.end());
  std::vector<double> UtE = hgemm_d(U, Ed, k, n, m, true, false);  // k x n
  Ed.clear(); Ed.shrink_to_fit();
  a.set_factors({{"U_k", (long long)m * k}, {"UtE", (long long)k * n}});
  put_i32(a.bytes, k);
  std::vector<float> Uf(U.size()), Tf(UtE.size());
  for (size_t i = 0; i < U.size(); ++i) { uint16_t u = f2h_log((float)U[i], a.h16f[0].st); put_u16(a.bytes, u); Uf[i] = h2f(u); }
  for (size_t i = 0; i < UtE.size(); ++i) { uint16_t u = f2h_log((float)UtE[i], a.h16f[1].st); put_u16(a.bytes, u); Tf[i] = h2f(u); }
  a.res = hgemm_f(Uf, Tf, m, n, k, false, false);
  return a;
}

// A3 (cn != nullptr: score |E_ij| * ||X_cal[:,j]||) and A5 (cn == nullptr: score |E_ij|). Same selection rule as 04bfd293 A3:
// top nE by score, ties to the lower flat index; stored sorted by index. Storage: int32 count, nE x int32 flat index, nE x FP16 value.
static ArmOut build_sparse(const char* arm, const std::vector<float>& E, const std::vector<double>* cn, size_t nE, int m, int n) {
  ArmOut a; a.arm = arm; a.entries = (long long)nE;
  const size_t N = (size_t)m * n;
  std::vector<double> score(N);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) score[(size_t)i * n + j] = fabs((double)E[(size_t)i * n + j]) * (cn ? (*cn)[j] : 1.0);
  std::vector<uint32_t> ord(N);
  std::iota(ord.begin(), ord.end(), 0u);
  std::nth_element(ord.begin(), ord.begin() + nE, ord.end(), [&](uint32_t x, uint32_t y) {
    return score[x] > score[y] || (score[x] == score[y] && x < y);
  });
  ord.resize(nE);
  std::sort(ord.begin(), ord.end());
  a.res.assign(N, 0.f);
  a.set_factors({{"values", (long long)nE}});
  put_i32(a.bytes, (int32_t)nE);
  for (size_t t = 0; t < nE; ++t) put_i32(a.bytes, (int32_t)ord[t]);
  for (size_t t = 0; t < nE; ++t) {
    uint16_t u = f2h_log(E[ord[t]], a.h16f[0].st);
    put_u16(a.bytes, u);
    a.res[ord[t]] = h2f(u);
  }
  return a;
}

// ---------- independent decoders: they see only the file (path), m and n ----------
struct Decoded { bool ok = false; std::string err; std::vector<float> res; };
static bool rd_i32(const std::vector<char>& b, size_t off, int32_t* v) {
  if (off + 4 > b.size()) return false;
  memcpy(v, b.data() + off, 4);
  return true;
}
static std::vector<float> rd_half_block(const std::vector<char>& b, size_t off, size_t cnt) {
  std::vector<float> o(cnt);
  for (size_t i = 0; i < cnt; ++i) { uint16_t u; memcpy(&u, b.data() + off + 2 * i, 2); o[i] = h2f(u); }
  return o;
}
// kind: "A1" (Us m x k, V n x k, residual Us V^T), "A4" (U_k m x k, U_k^T E k x n, residual U_k (U_k^T E)).
// transposed_broken: the two FP16 blocks are reinterpreted as k x m and k x n (transposed layouts) and multiplied as P^T Q.
static Decoded decode_lowrank(const std::string& path, int m, int n, const std::string& kind, bool transposed_broken) {
  Decoded d;
  if (file_size(path) < 0) { d.err = "cannot open file"; return d; }
  std::vector<char> b = read_all(path);
  int32_t k;
  if (!rd_i32(b, 0, &k) || k < 0) { d.err = "bad header"; return d; }
  if ((long long)b.size() != 4 + 2LL * k * (m + n)) { d.err = "length != 4 + 2k(m+n): shape-incompatible"; return d; }
  std::vector<float> f1 = rd_half_block(b, 4, (size_t)m * k);
  std::vector<float> f2 = rd_half_block(b, 4 + 2 * (size_t)m * k, (size_t)n * k);
  if (kind == "A1") {
    if (!transposed_broken) d.res = hgemm_f(f1, f2, m, n, k, false, true);
    else d.res = hgemm_f(f1, f2, m, n, k, true, false);  // f1 as k x m, f2 as k x n
  } else if (kind == "A4") {
    if (!transposed_broken) d.res = hgemm_f(f1, f2, m, n, k, false, false);
    else d.res = hgemm_f(f1, f2, m, n, k, true, false);
  } else { d.err = "unknown kind"; return d; }
  d.ok = true;
  return d;
}
// idx_shift != 0 is the broken "indices off by one" variant.
static Decoded decode_sparse(const std::string& path, int m, int n, int idx_shift) {
  Decoded d;
  if (file_size(path) < 0) { d.err = "cannot open file"; return d; }
  std::vector<char> b = read_all(path);
  int32_t c;
  if (!rd_i32(b, 0, &c) || c < 0) { d.err = "bad header"; return d; }
  if ((long long)b.size() != 4 + 6LL * c) { d.err = "length != 4 + 6c"; return d; }
  d.res.assign((size_t)m * n, 0.f);
  long long N = (long long)m * n, prev = -1;
  for (int32_t t = 0; t < c; ++t) {
    int32_t ix;
    memcpy(&ix, b.data() + 4 + 4 * (size_t)t, 4);
    long long j = (long long)ix + idx_shift;
    if (j < 0 || j >= N) { d.err = "index out of range"; d.res.clear(); return d; }
    if (idx_shift == 0 && ix <= prev) { d.err = "indices not strictly increasing"; d.res.clear(); return d; }
    prev = ix;
    uint16_t u; memcpy(&u, b.data() + 4 + 4 * (size_t)c + 2 * (size_t)t, 2);
    d.res[(size_t)j] = h2f(u);
  }
  d.ok = true;
  return d;
}
static Decoded decode_arm(const std::string& path, int m, int n, const std::string& arm) {
  if (arm == "A1" || arm == "A4") return decode_lowrank(path, m, n, arm, false);
  return decode_sparse(path, m, n, 0);
}
static long long bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return -1;
  long long c = 0;
  for (size_t i = 0; i < a.size(); ++i) if (memcmp(&a[i], &b[i], 4) != 0) ++c;
  return c;
}

// ---------- weights: per-projection row slices by byte range ----------
struct TInfo { std::string dtype; long long d0 = 0, d1 = 0, begin = 0, end = 0; };
struct Shards {
  FILE* f[2] = {nullptr, nullptr};
  std::string hdr[2], path[2];
  long long data0[2] = {0, 0};
};
static Shards open_shards(const std::string& p1, const std::string& p2) {
  Shards s;
  s.path[0] = p1; s.path[1] = p2;
  for (int i = 0; i < 2; ++i) {
    s.f[i] = fopen(s.path[i].c_str(), "rb");
    if (!s.f[i]) { fprintf(stderr, "cannot open %s\n", s.path[i].c_str()); exit(1); }
    uint64_t hl = 0;
    if (fread(&hl, 8, 1, s.f[i]) != 1) exit(1);
    s.hdr[i].resize(hl);
    if (fread(&s.hdr[i][0], 1, hl, s.f[i]) != hl) exit(1);
    s.data0[i] = 8 + (long long)hl;
  }
  return s;
}
static int find_tensor_info(const Shards& s, const std::string& name, TInfo* t) {
  for (int i = 0; i < 2; ++i) {
    const std::string& hdr = s.hdr[i];
    size_t p = hdr.find("\"" + name + "\"");
    if (p == std::string::npos) continue;
    size_t obj_end = hdr.find('}', p);
    auto value_at = [&](const char* key) -> size_t {
      std::string k = std::string("\"") + key + "\"";
      size_t q = hdr.find(k, p);
      if (q == std::string::npos || q > obj_end) { fprintf(stderr, "key %s missing for %s\n", key, name.c_str()); exit(1); }
      q += k.size();
      while (q < hdr.size() && (hdr[q] == ':' || hdr[q] == ' ')) ++q;
      return q;
    };
    size_t dq = value_at("dtype");
    t->dtype = hdr.substr(dq + 1, hdr.find('"', dq + 1) - dq - 1);
    size_t sq = value_at("shape");
    size_t lb = hdr.find('[', sq);
    t->d0 = atoll(hdr.c_str() + lb + 1);
    size_t comma = hdr.find(',', lb);
    t->d1 = (comma != std::string::npos && comma < hdr.find(']', lb)) ? atoll(hdr.c_str() + comma + 1) : 0;
    size_t oq = value_at("data_offsets");
    size_t ob = hdr.find('[', oq) + 1;
    t->begin = atoll(hdr.c_str() + ob);
    t->end = atoll(hdr.c_str() + hdr.find(',', ob) + 1);
    return i;
  }
  fprintf(stderr, "tensor %s not in either header\n", name.c_str());
  exit(1);
}
struct ProjSpec { std::string type, tensor; int row0, row1, full_rows, n; };
static ProjSpec proj_spec(int layer, const std::string& type) {
  char b[96];
  ProjSpec p;
  p.type = type;
  auto nm = [&](const char* suf) { snprintf(b, sizeof b, "model.layers.%d.%s", layer, suf); return std::string(b); };
  if (type == "q") p = {type, nm("self_attn.qkv_proj.weight"), 0, 3072, 5120, 3072};
  else if (type == "k") p = {type, nm("self_attn.qkv_proj.weight"), 3072, 4096, 5120, 3072};
  else if (type == "v") p = {type, nm("self_attn.qkv_proj.weight"), 4096, 5120, 5120, 3072};
  else if (type == "o") p = {type, nm("self_attn.o_proj.weight"), 0, 3072, 3072, 3072};
  else if (type == "gate") p = {type, nm("mlp.gate_up_proj.weight"), 0, 8192, 16384, 3072};
  else if (type == "up") p = {type, nm("mlp.gate_up_proj.weight"), 8192, 16384, 16384, 3072};
  else if (type == "down") p = {type, nm("mlp.down_proj.weight"), 0, 3072, 3072, 8192};
  else { fprintf(stderr, "unknown type %s\n", type.c_str()); exit(1); }
  return p;
}
static std::string input_kind_of(const std::string& type) {
  if (type == "q" || type == "k" || type == "v") return "qkv";
  if (type == "o") return "o";
  if (type == "gate" || type == "up") return "gu";
  return "down";
}
struct WSlice {
  int m = 0, n = 0;
  std::vector<uint16_t> bf;  // m x n BF16 bits
  std::string tensor, blob, sha;
  long long b0 = 0, b1 = 0;
};
static WSlice read_slice(const Shards& s, int layer, const std::string& type) {
  ProjSpec p = proj_spec(layer, type);
  TInfo t;
  int fi = find_tensor_info(s, p.tensor, &t);
  if (t.dtype != "BF16" || t.d0 != p.full_rows || t.d1 != p.n || t.end - t.begin != 2LL * p.full_rows * p.n) {
    fprintf(stderr, "OPEN_layout: %s dtype=%s shape=[%lld,%lld]\n", p.tensor.c_str(), t.dtype.c_str(), t.d0, t.d1);
    exit(3);
  }
  WSlice w;
  w.m = p.row1 - p.row0; w.n = p.n; w.tensor = p.tensor; w.blob = s.path[fi];
  w.b0 = s.data0[fi] + t.begin + 2LL * p.row0 * p.n;
  w.b1 = w.b0 + 2LL * w.m * w.n;
  w.bf.resize((size_t)w.m * w.n);
  if (fseek(s.f[fi], w.b0, SEEK_SET) != 0 || fread(w.bf.data(), 2, w.bf.size(), s.f[fi]) != w.bf.size()) {
    fprintf(stderr, "short read %s\n", p.tensor.c_str());
    exit(1);
  }
  w.sha = sha256_hex(w.bf.data(), w.bf.size() * 2);
  return w;
}
struct Fold {
  std::vector<float> Wf, deq, E;
  std::vector<uint8_t> Q;
  std::vector<double> absmax;
  std::string q_sha, absmax_sha;
};
// NF4 fold of one matrix (block 64 over the flat row-major slice), E = W - deQ(Q) in FP32 as in 04bfd293
static Fold nf4_fold(const std::vector<uint16_t>& bf) {
  Fold f;
  const int N = (int)bf.size();
  f.Wf.resize(N);
  for (int i = 0; i < N; ++i) { uint32_t bits = (uint32_t)bf[i] << 16; memcpy(&f.Wf[i], &bits, 4); }
  std::vector<double> Wd(f.Wf.begin(), f.Wf.end());
  const int nb = (N + kBlock - 1) / kBlock, nbytes = (N + 1) / 2;
  Dev<double> dW(N), dA(nb);
  Dev<uint8_t> dQ(nbytes);
  Dev<float> dDeq(N);
  dW.up(Wd.data(), N);
  nf4_quantize_device(dW.p, N, dA.p, dQ.p);
  dequant_kernel<<<(N + 255) / 256, 256>>>(dQ.p, dA.p, N, dDeq.p);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  f.deq.resize(N); f.Q.resize(nbytes); f.absmax.resize(nb);
  dDeq.down(f.deq.data(), N); dQ.down(f.Q.data(), nbytes); dA.down(f.absmax.data(), nb);
  f.q_sha = sha256_hex(f.Q.data(), f.Q.size());
  f.absmax_sha = sha256_hex(f.absmax.data(), f.absmax.size() * sizeof(double));
  f.E.resize(N);
  for (int i = 0; i < N; ++i) f.E[i] = f.Wf[i] - f.deq[i];
  return f;
}

static const char* kTypes[7] = {"q", "k", "v", "o", "gate", "up", "down"};
static void shape_of(const std::string& type, int* m, int* n) {
  ProjSpec p = proj_spec(0, type);
  *m = p.row1 - p.row0; *n = p.n;
}

// ---------- ga1dump: NF4 fold of the real matrices; Python cross-checks deQ against nf4_copy/nf4.py ----------
static int run_ga1dump(const std::map<std::string, std::string>& a) {
  init_handles();
  set_sources(need(a, "--srcs"));
  Shards sh = open_shards(need(a, "--st1"), need(a, "--st2"));
  std::vector<std::string> recs;
  for (auto& ls : split(need(a, "--layers"), ','))
    for (const char* ty : kTypes) {
      int L = atoi(ls.c_str());
      WSlice w = read_slice(sh, L, ty);
      Fold f = nf4_fold(w.bf);
      char base[512];
      snprintf(base, sizeof base, "%s/L%d_%s", need(a, "--outdir").c_str(), L, ty);
      write_bytes(std::string(base) + ".q.bin", f.Q);
      FILE* fo = fopen((std::string(base) + ".absmax.f64").c_str(), "wb");
      fwrite(f.absmax.data(), 8, f.absmax.size(), fo); fclose(fo);
      fo = fopen((std::string(base) + ".deq.f32").c_str(), "wb");
      fwrite(f.deq.data(), 4, f.deq.size(), fo); fclose(fo);
      recs.push_back(JObj().i("layer", L).s("type", ty).s("tensor", w.tensor).i("m", w.m).i("n", w.n).i("elements", (long long)w.bf.size())
                         .i("byte_begin", w.b0).i("byte_end", w.b1).s("w_slice_sha256", w.sha).s("q_sha256", f.q_sha)
                         .s("absmax_sha256", f.absmax_sha).s("deq_f32_sha256", sha256_hex(f.deq.data(), f.deq.size() * 4)).dump());
      printf("ga1dump L%d %s elements=%zu\n", L, ty, w.bf.size());
      fflush(stdout);
    }
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  write_text(need(a, "--out"), JObj().s("label", "PENDING Test Designer review").r("source_sha256", g_srcs_json).s("binary_sha256", g_bin_sha)
                                    .i("peak_rss_kb", ru.ru_maxrss).r("matrices", jarr(recs)).dump() + "\n");
  return 0;
}

// ---------- G-a apparatus (synthetic X only; never real captured X) ----------
static double fro(const std::vector<double>& v) { double s = 0; for (double x : v) s += x * x; return sqrt(s); }
static double rel_err_f(const std::vector<float>& a, const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) { double d = (double)a[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
  return sqrt(num / den);
}
struct SynX {
  std::vector<double> cn;  // column norms
  A4Ctx ctx;
  int rows = 2048;
  std::string tag, desc;
};
// salt 0 / rows 2048 reproduces the job #22 synthetic X exactly (same RNG stream). Other variants use other salts.
static SynX make_synth(uint64_t seed, int n, int rows, uint64_t salt, const char* tag) {
  SynX s;
  s.rows = rows; s.tag = tag;
  std::mt19937_64 rng(seed + (uint64_t)n + salt);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<float> X((size_t)s.rows * n);
  for (auto& x : X) x = (float)nd(rng);
  const int cols[3] = {7, 100, 1000};
  const float scl[3] = {50.f, 20.f, 5.f};
  for (int t = 0; t < s.rows; ++t)
    for (int q = 0; q < 3; ++q) X[(size_t)t * n + cols[q]] *= scl[q];
  s.cn.assign(n, 0.0);
  for (int t = 0; t < s.rows; ++t)
    for (int j = 0; j < n; ++j) s.cn[j] += (double)X[(size_t)t * n + j] * X[(size_t)t * n + j];
  for (auto& c : s.cn) c = sqrt(c);
  {  // C = X^T X in FP64, accumulated in <=2048-row chunks (one chunk for 2048 rows: identical to job #22)
    Dev<double> dC((size_t)n * n);
    for (int t0 = 0; t0 < s.rows; t0 += 2048)
      gram_accumulate(dC.p, X.data() + (size_t)t0 * n, std::min(2048, s.rows - t0), n, t0 == 0);
    s.ctx = a4_ctx_from_dev(dC.p, n);
  }
  s.desc = std::string(tag) + ": Gaussian " + std::to_string(rows) + " x n, seed " + std::to_string(seed + (uint64_t)n + salt) +
           " (mt19937_64, seed+n+salt), columns 7,100,1000 scaled by 50,20,5";
  return s;
}
static A4Ctx identity_ctx(int n) {
  std::vector<double> I((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) I[(size_t)i * n + i] = 1.0;
  return a4_ctx_from_host(I, n);
}
static std::string eig_json(const SynX& sx) {
  const A4Ctx& c = sx.ctx;
  return JObj().s("x_variant", sx.tag).i("x_rows", sx.rows).i("n", c.n).i("C_rank", c.rank_c)
      .s("C_rank_tolerance", "eigenvalue > n * eps_fp64 * max(Lambda)").n("C_rank_tol_value", (double)c.n * kEpsD * c.lam_max)
      .i("C_eigenvalues_le_0", c.n_nonpos).i("C_negative_eigenvalues", c.n_negative).n("C_min_eig", c.w[0]).n("C_max_eig", c.lam_max).dump();
}

// A4 G-a2 (stored-bytes round trip) and G-a3 (before FP16 rounding) for one X variant and one budget.
// Mw = E Q Lambda_+^{1/2}; G-a3: ||Mw - U_k U_k^T Mw||_F against sqrt(sum of dropped sigma^2 of Mw) (Gram eigenvalues of the SVD).
static std::string a4_checks(const SynX& sx, const Svd& s4, const std::vector<double>& Mw, const std::vector<float>& E, int m, int n, int k,
                             int pct, const char* ty, const std::string& od, long long a1b, bool* ok_out) {
  ArmOut ar = build_A4(s4, E, k);
  char fn[600];
  snprintf(fn, sizeof fn, "%s/ga2_%s_p%02d_A4_%s.bin", od.c_str(), ty, pct, sx.tag.c_str());
  write_bytes(fn, ar.bytes);
  long long fl = file_size(fn);
  Decoded d = decode_arm(fn, m, n, "A4");
  long long mm = d.ok ? bit_mismatches(ar.res, d.res) : -1;
  bool ok2 = d.ok && mm == 0 && fl == ar.added();
  bool bytes_rule = ar.added() == a1b;
  std::vector<double> U = a4_uk64(s4, k);
  std::vector<double> UtM = hgemm_d(U, Mw, k, n, m, true, false);
  std::vector<double> P = hgemm_d(U, UtM, m, n, k, false, false);
  UtM.clear(); UtM.shrink_to_fit();
  for (size_t i = 0; i < P.size(); ++i) P[i] = Mw[i] - P[i];
  double a4_fro = fro(P), a4_opt = s4.tail_fro(k), mfro = fro(Mw);
  P.clear(); P.shrink_to_fit();
  std::vector<double> G = hgemm_d(U, U, k, k, m, true, false);
  double ortho = 0;
  for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) ortho = std::max(ortho, fabs(G[(size_t)i * k + j] - (i == j ? 1.0 : 0.0)));
  double g4 = fabs(a4_fro - a4_opt) / a4_opt;
  bool ok3 = g4 <= 1e-6;
  *ok_out = ok2 && bytes_rule && ok3;
  return JObj().r("eig", eig_json(sx))
      .r("G_a2", JObj().s("arm", "A4").i("added_bytes", ar.added()).i("file_length", fl).b("length_equal", fl == ar.added()).b("decode_ok", d.ok)
                     .s("decode_err", d.err).i("bit_mismatches", mm).b("pass", ok2).b("bytes_vs_A1_rule_ok", bytes_rule).i("a1_bytes", a1b)
                     .i("fp16_overflow", ar.overflow_total()).i("fp16_underflow_to_zero_or_subnormal", ar.underflow_total())
                     .r("fp16_per_factor", ar.h16_json()).dump())
      .r("G_a3", JObj().s("svd_gram_side", m <= n ? "left: eig(M M^T), m x m; U_k = eigenvectors" : "right: eig(M^T M), n x n; U_k = M V_k / sigma")
                     .n("A4_fro_M_minus_UkUkTM", a4_fro).n("A4_sqrt_sum_dropped_sigma2", a4_opt).n("A4_rel_gap", g4).n("fro_M", mfro)
                     .n("U_k_orthonormality_max_abs_UtU_minus_I", ortho).b("pass", ok3).dump())
      .dump();
}

static int run_ga(const std::map<std::string, std::string>& a) {
  init_handles();
  set_sources(need(a, "--srcs"));
  const uint64_t seed = a.count("--seed") ? strtoull(a.at("--seed").c_str(), nullptr, 10) : 20261022ULL;
  const int layer = a.count("--layer") ? atoi(a.at("--layer").c_str()) : 16;
  const std::string od = need(a, "--outdir");
  Shards sh = open_shards(need(a, "--st1"), need(a, "--st2"));
  const int pcts[3] = {2, 5, 10};
  // X variants per n: [0] as job #22 (2048 rows; note 2048 < n, so C is rank-deficient), [1] full rank (n + 1024 rows), [2] rank-deficient (n/2 rows)
  std::vector<SynX> xv;
  A4Ctx ident;
  int cur_n = -1, cur_ident_n = -1;
  std::vector<std::string> gc_rows, cells;
  bool all_ok = true;
  for (const char* ty : kTypes) {
    int m, n;
    shape_of(ty, &m, &n);
    for (int pct : pcts) {
      long long R = budget_R(pct, m, n);
      int k = rank_k(R, m, n);
      long long ab = a1_bytes(k, m, n), nE = n_entries(ab);
      bool largest = a1_bytes(k + 1, m, n) > R;
      bool c_rule = (4 + 6 * nE) <= ab && ab - (4 + 6 * nE) < 6 && ab <= R && largest;
      all_ok = all_ok && c_rule;
      gc_rows.push_back(JObj().s("type", ty).i("m", m).i("n", n).i("pct", pct).i("R", R).i("k", k).i("A1_bytes", ab).i("A4_bytes", ab)
                            .i("A3_entries", nE).i("A5_entries", nE).i("A3_bytes", 4 + 6 * nE).i("A5_bytes", 4 + 6 * nE)
                            .i("sparse_slack_bytes", ab - (4 + 6 * nE)).b("A1_le_R", ab <= R).b("k_is_largest_fitting", largest)
                            .b("G_c_B2_pass", c_rule).dump());
    }
  }
  std::vector<std::string> sx_desc;
  for (const char* ty : kTypes) {
    int m, n;
    shape_of(ty, &m, &n);
    printf("ga shape %s %dx%d\n", ty, m, n); fflush(stdout);
    if (n != cur_n) { xv.clear(); cur_n = n; }
    WSlice w = read_slice(sh, layer, ty);
    Fold f = nf4_fold(w.bf);
    std::vector<double> Ed(f.E.begin(), f.E.end());
    const int kmax = rank_k(budget_R(10, m, n), m, n);
    Svd s1 = svd_topk(Ed, m, n, kmax);
    std::vector<Svd> s4v(3);
    std::vector<std::vector<double>> Mwv(3);
    for (size_t xi = 0; xi < 3; ++xi) {
      if (xv.size() <= xi) {
        if (xi == 0) xv.push_back(make_synth(seed, n, 2048, 0ULL, "X_job22_2048rows"));
        else if (xi == 1) xv.push_back(make_synth(seed, n, n + 1024, 7000003ULL, "X_fullrank_n_plus_1024rows"));
        else xv.push_back(make_synth(seed, n, n / 2, 9000011ULL, "X_rankdef_n_over_2_rows"));
        sx_desc.push_back(JObj().i("n", n).s("desc", xv.back().desc).r("eig", eig_json(xv.back())).dump());
      }
      Mwv[xi] = a4_weighted(Ed, m, n, xv[xi].ctx);
      s4v[xi] = svd_topk(Mwv[xi], m, n, kmax);
      if (n > 4096) { xv[xi].ctx.Qt.clear(); xv[xi].ctx.Qt.shrink_to_fit(); }  // RAM: Qt is only needed to form Mw
    }
    SynX& sx = xv[0];
    if (cur_ident_n != n) { ident = identity_ctx(n); cur_ident_n = n; }
    const A4Ctx& cI = ident;
    for (int pct : pcts) {
      long long R = budget_R(pct, m, n);
      int k = rank_k(R, m, n);
      long long ab = a1_bytes(k, m, n), nE = n_entries(ab);
      ArmOut arms[3] = {build_A1(s1, k), build_sparse("A3", f.E, &sx.cn, (size_t)nE, m, n), build_sparse("A5", f.E, nullptr, (size_t)nE, m, n)};
      std::vector<std::string> a2;
      for (ArmOut& ar : arms) {
        char fn[600];
        snprintf(fn, sizeof fn, "%s/ga2_%s_p%02d_%s.bin", od.c_str(), ty, pct, ar.arm.c_str());
        write_bytes(fn, ar.bytes);
        long long fl = file_size(fn);
        Decoded d = decode_arm(fn, m, n, ar.arm);
        long long mm = d.ok ? bit_mismatches(ar.res, d.res) : -1;
        bool ok = d.ok && mm == 0 && fl == ar.added();
        bool bytes_rule = (ar.arm == "A1") ? ar.added() == ab : (ar.added() <= ab && ab - ar.added() < 6);
        all_ok = all_ok && ok && bytes_rule;
        a2.push_back(JObj().s("arm", ar.arm).i("added_bytes", ar.added()).i("file_length", fl).b("length_equal", fl == ar.added())
                         .b("decode_ok", d.ok).s("decode_err", d.err).i("bit_mismatches", mm).b("pass", ok)
                         .b("bytes_vs_A1_rule_ok", bytes_rule).i("a1_bytes", ab).i("sparse_slack_bytes_vs_A1", ab - ar.added())
                         .i("fp16_overflow", ar.overflow_total()).i("fp16_underflow_to_zero_or_subnormal", ar.underflow_total())
                         .r("fp16_per_factor", ar.h16_json()).dump());
      }
      // A4 on the three synthetic X variants (G-a2 + G-a3)
      std::vector<std::string> a4cells;
      for (size_t xi = 0; xi < xv.size(); ++xi) {
        bool ok4 = false;
        a4cells.push_back(a4_checks(xv[xi], s4v[xi], Mwv[xi], f.E, m, n, k, pct, ty, od, ab, &ok4));
        all_ok = all_ok && ok4;
      }
      // G-a3 for A1 (before FP16 rounding), unchanged
      std::vector<double> Usd((size_t)m * k), Vd((size_t)n * k);
      for (int r = 0; r < m; ++r) for (int j = 0; j < k; ++j) Usd[(size_t)r * k + j] = s1.U[(size_t)r * s1.kmax + j] * s1.sig[j];
      for (int r = 0; r < n; ++r) for (int j = 0; j < k; ++j) Vd[(size_t)r * k + j] = s1.V[(size_t)r * s1.kmax + j];
      std::vector<double> Ek = hgemm_d(Usd, Vd, m, n, k, false, true);
      for (size_t i = 0; i < Ek.size(); ++i) Ek[i] = Ed[i] - Ek[i];
      double a1_fro = fro(Ek), a1_opt = s1.tail_fro(k);
      Ek.clear(); Ek.shrink_to_fit();
      double g1 = fabs(a1_fro - a1_opt) / a1_opt;
      bool a3ok = g1 <= 1e-6;
      all_ok = all_ok && a3ok;
      std::string ga3 = JObj().n("A1_fro_E_minus_Ek", a1_fro).n("A1_sqrt_sum_dropped_sigma2", a1_opt).n("A1_rel_gap", g1).b("A1_pass", a3ok).dump();
      // G-a4 planted
      std::mt19937_64 rng(seed * 7919ULL + (uint64_t)(m * 31 + n) * 17ULL + (uint64_t)pct);
      std::normal_distribution<double> nd(0.0, 1.0);
      const int r0 = k / 2;
      std::vector<double> P((size_t)m * r0), Qm((size_t)r0 * n);
      for (auto& x : P) x = nd(rng);
      for (auto& x : Qm) x = nd(rng);
      std::vector<double> E0 = hgemm_d(P, Qm, m, n, r0, false, false);
      P.clear(); Qm.clear();
      auto write_dec = [&](ArmOut& ar, const char* tag) {
        char fn[600];
        snprintf(fn, sizeof fn, "%s/ga4_%s_p%02d_%s.bin", od.c_str(), ty, pct, tag);
        write_bytes(fn, ar.bytes);
        return std::string(fn);
      };
      JObj p4;
      {
        Svd sp = svd_topk(E0, m, n, k);
        ArmOut a1p = build_A1(sp, k);
        std::string fn = write_dec(a1p, "A1");
        Decoded d = decode_lowrank(fn, m, n, "A1", false), dbk = decode_lowrank(fn, m, n, "A1", true);
        double e = d.ok ? rel_err_f(d.res, E0) : 1e30, ebk = dbk.ok ? rel_err_f(dbk.res, E0) : 1e30;
        p4.n("A1_rel_fro_err", e).b("A1_pass", d.ok && e <= 2e-3).n("A1_broken_transposed_rel_fro_err", ebk).b("A1_broken_decode_ok", dbk.ok)
            .s("A1_broken_decode_err", dbk.err).b("A1_broken_fails_as_required", !(dbk.ok && ebk <= 2e-3)).r("A1_fp16_per_factor", a1p.h16_json());
        all_ok = all_ok && d.ok && e <= 2e-3 && !(dbk.ok && ebk <= 2e-3);
        std::vector<double> Mw = a4_weighted(E0, m, n, cI);
        Svd sp4 = svd_topk(Mw, m, n, k);
        Mw.clear(); Mw.shrink_to_fit();
        std::vector<float> E0f(E0.begin(), E0.end());
        ArmOut a4p = build_A4(sp4, E0f, k);
        fn = write_dec(a4p, "A4");
        d = decode_lowrank(fn, m, n, "A4", false);
        dbk = decode_lowrank(fn, m, n, "A4", true);
        e = d.ok ? rel_err_f(d.res, E0) : 1e30; ebk = dbk.ok ? rel_err_f(dbk.res, E0) : 1e30;
        p4.n("A4_C_identity_rel_fro_err", e).b("A4_pass", d.ok && e <= 2e-3).n("A4_broken_transposed_rel_fro_err_EXTRA", ebk)
            .b("A4_broken_transposed_fails_EXTRA", !(dbk.ok && ebk <= 2e-3)).r("A4_fp16_per_factor", a4p.h16_json());
        all_ok = all_ok && d.ok && e <= 2e-3 && !(dbk.ok && ebk <= 2e-3);
      }
      p4.i("planted_rank", r0);
      // planted sparse
      {
        const size_t np = (size_t)nE / 2, N = (size_t)m * n;
        std::set<uint32_t> planted;
        std::uniform_int_distribution<uint64_t> ud(0, N - 1);
        while (planted.size() < np) planted.insert((uint32_t)ud(rng));
        std::vector<float> Es(N, 0.f);
        std::vector<double> Esd(N, 0.0);
        for (uint32_t ix : planted) {
          double v = nd(rng);
          v = (v < 0 ? -1.0 : 1.0) * (0.5 + fabs(v));
          Es[ix] = (float)v; Esd[ix] = (double)Es[ix];
        }
        for (int arm = 0; arm < 2; ++arm) {
          const char* nm = arm == 0 ? "A3" : "A5";
          ArmOut ar = build_sparse(nm, Es, arm == 0 ? &sx.cn : nullptr, (size_t)nE, m, n);
          std::string fn = write_dec(ar, nm);
          auto check = [&](int shift, double* relerr, bool* setmatch, std::string* err) {
            Decoded d = decode_sparse(fn, m, n, shift);
            if (!d.ok) { *err = d.err; *relerr = 1e30; *setmatch = false; return false; }
            std::set<uint32_t> got;
            for (size_t i = 0; i < N; ++i) if (d.res[i] != 0.f) got.insert((uint32_t)i);
            *setmatch = (got == planted);
            *relerr = rel_err_f(d.res, Esd);
            return true;
          };
          double e, ebk; bool sm, smbk; std::string err, errbk;
          bool dok = check(0, &e, &sm, &err), dbk = check(1, &ebk, &smbk, &errbk);
          bool pass = dok && sm && e <= 2e-3, brokenpass = dbk && smbk && ebk <= 2e-3;
          std::string pre = nm;
          p4.b((pre + "_index_set_exact").c_str(), dok && sm).n((pre + "_rel_fro_err").c_str(), e).b((pre + "_pass").c_str(), pass)
              .b((pre + "_broken_offbyone_decode_ok").c_str(), dbk).s((pre + "_broken_offbyone_err").c_str(), errbk)
              .b((pre + "_broken_offbyone_index_set_exact").c_str(), dbk && smbk)
              .b((pre + "_broken_fails_as_required").c_str(), !brokenpass).r((pre + "_fp16_per_factor").c_str(), ar.h16_json());
          all_ok = all_ok && pass && !brokenpass;
        }
        p4.i("planted_nonzeros", (long long)np);
      }
      cells.push_back(JObj().s("type", ty).i("m", m).i("n", n).i("pct", pct).i("R", R).i("k", k).i("A1_bytes", ab).i("sparse_entries", nE)
                          .r("G_a2_A1_A3_A5_on_X_job22", jarr(a2)).r("G_a3_A1", ga3).r("A4_on_each_synthetic_X", jarr(a4cells, ",\n  "))
                          .r("G_a4", p4.dump()).dump());
      printf("ga cell %s p%d done\n", ty, pct); fflush(stdout);
    }
  }
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  write_text(need(a, "--out"),
             JObj().s("label", "PENDING Test Designer review").s("rules_sha256", "03f97724426223451907581c4d690e0471b9f67d8f746dc6f49a5e69adbd054a")
                 .s("what", "G-a2..G-a4 apparatus checks on the Amendment B-1 A4 storage; synthetic X only; E = real NF4 error of real layer weights (weights only, no X)")
                 .i("e_layer", layer).i("seed", (long long)seed).r("synthetic_x", jarr(sx_desc)).r("source_sha256", g_srcs_json)
                 .s("binary_sha256", g_bin_sha).s("gpu", prop.name).i("peak_rss_kb", ru.ru_maxrss).b("all_checks_pass", all_ok)
                 .r("G_c_arithmetic", jarr(gc_rows)).r("cells", jarr(cells)).dump() + "\n");
  printf("ga all_checks_pass=%d\n", all_ok ? 1 : 0);
  return all_ok ? 0 : 6;
}

// ---------- crank: rank(C) and floored-eigenvalue count of C = X_cal^T X_cal (logging only; C is not used for any arm here) ----------
static int run_crank(const std::map<std::string, std::string>& a) {
  init_handles();
  set_sources(need(a, "--srcs"));
  const int n = atoi(need(a, "--n").c_str());
  std::vector<int> lens;
  {
    std::vector<char> raw = read_all(need(a, "--seqlens"));
    raw.push_back(0);
    for (auto& s : split(raw.data(), ',')) lens.push_back(atoi(s.c_str()));
  }
  FILE* f = fopen(need(a, "--xcal").c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open xcal\n"); return 1; }
  Dev<double> dC((size_t)n * n);
  long long tokens = 0;
  bool first = true;
  std::vector<float> buf;
  for (int T : lens) {
    buf.resize((size_t)T * n);
    if (fread(buf.data(), 4, buf.size(), f) != buf.size()) { fprintf(stderr, "short xcal\n"); return 1; }
    gram_accumulate(dC.p, buf.data(), T, n, first);
    first = false;
    tokens += T;
  }
  char probe;
  if (fread(&probe, 1, 1, f) == 1) { fprintf(stderr, "xcal longer than seqlens\n"); return 1; }
  fclose(f);
  A4Ctx c = a4_ctx_from_dev(dC.p, n);
  double tr = 0; for (double x : c.w) tr += x;
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  write_text(need(a, "--out"),
             JObj().s("label", "PENDING Test Designer review (logging only; C not used for any arm; no floor, no flag: retired by B-1)").s("capture", need(a, "--label")).i("n", n)
                 .i("tokens", tokens).i("sequences", (long long)lens.size())
                 .s("rank_tolerance", "eigenvalue > n * eps_fp64 * max(Lambda)").n("rank_tol_value", (double)n * kEpsD * c.lam_max)
                 .i("rank_C", c.rank_c).n("max_Lambda", c.lam_max)
                 .i("eigenvalues_le_0", c.n_nonpos).i("negative_eigenvalues", c.n_negative).n("min_eigenvalue", c.w[0]).n("trace", tr)
                 .r("source_sha256", g_srcs_json).s("binary_sha256", g_bin_sha)
                 .i("peak_rss_kb", ru.ru_maxrss).dump() + "\n");
  printf("crank %s n=%d tokens=%lld rank=%d eig_le_0=%d\n", need(a, "--label").c_str(), n, tokens, c.rank_c, c.n_nonpos);
  return 0;
}

// ---------- measure: per-cell arms A0, A1, A4, A3, A5 (NOT RUN on real data in job #22) ----------
static uint16_t f2bf(float x) {
  uint32_t b; memcpy(&b, &x, 4);
  b += 0x7FFF + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
static int run_measure(const std::map<std::string, std::string>& a) {
  init_handles();
  set_sources(need(a, "--srcs"));
  const std::string type = need(a, "--type");
  const int layer = atoi(need(a, "--layer").c_str());
  const std::string note = a.count("--note") ? a.at("--note") : "";
  WSlice w;
  if (a.count("--synthetic-w")) {  // smoke only: synthetic Gaussian weights, no real projection
    int m, n;
    shape_of(type, &m, &n);
    if (a.count("--m")) m = atoi(a.at("--m").c_str());
    if (a.count("--n")) n = atoi(a.at("--n").c_str());
    std::mt19937_64 rng(strtoull(a.at("--synthetic-w").c_str(), nullptr, 10));
    std::normal_distribution<float> nd(0.f, 0.02f);
    w.m = m; w.n = n; w.bf.resize((size_t)m * n);
    for (auto& u : w.bf) u = f2bf(nd(rng));
    w.tensor = "SYNTHETIC"; w.blob = "SYNTHETIC"; w.sha = sha256_hex(w.bf.data(), w.bf.size() * 2);
  } else {
    Shards sh = open_shards(need(a, "--st1"), need(a, "--st2"));
    w = read_slice(sh, layer, type);
  }
  const int m = w.m, n = w.n;
  std::vector<int> pcts;
  for (auto& s : split(need(a, "--pcts"), ',')) pcts.push_back(atoi(s.c_str()));
  Fold f = nf4_fold(w.bf);
  // calibration X: column norms (A3) and C (A4), streamed in <=2048-row chunks
  std::vector<double> cn(n, 0.0);
  size_t calcount;
  std::vector<float> xcal = read_f32_file(need(a, "--xcal"), &calcount);
  if (calcount % n) { fprintf(stderr, "xcal not a multiple of n\n"); return 1; }
  const int Tcal = (int)(calcount / n);
  for (int t = 0; t < Tcal; ++t)
    for (int j = 0; j < n; ++j) cn[j] += (double)xcal[(size_t)t * n + j] * xcal[(size_t)t * n + j];
  std::vector<double> cn2 = cn;  // squared, for the FP5 outlier share
  for (auto& c : cn) c = sqrt(c);
  std::string cal_sha = sha256_hex(xcal.data(), calcount * 4);
  std::vector<std::string> arms = split(need(a, "--arms"), ',');
  auto want = [&](const char* x) { return std::find(arms.begin(), arms.end(), x) != arms.end(); };
  A4Ctx ctx;
  if (want("A4")) {
    Dev<double> dC((size_t)n * n);
    for (int t0 = 0; t0 < Tcal; t0 += 2048) {
      int rows = std::min(2048, Tcal - t0);
      gram_accumulate(dC.p, xcal.data() + (size_t)t0 * n, rows, n, t0 == 0);
    }
    ctx = a4_ctx_from_dev(dC.p, n);
  }
  double outlier_share = 0;  // FP5: share of ||X_cal||^2 in the top 1% of columns
  {
    std::vector<double> s2 = cn2;
    std::sort(s2.begin(), s2.end(), std::greater<double>());
    size_t top = (size_t)std::ceil(0.01 * n);
    double tot = 0, tp = 0;
    for (size_t j = 0; j < s2.size(); ++j) { tot += s2[j]; if (j < top) tp += s2[j]; }
    outlier_share = tot > 0 ? tp / tot : 0;
  }
  xcal.clear(); xcal.shrink_to_fit();
  // eval sets
  std::vector<std::string> evpaths = split(need(a, "--xeval"), ','), evids = split(need(a, "--set-ids"), ',');
  std::vector<std::vector<float>> Xe, Yref;
  std::vector<std::string> ev_sha;
  std::vector<int> ev_rows;
  for (auto& p : evpaths) {
    size_t cnt;
    Xe.push_back(read_f32_file(p, &cnt));
    ev_rows.push_back((int)(cnt / n));
    ev_sha.push_back(sha256_hex(Xe.back().data(), cnt * 4));
    Yref.push_back(hgemm_f(Xe.back(), f.Wf, ev_rows.back(), m, n, false, true));
  }
  auto eval_e = [&](const std::vector<float>& hat, size_t si) {
    std::vector<float> yh = hgemm_f(Xe[si], hat, ev_rows[si], m, n, false, true);
    double num = 0, den = 0;
    for (size_t i = 0; i < yh.size(); ++i) { double r = Yref[si][i], d = r - (double)yh[i]; num += d * d; den += r * r; }
    return sqrt(num) / sqrt(den);
  };
  auto wrmse = [&](const std::vector<float>& hat) {
    double s = 0;
    for (size_t i = 0; i < hat.size(); ++i) { double d = (double)f.Wf[i] - (double)hat[i]; s += d * d; }
    return sqrt(s / hat.size());
  };
  const int pmax = *std::max_element(pcts.begin(), pcts.end());
  const int kmax = rank_k(budget_R(pmax, m, n), m, n);
  Svd s1, s4;
  std::vector<double> Ed(f.E.begin(), f.E.end());
  if (want("A1") || want("A4")) s1 = svd_topk(Ed, m, n, kmax);
  if (want("A4")) {
    std::vector<double> Mw = a4_weighted(Ed, m, n, ctx);
    s4 = svd_topk(Mw, m, n, kmax);
  }
  Ed.clear(); Ed.shrink_to_fit();
  struct rusage ru0; getrusage(RUSAGE_SELF, &ru0);
  cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  std::vector<float> hat(f.deq.size());
  std::vector<double> e_a0(Xe.size());
  for (size_t si = 0; si < Xe.size(); ++si) e_a0[si] = eval_e(f.deq, si);
  for (int pct : pcts) {
    long long R = budget_R(pct, m, n);
    int k = rank_k(R, m, n);
    long long ab = a1_bytes(k, m, n), nE = n_entries(ab);
    std::vector<ArmOut> outs;
    if (want("A1")) outs.push_back(build_A1(s1, k));
    if (want("A4")) outs.push_back(build_A4(s4, f.E, k));
    if (want("A3")) outs.push_back(build_sparse("A3", f.E, &cn, (size_t)nE, m, n));
    if (want("A5")) outs.push_back(build_sparse("A5", f.E, nullptr, (size_t)nE, m, n));
    std::vector<std::vector<std::string>> per_set(Xe.size());
    for (size_t si = 0; si < Xe.size(); ++si)
      if (want("A0")) per_set[si].push_back(JObj().s("arm", "A0").n("e", e_a0[si]).i("added_bytes", 0).n("weight_rmse", wrmse(f.deq)).dump());
    for (ArmOut& ar : outs) {
      char fn[700];
      snprintf(fn, sizeof fn, "%s/L%d_%s_p%02d_%s.bin", need(a, "--decode-dir").c_str(), layer, type.c_str(), pct, ar.arm.c_str());
      write_bytes(fn, ar.bytes);
      Decoded d = decode_arm(fn, m, n, ar.arm);
      long long mm = d.ok ? bit_mismatches(ar.res, d.res) : -1;
      for (size_t i = 0; i < hat.size(); ++i) hat[i] = f.deq[i] + ar.res[i];
      double wr = wrmse(hat);
      for (size_t si = 0; si < Xe.size(); ++si)
        per_set[si].push_back(JObj().s("arm", ar.arm).n("e", eval_e(hat, si)).i("added_bytes", ar.added()).n("weight_rmse", wr).i("rank", ar.k)
                                  .i("n_entries", ar.entries).i("file_length", file_size(fn)).b("decode_ok", d.ok).i("decode_bit_mismatches", mm)
                                  .b("storage_roundtrip_pass", d.ok && mm == 0 && file_size(fn) == ar.added()).i("fp16_overflow", ar.overflow_total())
                                  .i("fp16_underflow_to_zero_or_subnormal", ar.underflow_total()).r("fp16_per_factor", ar.h16_json()).dump());
    }
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    for (size_t si = 0; si < Xe.size(); ++si) {
      JObj cell;
      if (!note.empty()) cell.b("smoke", true).s("note", note);
      cell.i("layer", layer).s("type", type).i("m", m).i("n", n).s("set_id", evids[si]).i("pct", pct).i("R", R).i("k", k).i("a1_bytes", ab)
          .i("a4_bytes", want("A4") ? ab : -1).i("sparse_entries", nE).i("sparse_bytes", 4 + 6 * nE).i("sparse_slack_bytes", ab - (4 + 6 * nE))
          .b("sparse_bytes_rule_B2_ok", ab - (4 + 6 * nE) >= 0 && ab - (4 + 6 * nE) < 6).b("a1_le_R", ab <= R)
          .b("k_is_largest_fitting", a1_bytes(k + 1, m, n) > R);
      if (want("A4"))
        cell.i("C_rank", ctx.rank_c).s("C_rank_tolerance", "eigenvalue > n * eps_fp64 * max(Lambda)").i("C_eigenvalues_le_0", ctx.n_nonpos)
            .i("C_negative_eigenvalues", ctx.n_negative).n("C_max_eig", ctx.lam_max);
      cell.n("fp5_top1pct_column_share_of_Xcal_sq", outlier_share).i("cal_tokens", Tcal).i("eval_tokens", ev_rows[si]).s("cal_x_sha256", cal_sha)
          .s("eval_x_sha256", ev_sha[si]).s("w_slice_sha256", w.sha).s("w_blob", w.blob).i("w_byte_begin", w.b0).i("w_byte_end", w.b1)
          .s("nf4_qweight_sha256", f.q_sha).s("nf4_absmax_sha256", f.absmax_sha).r("source_sha256", g_srcs_json).s("binary_sha256", g_bin_sha)
          .s("gpu", prop.name).i("peak_rss_kb", ru.ru_maxrss).i("nf4_blocksize", 64).r("arms", jarr(per_set[si]));
      char fn[700];
      snprintf(fn, sizeof fn, "%s/L%d_%s_p%02d_%s.json", need(a, "--out-dir").c_str(), layer, type.c_str(), pct, evids[si].c_str());
      write_text(fn, cell.dump() + "\n");
    }
    printf("measure L%d %s p%d done\n", layer, type.c_str(), pct); fflush(stdout);
  }
  (void)ru0;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: p7fold3 parity|ga1dump|ga|crank|measure [flags]\n");
    return 1;
  }
  std::string mode = argv[1];
  if (mode == "parity") {
    if (argc != 5) return 1;
    return run_parity(argv[2], argv[3], argv[4]);
  }
  auto args = parse_args(argc, argv, 2);
  if (mode == "ga1dump") return run_ga1dump(args);
  if (mode == "ga") return run_ga(args);
  if (mode == "crank") return run_crank(args);
  if (mode == "measure") return run_measure(args);
  fprintf(stderr, "unknown mode %s (A2 is retired)\n", mode.c_str());
  return 1;
}

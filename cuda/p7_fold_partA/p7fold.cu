// p7fold.cu -- P7 one-layer fold harness: NF4 fold (block 64, bit parity with nf4.py) and arms A0-A3 at byte budget R.
// Modes:
//   p7fold parity <in.f64> <out_packed.bin> <out_absmax.f64>
//   p7fold smoke --weights F --xcal F --xeval F --cal-ids F --eval-ids F --arms A0,A1,A2,A3
//                --set-id S --input-kind K --src PATH --out JSON [--smoke]
// Build: nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o p7fold src/p7fold.cu -lcublas -lcusolver -lcrypto
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <openssl/sha.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
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

static const int kRows = 3072;
static const int kCols = 3072;
static const int kN = kRows * kCols;
static const int kBlock = 64;
static const long long kBudget = 943718;  // R, bytes
static const int kTokCount = 2048;  // rules sha 510a9dd1: 2,048 calibration tokens, 2,048 per eval set
static const int kHidden = 3072;
static const char* kOprojName = "model.layers.0.self_attn.o_proj.weight";

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

// ---------- smoke mode ----------
struct OprojRef {
  std::string blob_path;
  long long begin = 0, end = 0;
};

// Reads the o_proj byte range only (seek + fread), checks dtype/shape against the spec.
static std::vector<uint16_t> read_oproj(const std::string& path, OprojRef* ref) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  uint64_t hlen = 0;
  if (fread(&hlen, 8, 1, f) != 1) exit(1);
  std::string hdr(hlen, '\0');
  if (fread(&hdr[0], 1, hlen, f) != hlen) exit(1);
  size_t p = hdr.find(std::string("\"") + kOprojName + "\"");
  if (p == std::string::npos) {
    fprintf(stderr, "tensor %s not in header\n", kOprojName);
    exit(1);
  }
  size_t obj_end = hdr.find('}', p);
  auto value_at = [&](const char* key) -> size_t {
    std::string k = std::string("\"") + key + "\"";
    size_t q = hdr.find(k, p);
    if (q == std::string::npos || q > obj_end) {
      fprintf(stderr, "key %s missing in tensor header\n", key);
      exit(1);
    }
    q += k.size();
    while (q < hdr.size() && (hdr[q] == ':' || hdr[q] == ' ')) ++q;
    return q;
  };
  size_t dq = value_at("dtype");
  std::string dtype = hdr.substr(dq + 1, hdr.find('"', dq + 1) - dq - 1);
  size_t sq = value_at("shape");
  long long s0 = atoll(hdr.c_str() + hdr.find('[', sq) + 1);
  long long s1 = atoll(hdr.c_str() + hdr.find(',', hdr.find('[', sq)) + 1);
  size_t oq = value_at("data_offsets");
  size_t ob = hdr.find('[', oq) + 1;
  long long b = atoll(hdr.c_str() + ob);
  long long e = atoll(hdr.c_str() + hdr.find(',', ob) + 1);
  if (dtype != "BF16" || s0 != kRows || s1 != kCols || e - b != 2LL * kN) {
    fprintf(stderr, "unexpected o_proj header: dtype=%s shape=[%lld,%lld] range=%lld\n", dtype.c_str(), s0, s1, e - b);
    exit(1);
  }
  std::vector<uint16_t> w(kN);
  fseek(f, (long)(8 + hlen + b), SEEK_SET);
  if (fread(w.data(), 2, kN, f) != (size_t)kN) {
    fprintf(stderr, "short read o_proj\n");
    exit(1);
  }
  fclose(f);
  ref->blob_path = path;
  ref->begin = 8 + (long long)hlen + b;
  ref->end = 8 + (long long)hlen + e;
  return w;
}

static std::vector<float> read_f32(const std::string& path, size_t count) {
  std::vector<char> raw = read_all(path);
  if (raw.size() != count * sizeof(float)) {
    fprintf(stderr, "%s: expected %zu floats, got %zu bytes\n", path.c_str(), count, raw.size());
    exit(1);
  }
  std::vector<float> v(count);
  memcpy(v.data(), raw.data(), raw.size());
  return v;
}

static std::vector<int32_t> read_i32(const std::string& path, size_t count) {
  std::vector<char> raw = read_all(path);
  if (raw.size() != count * sizeof(int32_t)) {
    fprintf(stderr, "%s: expected %zu int32, got %zu bytes\n", path.c_str(), count, raw.size());
    exit(1);
  }
  std::vector<int32_t> v(count);
  memcpy(v.data(), raw.data(), raw.size());
  return v;
}

static double rel_output_error(cublasHandle_t h, const float* dXe, const float* dYref, float* dHat, float* dYhat,
                               const std::vector<float>& hat_host, std::vector<float>& yref_host,
                               double* out_norm_ref) {
  (void)dYref;
  CUDA_OK(cudaMemcpy(dHat, hat_host.data(), (size_t)kN * sizeof(float), cudaMemcpyHostToDevice));
  gemm_nt(h, dXe, dHat, dYhat, kTokCount, kCols, kHidden);
  std::vector<float> yhat((size_t)kTokCount * kCols);
  CUDA_OK(cudaMemcpy(yhat.data(), dYhat, yhat.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < yhat.size(); ++i) {
    double r = yref_host[i], d = r - (double)yhat[i];
    num += d * d;
    den += r * r;
  }
  *out_norm_ref = sqrt(den);
  return sqrt(num) / sqrt(den);
}

struct ArmResult {
  std::string arm;
  double e = 0, weight_rmse = 0;
  long long added_bytes = 0;
  int rank = -1;           // A1
  long long n_entries = -1;  // A3
  double svd_resid_fro = -1, svd_opt_fro = -1;  // A1 diagnostics
};

// A1: rank-k truncated SVD of E. Left vectors: syevd on E E^T (double). Right vectors: V = E^T U / sigma.
// Stored factors: U*sigma as FP16 (n*k*2 B), V as FP16 (n*k*2 B), rank header 4 B.
static std::vector<float> a1_residual(cublasHandle_t hb, cusolverDnHandle_t hs, const std::vector<float>& E, int k,
                                      long long* added, double* svd_resid, double* svd_opt) {
  const int n = kRows;
  std::vector<double> Ed(E.begin(), E.end());
  double *dE, *dG, *dW, *dU, *dV, *work;
  CUDA_OK(cudaMalloc(&dE, (size_t)kN * sizeof(double)));
  CUDA_OK(cudaMalloc(&dG, (size_t)kN * sizeof(double)));
  CUDA_OK(cudaMalloc(&dW, n * sizeof(double)));
  CUDA_OK(cudaMalloc(&dU, (size_t)n * k * sizeof(double)));
  CUDA_OK(cudaMalloc(&dV, (size_t)n * k * sizeof(double)));
  CUDA_OK(cudaMemcpy(dE, Ed.data(), (size_t)kN * sizeof(double), cudaMemcpyHostToDevice));
  const double one = 1.0, zero = 0.0;
  // dE (column-major view) is E^T. A^T * A with A = E^T gives E * E^T.
  CUBLAS_OK(cublasDgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, dE, n, dE, n, &zero, dG, n));
  int lwork = 0;
  int* devInfo;
  CUDA_OK(cudaMalloc(&devInfo, sizeof(int)));
  CUSOLVER_OK(cusolverDnDsyevd_bufferSize(hs, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dG, n, dW, &lwork));
  CUDA_OK(cudaMalloc(&work, (size_t)lwork * sizeof(double)));
  CUSOLVER_OK(cusolverDnDsyevd(hs, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dG, n, dW, work, lwork, devInfo));
  int info = 0;
  CUDA_OK(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) {
    fprintf(stderr, "syevd info=%d\n", info);
    exit(2);
  }
  std::vector<double> G((size_t)kN), eig(n);
  CUDA_OK(cudaMemcpy(G.data(), dG, (size_t)kN * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(eig.data(), dW, n * sizeof(double), cudaMemcpyDeviceToHost));
  // eig ascending: bottom n-k are the discarded energies (sigma^2)
  double tail = 0.0;
  for (int j = 0; j < n - k; ++j) tail += eig[j];
  *svd_opt = sqrt(std::max(tail, 0.0));
  // top-k eigenvectors, descending sigma
  std::vector<double> Uk((size_t)n * k), sigma(k);
  for (int j = 0; j < k; ++j) {
    int col = n - 1 - j;
    sigma[j] = sqrt(std::max(eig[col], 0.0));
    memcpy(&Uk[(size_t)j * n], &G[(size_t)col * n], n * sizeof(double));
  }
  CUDA_OK(cudaMemcpy(dU, Uk.data(), (size_t)n * k * sizeof(double), cudaMemcpyHostToDevice));
  CUBLAS_OK(cublasDgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, dE, n, dU, n, &zero, dV, n));
  std::vector<double> V((size_t)n * k);
  CUDA_OK(cudaMemcpy(V.data(), dV, (size_t)n * k * sizeof(double), cudaMemcpyDeviceToHost));
  // Row-major factor buffers for the GEMM: Us[r][j] = fp16(U[r][j]*sigma_j), Vh[r][j] = fp16(V[r][j]/sigma_j)
  // (rules amendment 2026-10-22 ~2:55 AM EDT, sha256 97d34ab0...: both factors stored as FP16, 4*n bytes per rank)
  std::vector<float> Us((size_t)n * k), Vh((size_t)n * k);
  std::vector<uint16_t> Us_bits((size_t)n * k), Vh_bits((size_t)n * k);
  for (int r = 0; r < n; ++r) {
    for (int j = 0; j < k; ++j) {
      double u = Uk[(size_t)j * n + r];
      double s = sigma[j];
      double v = s > 0.0 ? V[(size_t)j * n + r] / s : 0.0;
      __half hu = __float2half((float)(u * s));
      memcpy(&Us_bits[(size_t)r * k + j], &hu, 2);
      Us[(size_t)r * k + j] = __half2float(hu);
      __half hv = __float2half((float)v);
      memcpy(&Vh_bits[(size_t)r * k + j], &hv, 2);
      Vh[(size_t)r * k + j] = __half2float(hv);
    }
  }
  // residual = Us * Vh^T on device (FP32 accumulation)
  float *dUs, *dVh, *dR;
  CUDA_OK(cudaMalloc(&dUs, Us.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dVh, Vh.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dR, (size_t)kN * sizeof(float)));
  CUDA_OK(cudaMemcpy(dUs, Us.data(), Us.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dVh, Vh.data(), Vh.size() * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> R((size_t)kN);
  gemm_nt(hb, dUs, dVh, dR, n, n, k);
  CUDA_OK(cudaMemcpy(R.data(), dR, (size_t)kN * sizeof(float), cudaMemcpyDeviceToHost));
  double rs = 0.0;
  for (int i = 0; i < kN; ++i) {
    double d = (double)E[i] - (double)R[i];
    rs += d * d;
  }
  *svd_resid = sqrt(rs);
  *added = (long long)Us_bits.size() * 2 + (long long)Vh_bits.size() * 2 + 4;  // Us fp16 + Vh fp16 + 4 B rank header
  cudaFree(dE); cudaFree(dG); cudaFree(dW); cudaFree(dU); cudaFree(dV); cudaFree(work); cudaFree(devInfo);
  cudaFree(dUs); cudaFree(dVh); cudaFree(dR);
  return R;
}

static int run_smoke(const std::map<std::string, std::string>& a) {
  const std::string weights = need(a, "--weights");
  std::vector<std::string> arms;
  {
    std::string s = need(a, "--arms"), t;
    for (char c : s + ",") {
      if (c == ',') {
        if (!t.empty()) arms.push_back(t);
        t.clear();
      } else {
        t += c;
      }
    }
  }
  CUDA_OK(cudaMemcpyToSymbol(kCB, kCodebook, sizeof(kCodebook)));

  // 1. o_proj byte range only
  OprojRef oref;
  std::vector<uint16_t> wbf = read_oproj(weights, &oref);
  std::string oproj_sha = sha256_hex(wbf.data(), wbf.size() * sizeof(uint16_t));
  std::vector<float> Wf(kN);
  for (int i = 0; i < kN; ++i) {
    uint32_t bits = (uint32_t)wbf[i] << 16;
    memcpy(&Wf[i], &bits, 4);
  }
  std::vector<double> Wd(Wf.begin(), Wf.end());

  // 2. inputs
  std::vector<float> xcal = read_f32(need(a, "--xcal"), (size_t)kTokCount * kHidden);
  std::vector<float> xeval = read_f32(need(a, "--xeval"), (size_t)kTokCount * kHidden);
  std::vector<char> cal_raw = read_all(need(a, "--cal-ids")), eval_raw = read_all(need(a, "--eval-ids"));
  std::vector<int32_t> cal_ids = read_i32(need(a, "--cal-ids"), kTokCount);
  std::vector<int32_t> eval_ids = read_i32(need(a, "--eval-ids"), kTokCount);
  std::string cal_sha = sha256_hex(cal_raw.data(), cal_raw.size());
  std::string eval_sha = sha256_hex(eval_raw.data(), eval_raw.size());
  // G-d: overlap between calibration and eval token windows (8-grams) and identical sequences
  std::set<std::array<int32_t, 8>> cal_grams;
  for (int i = 0; i + 8 <= kTokCount; ++i) {
    std::array<int32_t, 8> g;
    for (int t = 0; t < 8; ++t) g[t] = cal_ids[i + t];
    cal_grams.insert(g);
  }
  int shared_grams = 0;
  for (int i = 0; i + 8 <= kTokCount; ++i) {
    std::array<int32_t, 8> g;
    for (int t = 0; t < 8; ++t) g[t] = eval_ids[i + t];
    shared_grams += cal_grams.count(g) ? 1 : 0;
  }
  bool identical = (cal_sha == eval_sha);

  // 3. NF4 fold on device
  const int nb = (kN + kBlock - 1) / kBlock;
  const int nbytes = (kN + 1) / 2;
  double* dWd;
  double* dAbs;
  uint8_t* dQ;
  float *dDeq, *dWf;
  CUDA_OK(cudaMalloc(&dWd, (size_t)kN * sizeof(double)));
  CUDA_OK(cudaMalloc(&dAbs, nb * sizeof(double)));
  CUDA_OK(cudaMalloc(&dQ, nbytes));
  CUDA_OK(cudaMalloc(&dDeq, (size_t)kN * sizeof(float)));
  CUDA_OK(cudaMalloc(&dWf, (size_t)kN * sizeof(float)));
  CUDA_OK(cudaMemcpy(dWd, Wd.data(), (size_t)kN * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dWf, Wf.data(), (size_t)kN * sizeof(float), cudaMemcpyHostToDevice));
  nf4_quantize_device(dWd, kN, dAbs, dQ);
  dequant_kernel<<<(kN + 255) / 256, 256>>>(dQ, dAbs, kN, dDeq);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> deq(kN);
  std::vector<uint8_t> Q(nbytes);
  std::vector<double> absmax(nb);
  CUDA_OK(cudaMemcpy(deq.data(), dDeq, (size_t)kN * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(Q.data(), dQ, nbytes, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(absmax.data(), dAbs, nb * sizeof(double), cudaMemcpyDeviceToHost));
  std::string q_sha = sha256_hex(Q.data(), Q.size());
  std::string absmax_sha = sha256_hex(absmax.data(), absmax.size() * sizeof(double));

  // E = W - deQ(Q) in FP32
  std::vector<float> E(kN);
  for (int i = 0; i < kN; ++i) E[i] = Wf[i] - deq[i];

  // 4. metric setup: Y_ref = X_eval W^T
  cublasHandle_t hb;
  cusolverDnHandle_t hs;
  CUBLAS_OK(cublasCreate(&hb));
  CUBLAS_OK(cublasSetMathMode(hb, CUBLAS_PEDANTIC_MATH));
  CUSOLVER_OK(cusolverDnCreate(&hs));
  float *dXe, *dYref, *dYhat, *dHat;
  CUDA_OK(cudaMalloc(&dXe, xeval.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dYref, (size_t)kTokCount * kCols * sizeof(float)));
  CUDA_OK(cudaMalloc(&dYhat, (size_t)kTokCount * kCols * sizeof(float)));
  CUDA_OK(cudaMalloc(&dHat, (size_t)kN * sizeof(float)));
  CUDA_OK(cudaMemcpy(dXe, xeval.data(), xeval.size() * sizeof(float), cudaMemcpyHostToDevice));
  gemm_nt(hb, dXe, dWf, dYref, kTokCount, kCols, kHidden);
  std::vector<float> yref((size_t)kTokCount * kCols);
  CUDA_OK(cudaMemcpy(yref.data(), dYref, yref.size() * sizeof(float), cudaMemcpyDeviceToHost));

  auto weight_rmse = [&](const std::vector<float>& hat) {
    double s = 0.0;
    for (int i = 0; i < kN; ++i) {
      double d = Wd[i] - (double)hat[i];
      s += d * d;
    }
    return sqrt(s / kN);
  };

  std::vector<ArmResult> results;
  std::vector<float> hat(kN);
  double ynorm = 0.0;
  // G-c (rules sha256 ea47b830..., unchanged in 510a9dd1): budgeted arms <= R; A3 capped at A1's MEASURED added_bytes; A3 bytes must equal A1 bytes
  long long a1_added = -1, a3_added = -1, a3_entries = -1;
  for (const std::string& arm : arms) {
    ArmResult r;
    r.arm = arm;
    std::vector<float> res(kN, 0.f);
    if (arm == "A0") {
      hat = deq;
      r.added_bytes = 0;
    } else if (arm == "A2") {
      res = E;
      r.added_bytes = (long long)kN * 4;  // full FP32 residual, no budget (apparatus ceiling)
      for (int i = 0; i < kN; ++i) hat[i] = deq[i] + res[i];
    } else if (arm == "A1") {
      int k = (int)((kBudget - 4) / (4LL * kRows));  // 4 bytes per rank-row pair: Us fp16 (2n) + Vh fp16 (2n)
      res = a1_residual(hb, hs, E, k, &r.added_bytes, &r.svd_resid_fro, &r.svd_opt_fro);
      r.rank = k;
      // layout unit check: bytes = 4*n*k + 4, must not exceed R, and k must be the largest rank that fits
      if (r.added_bytes != 4LL * kRows * k + 4 || 4LL * kRows * (k + 1) + 4 <= kBudget) {
        fprintf(stderr, "A1 layout check failed: k=%d added=%lld R=%lld\n", k, r.added_bytes, (long long)kBudget);
        return 3;
      }
      a1_added = r.added_bytes;
      for (int i = 0; i < kN; ++i) hat[i] = deq[i] + res[i];
    } else if (arm == "A3") {
      // calibration column norms ||X_cal[:, j]||_2 (map built from calibration tokens only)
      std::vector<double> cn(kCols, 0.0);
      for (int t = 0; t < kTokCount; ++t)
        for (int j = 0; j < kCols; ++j) cn[j] += (double)xcal[(size_t)t * kCols + j] * xcal[(size_t)t * kCols + j];
      for (int j = 0; j < kCols; ++j) cn[j] = sqrt(cn[j]);
      std::vector<double> score(kN);
      for (int i = 0; i < kRows; ++i)
        for (int j = 0; j < kCols; ++j) score[(size_t)i * kCols + j] = fabs((double)E[(size_t)i * kCols + j]) * cn[j];
      if (a1_added < 0) {
        fprintf(stderr, "A3 needs A1 to run first: its cap comes from A1's measured added_bytes (G-c)\n");
        return 4;
      }
      size_t nE = (size_t)((a1_added - 4) / 6);  // 4 B index + 2 B FP16 value per entry, + 4 B count header; capped at A1's measured bytes
      std::vector<uint32_t> ord(kN);
      std::iota(ord.begin(), ord.end(), 0u);
      std::nth_element(ord.begin(), ord.begin() + nE, ord.end(), [&](uint32_t x, uint32_t y) {
        return score[x] > score[y] || (score[x] == score[y] && x < y);
      });
      ord.resize(nE);
      std::sort(ord.begin(), ord.end());
      std::vector<uint32_t> idx_store(ord);
      std::vector<uint16_t> val_store(nE);
      for (size_t t = 0; t < nE; ++t) {
        __half hv = __float2half(E[ord[t]]);
        memcpy(&val_store[t], &hv, 2);
        res[ord[t]] = __half2float(hv);
      }
      r.n_entries = (long long)nE;
      r.added_bytes = (long long)idx_store.size() * 4 + (long long)val_store.size() * 2 + 4;
      a3_added = r.added_bytes;
      a3_entries = (long long)nE;
      for (int i = 0; i < kN; ++i) hat[i] = deq[i] + res[i];
    } else {
      fprintf(stderr, "unknown arm %s\n", arm.c_str());
      return 1;
    }
    if (arm == "A0") {
      // nothing else
    }
    r.e = rel_output_error(hb, dXe, dYref, dHat, dYhat, hat, yref, &ynorm);
    r.weight_rmse = weight_rmse(hat);
    results.push_back(r);
    printf("arm=%s e=%.9g weight_rmse=%.9g added_bytes=%lld budget_ratio=%.6f\n", arm.c_str(), r.e, r.weight_rmse,
           r.added_bytes, (double)r.added_bytes / kBudget);
    fflush(stdout);
  }

  bool gc_miss = false;
  for (const ArmResult& r : results)
    if ((r.arm == "A1" || r.arm == "A3") && r.added_bytes > kBudget) gc_miss = true;
  if (a1_added >= 0 && a3_added >= 0 && a3_added != a1_added) gc_miss = true;
  const char* gc_status = gc_miss ? "INCONCLUSIVE: apparatus miss (G-c)" : "ok";
  printf("G-c: a1_added_bytes=%lld a3_added_bytes=%lld a3_entries=%lld status=%s\n", a1_added, a3_added, a3_entries, gc_status);

  // 5. JSON records (one object per arm, wrapped in an array)
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  const std::string set_id = need(a, "--set-id");
  const std::string kind = need(a, "--input-kind");
  const std::vector<char> src_bytes = read_all(need(a, "--src"));
  const std::string src_sha = sha256_hex(src_bytes.data(), src_bytes.size());
  const bool smoke = a.count("--smoke") > 0;
  FILE* out = fopen(need(a, "--out").c_str(), "w");
  if (!out) return 1;
  fprintf(out, "[\n");
  for (size_t i = 0; i < results.size(); ++i) {
    const ArmResult& r = results[i];
    fprintf(out, "  {\n");
    if (smoke) fprintf(out, "    \"smoke\": true,\n    \"note\": \"SMOKE: build check only, not a measurement, no verdict\",\n");
    fprintf(out, "    \"arm\": \"%s\",\n", r.arm.c_str());
    fprintf(out, "    \"set_id\": \"%s\",\n", set_id.c_str());
    fprintf(out, "    \"e\": %.12g,\n", r.e);
    fprintf(out, "    \"weight_rmse\": %.12g,\n", r.weight_rmse);
    fprintf(out, "    \"added_bytes\": %lld,\n", r.added_bytes);
    fprintf(out, "    \"R\": %lld,\n", kBudget);
    fprintf(out, "    \"budget_ratio\": %.9f,\n", (double)r.added_bytes / (double)kBudget);
    fprintf(out, "    \"rank\": %d,\n", r.rank);
    fprintf(out, "    \"n_entries\": %lld,\n", r.n_entries);
    fprintf(out, "    \"a1_added_bytes\": %lld,\n    \"a3_added_bytes\": %lld,\n    \"a3_entries\": %lld,\n", a1_added, a3_added, a3_entries);
    fprintf(out, "    \"gc_status\": \"%s\",\n", gc_status);
    fprintf(out, "    \"svd_resid_fro\": %.12g,\n    \"svd_optimal_fro\": %.12g,\n", r.svd_resid_fro, r.svd_opt_fro);
    fprintf(out, "    \"ref_out_fro_norm\": %.12g,\n", ynorm);
    fprintf(out, "    \"oproj_blob\": \"%s\",\n", oref.blob_path.c_str());
    fprintf(out, "    \"oproj_range_bytes\": [%lld, %lld],\n", oref.begin, oref.end);
    fprintf(out, "    \"oproj_sha256\": \"%s\",\n", oproj_sha.c_str());
    fprintf(out, "    \"nf4_qweight_sha256\": \"%s\",\n", q_sha.c_str());
    fprintf(out, "    \"nf4_absmax_sha256\": \"%s\",\n", absmax_sha.c_str());
    fprintf(out, "    \"cal_token_sha256\": \"%s\",\n    \"eval_token_sha256\": \"%s\",\n", cal_sha.c_str(), eval_sha.c_str());
    fprintf(out, "    \"cal_tokens\": %d,\n    \"eval_tokens\": %d,\n", kTokCount, kTokCount);
    fprintf(out, "    \"cal_eval_identical_sequences\": %s,\n", identical ? "true" : "false");
    fprintf(out, "    \"cal_eval_shared_8grams\": %d,\n", shared_grams);
    fprintf(out, "    \"input_kind\": \"%s\",\n", kind.c_str());
    fprintf(out, "    \"source_sha256\": \"%s\",\n", src_sha.c_str());
    fprintf(out, "    \"gpu\": \"%s\",\n", prop.name);
    fprintf(out, "    \"peak_rss_kb\": %ld,\n", ru.ru_maxrss);
    fprintf(out, "    \"nf4_blocksize\": 64,\n    \"svd\": \"cuSOLVER syevd fp64 on E E^T; V=E^T U/sigma\"\n");
    fprintf(out, "  }%s\n", i + 1 < results.size() ? "," : "");
  }
  fprintf(out, "]\n");
  fclose(out);
  printf("wrote %s (%zu arm records)\n", need(a, "--out").c_str(), results.size());
  cublasDestroy(hb);
  cusolverDnDestroy(hs);
  if (gc_miss) {
    fprintf(stderr, "G-c self-check failed: A3 bytes %lld vs A1 bytes %lld (R=%lld)\n", a3_added, a1_added, (long long)kBudget);
    return 5;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: p7fold parity <in> <out_packed> <out_absmax> | p7fold smoke [flags]\n");
    return 1;
  }
  std::string mode = argv[1];
  if (mode == "parity") {
    if (argc != 5) return 1;
    return run_parity(argv[2], argv[3], argv[4]);
  }
  if (mode == "smoke" || mode == "measure") {  // "measure": same path; JSON carries no smoke flag unless --smoke is passed
    return run_smoke(parse_args(argc, argv, 2));
  }
  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 1;
}

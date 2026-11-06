/* tests/test_arm0_bf16_full.cu — Full 32-Layer Arm (0) BF16 Evaluation on 2048 Tokens
 * Loads unquantized BF16 weights directly from model-00001-of-00002.safetensors and model-00002-of-00002.safetensors.
 * Executes on RTX 3060 via Tensor Core cuBLAS and online softmax causal attention.
 * Verifies Gate 1:
 *   - Top-1 argmax token agreement >= 95.0% vs HuggingFace (amended 2026-11-02, GATE_RULES.md C.1) Oracle
 *   - Perplexity within 1.0% of HuggingFace Oracle
 * Exit: 0 gate PASS (or INT8 report), 1 gate FAIL, 2 usage, 3 setup, I/O or CUDA error.
 */

#include "arm0_forward.h"
#include "act_map.h"
#include "nf4.h"
#include "../include/glint_htile.h"
#include "../include/glint_plug_streamer.h"
#include "../include/glint_cells.h"
#include <algorithm>
#include <vector>

#define INT8_BLOCK 64

__global__ void int8_block_dequant_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                          __nv_bfloat16* __restrict__ out, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = __float2bfloat16((float)q[i] * scale[i / INT8_BLOCK]);
}

static double g_nf4_se, g_nf4_ref2, g_nf4_max; static uint64_t g_nf4_n; static FILE *g_nf4_map; static int g_nf4_first;
static __nv_bfloat16* upload_linear(const SafeTensors *st1, const SafeTensors *st2, const SafeTensors *int8_pin,
                                    const char *name, size_t numel) {
    __nv_bfloat16 *d_w;
    CK(cudaMalloc(&d_w, numel * sizeof(__nv_bfloat16)));
    if (!int8_pin) {
        const TensorEntry *e = need_tensor(st1, st2, name, "BF16", numel * sizeof(__nv_bfloat16));
        if (getenv("ARM0_NF4")) { /* NF4 comparison point and S0-WMAP-NF4 */
            std::vector<uint16_t> rt(numel);
            double se = 0, ref2 = 0, mx = 0;
            nf4_roundtrip((const uint16_t *)e->data, rt.data(), numel, &se, &ref2, &mx);
            g_nf4_se += se; g_nf4_ref2 += ref2; g_nf4_n += numel; if (mx > g_nf4_max) g_nf4_max = mx;
            if (g_nf4_map) fprintf(g_nf4_map, "%s{\"name\":\"%s\",\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}", g_nf4_first++ ? "," : "", name, sqrt(se / numel), mx, sqrt(se / ref2));
            CK(cudaMemcpy(d_w, rt.data(), numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            return d_w;
        }
        CK(cudaMemcpy(d_w, e->data, numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        return d_w;
    }
    char scale_name[300];
    snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
    size_t scale_bytes = (numel + INT8_BLOCK - 1) / INT8_BLOCK * sizeof(float);
    const TensorEntry *eq = need_tensor(int8_pin, NULL, name, "I8", numel);
    const TensorEntry *es = need_tensor(int8_pin, NULL, scale_name, "F32", scale_bytes);
    int8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    int8_block_dequant_kernel<<<(unsigned)((numel + 255) / 256), 256>>>(d_q, d_s, d_w, numel);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    return d_w;
}

/* --glint PINDIR [--hole-only]: linears through the H-TILE kernel on a glint_pin_v1 (hole + plug + absmax). */
struct GlintLin { uint8_t *hole, *plug; float *am; };
static int g_glint = 0, g_hole_only = 0;
static GlintLin g_gl[N_LAYERS][ARM0_N_LINEARS];
static uint8_t *g_zero_plug = NULL;
/* --glint-stream PINDIR: hole resident, plugs staged in pinned host RAM and brought in one layer ahead through the
 * 2-slot ring (glint_plug_streamer). One layer's plug is its four linears back to back (QKV, O, GATE_UP, DOWN). */
static uint8_t *g_sf_plugk_h[N_LAYERS][ARM0_N_LINEARS], *g_sf_plugk_d[N_LAYERS][ARM0_N_LINEARS];
static uint8_t *g_sf_holek[N_LAYERS][ARM0_N_LINEARS]; static float *g_sf_amk[N_LAYERS][ARM0_N_LINEARS]; /* K-major hole and absmax in VRAM (ARM0_SELFETCH_OPT=1) */
static int g_sf_opt = 0;
static unsigned g_sf_step = 0; /* per-step salt for the random selection rule */
static int g_stream = 0, g_res_layers = 0; /* ARM0_STREAM_RESIDENT_LAYERS=N: the first N layers keep their plug in VRAM, the rest stream */
static size_t g_vram_base_free = 0;
static GlintPlugStreamer *g_streamer = NULL;
static cudaStream_t g_copy_stream = NULL;
static const uint8_t *g_plug_src[N_LAYERS][ARM0_N_LINEARS];
static uint64_t g_plug_off[N_LAYERS][ARM0_N_LINEARS], g_layer_bytes[N_LAYERS];
static const uint8_t *g_cur_dev_plug = NULL;
static void glint_upload(const SafeTensors *a, const SafeTensors *b, int l, int j) {
    size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
    char nm[256], pn[300], an[300];
    snprintf(nm, sizeof(nm), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
    snprintf(pn, sizeof(pn), "%s.glint_plug", nm); snprintf(an, sizeof(an), "%s.absmax", nm);
    size_t nb = (n + 1) / 2, ab = (n + 63) / 64 * sizeof(float);
    const TensorEntry *eh = need_tensor(a, b, nm, "U8", nb), *ep = need_tensor(a, b, pn, "U8", nb), *ea = need_tensor(a, b, an, "F32", ab);
    GlintLin &g = g_gl[l][j];
    CK(cudaMalloc(&g.hole, nb)); CK(cudaMemcpy(g.hole, eh->data, nb, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g.am, ab)); CK(cudaMemcpy(g.am, ea->data, ab, cudaMemcpyHostToDevice));
    if (getenv("ARM0_SELFETCH")) { /* K-major copy of the plug in pinned, mapped host memory */
        const int Nn = ARM0_LINEAR[j].n_out, Kk = ARM0_LINEAR[j].k_in; const uint8_t *src = (const uint8_t *)ep->data;
        uint8_t *hp = NULL, *dp = NULL; CK(cudaHostAlloc((void **)&hp, (size_t)Kk * (Nn / 2), cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostGetDevicePointer((void **)&dp, hp, 0)); memset(hp, 0, (size_t)Kk * (Nn / 2));
        for (int n = 0; n < Nn; n++) for (int c = 0; c < Kk; c++) {
            size_t jj = (size_t)n * Kk + c; uint8_t nib = (src[jj >> 1] >> ((jj & 1) * 4)) & 15;
            hp[(size_t)c * (Nn / 2) + (n >> 1)] |= (uint8_t)(nib << ((n & 1) * 4));
        }
        g_sf_plugk_h[l][j] = hp; g_sf_plugk_d[l][j] = dp;
        if (getenv("ARM0_SELFETCH_OPT")) { /* K-major hole [K][N/2] and absmax [K/64][N] on the device */
            const uint8_t *hsrc = (const uint8_t *)eh->data; const float *asrc = (const float *)ea->data;
            std::vector<uint8_t> hk((size_t)Kk * (Nn / 2), 0); std::vector<float> ak((size_t)(Kk / 64) * Nn);
            for (int n = 0; n < Nn; n++) {
                for (int c = 0; c < Kk; c++) {
                    size_t jj = (size_t)n * Kk + c; uint8_t nib = (hsrc[jj >> 1] >> ((jj & 1) * 4)) & 15;
                    hk[(size_t)c * (Nn / 2) + (n >> 1)] |= (uint8_t)(nib << ((n & 1) * 4));
                }
                for (int b = 0; b < Kk / 64; b++) ak[(size_t)b * Nn + n] = asrc[(size_t)n * (Kk / 64) + b];
            }
            CK(cudaMalloc(&g_sf_holek[l][j], hk.size())); CK(cudaMemcpy(g_sf_holek[l][j], hk.data(), hk.size(), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&g_sf_amk[l][j], ak.size() * 4)); CK(cudaMemcpy(g_sf_amk[l][j], ak.data(), ak.size() * 4, cudaMemcpyHostToDevice));
        }
    }
    if (g_stream && l >= g_res_layers) { /* plug stays on the host; the ring brings it in */
        g_plug_src[l][j] = (const uint8_t *)ep->data; g_plug_off[l][j] = g_layer_bytes[l]; g_layer_bytes[l] += nb; g.plug = NULL;
    } else if (g_hole_only) { if (!g_zero_plug) { CK(cudaMalloc(&g_zero_plug, 8192ull * 3072 / 2 * 2)); CK(cudaMemset(g_zero_plug, 0, 8192ull * 3072)); } g.plug = g_zero_plug; }
    else { CK(cudaMalloc(&g.plug, nb)); CK(cudaMemcpy(g.plug, ep->data, nb, cudaMemcpyHostToDevice)); }
}

/* --glint-demand PIN: demand-map harness, docs/p7_mode3_fold/DEMAND_MAP_SPEC_2026-11-04.md.
 * Each GLINT linear is y = x*Wparent^T + mask(x)*Wcorr^T with Wparent = PARENT[hole]*absmax and
 * Wcorr = (GLINT_CELLS[hole][plug] - PARENT[hole])*absmax, both decoded to bf16 for one linear at a time and run
 * through cuBLAS. mask(x) keeps, per token, the top fraction f of 64-channel input blocks by the chosen rule.
 * Env: ARM0_DEMAND_RULE=act|random|static|oracle, ARM0_DEMAND_F (default 1), ARM0_DEMAND_SEED (42),
 * ARM0_DEMAND_ONLY_LAYER (restrict one layer, others full), ARM0_DEMAND_STATIC_OUT/IN (calibration file),
 * ARM0_DEMAND_STATS (json: bytes fetched, per-layer churn and predictability). */
enum { DEM_MAXBLK = 128, DEM_CALIB_TOKENS = 256 };
static int g_demand = 0, g_rule = 0, g_only_layer = -1, g_dem_m = 0, g_mask_parent = 0, g_gran_ch = 0;
static float g_frac = 1.0f;
static unsigned g_seed = 42;
static const char *g_static_out = NULL;
static __constant__ float c_cells[16][16];
static __constant__ float c_cellsB[16][16]; /* band-B table of the tiered plug (spec Amendment 5): child cells merged to ARM0_TIER_BITS_B bits */
static __constant__ float c_parent[16];
static __nv_bfloat16 *g_wp = NULL, *g_wd = NULL, *g_xm = NULL;
static uint8_t *g_sel = NULL;
static float *g_colnorm = NULL, *g_static_dev = NULL;
static double *g_dstats = NULL;                                  /* [N_LAYERS][N_LINEARS][4]: sel, jaccard, recall, pairs */
static double g_host_sel[N_LAYERS][ARM0_N_LINEARS];              /* selected blocks summed over tokens for f in {0,1} */

__global__ void k_dem_decode(const uint8_t *hole, const uint8_t *plug, const float *am, __nv_bfloat16 *wp, __nv_bfloat16 *wd, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t hb = hole[i >> 1], pb = plug[i >> 1];
    int h = (i & 1) ? (hb >> 4) : (hb & 15), p = (i & 1) ? (pb >> 4) : (pb & 15);
    float a = am[i >> 6], par = c_parent[h];
    wp[i] = __float2bfloat16(par * a);
    wd[i] = __float2bfloat16((c_cells[h][p] - par) * a);
}
__global__ void k_dem_colnorm(const __nv_bfloat16 *wd, int n_out, int k_in, float *out) { /* grid = k_in/64, block 256 */
    int b = blockIdx.x; double s = 0;
    for (int r = threadIdx.x; r < n_out; r += blockDim.x) {
        const __nv_bfloat16 *row = wd + (size_t)r * k_in + (size_t)b * 64;
        for (int c = 0; c < 64; c++) { float v = __bfloat162float(row[c]); s += (double)v * v; }
    }
    __shared__ double sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) out[b] = (float)sqrt(sh[0]);
}
__device__ static float dem_hash01(unsigned seed, unsigned a, unsigned t, unsigned b) {
    unsigned h = seed * 0x9E3779B1u ^ a * 0x85EBCA6Bu ^ t * 0xC2B2AE35u ^ b * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h >> 8) * (1.0f / 16777216.0f);
}
/* grid = tokens, block 128. rule: 0 act, 1 random, 2 static, 3 oracle. Writes sel[t][blk] and the masked input xm. */
__global__ void k_dem_select(const __nv_bfloat16 *x, __nv_bfloat16 *xm, uint8_t *sel, int k_in, int nblk, int nsel, int rule,
                             unsigned seed, unsigned salt, const float *stat, const float *colnorm) {
    int t = blockIdx.x; __shared__ float sc[DEM_MAXBLK]; __shared__ uint8_t sl[DEM_MAXBLK];
    const __nv_bfloat16 *xr = x + (size_t)t * k_in;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        float s = 0.f;
        for (int c = 0; c < 64; c++) { float v = __bfloat162float(xr[b * 64 + c]); s += v * v; }
        float act = sqrtf(s);
        sc[b] = rule == 0 ? act : rule == 1 ? dem_hash01(seed, salt, (unsigned)t, (unsigned)b) : rule == 2 ? stat[b] : act * colnorm[b];
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        int rank = 0; float me = sc[b];
        for (int j = 0; j < nblk; j++) rank += (sc[j] > me) || (sc[j] == me && j < b);
        sl[b] = rank < nsel; sel[(size_t)t * nblk + b] = sl[b];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < k_in; c += blockDim.x) xm[(size_t)t * k_in + c] = sl[c >> 6] ? xr[c] : __float2bfloat16(0.f);
}
/* Channel granularity (spec: act and random only). Keeps the nsel input channels of each token with the largest key
 * (|x| for act, a hash for random), found by a binary search on the float bit pattern; ties may keep a few more.
 * grid = tokens, block 256. acc[0] += kept channels. */
__global__ void k_dem_select_ch(const __nv_bfloat16 *x, __nv_bfloat16 *xm, int k_in, int nsel, int rule, unsigned seed, unsigned salt, double *acc) {
    int t = blockIdx.x; const __nv_bfloat16 *xr = x + (size_t)t * k_in;
    __shared__ int cnt; __shared__ unsigned lo, hi;
    if (threadIdx.x == 0) { lo = 0; hi = 0x7f800000u; }
    __syncthreads();
    for (int it = 0; it < 32; it++) {
        if (lo >= hi) break;
        unsigned mid = lo + (hi - lo + 1) / 2;
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        int c = 0;
        for (int i = threadIdx.x; i < k_in; i += blockDim.x) {
            unsigned key = rule == 0 ? __float_as_uint(fabsf(__bfloat162float(xr[i]))) : __float_as_uint(dem_hash01(seed, salt, (unsigned)t, (unsigned)i));
            c += key >= mid;
        }
        atomicAdd(&cnt, c);
        __syncthreads();
        if (threadIdx.x == 0) { if (cnt >= nsel) lo = mid; else hi = mid - 1; }
        __syncthreads();
    }
    int kept = 0;
    for (int i = threadIdx.x; i < k_in; i += blockDim.x) {
        unsigned key = rule == 0 ? __float_as_uint(fabsf(__bfloat162float(xr[i]))) : __float_as_uint(dem_hash01(seed, salt, (unsigned)t, (unsigned)i));
        bool keep = key >= lo; kept += keep;
        xm[(size_t)t * k_in + i] = keep ? xr[i] : __float2bfloat16(0.f);
    }
    __shared__ int ktot; if (threadIdx.x == 0) ktot = 0; __syncthreads();
    atomicAdd(&ktot, kept); __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&acc[0], (double)ktot);
}
__global__ void k_dem_stats(const uint8_t *sel, int nblk, double *acc) { /* grid = tokens, block 128 */
    int t = blockIdx.x; __shared__ int cnt[3];
    if (threadIdx.x == 0) cnt[0] = cnt[1] = cnt[2] = 0;
    __syncthreads();
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        int a = sel[(size_t)t * nblk + b], p = t > 0 ? sel[(size_t)(t - 1) * nblk + b] : 0;
        atomicAdd(&cnt[0], a); if (t > 0) { atomicAdd(&cnt[1], a & p); atomicAdd(&cnt[2], a | p); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&acc[0], (double)cnt[0]);
        if (t > 0) { atomicAdd(&acc[1], cnt[2] ? (double)cnt[1] / cnt[2] : 1.0); atomicAdd(&acc[2], cnt[0] ? (double)cnt[1] / cnt[0] : 1.0); atomicAdd(&acc[3], 1.0); }
    }
}
__global__ void k_dem_static_acc(const __nv_bfloat16 *x, int m, int k_in, float *out) { /* grid = nblk, block 128: mean act norm over calibration tokens */
    int b = blockIdx.x; double s = 0; int mm = m < DEM_CALIB_TOKENS ? m : DEM_CALIB_TOKENS;
    for (int t = threadIdx.x; t < mm; t += blockDim.x) {
        double q = 0; for (int c = 0; c < 64; c++) { float v = __bfloat162float(x[(size_t)t * k_in + b * 64 + c]); q += (double)v * v; }
        s += sqrt(q);
    }
    __shared__ double sh[128]; sh[threadIdx.x] = s; __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) out[b] = (float)(sh[0] / mm);
}
/* GLINT_PLUG_BITS=b (spec Amendment 4): emulate a b-bit child code. The 2^(4-b) adjacent child indices that would share a b-bit code all take the
 * mean of their cell values (the merged bin's reconstruction level). b = 4 (default) is the real table. Bytes are counted analytically elsewhere. */
static void cells_bits_fixed(float out[16][16], int b) {
    memcpy(out, GLINT_CELLS, sizeof(float) * 256); if (b >= 4 || b < 1) return;
    const int bin = 1 << (4 - b);
    for (int h = 0; h < 16; h++) for (int s0 = 0; s0 < 16; s0 += bin) { double m = 0; for (int k = 0; k < bin; k++) m += out[h][s0 + k]; m /= bin; for (int k = 0; k < bin; k++) out[h][s0 + k] = (float)m; }
}
static void cells_for_bits(float out[16][16]) {
    memcpy(out, GLINT_CELLS, sizeof(float) * 256);
    if (const char *cf = getenv("ARM0_CELLS_FILE")) { /* spec Amendment 11b: a custom 16x16 child table (256 floats, text) replaces the built-in one */
        FILE *f = fopen(cf, "r"); if (!f) setup_fail("cannot open ARM0_CELLS_FILE");
        for (int i = 0; i < 256; i++) if (fscanf(f, "%f", &out[i >> 4][i & 15]) != 1) setup_fail("ARM0_CELLS_FILE needs 256 floats");
        fclose(f); printf("ARM0_CELLS_FILE=%s: custom child table\n", cf);
    }
    const char *e = getenv("GLINT_PLUG_BITS"); int b = e ? atoi(e) : 4; if (b >= 4 || b < 1) return;
    const int bin = 1 << (4 - b);
    for (int h = 0; h < 16; h++) for (int s0 = 0; s0 < 16; s0 += bin) {
        double m = 0; for (int k = 0; k < bin; k++) m += out[h][s0 + k];
        m /= bin; for (int k = 0; k < bin; k++) out[h][s0 + k] = (float)m;
    }
    printf("GLINT_PLUG_BITS=%d: child cells merged in bins of %d\n", b, bin);
}
static void demand_init() {
    { float tbl[16][16]; cells_for_bits(tbl); CK(cudaMemcpyToSymbol(c_cells, tbl, sizeof(float) * 256)); }
    CK(cudaMemcpyToSymbol(c_parent, NF4_CODE, sizeof(float) * 16));
    size_t maxn = 0; for (int j = 0; j < ARM0_N_LINEARS; j++) { size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in; if (n > maxn) maxn = n; }
    CK(cudaMalloc(&g_wp, maxn * 2)); CK(cudaMalloc(&g_wd, maxn * 2));
    CK(cudaMalloc(&g_xm, 2048ull * 8192 * 2)); CK(cudaMalloc(&g_sel, 2048ull * DEM_MAXBLK));
    CK(cudaMalloc(&g_colnorm, DEM_MAXBLK * 4));
    size_t sn = (size_t)N_LAYERS * ARM0_N_LINEARS * DEM_MAXBLK;
    CK(cudaMalloc(&g_static_dev, sn * 4)); CK(cudaMemset(g_static_dev, 0, sn * 4));
    CK(cudaMalloc(&g_dstats, (size_t)N_LAYERS * ARM0_N_LINEARS * 4 * 8)); CK(cudaMemset(g_dstats, 0, (size_t)N_LAYERS * ARM0_N_LINEARS * 4 * 8));
    const char *r = getenv("ARM0_DEMAND_RULE");
    g_rule = !r || !strcmp(r, "act") ? 0 : !strcmp(r, "random") ? 1 : !strcmp(r, "static") ? 2 : !strcmp(r, "oracle") ? 3 : -1;
    if (g_rule < 0) setup_fail("ARM0_DEMAND_RULE must be act|random|static|oracle");
    if (getenv("ARM0_DEMAND_F")) g_frac = (float)atof(getenv("ARM0_DEMAND_F"));
    if (getenv("ARM0_DEMAND_SEED")) g_seed = (unsigned)atoi(getenv("ARM0_DEMAND_SEED"));
    if (getenv("ARM0_DEMAND_ONLY_LAYER")) g_only_layer = atoi(getenv("ARM0_DEMAND_ONLY_LAYER"));
    g_static_out = getenv("ARM0_DEMAND_STATIC_OUT");
    g_mask_parent = getenv("ARM0_DEMAND_MASK_PARENT") && atoi(getenv("ARM0_DEMAND_MASK_PARENT")) == 1;
    g_gran_ch = getenv("ARM0_DEMAND_GRAN") && !strcmp(getenv("ARM0_DEMAND_GRAN"), "channel");
    if (g_gran_ch && g_rule != 0 && g_rule != 1) setup_fail("channel granularity supports rules act and random only");
    if (g_rule == 2) {
        const char *in = getenv("ARM0_DEMAND_STATIC_IN"); if (!in) setup_fail("rule static needs ARM0_DEMAND_STATIC_IN");
        FILE *f = fopen(in, "rb"); if (!f) setup_fail("cannot read %s", in);
        std::vector<float> h(sn); if (fread(h.data(), 4, sn, f) != sn) setup_fail("short read %s", in); fclose(f);
        CK(cudaMemcpy(g_static_dev, h.data(), sn * 4, cudaMemcpyHostToDevice));
    }
    printf("Demand map: rule=%d f=%.4f seed=%u only_layer=%d\n", g_rule, g_frac, g_seed, g_only_layer);
}
static void demand_linear(cublasHandle_t cb, int layer, int which, const __nv_bfloat16 *x, float *y, int m) {
    const int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in, nblk = k_in / 64; const size_t n = (size_t)n_out * k_in;
    g_dem_m = m; GlintLin &g = g_gl[layer][which];
    k_dem_decode<<<(unsigned)((n + 255) / 256), 256>>>(g.hole, g.plug, g.am, g_wp, g_wd, n);
    float one = 1.0f, zero = 0.0f;
    if (g_static_out) k_dem_static_acc<<<nblk, 128>>>(x, m, k_in, g_static_dev + ((size_t)layer * ARM0_N_LINEARS + which) * DEM_MAXBLK);
    float f = (g_only_layer >= 0 && layer != g_only_layer) ? 1.0f : g_frac;
    if (g_mask_parent && f <= 0.0f) setup_fail("ARM0_DEMAND_MASK_PARENT needs f > 0");
    const __nv_bfloat16 *xin = x;
    if (f >= 1.0f) g_host_sel[layer][which] += (double)m * (g_gran_ch ? k_in : nblk);
    else if (f > 0.0f && g_gran_ch) {
        int nsel = (int)ceil((double)f * k_in - 1e-6); if (nsel < 1) nsel = 1; if (nsel > k_in) nsel = k_in;
        k_dem_select_ch<<<m, 256>>>(x, g_xm, k_in, nsel, g_rule, g_seed, (unsigned)(layer * 8 + which), g_dstats + ((size_t)layer * ARM0_N_LINEARS + which) * 4);
        xin = g_xm;
    } else if (f > 0.0f) {
        int nsel = (int)ceil((double)f * nblk - 1e-6); if (nsel < 1) nsel = 1; if (nsel > nblk) nsel = nblk;
        if (g_rule == 3) k_dem_colnorm<<<nblk, 256>>>(g_wd, n_out, k_in, g_colnorm);
        const float *stat = g_static_dev + ((size_t)layer * ARM0_N_LINEARS + which) * DEM_MAXBLK;
        k_dem_select<<<m, 128>>>(x, g_xm, g_sel, k_in, nblk, nsel, g_rule, g_seed, (unsigned)(layer * 8 + which), stat, g_colnorm);
        k_dem_stats<<<m, 128>>>(g_sel, nblk, g_dstats + ((size_t)layer * ARM0_N_LINEARS + which) * 4);
        xin = g_xm;
    }
    /* ARM0_DEMAND_MASK_PARENT=1 (exploratory, not in the spec): the parent GEMM sees the masked input too, so low-activation
     * channels are skipped for the whole linear (a TEAL-style activation-sparsity test), not only for the plug correction. */
    const __nv_bfloat16 *xp = (g_mask_parent && f < 1.0f) ? xin : x;
    CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &one, g_wp, CUDA_R_16BF, k_in, xp, CUDA_R_16BF, k_in, &zero, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    if (f > 0.0f)
        CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &one, g_wd, CUDA_R_16BF, k_in, xin, CUDA_R_16BF, k_in, &one, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}
static void demand_report() {
    CK(cudaDeviceSynchronize());
    std::vector<double> h((size_t)N_LAYERS * ARM0_N_LINEARS * 4);
    CK(cudaMemcpy(h.data(), g_dstats, h.size() * 8, cudaMemcpyDeviceToHost));
    double sel_bytes = 0, all_bytes = 0; const int m = g_dem_m;
    const char *sp = getenv("ARM0_DEMAND_STATS"); FILE *js = sp ? fopen(sp, "w") : NULL;
    if (js) fprintf(js, "{\"rule\":%d,\"f\":%.6f,\"seed\":%u,\"only_layer\":%d,\"m\":%d,\"layers\":[", g_rule, g_frac, g_seed, g_only_layer, m);
    for (int l = 0; l < N_LAYERS; l++) {
        double jac = 0, rec = 0, pairs = 0, lsel = 0, lall = 0;
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            const double *a = &h[((size_t)l * ARM0_N_LINEARS + j) * 4]; int nblk = g_gran_ch ? ARM0_LINEAR[j].k_in : ARM0_LINEAR[j].k_in / 64;
            double pb = (double)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in / 2.0, frac = (a[0] + g_host_sel[l][j]) / ((double)m * nblk);
            sel_bytes += frac * pb; all_bytes += pb; lsel += frac * pb; lall += pb; jac += a[1]; rec += a[2]; pairs += a[3];
        }
        if (js) fprintf(js, "%s{\"layer\":%d,\"bytes_frac\":%.6f,\"jaccard_prev_token\":%.6f,\"recall_prev_token\":%.6f}", l ? "," : "", l, lsel / lall, pairs > 0 ? jac / pairs : -1.0, pairs > 0 ? rec / pairs : -1.0);
    }
    if (js) { fprintf(js, "],\"bytes_frac\":%.6f}\n", sel_bytes / all_bytes); fclose(js); }
    printf("DEMAND_STATS bytes_frac=%.6f\n", sel_bytes / all_bytes);
    if (g_static_out) {
        size_t sn = (size_t)N_LAYERS * ARM0_N_LINEARS * DEM_MAXBLK; std::vector<float> hs(sn);
        CK(cudaMemcpy(hs.data(), g_static_dev, sn * 4, cudaMemcpyDeviceToHost));
        FILE *f = fopen(g_static_out, "wb"); if (f) { fwrite(hs.data(), 4, sn, f); fclose(f); printf("static calibration written: %s\n", g_static_out); }
    }
}

/* ---- Decode path (glint #84): one token per step with a per-layer KV cache. Same arithmetic as the prefill kernels. ---- */
__global__ void k_dec_rope(const float *qkv, float *q, float *kc, float *vc, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM) return;
    int d = idx % HEAD_DIM, head = idx / HEAD_DIM;
    const float *qkv_row = qkv;
    int i = d % (ROTARY_DIM / 2);
    float inv_freq = 1.0f / powf(ROPE_THETA, (2.0f * (float)i) / (float)ROTARY_DIM);
    float angle = (float)pos * inv_freq;
    float cos_v = bf16r(cosf(angle) * ATTENTION_SCALING);
    float sin_v = bf16r(sinf(angle) * ATTENTION_SCALING);
    if (head < NUM_Q_HEADS + NUM_KV_HEADS) {
        int in_base = head * HEAD_DIM; /* q heads then k heads are contiguous in the fused row */
        float val;
        if (d < ROTARY_DIM / 2) {
            float x0 = bf16r(qkv_row[in_base + d]), x1 = bf16r(qkv_row[in_base + d + ROTARY_DIM / 2]);
            val = bf16r(bf16r(x0 * cos_v) - bf16r(x1 * sin_v));
        } else if (d < ROTARY_DIM) {
            float x0 = bf16r(qkv_row[in_base + d - ROTARY_DIM / 2]), x1 = bf16r(qkv_row[in_base + d]);
            val = bf16r(bf16r(x1 * cos_v) + bf16r(x0 * sin_v));
        } else {
            val = bf16r(qkv_row[in_base + d]);
        }
        if (head < NUM_Q_HEADS) q[head * HEAD_DIM + d] = val;
        else kc[(size_t)pos * NUM_KV_HEADS * HEAD_DIM + (head - NUM_Q_HEADS) * HEAD_DIM + d] = val;
    } else {
        int h = head - (NUM_Q_HEADS + NUM_KV_HEADS);
        int in_base = (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM + h * HEAD_DIM;
        vc[(size_t)pos * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d] = bf16r(qkv_row[in_base + d]);
    }
}
__global__ void k_dec_attn(const float *q, const float *k, const float *v, float *out, __nv_bfloat16 *out_bf16, int npos) {
    int hq = blockIdx.x; if (hq >= NUM_Q_HEADS) return;
    int hkv = hq / 3, d = threadIdx.x;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float q_val = q[hq * HEAD_DIM + d] * scale;
    float m_val = -1e30f, d_val = 0.0f, acc = 0.0f;
    __shared__ float s_red[128];
    for (int tk = 0; tk < npos; tk++) {
        const float *k_row = k + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        float prod = q_val * k_row[d];
        s_red[d] = prod; __syncthreads();
        if (d < 64) s_red[d] += s_red[d + 64];
        __syncthreads();
        if (d < 32) {
            float b = s_red[d] + s_red[d + 32];
            for (int offset = 16; offset > 0; offset /= 2) b += __shfl_down_sync(0xffffffff, b, offset);
            s_red[d] = b;
        }
        __syncthreads();
        float score = s_red[0];
        float m_new = fmaxf(m_val, score), alpha = expf(m_val - m_new), beta = expf(score - m_new);
        d_val = d_val * alpha + beta; m_val = m_new;
        const float *v_row = v + ((size_t)tk * NUM_KV_HEADS + hkv) * HEAD_DIM;
        acc = acc * alpha + bf16r(beta) * v_row[d];
        __syncthreads();
    }
    float final_out = (d_val > 0.0f) ? (acc / d_val) : 0.0f;
    out[hq * HEAD_DIM + d] = final_out;
    if (out_bf16) out_bf16[hq * HEAD_DIM + d] = __float2bfloat16(final_out);
}
struct DecBufs {
    float *x, *h, *qkv, *q, *attn, *proj, *gu, *act, *down, *logits, *loss; __nv_bfloat16 *hb, *attnb, *actb, *xfb;
    float *kc[N_LAYERS], *vc[N_LAYERS]; int32_t *tgt, *top1;
};
static void dec_alloc(DecBufs &b) {
    CK(cudaMalloc(&b.x, HIDDEN_DIM * 4)); CK(cudaMalloc(&b.h, HIDDEN_DIM * 4)); CK(cudaMalloc(&b.qkv, 5120 * 4));
    CK(cudaMalloc(&b.q, NUM_Q_HEADS * HEAD_DIM * 4)); CK(cudaMalloc(&b.attn, NUM_Q_HEADS * HEAD_DIM * 4));
    CK(cudaMalloc(&b.proj, HIDDEN_DIM * 4)); CK(cudaMalloc(&b.gu, 2 * INTERMEDIATE_DIM * 4)); CK(cudaMalloc(&b.act, INTERMEDIATE_DIM * 4));
    CK(cudaMalloc(&b.down, HIDDEN_DIM * 4)); CK(cudaMalloc(&b.logits, (size_t)VOCAB_SIZE * 4)); CK(cudaMalloc(&b.loss, 4));
    CK(cudaMalloc(&b.hb, HIDDEN_DIM * 2)); CK(cudaMalloc(&b.attnb, NUM_Q_HEADS * HEAD_DIM * 2));
    CK(cudaMalloc(&b.actb, INTERMEDIATE_DIM * 2)); CK(cudaMalloc(&b.xfb, HIDDEN_DIM * 2));
    CK(cudaMalloc(&b.tgt, 4)); CK(cudaMalloc(&b.top1, 4));
    for (int l = 0; l < N_LAYERS; l++) { CK(cudaMalloc(&b.kc[l], 2048ull * NUM_KV_HEADS * HEAD_DIM * 4)); CK(cudaMalloc(&b.vc[l], 2048ull * NUM_KV_HEADS * HEAD_DIM * 4)); }
}
/* One token at position pos (the KV cache holds positions 0..pos-1). Returns the argmax and the CE loss against `target`. */
static void dec_step(const Arm0Model *model, DecBufs &b, int token, int pos, int target, Arm0LinearFn linear, void *ctx,
                     cublasHandle_t cb, int32_t *top1, float *loss) {
    g_sf_step = (unsigned)pos * 7919u + (unsigned)token;
    if (g_stream && g_streamer && glint_plug_streamer_start(g_streamer) != 0) setup_fail("streamer start failed (decode step %d)", pos); /* each token is one pass over the layers */
    float hx[HIDDEN_DIM];
    for (int d = 0; d < HIDDEN_DIM; d++) hx[d] = __bfloat162float(model->h_embed[(size_t)token * HIDDEN_DIM + d]);
    CK(cudaMemcpy(b.x, hx, sizeof(hx), cudaMemcpyHostToDevice));
    for (int l = 0; l < N_LAYERS; l++) {
        rms_norm_kernel<<<1, 128>>>(b.x, model->d_input_norm[l], b.h, b.hb, 1, HIDDEN_DIM, RMS_EPS);
        linear(ctx, l, ARM0_QKV, b.hb, b.qkv, 1);
        k_dec_rope<<<((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM + 255) / 256, 256>>>(b.qkv, b.q, b.kc[l], b.vc[l], pos);
        k_dec_attn<<<NUM_Q_HEADS, 128>>>(b.q, b.kc[l], b.vc[l], b.attn, b.attnb, pos + 1);
        linear(ctx, l, ARM0_O, b.attnb, b.proj, 1);
        residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256>>>(b.x, b.proj, HIDDEN_DIM);
        rms_norm_kernel<<<1, 128>>>(b.x, model->d_post_norm[l], b.h, b.hb, 1, HIDDEN_DIM, RMS_EPS);
        linear(ctx, l, ARM0_GATE_UP, b.hb, b.gu, 1);
        swiglu_kernel<<<(INTERMEDIATE_DIM + 255) / 256, 256>>>(b.gu, b.act, b.actb, 1);
        linear(ctx, l, ARM0_DOWN, b.actb, b.down, 1);
        residual_add_kernel<<<(HIDDEN_DIM + 255) / 256, 256>>>(b.x, b.down, HIDDEN_DIM);
    }
    rms_norm_kernel<<<1, 128>>>(b.x, model->d_final_norm, b.h, b.xfb, 1, HIDDEN_DIM, RMS_EPS);
    float one = 1.0f, zero = 0.0f;
    CKB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, 1, HIDDEN_DIM, &one, model->d_embed, CUDA_R_16BF, HIDDEN_DIM,
                     b.xfb, CUDA_R_16BF, HIDDEN_DIM, &zero, b.logits, CUDA_R_32F, VOCAB_SIZE, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    int32_t tg = target; CK(cudaMemcpy(b.tgt, &tg, 4, cudaMemcpyHostToDevice));
    compute_ce_and_argmax_kernel<<<1, 256>>>(b.logits, b.tgt, b.loss, b.top1, 1, VOCAB_SIZE);
    CK(cudaGetLastError());
    CK(cudaMemcpy(top1, b.top1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(loss, b.loss, 4, cudaMemcpyDeviceToHost));
}

/* ---- GLINT decode GEMV (glint #89): y[n] = sum_k GLINT_CELLS[hole][plug] * absmax * x[k] for one token (M = 1). One warp per output row;
 * each lane reads 16 B of hole and 16 B of plug (32 weights) per step, so a warp's loads are contiguous. The 16x16 table is in shared memory
 * (constant memory would serialize on the divergent nibble index). f32 accumulation. parent_only decodes PARENT[hole] (the fold-only arm). ---- */
static int g_gemv = 0, g_gemv_parent = 0;
__global__ void k_glint_gemv(const uint8_t *hole, const uint8_t *plug, const float *am, const __nv_bfloat16 *x, float *y, int N, int K, int parent_only) {
    extern __shared__ float sm[];
    float *s_cells = sm, *s_x = sm + 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_cells[i] = parent_only ? c_parent[i >> 4] : c_cells[i >> 4][i & 15];
    for (int k = threadIdx.x; k < K; k += blockDim.x) s_x[k] = __bfloat162float(x[k]);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const uint4 *hp = (const uint4 *)(hole + ((size_t)row * K) / 2), *pp = (const uint4 *)(plug + ((size_t)row * K) / 2);
    float acc = 0.f;
    for (int c = lane; c < K / 32; c += 32) {
        uint4 hv = hp[c], pv = pp[c];
        float a = am[((size_t)row * K + (size_t)c * 32) / 64];
        const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w}, pw[4] = {pv.x, pv.y, pv.z, pv.w};
        const float *xs = s_x + c * 32;
#pragma unroll
        for (int w = 0; w < 4; w++)
#pragma unroll
            for (int b = 0; b < 8; b++) {
                int h = (hw[w] >> (4 * b)) & 15, p = (pw[w] >> (4 * b)) & 15;
                acc += (s_cells[(h << 4) | p] * a) * xs[w * 8 + b];
            }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = acc;
}
static void gemv_init() {
    { float tbl[16][16]; cells_for_bits(tbl); CK(cudaMemcpyToSymbol(c_cells, tbl, sizeof(float) * 256)); }
    CK(cudaMemcpyToSymbol(c_parent, NF4_CODE, sizeof(float) * 16));
    const char *po = getenv("GLINT_PARENT_ONLY"); g_gemv_parent = po && po[0] == '1';
}
static void gemv_launch(const uint8_t *hole, const uint8_t *plug, const float *am, const __nv_bfloat16 *x, float *y, int N, int K) {
    int blocks = (N + 3) / 4; size_t smem = (256 + (size_t)K) * 4;
    k_glint_gemv<<<blocks, 128, smem>>>(hole, plug, am, x, y, N, K, g_gemv_parent);
    CK(cudaGetLastError());
}

/* ARM0_GEMV_CHECK=1: for every M = 1 linear also run H-TILE and a double-precision reference on the same input, and record the largest and
 * mean error of the GEMV and of H-TILE against the double reference, and of the GEMV against H-TILE, each relative to the RMS of the reference. */
static int g_gemv_check = 0; static long g_gc_n = 0;
static double g_gc_max_gh = 0, g_gc_sum_gh = 0, g_gc_max_gr = 0, g_gc_sum_gr = 0, g_gc_max_hr = 0, g_gc_sum_hr = 0;
static float *g_gc_y2 = NULL; static double *g_gc_yr = NULL;
__global__ void k_gemv_ref(const uint8_t *hole, const uint8_t *plug, const float *am, const __nv_bfloat16 *x, double *y, int N, int K, int parent_only) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; if (row >= N) return;
    double acc = 0;
    for (int k = 0; k < K; k++) {
        size_t j = (size_t)row * K + k; uint8_t hb = hole[j >> 1], pb = plug[j >> 1];
        int h = (j & 1) ? (hb >> 4) : (hb & 15), p = (j & 1) ? (pb >> 4) : (pb & 15);
        float w = (parent_only ? c_parent[h] : c_cells[h][p]) * am[j >> 6];
        acc += (double)w * (double)__bfloat162float(x[k]);
    }
    y[row] = acc;
}
static void gemv_check(const uint8_t *hole, const uint8_t *plug, const float *am, const __nv_bfloat16 *x, const float *y, int N, int K, int layer, int which) {
    if (!g_gc_y2) { CK(cudaMalloc(&g_gc_y2, 16384 * 4)); CK(cudaMalloc(&g_gc_yr, 16384 * 8)); }
    if (launch_glint_htile_s3_f32(hole, plug, am, x, g_gc_y2, 1, N, K, 64) != 0) setup_fail("htile check launch failed");
    k_gemv_ref<<<(N + 127) / 128, 128>>>(hole, plug, am, x, g_gc_yr, N, K, g_gemv_parent);
    std::vector<float> a(N), b(N); std::vector<double> r(N);
    CK(cudaMemcpy(a.data(), y, N * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), g_gc_y2, N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.data(), g_gc_yr, N * 8, cudaMemcpyDeviceToHost));
    double ss = 0, mgh = 0, mgr = 0, mhr = 0;
    for (int i = 0; i < N; i++) {
        ss += r[i] * r[i];
        mgh = fmax(mgh, fabs((double)a[i] - b[i])); mgr = fmax(mgr, fabs((double)a[i] - r[i])); mhr = fmax(mhr, fabs((double)b[i] - r[i]));
    }
    double rms = sqrt(ss / N); if (rms <= 0) return;
    g_gc_max_gh = fmax(g_gc_max_gh, mgh / rms); g_gc_sum_gh += mgh / rms;
    g_gc_max_gr = fmax(g_gc_max_gr, mgr / rms); g_gc_sum_gr += mgr / rms;
    g_gc_max_hr = fmax(g_gc_max_hr, mhr / rms); g_gc_sum_hr += mhr / rms; g_gc_n++;
}

/* ---- Selected-channel fetch decode (glint #86). For one token (M = 1) a linear is  y = parent_gemv(hole, x) + correction over the top-f
 * input channels of x by |x|. The plug of a selected channel is read from a K-major copy (channel c holds its N plug nibbles contiguously)
 * in pinned host memory, by the kernel itself over PCIe (zero-copy gather), not staged through VRAM. The hole stays row-major in VRAM.
 * ARM0_SELFETCH=f (needs --glint-demand for a fast prefill; f in (0,1); 1 = full model, 0 = fold only), ARM0_SELFETCH_STATS=1 counts bytes,
 * ARM0_SELFETCH_CHECK=1 compares each result with a double-precision reference built from the same selection. ---- */
static double g_sf_frac = -1; static int g_sf_stats = 0, g_sf_check = 0, g_sf_rule = 0;
static int g_sf_hi[N_LAYERS]; static double g_sf_hi_f = -1;
static int *g_sf_idx = NULL, *g_sf_cnt = NULL; static double g_sf_bytes = 0, g_sf_total = 0; static long g_sf_calls = 0;
static double g_sfc_max = 0, g_sfc_sum = 0; static long g_sfc_n = 0; static double *g_sfc_yr = NULL; static float *g_sfc_y = NULL;
__device__ static unsigned sf_key(const __nv_bfloat16 *x, int i, int rule, unsigned salt) {
    if (rule == 0) return __float_as_uint(fabsf(__bfloat162float(x[i])));
    unsigned h = salt * 0x9E3779B1u ^ (unsigned)i * 0x27D4EB2Fu; h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h >> 2; /* below 0x3fffffff, a valid float-bit threshold range */
}
__global__ void k_sf_select(const __nv_bfloat16 *x, int K, int nsel, int *idx, int *cnt_out, int rule, unsigned salt) { /* one block of 1024 threads */
    __shared__ int cnt; __shared__ unsigned lo, hi; __shared__ int kept;
    if (threadIdx.x == 0) { lo = 0; hi = 0x7f800000u; kept = 0; }
    __syncthreads();
    for (int it = 0; it < 32; it++) {
        if (lo >= hi) break;
        unsigned mid = lo + (hi - lo + 1) / 2;
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        int c = 0; for (int i = threadIdx.x; i < K; i += blockDim.x) c += sf_key(x, i, rule, salt) >= mid;
        atomicAdd(&cnt, c); __syncthreads();
        if (threadIdx.x == 0) { if (cnt >= nsel) lo = mid; else hi = mid - 1; }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        if (sf_key(x, i, rule, salt) >= lo) { int p = atomicAdd(&kept, 1); idx[p] = i; }
    __syncthreads();
    if (threadIdx.x == 0) *cnt_out = kept;
}
__global__ void k_sf_corr(const uint8_t *hole, const float *am, const uint8_t *plugk, const __nv_bfloat16 *x, const int *idx, const int *cnt_p,
                          float *y, int N, int K) {
    __shared__ float s_d[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_d[i] = c_cells[i >> 4][i & 15] - c_parent[i >> 4];
    __syncthreads();
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    const int cnt = *cnt_p; float acc = 0.f; const size_t half = (size_t)N / 2;
    for (int i = 0; i < cnt; i++) {
        int c = idx[i]; size_t j = (size_t)n * K + c;
        uint8_t hb = hole[j >> 1], pb = plugk[(size_t)c * half + (n >> 1)];
        int h = (j & 1) ? (hb >> 4) : (hb & 15), p = (n & 1) ? (pb >> 4) : (pb & 15);
        acc += (s_d[(h << 4) | p] * am[j >> 6]) * __bfloat162float(x[c]);
    }
    y[n] += acc;
}
__global__ void k_sf_ref(const uint8_t *hole, const uint8_t *plug, const float *am, const __nv_bfloat16 *x, const int *idx, const int *cnt_p, double *y, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    double acc = 0; const int cnt = *cnt_p;
    for (int k = 0; k < K; k++) {
        size_t j = (size_t)n * K + k; uint8_t hb = hole[j >> 1]; int h = (j & 1) ? (hb >> 4) : (hb & 15);
        acc += (double)(c_parent[h] * am[j >> 6]) * (double)__bfloat162float(x[k]);
    }
    for (int i = 0; i < cnt; i++) {
        int c = idx[i]; size_t j = (size_t)n * K + c; uint8_t hb = hole[j >> 1], pb = plug[j >> 1];
        int h = (j & 1) ? (hb >> 4) : (hb & 15), p = (j & 1) ? (pb >> 4) : (pb & 15);
        acc += (double)((c_cells[h][p] - c_parent[h]) * am[j >> 6]) * (double)__bfloat162float(x[c]);
    }
    y[n] = acc;
}
__global__ void k_sf_gather(const uint8_t *src, uint8_t *dst, const int *idx, const int *cnt_p, int half, const uint8_t *skip = nullptr) { /* grid = K blocks, block 128; skip[c] != 0: channel already staged by another band */
    int i = blockIdx.x; if (i >= *cnt_p) return; if (skip && skip[idx[i]]) return;
    const uint4 *s = (const uint4 *)(src + (size_t)idx[i] * half); uint4 *d = (uint4 *)(dst + (size_t)i * half);
    for (int q = threadIdx.x; q < half / 16; q += blockDim.x) d[q] = s[q];
}
/* Correction over the staged channels with K-major hole and absmax: a thread owns an output pair (2t, 2t+1); blockIdx.y splits the channel list
 * and writes its own partial sum (deterministic; summed by k_sf_sum). */
__global__ void k_sf_corr2(const uint8_t *hole_k, const float *am_k, const uint8_t *stage, const __nv_bfloat16 *x, const int *idx, const int *cnt_p,
                           float *partial, int N, int nsplit, int tabB, const uint8_t *skip) {
    __shared__ float s_d[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_d[i] = (tabB ? c_cellsB[i >> 4][i & 15] : c_cells[i >> 4][i & 15]) - c_parent[i >> 4];
    __syncthreads();
    const int half = N / 2, t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= half) return;
    const int cnt = *cnt_p; float a0 = 0.f, a1 = 0.f;
    for (int i = blockIdx.y; i < cnt; i += nsplit) {
        const int c = idx[i]; if (skip && skip[c]) continue; const float xv = __bfloat162float(x[c]);
        const uint8_t hb = hole_k[(size_t)c * half + t], pb = stage[(size_t)i * half + t];
        const float2 a = *(const float2 *)(am_k + (size_t)(c >> 6) * N + 2 * t);
        a0 += (s_d[((hb & 15) << 4) | (pb & 15)] * a.x) * xv;
        a1 += (s_d[((hb >> 4) << 4) | (pb >> 4)] * a.y) * xv;
    }
    partial[(size_t)blockIdx.y * N + 2 * t] = a0; partial[(size_t)blockIdx.y * N + 2 * t + 1] = a1;
}
__global__ void k_sf_sum(const float *partial, float *y, int N, int nsplit) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    float s = y[n]; for (int sp = 0; sp < nsplit; sp++) s += partial[(size_t)sp * N + n]; y[n] = s;
}
__global__ void k_sf_mark(const int *idx, const int *cnt_p, uint8_t *mask) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < *cnt_p) mask[idx[i]] = 1; }
static double g_tier_fracB = 0; static int *g_sf_idx2 = NULL, *g_sf_cnt2 = NULL; static uint8_t *g_sf_stage2 = NULL, *g_sf_mask = NULL; static float *g_sf_partial2 = NULL;
static uint8_t *g_sf_stage = NULL; static float *g_sf_partial = NULL; static cudaStream_t g_sf_sB = NULL; static cudaEvent_t g_sf_evSel, g_sf_evGat;
static int g_sf_stale = 0; static __nv_bfloat16 *g_sf_prevx[64][8]; /* ARM0_SELFETCH_RULE=stale (spec Amendment 10) */
static const int SF_NSPLIT = 4; static int g_sf_abl = -1; /* ARM0_SF_ABL bitmask (timing ablation only, output is wrong): 1 skip select, 2 skip gathers, 4 skip corrections+sums */
static void selfetch_linear(int layer, int which, const GlintLin &g, const __nv_bfloat16 *x, float *y) {
    const int N = ARM0_LINEAR[which].n_out, K = ARM0_LINEAR[which].k_in;
    if (!g_sf_idx) { CK(cudaMalloc(&g_sf_idx, 8192 * 4)); CK(cudaMalloc(&g_sf_cnt, 4)); }
    g_sf_total += (double)K * (N / 2); g_sf_calls++;
    const double fl = (g_sf_hi_f >= 0 && g_sf_hi[layer]) ? g_sf_hi_f : g_sf_frac; const unsigned salt = g_sf_step * 131u + (unsigned)(layer * 8 + which);
    int nsel = (int)ceil(fl * K - 1e-6); if (nsel < 1) nsel = 1; if (nsel > K) nsel = K;
    const __nv_bfloat16 *xs = x; if (g_sf_stale) { __nv_bfloat16 *&pv = g_sf_prevx[layer][which]; if (!pv) { CK(cudaMalloc(&pv, (size_t)K * 2)); CK(cudaMemset(pv, 0, (size_t)K * 2)); } if (g_sf_stale == 2 && layer > 0 && g_sf_prevx[layer - 1][which]) xs = g_sf_prevx[layer - 1][which]; else xs = pv; } /* stale rule (2 = prevlayer: this token's input to the same linear one layer earlier): rank channels by the previous token's input to this linear, known before this layer runs */
    if (g_sf_opt && fl > 0.0 && g_tier_fracB > 0.0) { /* tiered plug: top nA at the table in c_cells, the next band at c_cellsB, the rest base-only */
        if (!g_sf_stage) { CK(cudaMalloc(&g_sf_stage, 26ull << 20)); CK(cudaMalloc(&g_sf_partial, (size_t)SF_NSPLIT * 16384 * 4)); }
        if (!g_sf_stage2) { CK(cudaMalloc(&g_sf_stage2, 26ull << 20)); CK(cudaMalloc(&g_sf_partial2, (size_t)SF_NSPLIT * 16384 * 4)); CK(cudaMalloc(&g_sf_idx2, 8192 * 4)); CK(cudaMalloc(&g_sf_cnt2, 4)); CK(cudaMalloc(&g_sf_mask, 8192)); }
        int nAll = (int)ceil((fl + g_tier_fracB) * K - 1e-6); if (nAll < nsel) nAll = nsel; if (nAll > K) nAll = K;
        if (g_sf_abl < 0) g_sf_abl = getenv("ARM0_SF_ABL") ? atoi(getenv("ARM0_SF_ABL")) : 0;
        if (!g_sf_sB) { CK(cudaStreamCreateWithFlags(&g_sf_sB, cudaStreamNonBlocking)); CK(cudaEventCreateWithFlags(&g_sf_evSel, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&g_sf_evGat, cudaEventDisableTiming)); }
        k_sf_select<<<1, 1024>>>(xs, K, nsel, g_sf_idx, g_sf_cnt, g_sf_rule, salt);
        k_sf_select<<<1, 1024>>>(xs, K, nAll, g_sf_idx2, g_sf_cnt2, g_sf_rule, salt);
        CK(cudaMemsetAsync(g_sf_mask, 0, K)); k_sf_mark<<<(K + 255) / 256, 256>>>(g_sf_idx, g_sf_cnt, g_sf_mask);
        CK(cudaEventRecord(g_sf_evSel, 0)); CK(cudaStreamWaitEvent(g_sf_sB, g_sf_evSel, 0));
        if (!(g_sf_abl & 2)) k_sf_gather<<<K, 128, 0, g_sf_sB>>>(g_sf_plugk_d[layer][which], g_sf_stage, g_sf_idx, g_sf_cnt, N / 2);                        /* band A, PCIe, concurrent with the parent GEMV */
        if (!(g_sf_abl & 2) && !(g_sf_abl & 8)) k_sf_gather<<<K, 128, 0, g_sf_sB>>>(g_sf_plugk_d[layer][which], g_sf_stage2, g_sf_idx2, g_sf_cnt2, N / 2, g_sf_mask);          /* band B only: channels of band A are not fetched twice */
        CK(cudaEventRecord(g_sf_evGat, g_sf_sB));
        int save = g_gemv_parent; g_gemv_parent = 1; gemv_launch(g.hole, g.plug, g.am, x, y, N, K); g_gemv_parent = save;
        CK(cudaStreamWaitEvent(0, g_sf_evGat, 0));
        if (!(g_sf_abl & 4)) k_sf_corr2<<<dim3((N / 2 + 127) / 128, SF_NSPLIT), 128>>>(g_sf_holek[layer][which], g_sf_amk[layer][which], g_sf_stage, x, g_sf_idx, g_sf_cnt, g_sf_partial, N, SF_NSPLIT, 0, nullptr);
        if (!(g_sf_abl & 4)) k_sf_sum<<<(N + 127) / 128, 128>>>(g_sf_partial, y, N, SF_NSPLIT);
        if (!(g_sf_abl & 4) && !(g_sf_abl & 8)) k_sf_corr2<<<dim3((N / 2 + 127) / 128, SF_NSPLIT), 128>>>(g_sf_holek[layer][which], g_sf_amk[layer][which], g_sf_stage2, x, g_sf_idx2, g_sf_cnt2, g_sf_partial2, N, SF_NSPLIT, 1, g_sf_mask);
        if (!(g_sf_abl & 4) && !(g_sf_abl & 8)) k_sf_sum<<<(N + 127) / 128, 128>>>(g_sf_partial2, y, N, SF_NSPLIT);
        CK(cudaGetLastError());
    } else if (g_sf_opt && fl > 0.0) {
        if (!g_sf_stage) { CK(cudaMalloc(&g_sf_stage, 26ull << 20)); CK(cudaMalloc(&g_sf_partial, (size_t)SF_NSPLIT * 16384 * 4)); CK(cudaStreamCreateWithFlags(&g_sf_sB, cudaStreamNonBlocking));
                           CK(cudaEventCreateWithFlags(&g_sf_evSel, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&g_sf_evGat, cudaEventDisableTiming)); }
        k_sf_select<<<1, 1024>>>(xs, K, nsel, g_sf_idx, g_sf_cnt, g_sf_rule, salt);     /* stream 0 */
        CK(cudaEventRecord(g_sf_evSel, 0)); CK(cudaStreamWaitEvent(g_sf_sB, g_sf_evSel, 0));
        k_sf_gather<<<K, 128, 0, g_sf_sB>>>(g_sf_plugk_d[layer][which], g_sf_stage, g_sf_idx, g_sf_cnt, N / 2); /* PCIe gather, concurrent with the parent GEMV */
        CK(cudaEventRecord(g_sf_evGat, g_sf_sB));
        int save = g_gemv_parent; g_gemv_parent = 1; gemv_launch(g.hole, g.plug, g.am, x, y, N, K); g_gemv_parent = save;
        CK(cudaStreamWaitEvent(0, g_sf_evGat, 0));
        k_sf_corr2<<<dim3((N / 2 + 127) / 128, SF_NSPLIT), 128>>>(g_sf_holek[layer][which], g_sf_amk[layer][which], g_sf_stage, x, g_sf_idx, g_sf_cnt, g_sf_partial, N, SF_NSPLIT, 0, nullptr);
        k_sf_sum<<<(N + 127) / 128, 128>>>(g_sf_partial, y, N, SF_NSPLIT);
        CK(cudaGetLastError());
    } else {
    int save = g_gemv_parent; g_gemv_parent = 1; gemv_launch(g.hole, g.plug, g.am, x, y, N, K); g_gemv_parent = save; /* parent value for every weight */
    if (fl <= 0.0) return;
    k_sf_select<<<1, 1024>>>(xs, K, nsel, g_sf_idx, g_sf_cnt, g_sf_rule, salt);
    k_sf_corr<<<(N + 127) / 128, 128>>>(g.hole, g.am, g_sf_plugk_d[layer][which], x, g_sf_idx, g_sf_cnt, y, N, K);
    CK(cudaGetLastError());
    }
    if (g_sf_stale) CK(cudaMemcpyAsync(g_sf_prevx[layer][which], x, (size_t)K * 2, cudaMemcpyDeviceToDevice, 0)); /* next token ranks channels by this token's input */
    if (g_sf_stats || g_sf_check) {
        int cnt = 0; CK(cudaMemcpy(&cnt, g_sf_cnt, 4, cudaMemcpyDeviceToHost)); g_sf_bytes += (double)cnt * (N / 2);
        if (g_sf_check) {
            if (!g_sfc_yr) { CK(cudaMalloc(&g_sfc_yr, 16384 * 8)); g_sfc_y = (float *)malloc(16384 * 4); }
            k_sf_ref<<<(N + 127) / 128, 128>>>(g.hole, g.plug, g.am, x, g_sf_idx, g_sf_cnt, g_sfc_yr, N, K);
            std::vector<double> r(N); std::vector<float> a(N);
            CK(cudaMemcpy(r.data(), g_sfc_yr, N * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(a.data(), y, N * 4, cudaMemcpyDeviceToHost));
            double ss = 0, md = 0; for (int i = 0; i < N; i++) { ss += r[i] * r[i]; md = fmax(md, fabs((double)a[i] - r[i])); }
            double rms = sqrt(ss / N); if (rms > 0) { g_sfc_max = fmax(g_sfc_max, md / rms); g_sfc_sum += md / rms; g_sfc_n++; }
        }
    }
}

struct CublasLinears {
    cublasHandle_t cublas;
    __nv_bfloat16 *w[N_LAYERS][ARM0_N_LINEARS];
};

static void cublas_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const CublasLinears *lin = (const CublasLinears *)ctx;
    act_capture(layer, which, x, m);
    if (g_glint) {
        const GlintLin &g = g_gl[layer][which];
        if (g_sf_frac >= 0 && g_sf_frac < 1.0 && m == 1) { selfetch_linear(layer, which, g, x, y); return; }
        if (g_gemv && g_sf_frac >= 1.0 && m == 1) { gemv_launch(g.hole, g.plug, g.am, x, y, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in); return; }
        if (g_demand) { demand_linear(lin->cublas, layer, which, x, y, m); return; }
        if (g_stream && layer >= g_res_layers) {
            if (which == ARM0_QKV && glint_plug_streamer_acquire(g_streamer, layer - g_res_layers, &g_cur_dev_plug) != 0)
                setup_fail("plug acquire failed layer %d", layer);
            if (!g_cur_dev_plug) setup_fail("plug for layer %d not acquired (linear order changed)", layer);
            if (g_gemv && m == 1) gemv_launch(g.hole, g_cur_dev_plug + g_plug_off[layer][which], g.am, x, y, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in);
            else if (launch_glint_htile_s3_f32_stream(g.hole, g_cur_dev_plug + g_plug_off[layer][which], g.am, x, y, m,
                                                 ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64, 0) != 0)
                setup_fail("htile launch failed layer %d", layer);
            if (which == ARM0_DOWN) {
                if (glint_plug_streamer_release(g_streamer, layer - g_res_layers) != 0) setup_fail("plug release failed layer %d", layer);
                g_cur_dev_plug = NULL;
            }
            return;
        }
        if (g_gemv && m == 1) {
            gemv_launch(g.hole, g.plug, g.am, x, y, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in);
            if (g_gemv_check) gemv_check(g.hole, g.plug, g.am, x, y, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, layer, which);
            return;
        }
        if (launch_glint_htile_s3_f32(g.hole, g.plug, g.am, x, y, m, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64) != 0)
            setup_fail("htile launch failed layer %d", layer);
        return;
    }
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(lin->cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &alpha, lin->w[layer][which], CUDA_R_16BF,
                     k_in, x, CUDA_R_16BF, k_in, &beta, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static Arm0Result g_result;
static float g_dump[N_LAYERS + 1][N_DUMP][HIDDEN_DIM];

int main(int argc, char** argv) {
    const char* dump_dir = NULL;
    const char* int8_pin_path = NULL;
    const char* glint_dir = NULL;
    int argi = 1;
    for (; argi + 1 < argc && argv[argi][0] == '-'; argi += 2) {
        if (strcmp(argv[argi], "--dump") == 0) dump_dir = argv[argi + 1];
        else if (strcmp(argv[argi], "--int8") == 0) int8_pin_path = argv[argi + 1];
        else if (strcmp(argv[argi], "--glint") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; }
        else if (strcmp(argv[argi], "--glint-hole-only") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_hole_only = 1; }
        else if (strcmp(argv[argi], "--glint-stream") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_stream = 1; }
        else if (strcmp(argv[argi], "--glint-demand") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_demand = 1; }
        else break;
    }
    if (argc - argi != 2) {
        printf("usage: %s [--dump DIR] [--int8 PIN.safetensors] <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    if (g_stream && getenv("ARM0_STREAM_RESIDENT_LAYERS")) { g_res_layers = atoi(getenv("ARM0_STREAM_RESIDENT_LAYERS")); if (g_res_layers < 0) g_res_layers = 0; if (g_res_layers > N_LAYERS) g_res_layers = N_LAYERS; }
    { size_t tot = 0; CK(cudaMemGetInfo(&g_vram_base_free, &tot)); }
    const char* model_dir = argv[argi];
    const char* artifacts_dir = argv[argi + 1];
    char path[1024];

    printf("=========================================================\n");
    printf("Arm (0) Full 32-Layer BF16 Evaluator (2048 Tokens)\n");
    printf("=========================================================\n");

    if (dump_dir && mkdir(dump_dir, 0755) != 0 && errno != EEXIST)
        setup_fail("cannot create dump dir %s: %s", dump_dir, strerror(errno));

    static Arm0Eval ev;
    arm0_read_eval(artifacts_dir, &ev);

    SafeTensors st1, st2;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);

    SafeTensors st_int8;
    const SafeTensors* int8_pin = NULL;
    if (int8_pin_path) {
        if (safetensors_open(&st_int8, int8_pin_path) != 0) setup_fail("cannot open INT8 pin %s", int8_pin_path);
        int8_pin = &st_int8;
    }

    static CublasLinears lin;
    SafeTensors gs1, gs2;
    if (g_glint) {
        char a[1024], b[1024];
        snprintf(a, sizeof(a), "%s/model-00001-of-00002.safetensors", glint_dir);
        snprintf(b, sizeof(b), "%s/model-00002-of-00002.safetensors", glint_dir);
        if (safetensors_open(&gs1, a) != 0 || safetensors_open(&gs2, b) != 0) setup_fail("cannot open glint pin %s", glint_dir);
        for (int l = 0; l < N_LAYERS; l++) for (int j = 0; j < ARM0_N_LINEARS; j++) glint_upload(&gs1, &gs2, l, j);
        if (g_demand) demand_init();
        if (getenv("ARM0_SELFETCH")) { g_sf_frac = atof(getenv("ARM0_SELFETCH")); g_sf_stats = getenv("ARM0_SELFETCH_STATS") != NULL; g_sf_check = getenv("ARM0_SELFETCH_CHECK") != NULL; g_sf_opt = getenv("ARM0_SELFETCH_OPT") != NULL;
            if (getenv("ARM0_TIER_FRAC_B")) { g_tier_fracB = atof(getenv("ARM0_TIER_FRAC_B")); int bB = getenv("ARM0_TIER_BITS_B") ? atoi(getenv("ARM0_TIER_BITS_B")) : 3;
                float tb[16][16]; cells_bits_fixed(tb, bB); CK(cudaMemcpyToSymbol(c_cellsB, tb, sizeof(float) * 256)); printf("Tiered plug: band A = ARM0_SELFETCH at the main table, band B = next %.3f of channels at %d-bit merged cells\n", g_tier_fracB, bB); }
            { const char *rl = getenv("ARM0_SELFETCH_RULE"); g_sf_rule = rl && !strcmp(rl, "random") ? 1 : 0; g_sf_stale = rl && !strcmp(rl, "stale") ? 1 : (rl && !strcmp(rl, "prevlayer") ? 2 : 0);
              const char *hl = getenv("ARM0_SELFETCH_HI_LAYERS"); if (hl) { g_sf_hi_f = getenv("ARM0_SELFETCH_HI_F") ? atof(getenv("ARM0_SELFETCH_HI_F")) : -1;
                  char buf[256]; strncpy(buf, hl, 255); buf[255] = 0; for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) { int li = atoi(tok); if (li >= 0 && li < N_LAYERS) g_sf_hi[li] = 1; } }
              printf("Selected fetch: rule=%s opt=%d hi_layers_f=%.3f\n", g_sf_rule ? "random" : "act", g_sf_opt, g_sf_hi_f); } g_gemv = 1; gemv_init(); printf("Selected-channel fetch decode: f=%.4f (K-major plug in pinned host memory, zero-copy)\n", g_sf_frac); }
        if (getenv("ARM0_GEMV") && atoi(getenv("ARM0_GEMV")) == 1) { g_gemv = 1; gemv_init(); g_gemv_check = getenv("ARM0_GEMV_CHECK") && atoi(getenv("ARM0_GEMV_CHECK")) == 1; printf("GLINT decode GEMV enabled for M = 1 (parent_only=%d)\n", g_gemv_parent); }
        if (g_stream && g_res_layers < N_LAYERS) {
            CK(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking));
            if (glint_plug_streamer_create(&g_streamer, N_LAYERS - g_res_layers, g_layer_bytes + g_res_layers, g_copy_stream, 0) != 0) setup_fail("streamer create failed");
            for (int l = g_res_layers; l < N_LAYERS; l++) {
                std::vector<uint8_t> cat(g_layer_bytes[l]);
                for (int j = 0; j < ARM0_N_LINEARS; j++) {
                    size_t nb = ((size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in + 1) / 2;
                    memcpy(cat.data() + g_plug_off[l][j], g_plug_src[l][j], nb);
                }
                if (glint_plug_streamer_set_host_data(g_streamer, l - g_res_layers, cat.data(), g_layer_bytes[l]) != 0) setup_fail("streamer host data layer %d", l);
            }
            if (glint_plug_streamer_start(g_streamer) != 0) setup_fail("streamer start failed");
        }
        printf("Glint pin loaded (%s).\n", g_stream ? "hole resident; first N layers' plug resident, the rest streamed through the 2-slot ring" : (g_hole_only ? "hole only, plug zeroed" : "hole + plug"));
    }
    if (getenv("ARM0_NF4") && getenv("ARM0_NF4_WMAP")) {
        g_nf4_map = fopen(getenv("ARM0_NF4_WMAP"), "w");
        if (g_nf4_map) fprintf(g_nf4_map, "{\"schema\":\"nf4_weight_diff_map_v1\",\"scheme\":\"NF4 blocksize 64 absmax, bf16 dequant\",\"tensors\":[");
    }
    for (int l = 0; l < N_LAYERS && !g_glint; l++) {
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
            lin.w[l][j] = upload_linear(&st1, &st2, int8_pin, name, (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in);
        }
    }
    if (getenv("ARM0_NF4")) {
        printf("NF4 weights: %llu linear elems, RMSE %.6e, max abs %.6e, rel L2 %.6e\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2));
        if (g_nf4_map) { fprintf(g_nf4_map, "],\"n_elems\":%llu,\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2)); fclose(g_nf4_map); }
    }
    printf("Uploaded all 32 layers of %s weights to VRAM.\n", int8_pin ? "INT8-dequantized" : "BF16");

    const char *act_map = getenv("ARM0_ACT_MAP");
    act_map_init(act_map);
    CKB(cublasCreate(&lin.cublas));
    arm0_forward(&model, ev.tokens, cublas_linear, &lin, lin.cublas, dump_dir ? g_dump : NULL, &g_result);
    if (g_demand) demand_report();
    if (g_stream) {
        size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
        printf("VRAM_USED_MIB %.1f (resident plug layers %d of %d)\n", ((double)g_vram_base_free - (double)fr) / 1048576.0, g_res_layers, N_LAYERS);
    }
    if (g_stream && g_streamer) {
        GlintStreamerStats ss;
        if (glint_plug_streamer_get_stats(g_streamer, &ss) == 0) {
            double mx = 0; for (int l = 1; l < ss.n_layers; l++) if (ss.per_layer_stall_ms[l] > mx) mx = ss.per_layer_stall_ms[l];
            printf("STREAM_STATS h2d_bytes=%llu total_stall_ms=%.3f layer0_stall_ms=%.3f max_stall_ms_layers_1plus=%.3f\n",
                   (unsigned long long)ss.total_h2d_bytes, ss.total_stall_ms, ss.per_layer_stall_ms[0], mx);
        }
    }
    act_map_write(act_map, int8_pin ? "BF16 cuBLAS (INT8-dequantized weights)" : "BF16 cuBLAS (original BF16 weights)");
    const float* h_losses = g_result.losses;
    const int32_t* h_top1 = g_result.top1;
    const uint32_t* oracle_top1 = ev.oracle_top1;
    const double oracle_ppl = ev.oracle_ppl;
    double total_sec = g_result.seconds;

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

    double ppl_delta_pct = fabs(ppl - oracle_ppl) / oracle_ppl * 100.0;

    printf("\n=== %s Evaluation Results ===\n", int8_pin ? "INT8 pin" : "Arm (0) BF16");
    printf("Forward Pass Time:      %.3f s (%.1f tok/s)\n", total_sec, 2048.0 / total_sec);
    printf("Mean Cross Entropy:     %.6f\n", mean_ce);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", ppl_delta_pct);
    printf("Top-1 Matches:          %d / 2047\n", top1_matches);
    printf("Top-1 Agreement:        %.2f %%\n", top1_acc);
    if (g_result.has_kl) { /* KL(ref || arm) per token against ARM0_LOGITS_REF */
        std::vector<float> kv(g_result.kl, g_result.kl + 2047); double ks = 0; for (float v : kv) ks += v;
        std::sort(kv.begin(), kv.end());
        printf("KL vs reference: mean %.6e p99 %.6e max %.6e (2047 tokens)\n", ks / 2047.0, kv[(size_t)(0.99 * 2046)], kv[2046]);
    }

    int pass = top1_acc >= 95.0 && ppl_delta_pct <= 1.0;
    if (g_glint) printf("VERDICT: REPORT ONLY (glint %s)\n", g_hole_only ? "hole-only" : "hole+plug");
    else if (int8_pin) printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", int8_pin_path);
    else printf(pass ? "VERDICT: PASS (Gate 1 Anchor Verified)\n" : "VERDICT: FAIL (Gate 1 Anchor Mismatch)\n");
    fflush(stdout);

    if (dump_dir) {
        snprintf(path, sizeof(path), "%s/cuda_hidden.bin", dump_dir);
        FILE* fh = fopen(path, "wb");
        if (!fh) setup_fail("cannot write %s", path);
        fwrite(g_dump, sizeof(g_dump), 1, fh);
        fclose(fh);
        snprintf(path, sizeof(path), "%s/cuda_pos.bin", dump_dir);
        FILE* fp = fopen(path, "wb");
        if (!fp) setup_fail("cannot write %s", path);
        fwrite(h_losses, sizeof(float), 2047, fp);
        fwrite(h_top1, sizeof(int32_t), 2047, fp);
        fclose(fp);
    }

    /* ARM0_DECODE=PL,S: decode path check and timing (glint #84). (1) teacher-forced: step the true tokens 0..PL+S-1 through the KV-cache
     * path and compare top-1 and CE loss with the prefill results of this same run; (2) free-running: greedy-generate S tokens after the
     * PL-token prompt, timing each step; the generated ids go to ARM0_DECODE_OUT. */
    if (getenv("ARM0_DECODE")) {
        int PL = 64, S = 16; sscanf(getenv("ARM0_DECODE"), "%d,%d", &PL, &S);
        DecBufs db; dec_alloc(db);
        const int N = PL + S;
        int t1_eq = 0, loss_bit_eq = 0; double max_dl = 0, sum_dl = 0; std::vector<int> dt1(N);
        for (int pos = 0; pos < N; pos++) {
            int32_t t1; float ls;
            dec_step(&model, db, ev.tokens[pos], pos, ev.tokens[pos + 1], cublas_linear, &lin, lin.cublas, &t1, &ls);
            dt1[pos] = t1;
            t1_eq += (t1 == h_top1[pos]);
            double dl = fabs((double)ls - (double)h_losses[pos]); sum_dl += dl; if (dl > max_dl) max_dl = dl;
            loss_bit_eq += (memcmp(&ls, &h_losses[pos], 4) == 0);
        }
        printf("DECODE_CHECK positions=%d top1_equal=%d loss_bitwise_equal=%d max_abs_dCE=%.3e mean_abs_dCE=%.3e\n", N, t1_eq, loss_bit_eq, max_dl, sum_dl / N);
        std::vector<int> gen; gen.push_back(dt1[PL - 1]);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); double tot_ms = 0; int nt = 0;
        for (int st = 1; st < S; st++) {
            int32_t t1; float ls; float ms = 0;
            CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
            dec_step(&model, db, gen[st - 1], PL + st - 1, 0, cublas_linear, &lin, lin.cublas, &t1, &ls);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
            gen.push_back(t1); tot_ms += ms; nt++;
        }
        printf("DECODE_TIMING steps=%d mean_ms_per_token=%.3f mean_ms_per_layer=%.4f tokens_per_s=%.2f\n", nt, tot_ms / nt, tot_ms / nt / N_LAYERS, 1000.0 * nt / tot_ms);
        if (g_gemv_check) printf("GEMV_CHECK linears=%ld | max/mean error relative to RMS of the double reference: gemv %.3e/%.3e  htile %.3e/%.3e  gemv-vs-htile %.3e/%.3e\n", g_gc_n, g_gc_max_gr, g_gc_n ? g_gc_sum_gr / g_gc_n : 0.0, g_gc_max_hr, g_gc_n ? g_gc_sum_hr / g_gc_n : 0.0, g_gc_max_gh, g_gc_n ? g_gc_sum_gh / g_gc_n : 0.0);
        if (g_sf_frac >= 0 && g_sf_frac < 1.0) printf("SELFETCH f=%.4f bytes_per_token_MB=%.2f frac_of_plug=%.4f%s\n", g_sf_frac, g_sf_bytes / (g_sf_calls / (double)(N_LAYERS * ARM0_N_LINEARS)) / 1e6, g_sf_total > 0 ? g_sf_bytes / g_sf_total : 0.0, g_sf_check ? "" : " (only meaningful with ARM0_SELFETCH_STATS=1)");
        if (g_sf_check && g_sfc_n) printf("SELFETCH_CHECK linears=%ld max_err_to_rms=%.3e mean_err_to_rms=%.3e (vs double reference with the same selection)\n", g_sfc_n, g_sfc_max, g_sfc_sum / g_sfc_n);
        printf("DECODE_GEN:"); for (int g : gen) printf(" %d", g); printf("\n");
        const char *dro = getenv("ARM0_DECODE_OUT"); if (dro) { FILE *fo = fopen(dro, "w"); for (size_t i = 0; i < gen.size(); i++) fprintf(fo, "%s%d", i ? " " : "", gen[i]); fprintf(fo, "\n"); fclose(fo); }
        fflush(stdout);
    }

    /* ARM0_DECODE_GATE=N (spec Amendment 12): the recorded teacher-forced gate (GATE_RULES.md) through the KV-cache decode path. The true tokens
     * 0..N-1 are stepped in order, so selection (ARM0_SELFETCH*) acts per token exactly as in decode. Reports PPL and top-1 against the HF
     * oracle and, with ARM0_LOGITS_REF, KL(ref || arm) per token; ARM0_DECODE_GATE_OUT receives one line per position (pos top1 oracle_top1 CE kl). */
    if (getenv("ARM0_DECODE_GATE")) {
        int N = atoi(getenv("ARM0_DECODE_GATE")); if (N > 2047) N = 2047;
        DecBufs gb; dec_alloc(gb);
        const char *gref = getenv("ARM0_LOGITS_REF"), *gout = getenv("ARM0_DECODE_GATE_OUT");
        FILE *fr = gref ? fopen(gref, "rb") : NULL; FILE *fo = gout ? fopen(gout, "w") : NULL;
        float *hrow = (float *)malloc((size_t)VOCAB_SIZE * 4), *dref = NULL, *dkl = NULL; int32_t *drt = NULL, *dqt = NULL;
        if (fr) { CK(cudaMalloc(&dref, (size_t)VOCAB_SIZE * 4)); CK(cudaMalloc(&dkl, 4)); CK(cudaMalloc(&drt, 4)); CK(cudaMalloc(&dqt, 4)); }
        double sce = 0; int eq = 0; std::vector<float> kls;
        for (int pos = 0; pos < N; pos++) {
            int32_t t1; float ls; dec_step(&model, gb, ev.tokens[pos], pos, ev.tokens[pos + 1], cublas_linear, &lin, lin.cublas, &t1, &ls);
            float kl = -1.f;
            if (fr) {
                if (fread(hrow, 4, (size_t)VOCAB_SIZE, fr) != (size_t)VOCAB_SIZE) setup_fail("short read %s", gref);
                CK(cudaMemcpy(dref, hrow, (size_t)VOCAB_SIZE * 4, cudaMemcpyHostToDevice));
                kl_and_argmax_kernel<<<1, 256>>>(dref, gb.logits, dkl, drt, dqt, 1, VOCAB_SIZE); CK(cudaMemcpy(&kl, dkl, 4, cudaMemcpyDeviceToHost)); kls.push_back(kl);
            }
            sce += ls; eq += (t1 == (int32_t)oracle_top1[pos]);
            if (fo) fprintf(fo, "%d %d %d %.7g %.7g\n", pos, t1, (int)oracle_top1[pos], ls, kl);
        }
        const double ppl_g = exp(sce / N), dpct = (ppl_g - oracle_ppl) / oracle_ppl * 100.0, acc = 100.0 * eq / N;
        printf("DECODE_GATE positions=%d ppl=%.6f oracle_ppl=%.6f ppl_delta_pct=%.3f top1_vs_oracle=%d/%d (%.2f%%)", N, ppl_g, oracle_ppl, dpct, eq, N, acc);
        if (!kls.empty()) { double ks = 0; for (float v : kls) ks += v; std::sort(kls.begin(), kls.end()); printf(" kl_mean=%.4e kl_p99=%.4e kl_max=%.4e", ks / kls.size(), kls[(size_t)(0.99 * (kls.size() - 1))], kls.back()); }
        printf(" gate=%s\n", (acc >= 95.0 && fabs(dpct) <= 2.0) ? "PASS" : ((acc >= 95.0 && fabs(dpct) <= 5.0) ? "INCONCLUSIVE" : "FAIL"));
        if (fr) fclose(fr); if (fo) fclose(fo); free(hrow); fflush(stdout);
    }

    /* ARM0_DECODE_ROLL=P,S,PL,STRIDE (spec Amendment 3): greedy generation through the KV-cache decode path for P prompts (PL tokens of the
     * evaluation text at offsets p*STRIDE), S generated tokens each; one line of ids per prompt to ARM0_DECODE_ROLL_OUT. */
    if (getenv("ARM0_DECODE_ROLL")) {
        int P = 12, S = 32, PL = 64, STR = 128; sscanf(getenv("ARM0_DECODE_ROLL"), "%d,%d,%d,%d", &P, &S, &PL, &STR);
        static DecBufs rb; static int rb_alloc = 0; if (!rb_alloc) { dec_alloc(rb); rb_alloc = 1; }
        const char *rro = getenv("ARM0_DECODE_ROLL_OUT"); FILE *rf = rro ? fopen(rro, "w") : NULL; double t_all = 0; long n_all = 0;
        for (int p = 0; p < P; p++) {
            const int base = p * STR; if (base + PL >= 2048) break;
            int32_t t1 = 0; float ls = 0;
            for (int pos = 0; pos < PL; pos++) dec_step(&model, rb, ev.tokens[base + pos], pos, 0, cublas_linear, &lin, lin.cublas, &t1, &ls);
            std::vector<int> gen; gen.push_back(t1);
            for (int st = 1; st < S; st++) { dec_step(&model, rb, gen[st - 1], PL + st - 1, 0, cublas_linear, &lin, lin.cublas, &t1, &ls); gen.push_back(t1); }
            if (rf) { for (size_t i = 0; i < gen.size(); i++) fprintf(rf, "%s%d", i ? " " : "", gen[i]); fprintf(rf, "\n"); fflush(rf); }
            n_all += gen.size();
        }
        if (rf) fclose(rf);
        printf("DECODE_ROLL prompts=%d steps=%d generated=%ld\n", P, S, n_all); fflush(stdout);
    }

    /* ARM0_INDUCT=N,L (spec Amendment 8): induction/retrieval probe. N sequences of L pseudo-random token ids (fixed LCG, ids in
     * [1000,30000)) are shown twice in a row through the KV-cache decode path; at every position of the second copy after its first
     * token the model's greedy next token is compared with the true next token of the copy (teacher forced). Reports accuracy. */
    if (getenv("ARM0_INDUCT")) {
        int N = 8, L = 48; sscanf(getenv("ARM0_INDUCT"), "%d,%d", &N, &L);
        static DecBufs ib; static int ib_alloc = 0; if (!ib_alloc) { dec_alloc(ib); ib_alloc = 1; }
        long hit = 0, tot = 0; uint32_t s = 12345u;
        for (int n = 0; n < N; n++) {
            std::vector<int32_t> seq(L); for (int i = 0; i < L; i++) { s = s * 1664525u + 1013904223u; seq[i] = 1000 + (int)((s >> 8) % 29000u); }
            int32_t t1 = 0; float ls = 0; long h = 0, t = 0;
            for (int pos = 0; pos < 2 * L - 1; pos++) {
                int32_t tok = seq[pos % L];
                dec_step(&model, ib, tok, pos, 0, cublas_linear, &lin, lin.cublas, &t1, &ls);
                if (pos >= L) { t++; if (t1 == seq[(pos + 1) % L]) h++; }
            }
            hit += h; tot += t; printf("INDUCT seq=%d hits=%ld/%ld\n", n, h, t);
        }
        printf("INDUCT_ACC %.4f (%ld/%ld) sequences=%d length=%d\n", (double)hit / tot, hit, tot, N, L); fflush(stdout);
    }

    /* ARM0_ROLLOUT=P,S,PL: greedy generation by repeated full forwards on a padded 2048-token buffer. Attention is causal, so
     * the logits at the last real position do not depend on what follows it. Prompt p is eval tokens [p*512, p*512+PL) placed at
     * position 0 (the rest of the buffer is filler that is overwritten as tokens are generated). Writes the generated ids, one
     * line per prompt, to ARM0_ROLLOUT_OUT. Run without ARM0_LOGITS_OUT/REF. See DEMAND_MAP_SPEC (rollout amendment). */
    if (getenv("ARM0_ROLLOUT")) {
        int P = 3, S = 16, PL = 64; sscanf(getenv("ARM0_ROLLOUT"), "%d,%d,%d", &P, &S, &PL);
        const char *ro = getenv("ARM0_ROLLOUT_OUT"); FILE *fo = ro ? fopen(ro, "w") : NULL;
        unsetenv("ARM0_LOGITS_OUT"); unsetenv("ARM0_LOGITS_REF");
        static int32_t rbuf[2048];
        for (int p = 0; p < P; p++) {
            for (int i = 0; i < 2048; i++) rbuf[i] = ev.tokens[(p * 512 + i) % 2048];
            std::vector<int> gen;
            for (int st = 0; st < S; st++) {
                int len = PL + st;
                arm0_forward(&model, rbuf, cublas_linear, &lin, lin.cublas, NULL, &g_result);
                int nxt = g_result.top1[len - 1]; rbuf[len] = nxt; gen.push_back(nxt);
            }
            printf("ROLLOUT prompt %d:", p); for (int g : gen) printf(" %d", g); printf("\n");
            if (fo) { for (size_t i = 0; i < gen.size(); i++) fprintf(fo, "%s%d", i ? " " : "", gen[i]); fprintf(fo, "\n"); }
            fflush(stdout);
        }
        if (fo) fclose(fo);
    }

    safetensors_close(&st1);
    safetensors_close(&st2);
    cublasDestroy(lin.cublas);
    return (pass || int8_pin || g_glint) ? EXIT_GATE_PASS : EXIT_GATE_FAIL;
}

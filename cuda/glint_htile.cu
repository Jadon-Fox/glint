/* GLINT H-TILE S3 — K-contiguous packed loads, smem tiles, cp.async D=2.
 *
 * Coalescing: packed hole/plug are sequential in K. Threads load 16B along K
 * for a fixed N-row. Old sketch walked N first (stride K/2) — uncoalesced.
 *
 * Tile: M=16 N=16 K=64. Block 16×16 = 256 threads. Ampere sm_86.
 * nvcc -O3 -arch=sm_86 -I../include -c cuda/glint_htile.cu
 */
#include "../include/glint_meta.h"
#include "../include/glint_htile_load.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

static __constant__ float d_cells[16][16];

static int load_cells() {
    static int once = 0;
    if (once)
        return 0;
    float host_cells[16][16];
    memcpy(host_cells, GLINT_CELLS, sizeof(host_cells));
    if (cudaMemcpyToSymbol(d_cells, host_cells, sizeof(host_cells)) != cudaSuccess)
        return -1;
    once = 1;
    return 0;
}
#define TN 16
#define TM 16
#define TK 64
#define STAGES 2
#define HOLE_B (TK / 2) /* 32 */
static_assert(TK % 2 == 0, "even K");
static_assert(HOLE_B % 16 == 0, "16B cp.async");
__device__ __forceinline__ void cp16(void *smem, const void *gmem) {
#if __CUDA_ARCH__ >= 800
    unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sm), "l"(gmem));
#else
    *reinterpret_cast<uint4 *>(smem) = *reinterpret_cast<const uint4 *>(gmem);
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
__device__ __forceinline__ void cp_wait(int n) {
#if __CUDA_ARCH__ >= 800
    if (n == 0)
        asm volatile("cp.async.wait_group 0;");
    else
        asm volatile("cp.async.wait_group 1;");
#else
    (void)n;
#endif
    __syncthreads();
}
/* cp.async 16 B needs a 16-byte-aligned global address, not just an aligned index:
 * a caller may pass a pointer into a larger buffer (e.g. a whole uploaded shard). */
__device__ __forceinline__ bool aligned16(const void *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}
__device__ void load_k_tile(
    int s, int n0, int m0, int k0, int N, int M, int K,
    const uint8_t *hole, const uint8_t *plug, const __nv_bfloat16 *x,
    uint8_t sh_h[STAGES][TN][HOLE_B],
    uint8_t sh_p[STAGES][TN][HOLE_B],
    __nv_bfloat16 sh_x[STAGES][TM][TK])
{
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; /* 0..255 */
    /* hole: 16 rows × 32 B = 512 B → 32 × 16B. tid 0..31 */
    if (tid < 32) {
        const int row = tid / 2; /* 0..15 */
        const int off = (tid % 2) * 16; /* 0 or 16 */
        const int n = n0 + row;
        const int k = k0 + 2 * off;
        const int64_t i = (int64_t)n * K + k;
        if (n < N && k + 32 <= K && (i & 1) == 0 && aligned16(hole + (i >> 1))) {
            cp16(&sh_h[s][row][off], hole + (i >> 1));
        } else {
            for (int b = 0; b < 16; ++b)
                sh_h[s][row][off + b] = glint_packed_tile_byte(hole, n, N, K, k + 2 * b);
        }
    }
    /* plug: tid 32..63, same map */
    if (tid >= 32 && tid < 64) {
        const int t = tid - 32;
        const int row = t / 2;
        const int off = (t % 2) * 16;
        const int n = n0 + row;
        const int k = k0 + 2 * off;
        const int64_t i = (int64_t)n * K + k;
        if (n < N && k + 32 <= K && (i & 1) == 0 && aligned16(plug + (i >> 1))) {
            cp16(&sh_p[s][row][off], plug + (i >> 1));
        } else {
            for (int b = 0; b < 16; ++b)
                sh_p[s][row][off + b] = glint_packed_tile_byte(plug, n, N, K, k + 2 * b);
        }
    }
    /* X: 16×64 bf16 = 2048 B → 128 × 16B. tid walks 128 chunks. */
    {
        const int nch = (TM * TK * (int)sizeof(__nv_bfloat16)) / 16;
        for (int c = tid; c < nch; c += 256) {
            const int elem = (c * 16) / (int)sizeof(__nv_bfloat16);
            const int row = elem / TK;
            const int col = elem % TK;
            const int m = m0 + row;
            const int k = k0 + col;
            const int64_t i = (int64_t)m * K + k;
            if (m < M && k + 8 <= K && aligned16(x + i)) {
                cp16(&sh_x[s][row][col], x + i);
            } else {
                for (int e = 0; e < 8; ++e)
                    sh_x[s][row][col + e] = glint_x_tile_element(x, m, M, K, k + e);
            }
        }
    }
    cp_commit();
}
__global__ void glint_htile_s3_f32(
    const uint8_t *__restrict__ hole,
    const uint8_t *__restrict__ plug,
    const float *__restrict__ absmax,
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ y,
    int M, int N, int K, int blocksize)
{
    const int n0 = blockIdx.x * TN;
    const int m0 = blockIdx.y * TM;
    const int tn = threadIdx.x;
    const int tm = threadIdx.y;
    __shared__ __align__(16) uint8_t sh_h[STAGES][TN][HOLE_B];
    __shared__ __align__(16) uint8_t sh_p[STAGES][TN][HOLE_B];
    __shared__ __align__(16) __nv_bfloat16 sh_x[STAGES][TM][TK];
    float acc = 0.f;
    int cur = 0;
    load_k_tile(0, n0, m0, 0, N, M, K, hole, plug, x, sh_h, sh_p, sh_x);
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int nxt = k0 + TK;
        if (nxt < K)
            load_k_tile(cur ^ 1, n0, m0, nxt, N, M, K, hole, plug, x, sh_h, sh_p, sh_x);
        cp_wait(nxt < K ? 1 : 0);
        if ((m0 + tm) < M && (n0 + tn) < N) {
            const int n = n0 + tn;
            for (int kk = 0; kk < TK && (k0 + kk) < K; ++kk) {
                const uint8_t hb = sh_h[cur][tn][kk >> 1];
                const uint8_t pb = sh_p[cur][tn][kk >> 1];
                const uint8_t h = (kk & 1) ? (uint8_t)((hb >> 4) & 15) : (uint8_t)(hb & 15);
                const uint8_t p = (kk & 1) ? (uint8_t)((pb >> 4) & 15) : (uint8_t)(pb & 15);
                const int64_t i = (int64_t)n * K + (k0 + kk);
                acc += d_cells[h][p] * absmax[i / blocksize] *
                       __bfloat162float(sh_x[cur][tm][kk]);
            }
        }
        __syncthreads();
        cur ^= 1;
    }
    if ((m0 + tm) < M && (n0 + tn) < N)
        y[(int64_t)(m0 + tm) * N + (n0 + tn)] = acc;
}
extern "C" int launch_glint_htile_s3_f32(
    const uint8_t *d_hole, const uint8_t *d_plug, const float *d_absmax,
    const __nv_bfloat16 *d_x, float *d_y,
    int M, int N, int K, int blocksize)
{
    if (!d_hole || !d_plug || !d_absmax || !d_x || !d_y)
        return -1;
    if (blocksize <= 0)
        return -2;
    if (load_cells() != 0)
        return -4;
    dim3 blk(TN, TM);
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    glint_htile_s3_f32<<<grid, blk>>>(d_hole, d_plug, d_absmax, d_x, d_y, M, N, K, blocksize);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

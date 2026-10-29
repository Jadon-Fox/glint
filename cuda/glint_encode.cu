/* GLINT encode on GPU. One thread per weight after a block absmax reduce.
 * nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC -o libglint_encode.so \
 * cuda/glint_encode.cu -Iinclude
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "../include/glint_encode.h"
#include "../include/glint_cells.h"
#include "../include/glint_meta.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define GLINT_PARENT_N 16
#define GLINT_CHILD_N 16
__constant__ float d_parent[16];
__constant__ float d_cells[16][16];
static const float kParent[16] = {
    -1.0f,
    -0.696192801f,
    -0.525073051f,
    -0.394917488f,
    -0.284441382f,
    -0.184773430f,
    -0.091050036f,
    0.0f,
    0.079580300f,
    0.160930201f,
    0.246112302f,
    0.337915242f,
    0.440709829f,
    0.562617004f,
    0.722956836f,
    1.0f,
};
__global__ void k_bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    dst[i] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&src[i]));
}
__global__ void k_absmax(const float *w, float *absmax, int64_t n, int bs) {
    int b = (int)blockIdx.x;
    int64_t base = (int64_t)b * bs;
    float m = 0.f;
    for (int t = (int)threadIdx.x; t < bs; t += (int)blockDim.x) {
        int64_t i = base + t;
        if (i < n) {
            float a = fabsf(w[i]);
            if (a > m)
                m = a;
        }
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            float o = sh[threadIdx.x + s];
            if (o > sh[threadIdx.x])
                sh[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        absmax[b] = sh[0];
}
__device__ __forceinline__ int nearest16(const float *tab, float v) {
    int best = 0;
    float bd = fabsf(v - tab[0]);
#pragma unroll
    for (int k = 1; k < 16; ++k) {
        float d = fabsf(v - tab[k]);
        if (d < bd) {
            bd = d;
            best = k;
        }
    }
    return best;
}
__global__ void k_encode_pack(
    const float *w, const float *absmax, uint8_t *hole_pk, uint8_t *plug_pk, int64_t n, int bs)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t npack = (n + 1) / 2;
    if (e >= npack)
        return;
    int64_t i0 = e * 2;
    int64_t i1 = i0 + 1;
    float s0 = absmax[i0 / bs];
    float v0 = (s0 == 0.f) ? 0.f : (w[i0] / s0);
    int h0 = nearest16(d_parent, v0);
    int p0 = nearest16(d_cells[h0], v0);
    int h1 = 0, p1 = 0;
    if (i1 < n) {
        float s1 = absmax[i1 / bs];
        float v1 = (s1 == 0.f) ? 0.f : (w[i1] / s1);
        h1 = nearest16(d_parent, v1);
        p1 = nearest16(d_cells[h1], v1);
    }
    hole_pk[e] = (uint8_t)((h0 & 15) | ((h1 & 15) << 4));
    plug_pk[e] = (uint8_t)((p0 & 15) | ((p1 & 15) << 4));
}
__global__ void k_expand(
    const uint8_t *hole_pk, const uint8_t *plug_pk, const float *absmax, float *out, int64_t n, int bs)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint8_t hb = hole_pk[i >> 1];
    uint8_t pb = plug_pk[i >> 1];
    uint8_t h = (i & 1) ? (uint8_t)((hb >> 4) & 15) : (uint8_t)(hb & 15);
    uint8_t p = (i & 1) ? (uint8_t)((pb >> 4) & 15) : (uint8_t)(pb & 15);
    out[i] = d_cells[h][p] * absmax[i / bs];
}
static int load_cells() {
    static int once = 0;
    if (once)
        return 0;
    float host_cells[16][16];
    memcpy(host_cells, GLINT_CELLS, sizeof(host_cells));
    if (cudaMemcpyToSymbol(d_parent, kParent, sizeof(kParent)) != cudaSuccess)
        return -1;
    if (cudaMemcpyToSymbol(d_cells, host_cells, sizeof(host_cells)) != cudaSuccess)
        return -1;
    once = 1;
    return 0;
}
int glint_encode_n_absmax(int64_t n, int blocksize) {
    if (blocksize <= 0)
        return 0;
    return (int)((n + blocksize - 1) / blocksize);
}
int glint_encode_n_packed(int64_t n) { return (int)((n + 1) / 2); }
static int encode_dev_f32(const float *d_w, int64_t n, int blocksize, uint8_t *h_hole, uint8_t *h_plug, float *h_am) {
    if (load_cells() != 0)
        return -2;
    int nblocks = glint_encode_n_absmax(n, blocksize);
    int npack = glint_encode_n_packed(n);
    float *d_am = NULL;
    uint8_t *d_hp = NULL, *d_pp = NULL;
    int rc = -3;
    if (cudaMalloc(&d_am, (size_t)nblocks * sizeof(float)) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_hp, (size_t)npack) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_pp, (size_t)npack) != cudaSuccess)
        goto done;
    k_absmax<<<nblocks, 256>>>(d_w, d_am, n, blocksize);
    {
        int threads = 256;
        int pblocks = (int)((npack + threads - 1) / threads);
        k_encode_pack<<<pblocks, threads>>>(d_w, d_am, d_hp, d_pp, n, blocksize);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_am, d_am, (size_t)nblocks * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_hole, d_hp, (size_t)npack, cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    if (cudaMemcpy(h_plug, d_pp, (size_t)npack, cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    rc = 0;
done:
    cudaFree(d_am);
    cudaFree(d_hp);
    cudaFree(d_pp);
    return rc;
}
int glint_encode_f32_host(const float *w, int64_t n, int blocksize, uint8_t *hole_packed, uint8_t *plug_packed, float *absmax) {
    float *d_w = NULL;
    int rc = -1;
    if (cudaMalloc(&d_w, (size_t)n * sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMemcpy(d_w, w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_w);
        return -1;
    }
    rc = encode_dev_f32(d_w, n, blocksize, hole_packed, plug_packed, absmax);
    cudaFree(d_w);
    return rc;
}
int glint_encode_bf16_host(const uint16_t *w_bf16, int64_t n, int blocksize, uint8_t *hole_packed, uint8_t *plug_packed, float *absmax) {
    uint16_t *d_b = NULL;
    float *d_w = NULL;
    int rc = -1;
    if (cudaMalloc(&d_b, (size_t)n * sizeof(uint16_t)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_w, (size_t)n * sizeof(float)) != cudaSuccess) {
        cudaFree(d_b);
        return -1;
    }
    if (cudaMemcpy(d_b, w_bf16, (size_t)n * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)
        goto done;
    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        k_bf16_to_f32<<<blocks, threads>>>(d_b, d_w, n);
    }
    rc = encode_dev_f32(d_w, n, blocksize, hole_packed, plug_packed, absmax);
done:
    cudaFree(d_b);
    cudaFree(d_w);
    return rc;
}
int glint_expand_packed_to_f32_host(
    const uint8_t *hole_packed, const uint8_t *plug_packed, const float *absmax,
    int64_t n, int blocksize, float *out_f32)
{
    if (load_cells() != 0)
        return -2;
    int npack = glint_encode_n_packed(n);
    int nblocks = glint_encode_n_absmax(n, blocksize);
    uint8_t *d_h = NULL, *d_p = NULL;
    float *d_am = NULL, *d_o = NULL;
    int rc = -3;
    if (cudaMalloc(&d_h, (size_t)npack) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_p, (size_t)npack) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_am, (size_t)nblocks * sizeof(float)) != cudaSuccess)
        goto done;
    if (cudaMalloc(&d_o, (size_t)n * sizeof(float)) != cudaSuccess)
        goto done;
    if (cudaMemcpy(d_h, hole_packed, (size_t)npack, cudaMemcpyHostToDevice) != cudaSuccess)
        goto done;
    if (cudaMemcpy(d_p, plug_packed, (size_t)npack, cudaMemcpyHostToDevice) != cudaSuccess)
        goto done;
    if (cudaMemcpy(d_am, absmax, (size_t)nblocks * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        goto done;
    {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        k_expand<<<blocks, threads>>>(d_h, d_p, d_am, d_o, n, blocksize);
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        goto done;
    if (cudaMemcpy(out_f32, d_o, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        goto done;
    rc = 0;
done:
    cudaFree(d_h);
    cudaFree(d_p);
    cudaFree(d_am);
    cudaFree(d_o);
    return rc;
}
/* L0 load, device pointers: nothing crosses PCIe, f32 W stays on the GPU. */
int glint_expand_packed_to_f32_device(
    const uint8_t *d_hole_packed, const uint8_t *d_plug_packed, const float *d_absmax,
    int64_t n, int blocksize, float *d_out_f32, cudaStream_t stream)
{
    if (!d_hole_packed || !d_plug_packed || !d_absmax || !d_out_f32 || blocksize <= 0)
        return -1;
    if (load_cells() != 0)
        return -2;
    if (n <= 0)
        return 0;
    int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    k_expand<<<(unsigned)blocks, threads, 0, stream>>>(d_hole_packed, d_plug_packed, d_absmax, d_out_f32, n, blocksize);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

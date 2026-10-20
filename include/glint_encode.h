/* GLINT encode FFI — BF16/F32 on device → packed hole+plug.
 * Product path is CUDA. Host Python encode is tests only.
 */
#ifndef GLINT_ENCODE_H_
#define GLINT_ENCODE_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All pointers host. Internals cudaMalloc + kernels. Returns 0 on ok. */
int glint_encode_bf16_host(
    const uint16_t *w_bf16, int64_t n, int blocksize,
    uint8_t *hole_packed, uint8_t *plug_packed, float *absmax);
int glint_encode_f32_host(
    const float *w, int64_t n, int blocksize,
    uint8_t *hole_packed, uint8_t *plug_packed, float *absmax);
int glint_encode_n_absmax(int64_t n, int blocksize);
int glint_encode_n_packed(int64_t n);
/* L0 load: packed hole+plug → f32 on GPU, copy out. */
int glint_expand_packed_to_f32_host(
    const uint8_t *hole_packed, const uint8_t *plug_packed, const float *absmax,
    int64_t n, int blocksize, float *out_f32);
#ifdef __cplusplus
}
#endif
#endif

#ifndef GLINT_HTILE_LOAD_H_
#define GLINT_HTILE_LOAD_H_
#include <stdint.h>

#ifdef __CUDACC__
#define GLINT_LOAD_INLINE __host__ __device__ __forceinline__
#else
#define GLINT_LOAD_INLINE inline
#endif

GLINT_LOAD_INLINE uint8_t glint_packed_tile_byte(
    const uint8_t *packed, int row, int rows, int K, int k)
{
    uint8_t value = 0;
    if (row < rows) {
        for (int half = 0; half < 2; ++half) {
            if (k + half < K) {
                const int64_t i = (int64_t)row * K + k + half;
                const uint8_t nibble = (packed[i >> 1] >> (4 * (i & 1))) & 15;
                value |= (uint8_t)(nibble << (4 * half));
            }
        }
    }
    return value;
}

template <typename T>
GLINT_LOAD_INLINE T glint_x_tile_element(
    const T *x, int row, int rows, int K, int k)
{
    return row < rows && k < K ? x[(int64_t)row * K + k] : T(0);
}

#undef GLINT_LOAD_INLINE
#endif

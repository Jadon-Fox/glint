/* GLINT load ABI. hole=parent cell nibble, plug=child 0..15.
 * w = GLINT_CELLS[hole][plug] * absmax[i/blocksize]
 * L0 expand-at-load. L1 H-TILE lookup. CUDA.
 */
#ifndef GLINT_META_H_
#define GLINT_META_H_
#include "glint_cells.h"
#include <stdint.h>
#define DTYPE_GLINT 8
typedef struct {
    int32_t blocksize;
    int32_t nibble_order; /* 0=lo_then_hi */
    int32_t out_features;
    int32_t in_features;
    const uint8_t *hole;
    const uint8_t *plug;
    const float *absmax;
} GlintMeta;
static inline float glint_inflate(uint8_t hole, uint8_t plug, float absmax) {
    return GLINT_CELLS[hole & 15u][plug & 15u] * absmax;
}
#endif

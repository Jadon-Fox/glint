/* Test GLINT cell table integrity and cross-repo compatibility */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../include/glint_cells.h"

int main(void) {
    int errors = 0;
    printf("=== TEST GLINT CELLS TABLE INTEGRITY ===\n");

    /* 1. Self-consistency and range check [-1.0, 1.0] */
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float g = GLINT_CELLS[i][j];
            float p = P7_CELLS[i][j];

            uint32_t ug, up;
            memcpy(&ug, &g, sizeof(uint32_t));
            memcpy(&up, &p, sizeof(uint32_t));

            if (ug != up) {
                fprintf(stderr, "Mismatch between GLINT_CELLS and P7_CELLS at [%d][%d]: %e vs %e\n",
                        i, j, g, p);
                errors++;
            }

            if (isnan(g) || isinf(g) || g < -1.01f || g > 1.01f) {
                fprintf(stderr, "Value out of range at [%d][%d]: %e\n", i, j, g);
                errors++;
            }
        }
    }

    /* 2. Check monotonic properties across plug dimensions if expected */
    for (int i = 0; i < 16; i++) {
        for (int j = 1; j < 16; j++) {
            if (GLINT_CELLS[i][j] < GLINT_CELLS[i][j - 1]) {
                /* plugs in each hole bucket are monotonically increasing */
                fprintf(stderr, "Monotonicity violation at [%d][%d]: %e < %e\n",
                        i, j, GLINT_CELLS[i][j], GLINT_CELLS[i][j - 1]);
                errors++;
            }
        }
    }

    if (errors != 0) {
        fprintf(stderr, "TEST_CELLS_FAIL: %d errors found\n", errors);
        return 1;
    }

    printf("TEST_CELLS_OK: all 256 cells verified valid, monotonically ordered, and bit-identical to P7_CELLS.\n");
    return 0;
}

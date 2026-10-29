#include "../include/glint_htile_load.h"
#include "../include/glint_meta.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char *message, int K) {
    if (!ok) {
        std::fprintf(stderr, "FAIL K=%d %s\n", K, message);
        std::exit(1);
    }
}

int main() {
    constexpr int tile = 64;
    constexpr int M = 17, N = 19, blocksize = 23;
    for (int K : {1, tile - 1, tile, tile + 1, 3 * tile + 7}) {
        std::vector<uint8_t> hole((N * K + 1) / 2, 0), plug(hole.size(), 0);
        std::vector<float> x(M * K), weights(N * K), scales((N * K + blocksize - 1) / blocksize);
        for (size_t b = 0; b < scales.size(); ++b)
            scales[b] = 0.25f + (b % 7) * 0.125f;
        for (int i = 0; i < N * K; ++i) {
            const uint8_t h = (i * 7 + 3) % 16, p = (i * 11 + 5) % 16;
            hole[i / 2] |= h << (4 * (i % 2));
            plug[i / 2] |= p << (4 * (i % 2));
            weights[i] = GLINT_CELLS[h][p] * scales[i / blocksize];
        }
        for (int i = 0; i < M * K; ++i)
            x[i] = float((i * 3) % 17 - 8) * 0.0625f;
        float max_error = 0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float reference = 0, actual = 0;
                for (int k = 0; k < K; ++k)
                    reference += weights[n * K + k] * x[m * K + k];
                for (int k0 = 0; k0 < K; k0 += tile) {
                    uint8_t sh_h[tile / 2], sh_p[tile / 2];
                    float sh_x[tile];
                    for (int b = 0; b < tile / 2; ++b) {
                        sh_h[b] = glint_packed_tile_byte(hole.data(), n, N, K, k0 + 2 * b);
                        sh_p[b] = glint_packed_tile_byte(plug.data(), n, N, K, k0 + 2 * b);
                    }
                    for (int e = 0; e < tile; ++e) {
                        sh_x[e] = glint_x_tile_element(x.data(), m, M, K, k0 + e);
                        if (k0 + e >= K) {
                            require(sh_x[e] == 0, "X tail is not zero", K);
                            require(((sh_h[e / 2] >> (4 * (e % 2))) & 15) == 0,
                                    "hole tail is not zero", K);
                            require(((sh_p[e / 2] >> (4 * (e % 2))) & 15) == 0,
                                    "plug tail is not zero", K);
                        }
                    }
                    for (int kk = 0; kk < tile && k0 + kk < K; ++kk) {
                        const int h = (sh_h[kk / 2] >> (4 * (kk % 2))) & 15;
                        const int p = (sh_p[kk / 2] >> (4 * (kk % 2))) & 15;
                        actual += GLINT_CELLS[h][p] * scales[(n * K + k0 + kk) / blocksize] * sh_x[kk];
                    }
                }
                max_error = std::max(max_error, std::fabs(actual - reference));
            }
        }
        require(glint_packed_tile_byte(hole.data(), N, N, K, 0) == 0,
                "invalid packed row is not zero", K);
        require(glint_x_tile_element(x.data(), M, M, K, 0) == 0,
                "invalid X row is not zero", K);
        require(max_error <= 1e-6f, "tiled matmul differs from plain matmul", K);
        std::printf("PASS K=%d tile=%d M=%d N=%d max_abs_error=%.9g zero_padding=PASS\n",
                    K, tile, M, N, max_error);
    }
    std::puts("PASS 5 K-tail cases using production host/device loaders; GPU kernel NOT RUN (needs GPU)");
}

#include "../include/glint_meta.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>

extern "C" int launch_glint_htile_s3_f32(
    const uint8_t *, const uint8_t *, const float *, const __nv_bfloat16 *,
    float *, int, int, int, int);

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA FAIL %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

template <typename T>
static T *upload(const std::vector<T> &values) {
    T *device = nullptr;
    check(cudaMalloc(&device, values.size() * sizeof(T)));
    check(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

int main() {
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, 0));
    std::printf("DEVICE %s sm_%d%d\n", properties.name, properties.major, properties.minor);
    constexpr int blocksize = 23, guard = 16;
    constexpr float canary = 123456.0f;
    int cases = 0;
    const std::pair<int, int> shapes[] = {{1, 1}, {15, 15}, {16, 16}, {17, 19}, {33, 31}};
    for (int K : {1, 32, 40, 48, 63, 64, 65, 192, 199, 256}) {
        double worst_error = 0;
        for (const auto shape : shapes) {
            const int M = shape.first, N = shape.second;
            std::vector<uint8_t> hole((N * K + 1) / 2, 0), plug(hole.size(), 0);
            std::vector<float> weights(N * K), scales((N * K + blocksize - 1) / blocksize);
            std::vector<__nv_bfloat16> x(M * K);
            std::vector<float> y(M * N + 2 * guard, canary);
            for (size_t b = 0; b < scales.size(); ++b)
                scales[b] = 0.25f + (b % 7) * 0.125f;
            for (int i = 0; i < N * K; ++i) {
                const uint8_t h = (i * 7 + i / 17 + 3) % 16;
                const uint8_t p = (i * 11 + i / 13 + 5) % 16;
                hole[i / 2] |= h << (4 * (i % 2));
                plug[i / 2] |= p << (4 * (i % 2));
                weights[i] = GLINT_CELLS[h][p] * scales[i / blocksize];
            }
            for (int i = 0; i < M * K; ++i)
                x[i] = __float2bfloat16(float((i * 3 + i / 19) % 17 - 8) * 0.0625f);
            auto *d_hole = upload(hole);
            auto *d_plug = upload(plug);
            auto *d_scales = upload(scales);
            auto *d_x = upload(x);
            auto *d_y = upload(y);
            for (int repetition = 0; repetition < 3; ++repetition) {
                const int status = launch_glint_htile_s3_f32(
                    d_hole, d_plug, d_scales, d_x, d_y + guard, M, N, K, blocksize);
                if (status != 0) {
                    std::fprintf(stderr, "LAUNCH FAIL status=%d M=%d N=%d K=%d\n", status, M, N, K);
                    return 1;
                }
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int g = 0; g < guard; ++g) {
                    if (y[g] != canary || y[guard + M * N + g] != canary) {
                        std::fprintf(stderr, "CANARY FAIL M=%d N=%d K=%d\n", M, N, K);
                        return 1;
                    }
                }
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        double reference = 0;
                        for (int k = 0; k < K; ++k)
                            reference += double(weights[n * K + k]) * double(__bfloat162float(x[m * K + k]));
                        const double actual = y[guard + m * N + n];
                        const double error = std::fabs(actual - reference);
                        worst_error = std::max(worst_error, error);
                        if (!std::isfinite(actual) || error > 1e-5 * (1 + std::fabs(reference))) {
                            std::fprintf(stderr, "MATH FAIL M=%d N=%d K=%d m=%d n=%d actual=%.12g reference=%.12g\n",
                                         M, N, K, m, n, actual, reference);
                            return 1;
                        }
                    }
                }
            }
            check(cudaFree(d_hole));
            check(cudaFree(d_plug));
            check(cudaFree(d_scales));
            check(cudaFree(d_x));
            check(cudaFree(d_y));
            ++cases;
        }
        std::printf("PASS device K=%d shapes=5 repetitions=3 max_abs_error=%.9g canaries=PASS\n", K, worst_error);
    }
    std::printf("PASS device cases=%d launches=%d independent_matmul=PASS\n", cases, cases * 3);
}

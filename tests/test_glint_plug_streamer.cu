#include "../include/glint_plug_streamer.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

__global__ void k_verify_plug(const uint8_t *plug, int layer, uint64_t nbytes, int *d_err) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbytes) {
        uint8_t expected = (uint8_t)((layer * 37 + (idx % 251)) & 0xFF);
        if (plug[idx] != expected) {
            atomicAdd(d_err, 1);
        }
    }
}

int main(void) {
    printf("=== TEST GLINT PLUG STREAMER (Double-Buffer VRAM Transport) ===\n");

    const int n_layers = 32;
    const uint64_t bytes_per_layer = 4 * 1024 * 1024; /* 4 MiB test layer size (128 MiB total) */
    uint64_t layer_bytes[n_layers];
    for (int l = 0; l < n_layers; ++l) {
        layer_bytes[l] = bytes_per_layer;
    }

    cudaStream_t copy_stream, compute_stream;
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);

    GlintPlugStreamer *streamer = NULL;
    int rc = glint_plug_streamer_create(&streamer, n_layers, layer_bytes, copy_stream, compute_stream);
    assert(rc == 0);
    assert(streamer != NULL);

    /* Populate host buffers */
    uint8_t *host_buf = (uint8_t *)malloc(bytes_per_layer);
    for (int l = 0; l < n_layers; ++l) {
        for (uint64_t i = 0; i < bytes_per_layer; ++i) {
            host_buf[i] = (uint8_t)((l * 37 + (i % 251)) & 0xFF);
        }
        rc = glint_plug_streamer_set_host_data(streamer, l, host_buf, bytes_per_layer);
        assert(rc == 0);
    }
    free(host_buf);

    int *d_err = NULL;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));

    /* Start prefetch */
    rc = glint_plug_streamer_start(streamer);
    assert(rc == 0);

    /* Execute streaming pipeline */
    for (int l = 0; l < n_layers; ++l) {
        const uint8_t *dev_plug = NULL;
        rc = glint_plug_streamer_acquire(streamer, l, &dev_plug);
        assert(rc == 0);
        assert(dev_plug != NULL);

        int threads = 256;
        int blocks = (int)((bytes_per_layer + threads - 1) / threads);
        k_verify_plug<<<blocks, threads, 0, compute_stream>>>(dev_plug, l, bytes_per_layer, d_err);

        rc = glint_plug_streamer_release(streamer, l);
        assert(rc == 0);
    }

    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(copy_stream);

    int h_err = 0;
    cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_err);

    GlintStreamerStats stats;
    rc = glint_plug_streamer_get_stats(streamer, &stats);
    assert(rc == 0);

    glint_plug_streamer_destroy(streamer);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);

    printf("Total H2D bytes: %llu (expected %llu)\n",
           (unsigned long long)stats.total_h2d_bytes,
           (unsigned long long)(n_layers * bytes_per_layer));
    printf("Total compute stall time: %.3f ms\n", stats.total_stall_ms);
    for (int l = 0; l < 4; ++l) {
        printf("  Layer %d stall: %.3f ms\n", l, stats.per_layer_stall_ms[l]);
    }
    printf("Verification errors: %d\n", h_err);

    if (h_err == 0 && stats.total_h2d_bytes == n_layers * bytes_per_layer) {
        printf("GLINT PLUG STREAMER TEST PASSED GREEN [0 CHECKSUM MISMATCHES].\n");
        return 0;
    } else {
        printf("GLINT PLUG STREAMER TEST FAILED.\n");
        return 1;
    }
}

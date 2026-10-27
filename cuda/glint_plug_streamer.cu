/* glint_plug_streamer.cu — Generalized 2-slot VRAM ring plug streamer for GLINT.
 * Event-only synchronization overlapping PCIe plug streaming with compute.
 */
#include "glint_plug_streamer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct GlintPlugStreamer {
    int n_layers;
    uint64_t layer_bytes[64];
    uint64_t max_bytes;
    uint8_t *pinned[64];
    uint8_t *slot[2];
    cudaStream_t copy_stream;
    cudaStream_t compute_stream;
    cudaEvent_t h2d_b[64];
    cudaEvent_t h2d_e[64];
    cudaEvent_t cmp_b[64];
    cudaEvent_t cmp_e[64];
    cudaEvent_t stall_b[64];
    cudaEvent_t stall_e[64];
    int has_started;
    GlintStreamerStats stats;
};

extern "C" int glint_plug_streamer_create(
    GlintPlugStreamer **out, int n_layers, const uint64_t *layer_bytes,
    cudaStream_t copy_stream, cudaStream_t compute_stream)
{
    if (!out || n_layers <= 0 || n_layers > 64 || !layer_bytes)
        return -1;

    GlintPlugStreamer *s = (GlintPlugStreamer *)calloc(1, sizeof(GlintPlugStreamer));
    if (!s)
        return -2;

    s->n_layers = n_layers;
    s->copy_stream = copy_stream;
    s->compute_stream = compute_stream;

    uint64_t max_b = 0;
    for (int l = 0; l < n_layers; ++l) {
        s->layer_bytes[l] = layer_bytes[l];
        if (layer_bytes[l] > max_b)
            max_b = layer_bytes[l];
    }
    s->max_bytes = max_b;

    /* Allocate 2 device slots in VRAM */
    for (int b = 0; b < 2; ++b) {
        if (cudaMalloc((void **)&s->slot[b], max_b) != cudaSuccess) {
            glint_plug_streamer_destroy(s);
            return -3;
        }
    }

    /* Allocate pinned host memory for all layers and create sync events */
    for (int l = 0; l < n_layers; ++l) {
        if (cudaHostAlloc((void **)&s->pinned[l], s->layer_bytes[l], cudaHostAllocDefault) != cudaSuccess) {
            glint_plug_streamer_destroy(s);
            return -4;
        }
        if (cudaEventCreate(&s->h2d_b[l]) != cudaSuccess ||
            cudaEventCreate(&s->h2d_e[l]) != cudaSuccess ||
            cudaEventCreate(&s->cmp_b[l]) != cudaSuccess ||
            cudaEventCreate(&s->cmp_e[l]) != cudaSuccess ||
            cudaEventCreate(&s->stall_b[l]) != cudaSuccess ||
            cudaEventCreate(&s->stall_e[l]) != cudaSuccess)
        {
            glint_plug_streamer_destroy(s);
            return -5;
        }
    }

    s->stats.n_layers = n_layers;
    *out = s;
    return 0;
}

extern "C" void glint_plug_streamer_destroy(GlintPlugStreamer *s) {
    if (!s)
        return;
    for (int b = 0; b < 2; ++b) {
        if (s->slot[b])
            cudaFree(s->slot[b]);
    }
    for (int l = 0; l < s->n_layers; ++l) {
        if (s->pinned[l])
            cudaFreeHost(s->pinned[l]);
        if (s->h2d_b[l]) cudaEventDestroy(s->h2d_b[l]);
        if (s->h2d_e[l]) cudaEventDestroy(s->h2d_e[l]);
        if (s->cmp_b[l]) cudaEventDestroy(s->cmp_b[l]);
        if (s->cmp_e[l]) cudaEventDestroy(s->cmp_e[l]);
        if (s->stall_b[l]) cudaEventDestroy(s->stall_b[l]);
        if (s->stall_e[l]) cudaEventDestroy(s->stall_e[l]);
    }
    free(s);
}

extern "C" int glint_plug_streamer_set_host_data(
    GlintPlugStreamer *s, int layer, const void *host_src, uint64_t bytes)
{
    if (!s || layer < 0 || layer >= s->n_layers || !host_src)
        return -1;
    if (bytes != s->layer_bytes[layer])
        return -2;
    memcpy(s->pinned[layer], host_src, (size_t)bytes);
    return 0;
}

extern "C" int glint_plug_streamer_start(GlintPlugStreamer *s) {
    if (!s || s->n_layers <= 0)
        return -1;

    if (s->has_started && s->n_layers >= 2) {
        int last_slot0_layer = (s->n_layers - 1) & ~1;
        if (cudaStreamWaitEvent(s->copy_stream, s->cmp_e[last_slot0_layer], 0) != cudaSuccess)
            return -5;
    }
    s->has_started = 1;

    /* Prefetch layer 0 into slot 0 */
    if (cudaEventRecord(s->h2d_b[0], s->copy_stream) != cudaSuccess)
        return -2;
    if (cudaMemcpyAsync(s->slot[0], s->pinned[0], (size_t)s->layer_bytes[0],
                        cudaMemcpyHostToDevice, s->copy_stream) != cudaSuccess)
        return -3;
    if (cudaEventRecord(s->h2d_e[0], s->copy_stream) != cudaSuccess)
        return -4;

    s->stats.total_h2d_bytes += s->layer_bytes[0];
    return 0;
}

extern "C" int glint_plug_streamer_acquire(
    GlintPlugStreamer *s, int layer, const uint8_t **out_dev_plug)
{
    if (!s || layer < 0 || layer >= s->n_layers || !out_dev_plug)
        return -1;

    int cur_slot = layer & 1;

    /* Issue async copy for layer + 1 on copy_stream */
    if (layer + 1 < s->n_layers) {
        int next_layer = layer + 1;
        int next_slot = next_layer & 1;

        if (layer >= 1) {
            if (cudaStreamWaitEvent(s->copy_stream, s->cmp_e[layer - 1], 0) != cudaSuccess)
                return -2;
        }

        if (cudaEventRecord(s->h2d_b[next_layer], s->copy_stream) != cudaSuccess)
            return -3;
        if (cudaMemcpyAsync(s->slot[next_slot], s->pinned[next_layer],
                            (size_t)s->layer_bytes[next_layer],
                            cudaMemcpyHostToDevice, s->copy_stream) != cudaSuccess)
            return -4;
        if (cudaEventRecord(s->h2d_e[next_layer], s->copy_stream) != cudaSuccess)
            return -5;

        s->stats.total_h2d_bytes += s->layer_bytes[next_layer];
    }

    /* On compute_stream: record stall_b, wait for H2D event of layer, record stall_e */
    if (cudaEventRecord(s->stall_b[layer], s->compute_stream) != cudaSuccess)
        return -6;
    if (cudaStreamWaitEvent(s->compute_stream, s->h2d_e[layer], 0) != cudaSuccess)
        return -7;
    if (cudaEventRecord(s->stall_e[layer], s->compute_stream) != cudaSuccess)
        return -8;

    if (cudaEventRecord(s->cmp_b[layer], s->compute_stream) != cudaSuccess)
        return -9;

    *out_dev_plug = s->slot[cur_slot];
    return 0;
}

extern "C" int glint_plug_streamer_release(GlintPlugStreamer *s, int layer) {
    if (!s || layer < 0 || layer >= s->n_layers)
        return -1;
    if (cudaEventRecord(s->cmp_e[layer], s->compute_stream) != cudaSuccess)
        return -2;
    return 0;
}

extern "C" int glint_plug_streamer_get_stats(GlintPlugStreamer *s, GlintStreamerStats *stats) {
    if (!s || !stats)
        return -1;
    cudaDeviceSynchronize();
    double total_stall = 0.0;
    for (int l = 0; l < s->n_layers; ++l) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s->stall_b[l], s->stall_e[l]);
        s->stats.per_layer_stall_ms[l] = (double)ms;
        total_stall += (double)ms;
    }
    s->stats.total_stall_ms = total_stall;
    *stats = s->stats;
    return 0;
}

extern "C" void glint_plug_streamer_reset_stats(GlintPlugStreamer *s) {
    if (!s)
        return;
    cudaStreamSynchronize(s->copy_stream);
    cudaStreamSynchronize(s->compute_stream);
    s->stats.total_h2d_bytes = 0;
    s->stats.total_stall_ms = 0.0;
    memset(s->stats.per_layer_stall_ms, 0, sizeof(s->stats.per_layer_stall_ms));
    s->has_started = 0;
}

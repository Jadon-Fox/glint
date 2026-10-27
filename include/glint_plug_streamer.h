#ifndef GLINT_PLUG_STREAMER_H
#define GLINT_PLUG_STREAMER_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GlintPlugStreamer GlintPlugStreamer;

typedef struct {
    uint64_t total_h2d_bytes;
    double total_stall_ms;
    double per_layer_stall_ms[64];
    int n_layers;
} GlintStreamerStats;

/* Create a 2-slot VRAM ring streamer for n_layers.
 * Allocates 2 device slots of max_layer_plug_bytes and pinned host buffers.
 */
int glint_plug_streamer_create(GlintPlugStreamer **out, int n_layers, const uint64_t *layer_bytes,
                               cudaStream_t copy_stream, cudaStream_t compute_stream);

/* Destroy streamer and free all device and host resources. */
void glint_plug_streamer_destroy(GlintPlugStreamer *s);

/* Set host plug data for layer l into pinned host memory. */
int glint_plug_streamer_set_host_data(GlintPlugStreamer *s, int layer, const void *host_src, uint64_t bytes);

/* Start prefetching layer 0 before the layer loop begins. */
int glint_plug_streamer_start(GlintPlugStreamer *s);

/* Acquire device slot for layer l on compute_stream:
 * - If l + 1 < n_layers: issues async copy for layer l + 1 into slot (l + 1) & 1.
 * - On compute_stream: waits for layer l H2D copy to complete.
 * - Returns pointer to device plug buffer for layer l.
 */
int glint_plug_streamer_acquire(GlintPlugStreamer *s, int layer, const uint8_t **out_dev_plug);

/* Release layer l after compute finishes layer l.
 * Records compute end event so the copy stream knows when the slot can be reused.
 */
int glint_plug_streamer_release(GlintPlugStreamer *s, int layer);

/* Retrieve measured stats: total stall time, per-layer stall, total bytes. */
int glint_plug_streamer_get_stats(GlintPlugStreamer *s, GlintStreamerStats *stats);

/* Reset recorded stats and synchronize streams between passes. */
void glint_plug_streamer_reset_stats(GlintPlugStreamer *s);

/* Backward-compatible aliases for P7 callers */
typedef GlintPlugStreamer P7PlugStreamer;
typedef GlintStreamerStats P7StreamerStats;
#define p7_plug_streamer_create glint_plug_streamer_create
#define p7_plug_streamer_destroy glint_plug_streamer_destroy
#define p7_plug_streamer_set_host_data glint_plug_streamer_set_host_data
#define p7_plug_streamer_start glint_plug_streamer_start
#define p7_plug_streamer_acquire glint_plug_streamer_acquire
#define p7_plug_streamer_release glint_plug_streamer_release
#define p7_plug_streamer_get_stats glint_plug_streamer_get_stats
#define p7_plug_streamer_reset_stats glint_plug_streamer_reset_stats

#ifdef __cplusplus
}
#endif

#endif /* GLINT_PLUG_STREAMER_H */

/* tests/test_arm0_bf16_full.cu — Full 32-Layer Arm (0) BF16 Evaluation on 2048 Tokens
 * Loads unquantized BF16 weights directly from model-00001-of-00002.safetensors and model-00002-of-00002.safetensors.
 * Executes on RTX 3060 via Tensor Core cuBLAS and online softmax causal attention.
 * Verifies Gate 1:
 *   - Top-1 argmax token agreement >= 95.0% vs HuggingFace (amended 2026-11-02, GATE_RULES.md C.1) Oracle
 *   - Perplexity within 1.0% of HuggingFace Oracle
 * Exit: 0 gate PASS (or INT8 report), 1 gate FAIL, 2 usage, 3 setup, I/O or CUDA error.
 */

#include "arm0_forward.h"
#include "act_map.h"
#include "nf4.h"
#include "../include/glint_htile.h"
#include "../include/glint_plug_streamer.h"
#include <vector>

#define INT8_BLOCK 64

__global__ void int8_block_dequant_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                          __nv_bfloat16* __restrict__ out, size_t numel) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = __float2bfloat16((float)q[i] * scale[i / INT8_BLOCK]);
}

static double g_nf4_se, g_nf4_ref2, g_nf4_max; static uint64_t g_nf4_n; static FILE *g_nf4_map; static int g_nf4_first;
static __nv_bfloat16* upload_linear(const SafeTensors *st1, const SafeTensors *st2, const SafeTensors *int8_pin,
                                    const char *name, size_t numel) {
    __nv_bfloat16 *d_w;
    CK(cudaMalloc(&d_w, numel * sizeof(__nv_bfloat16)));
    if (!int8_pin) {
        const TensorEntry *e = need_tensor(st1, st2, name, "BF16", numel * sizeof(__nv_bfloat16));
        if (getenv("ARM0_NF4")) { /* NF4 comparison point and S0-WMAP-NF4 */
            std::vector<uint16_t> rt(numel);
            double se = 0, ref2 = 0, mx = 0;
            nf4_roundtrip((const uint16_t *)e->data, rt.data(), numel, &se, &ref2, &mx);
            g_nf4_se += se; g_nf4_ref2 += ref2; g_nf4_n += numel; if (mx > g_nf4_max) g_nf4_max = mx;
            if (g_nf4_map) fprintf(g_nf4_map, "%s{\"name\":\"%s\",\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}", g_nf4_first++ ? "," : "", name, sqrt(se / numel), mx, sqrt(se / ref2));
            CK(cudaMemcpy(d_w, rt.data(), numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            return d_w;
        }
        CK(cudaMemcpy(d_w, e->data, numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        return d_w;
    }
    char scale_name[300];
    snprintf(scale_name, sizeof(scale_name), "%s.int8_scale", name);
    size_t scale_bytes = (numel + INT8_BLOCK - 1) / INT8_BLOCK * sizeof(float);
    const TensorEntry *eq = need_tensor(int8_pin, NULL, name, "I8", numel);
    const TensorEntry *es = need_tensor(int8_pin, NULL, scale_name, "F32", scale_bytes);
    int8_t *d_q;
    float *d_s;
    CK(cudaMalloc(&d_q, numel));
    CK(cudaMalloc(&d_s, scale_bytes));
    CK(cudaMemcpy(d_q, eq->data, numel, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, es->data, scale_bytes, cudaMemcpyHostToDevice));
    int8_block_dequant_kernel<<<(unsigned)((numel + 255) / 256), 256>>>(d_q, d_s, d_w, numel);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d_q));
    CK(cudaFree(d_s));
    return d_w;
}

/* --glint PINDIR [--hole-only]: linears through the H-TILE kernel on a glint_pin_v1 (hole + plug + absmax). */
struct GlintLin { uint8_t *hole, *plug; float *am; };
static int g_glint = 0, g_hole_only = 0;
static GlintLin g_gl[N_LAYERS][ARM0_N_LINEARS];
static uint8_t *g_zero_plug = NULL;
/* --glint-stream PINDIR: hole resident, plugs staged in pinned host RAM and brought in one layer ahead through the
 * 2-slot ring (glint_plug_streamer). One layer's plug is its four linears back to back (QKV, O, GATE_UP, DOWN). */
static int g_stream = 0;
static GlintPlugStreamer *g_streamer = NULL;
static cudaStream_t g_copy_stream = NULL;
static const uint8_t *g_plug_src[N_LAYERS][ARM0_N_LINEARS];
static uint64_t g_plug_off[N_LAYERS][ARM0_N_LINEARS], g_layer_bytes[N_LAYERS];
static const uint8_t *g_cur_dev_plug = NULL;
static void glint_upload(const SafeTensors *a, const SafeTensors *b, int l, int j) {
    size_t n = (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in;
    char nm[256], pn[300], an[300];
    snprintf(nm, sizeof(nm), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
    snprintf(pn, sizeof(pn), "%s.glint_plug", nm); snprintf(an, sizeof(an), "%s.absmax", nm);
    size_t nb = (n + 1) / 2, ab = (n + 63) / 64 * sizeof(float);
    const TensorEntry *eh = need_tensor(a, b, nm, "U8", nb), *ep = need_tensor(a, b, pn, "U8", nb), *ea = need_tensor(a, b, an, "F32", ab);
    GlintLin &g = g_gl[l][j];
    CK(cudaMalloc(&g.hole, nb)); CK(cudaMemcpy(g.hole, eh->data, nb, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&g.am, ab)); CK(cudaMemcpy(g.am, ea->data, ab, cudaMemcpyHostToDevice));
    if (g_stream) { /* plug stays on the host; the ring brings it in */
        g_plug_src[l][j] = (const uint8_t *)ep->data; g_plug_off[l][j] = g_layer_bytes[l]; g_layer_bytes[l] += nb; g.plug = NULL;
    } else if (g_hole_only) { if (!g_zero_plug) { CK(cudaMalloc(&g_zero_plug, 8192ull * 3072 / 2 * 2)); CK(cudaMemset(g_zero_plug, 0, 8192ull * 3072)); } g.plug = g_zero_plug; }
    else { CK(cudaMalloc(&g.plug, nb)); CK(cudaMemcpy(g.plug, ep->data, nb, cudaMemcpyHostToDevice)); }
}

struct CublasLinears {
    cublasHandle_t cublas;
    __nv_bfloat16 *w[N_LAYERS][ARM0_N_LINEARS];
};

static void cublas_linear(void *ctx, int layer, Arm0Linear which, const __nv_bfloat16 *x, float *y, int m) {
    const CublasLinears *lin = (const CublasLinears *)ctx;
    act_capture(layer, which, x, m);
    if (g_glint) {
        const GlintLin &g = g_gl[layer][which];
        if (g_stream) {
            if (which == ARM0_QKV && glint_plug_streamer_acquire(g_streamer, layer, &g_cur_dev_plug) != 0)
                setup_fail("plug acquire failed layer %d", layer);
            if (!g_cur_dev_plug) setup_fail("plug for layer %d not acquired (linear order changed)", layer);
            if (launch_glint_htile_s3_f32_stream(g.hole, g_cur_dev_plug + g_plug_off[layer][which], g.am, x, y, m,
                                                 ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64, 0) != 0)
                setup_fail("htile launch failed layer %d", layer);
            if (which == ARM0_DOWN) {
                if (glint_plug_streamer_release(g_streamer, layer) != 0) setup_fail("plug release failed layer %d", layer);
                g_cur_dev_plug = NULL;
            }
            return;
        }
        if (launch_glint_htile_s3_f32(g.hole, g.plug, g.am, x, y, m, ARM0_LINEAR[which].n_out, ARM0_LINEAR[which].k_in, 64) != 0)
            setup_fail("htile launch failed layer %d", layer);
        return;
    }
    int n_out = ARM0_LINEAR[which].n_out, k_in = ARM0_LINEAR[which].k_in;
    float alpha = 1.0f, beta = 0.0f;
    CKB(cublasGemmEx(lin->cublas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, m, k_in, &alpha, lin->w[layer][which], CUDA_R_16BF,
                     k_in, x, CUDA_R_16BF, k_in, &beta, y, CUDA_R_32F, n_out, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static Arm0Result g_result;
static float g_dump[N_LAYERS + 1][N_DUMP][HIDDEN_DIM];

int main(int argc, char** argv) {
    const char* dump_dir = NULL;
    const char* int8_pin_path = NULL;
    const char* glint_dir = NULL;
    int argi = 1;
    for (; argi + 1 < argc && argv[argi][0] == '-'; argi += 2) {
        if (strcmp(argv[argi], "--dump") == 0) dump_dir = argv[argi + 1];
        else if (strcmp(argv[argi], "--int8") == 0) int8_pin_path = argv[argi + 1];
        else if (strcmp(argv[argi], "--glint") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; }
        else if (strcmp(argv[argi], "--glint-hole-only") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_hole_only = 1; }
        else if (strcmp(argv[argi], "--glint-stream") == 0) { glint_dir = argv[argi + 1]; g_glint = 1; g_stream = 1; }
        else break;
    }
    if (argc - argi != 2) {
        printf("usage: %s [--dump DIR] [--int8 PIN.safetensors] <model_snapshot_dir> <artifacts_dir>\n", argv[0]);
        return EXIT_USAGE;
    }
    const char* model_dir = argv[argi];
    const char* artifacts_dir = argv[argi + 1];
    char path[1024];

    printf("=========================================================\n");
    printf("Arm (0) Full 32-Layer BF16 Evaluator (2048 Tokens)\n");
    printf("=========================================================\n");

    if (dump_dir && mkdir(dump_dir, 0755) != 0 && errno != EEXIST)
        setup_fail("cannot create dump dir %s: %s", dump_dir, strerror(errno));

    static Arm0Eval ev;
    arm0_read_eval(artifacts_dir, &ev);

    SafeTensors st1, st2;
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", model_dir);
    if (safetensors_open(&st1, p1) != 0 || safetensors_open(&st2, p2) != 0)
        setup_fail("cannot open safetensors shards in %s", model_dir);
    Arm0Model model;
    arm0_upload_model(&st1, &st2, &model);

    SafeTensors st_int8;
    const SafeTensors* int8_pin = NULL;
    if (int8_pin_path) {
        if (safetensors_open(&st_int8, int8_pin_path) != 0) setup_fail("cannot open INT8 pin %s", int8_pin_path);
        int8_pin = &st_int8;
    }

    static CublasLinears lin;
    SafeTensors gs1, gs2;
    if (g_glint) {
        char a[1024], b[1024];
        snprintf(a, sizeof(a), "%s/model-00001-of-00002.safetensors", glint_dir);
        snprintf(b, sizeof(b), "%s/model-00002-of-00002.safetensors", glint_dir);
        if (safetensors_open(&gs1, a) != 0 || safetensors_open(&gs2, b) != 0) setup_fail("cannot open glint pin %s", glint_dir);
        for (int l = 0; l < N_LAYERS; l++) for (int j = 0; j < ARM0_N_LINEARS; j++) glint_upload(&gs1, &gs2, l, j);
        if (g_stream) {
            CK(cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking));
            if (glint_plug_streamer_create(&g_streamer, N_LAYERS, g_layer_bytes, g_copy_stream, 0) != 0) setup_fail("streamer create failed");
            for (int l = 0; l < N_LAYERS; l++) {
                std::vector<uint8_t> cat(g_layer_bytes[l]);
                for (int j = 0; j < ARM0_N_LINEARS; j++) {
                    size_t nb = ((size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in + 1) / 2;
                    memcpy(cat.data() + g_plug_off[l][j], g_plug_src[l][j], nb);
                }
                if (glint_plug_streamer_set_host_data(g_streamer, l, cat.data(), g_layer_bytes[l]) != 0) setup_fail("streamer host data layer %d", l);
            }
            if (glint_plug_streamer_start(g_streamer) != 0) setup_fail("streamer start failed");
        }
        printf("Glint pin loaded (%s).\n", g_stream ? "hole resident, plug streamed through the 2-slot ring" : (g_hole_only ? "hole only, plug zeroed" : "hole + plug"));
    }
    if (getenv("ARM0_NF4") && getenv("ARM0_NF4_WMAP")) {
        g_nf4_map = fopen(getenv("ARM0_NF4_WMAP"), "w");
        if (g_nf4_map) fprintf(g_nf4_map, "{\"schema\":\"nf4_weight_diff_map_v1\",\"scheme\":\"NF4 blocksize 64 absmax, bf16 dequant\",\"tensors\":[");
    }
    for (int l = 0; l < N_LAYERS && !g_glint; l++) {
        for (int j = 0; j < ARM0_N_LINEARS; j++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s.weight", l, ARM0_LINEAR[j].name);
            lin.w[l][j] = upload_linear(&st1, &st2, int8_pin, name, (size_t)ARM0_LINEAR[j].n_out * ARM0_LINEAR[j].k_in);
        }
    }
    if (getenv("ARM0_NF4")) {
        printf("NF4 weights: %llu linear elems, RMSE %.6e, max abs %.6e, rel L2 %.6e\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2));
        if (g_nf4_map) { fprintf(g_nf4_map, "],\"n_elems\":%llu,\"rmse\":%.6e,\"max_abs\":%.6e,\"rel_l2\":%.6e}\n", (unsigned long long)g_nf4_n, sqrt(g_nf4_se / g_nf4_n), g_nf4_max, sqrt(g_nf4_se / g_nf4_ref2)); fclose(g_nf4_map); }
    }
    printf("Uploaded all 32 layers of %s weights to VRAM.\n", int8_pin ? "INT8-dequantized" : "BF16");

    const char *act_map = getenv("ARM0_ACT_MAP");
    act_map_init(act_map);
    CKB(cublasCreate(&lin.cublas));
    arm0_forward(&model, ev.tokens, cublas_linear, &lin, lin.cublas, dump_dir ? g_dump : NULL, &g_result);
    if (g_stream) {
        GlintStreamerStats ss;
        if (glint_plug_streamer_get_stats(g_streamer, &ss) == 0) {
            double mx = 0; for (int l = 1; l < ss.n_layers; l++) if (ss.per_layer_stall_ms[l] > mx) mx = ss.per_layer_stall_ms[l];
            printf("STREAM_STATS h2d_bytes=%llu total_stall_ms=%.3f layer0_stall_ms=%.3f max_stall_ms_layers_1plus=%.3f\n",
                   (unsigned long long)ss.total_h2d_bytes, ss.total_stall_ms, ss.per_layer_stall_ms[0], mx);
        }
    }
    act_map_write(act_map, int8_pin ? "BF16 cuBLAS (INT8-dequantized weights)" : "BF16 cuBLAS (original BF16 weights)");
    const float* h_losses = g_result.losses;
    const int32_t* h_top1 = g_result.top1;
    const uint32_t* oracle_top1 = ev.oracle_top1;
    const double oracle_ppl = ev.oracle_ppl;
    double total_sec = g_result.seconds;

    double total_ce = 0.0;
    int top1_matches = 0;
    int printed = 0;
    for (int i = 0; i < 2047; i++) {
        total_ce += h_losses[i];
        if (h_top1[i] == (int32_t)oracle_top1[i]) {
            top1_matches++;
        } else if (printed < 10) {
            printf("Mismatch at pos %d: our_top1=%d oracle_top1=%d loss=%.4f\n",
                   i, h_top1[i], oracle_top1[i], h_losses[i]);
            printed++;
        }
    }
    double mean_ce = total_ce / 2047.0;
    double ppl = exp(mean_ce);
    double top1_acc = (double)top1_matches / 2047.0 * 100.0;

    double ppl_delta_pct = fabs(ppl - oracle_ppl) / oracle_ppl * 100.0;

    printf("\n=== %s Evaluation Results ===\n", int8_pin ? "INT8 pin" : "Arm (0) BF16");
    printf("Forward Pass Time:      %.3f s (%.1f tok/s)\n", total_sec, 2048.0 / total_sec);
    printf("Mean Cross Entropy:     %.6f\n", mean_ce);
    printf("Measured Perplexity:    %.6f\n", ppl);
    printf("HuggingFace Oracle PPL: %.6f\n", oracle_ppl);
    printf("PPL Delta vs Oracle:    %.4f %%\n", ppl_delta_pct);
    printf("Top-1 Matches:          %d / 2047\n", top1_matches);
    printf("Top-1 Agreement:        %.2f %%\n", top1_acc);

    int pass = top1_acc >= 95.0 && ppl_delta_pct <= 1.0;
    if (g_glint) printf("VERDICT: REPORT ONLY (glint %s)\n", g_hole_only ? "hole-only" : "hole+plug");
    else if (int8_pin) printf("VERDICT: REPORT ONLY (INT8 pin %s, no gate)\n", int8_pin_path);
    else printf(pass ? "VERDICT: PASS (Gate 1 Anchor Verified)\n" : "VERDICT: FAIL (Gate 1 Anchor Mismatch)\n");
    fflush(stdout);

    if (dump_dir) {
        snprintf(path, sizeof(path), "%s/cuda_hidden.bin", dump_dir);
        FILE* fh = fopen(path, "wb");
        if (!fh) setup_fail("cannot write %s", path);
        fwrite(g_dump, sizeof(g_dump), 1, fh);
        fclose(fh);
        snprintf(path, sizeof(path), "%s/cuda_pos.bin", dump_dir);
        FILE* fp = fopen(path, "wb");
        if (!fp) setup_fail("cannot write %s", path);
        fwrite(h_losses, sizeof(float), 2047, fp);
        fwrite(h_top1, sizeof(int32_t), 2047, fp);
        fclose(fp);
    }

    safetensors_close(&st1);
    safetensors_close(&st2);
    cublasDestroy(lin.cublas);
    return (pass || int8_pin || g_glint) ? EXIT_GATE_PASS : EXIT_GATE_FAIL;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "/home/workspace/p7-scratch/tests/test_st_loader.c"

int main() {
    SafeTensors st;
    int rc = safetensors_open(&st, "/home/workspace/training_orchestrator/artifacts/pins/Phi-4-mini-int8-from-bf16/model.safetensors");
    printf("int8 pin open rc=%d, n_entries=%d\n", rc, st.n_entries);
    const TensorEntry *e = safetensors_find(&st, "model.layers.0.self_attn.qkv_proj.weight");
    if (e) {
        printf("found qkv: dtype=%s shape=[%ld, %ld] bytes=%lu\n", e->dtype, e->shape[0], e->shape[1], e->end - e->begin);
    }
    const TensorEntry *s = safetensors_find(&st, "model.layers.0.self_attn.qkv_proj.weight.int8_scale");
    if (s) {
        printf("found qkv scale: dtype=%s shape=[%ld] bytes=%lu\n", s->dtype, s->shape[0], s->end - s->begin);
    }
    safetensors_close(&st);
    return 0;
}

#include <stdio.h>
#include "../core/safetensors_c.h"

int main() {
    SafeTensors st1, st2;
    if (safetensors_open(&st1, "/home/workspace/p7-scratch/artifacts/pins/Phi-4-mini-nested-nf8/model.safetensors") != 0) {
        printf("st1 open failed\n"); return 1;
    }
    if (safetensors_open(&st2, "/home/workspace/training_orchestrator/artifacts/pins/Phi-4-mini-int8-from-bf16/model.safetensors") != 0) {
        printf("st2 open failed\n"); return 1;
    }
    printf("st1 (nested-nf8): %d entries\n", st1.n_entries);
    printf("st2 (int8):       %d entries\n", st2.n_entries);
    const TensorEntry *e1 = safetensors_find(&st1, "model.layers.0.self_attn.qkv_proj.weight");
    const TensorEntry *e2 = safetensors_find(&st2, "model.layers.0.self_attn.qkv_proj.weight");
    printf("st1 qkv: %s, %ld bytes\n", e1 ? e1->dtype : "null", e1 ? e1->end - e1->begin : 0);
    printf("st2 qkv: %s, %ld bytes\n", e2 ? e2->dtype : "null", e2 ? e2->end - e2->begin : 0);
    safetensors_close(&st1);
    safetensors_close(&st2);
    printf("PASS\n");
    return 0;
}

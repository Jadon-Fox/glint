"""Job 22: SYNTHETIC inputs for apparatus smoke tests only (no real weights, no real X). Seeds fixed.
 reg_*: a synthetic safetensors file (tensor name model.layers.0.self_attn.o_proj.weight, BF16 3072x3072) + synthetic X, for the
        old-vs-new lineage regression (job20/p7fold built from 04bfd293 vs p7fold2 on identical synthetic inputs).
 sk_*, sd_*: synthetic X for the measure-mode smoke tests (synthetic weights are generated inside the harness)."""
import json, struct
import numpy as np
O = "/home/workspace/supervisor/scratch/p7-fold-harness/job22/smoke/"
rng = np.random.default_rng(20261022)
def bf16(a):
    u = a.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)
W = bf16(rng.standard_normal((3072, 3072)).astype(np.float32) * 0.02)
name = "model.layers.0.self_attn.o_proj.weight"
hdr = json.dumps({name: {"dtype": "BF16", "shape": [3072, 3072], "data_offsets": [0, W.nbytes]}}).encode()
with open(O + "reg_syn.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hdr))); f.write(hdr); f.write(W.tobytes())
def synthx(rows, n, scales):
    x = rng.standard_normal((rows, n)).astype(np.float32)
    for c, s in scales.items(): x[:, c] *= s
    return x
sc = {7: 40.0, 100: 15.0, 1000: 6.0}
synthx(2048, 3072, sc).tofile(O + "reg_xcal.f32"); synthx(2048, 3072, sc).tofile(O + "reg_xeval.f32")
rng.integers(0, 1000, 2048).astype("<i4").tofile(O + "reg_cal.i32"); rng.integers(0, 1000, 2048).astype("<i4").tofile(O + "reg_eval.i32")
synthx(4096, 3072, sc).tofile(O + "sk_xcal.f32"); synthx(2048, 3072, sc).tofile(O + "sk_xe1.f32"); synthx(2048, 3072, sc).tofile(O + "sk_xe2.f32")
synthx(4096, 8192, sc).tofile(O + "sd_xcal.f32"); synthx(2048, 8192, sc).tofile(O + "sd_xe1.f32")
print("ok")

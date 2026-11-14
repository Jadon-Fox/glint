#!/usr/bin/env python3
"""Independent check of mojo/p7_trace_smol.mojo (verification tooling only; not on any train/serve/eval path).

Decodes the SAME glint_pin_v1 pin with numpy, runs an f32 SmolLM2 forward for the same token ids, and compares the traced
tensors against it. Also runs a forward on the ORIGINAL BF16 weights, to separate GLINT quantisation error from tracer error.
Usage (container python with numpy): p7_trace_verify_numpy.py PIN_DIR BF16_SAFETENSORS TRACE_DIR   (a glint_trace_v2 directory)
"""
import json, struct, sys
import numpy as np

pin_dir, bf16_path, trace_dir = sys.argv[1], sys.argv[2], sys.argv[3]
H, NH, NKV, HD, INTER, NL, V = 576, 9, 3, 64, 1536, 30, 49152
THETA, EPS = 100000.0, 1e-5


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        data = np.memmap(path, dtype=np.uint8, mode="r")
    return hdr, base, data


def get(hdr, base, data, name):
    e = hdr[name]
    b, en = e["data_offsets"]
    return data[base + b: base + en], e


def bf16_to_f32(raw):
    u = raw.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


hdr, base, data = read_st(pin_dir + "/model.safetensors")
cells = get(hdr, base, data, "_glint_cells")[0].view(np.float32).reshape(16, 16)


def glint(name, n, k):
    hole, _ = get(hdr, base, data, name)
    plug, _ = get(hdr, base, data, name + ".glint_plug")
    sc = get(hdr, base, data, name + ".absmax")[0].view(np.float32)

    def nibbles(b):
        out = np.empty(b.size * 2, dtype=np.uint8)
        out[0::2] = b & 15
        out[1::2] = b >> 4
        return out
    h, p = nibbles(np.asarray(hole)), nibbles(np.asarray(plug))
    w = cells[h, p] * np.repeat(sc, 64)
    return w.reshape(n, k).astype(np.float32)


def norm_w(name):
    return bf16_to_f32(np.asarray(get(hdr, base, data, name)[0]))


def forward(get_w, get_norm, emb, ids):
    kcache = [[] for _ in range(NL)]
    vcache = [[] for _ in range(NL)]
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    last = {}
    for pos, tok in enumerate(ids):
        x = emb[tok].copy()
        cs = np.cos(pos * inv).astype(np.float32)
        sn = np.sin(pos * inv).astype(np.float32)
        for l in range(NL):
            p = f"model.layers.{l}."

            def rms(v, w):
                return w * (v * (1.0 / np.sqrt(np.mean(v * v) + EPS))).astype(np.float32)
            h1 = rms(x, get_norm(p + "input_layernorm.weight"))
            q = get_w(p + "self_attn.q_proj.weight", H, H) @ h1
            k = get_w(p + "self_attn.k_proj.weight", NKV * HD, H) @ h1
            v = get_w(p + "self_attn.v_proj.weight", NKV * HD, H) @ h1

            def rope(t, heads):
                t = t.reshape(heads, HD)
                a, b = t[:, :HD // 2], t[:, HD // 2:]
                return np.concatenate([a * cs - b * sn, b * cs + a * sn], axis=1).reshape(-1)
            qr, kr = rope(q, NH), rope(k, NKV)
            kcache[l].append(kr)
            vcache[l].append(v)
            K = np.stack(kcache[l]).reshape(len(kcache[l]), NKV, HD)
            Vv = np.stack(vcache[l]).reshape(len(vcache[l]), NKV, HD)
            attn = np.zeros((NH, HD), dtype=np.float32)
            for hq in range(NH):
                s = (K[:, hq // 3, :] @ qr.reshape(NH, HD)[hq]) * 0.125
                e = np.exp(s - s.max())
                attn[hq] = (e / e.sum()) @ Vv[:, hq // 3, :]
            xm = x + get_w(p + "self_attn.o_proj.weight", H, H) @ attn.reshape(-1)
            h2 = rms(xm, get_norm(p + "post_attention_layernorm.weight"))
            g = get_w(p + "mlp.gate_proj.weight", INTER, H) @ h2
            u = get_w(p + "mlp.up_proj.weight", INTER, H) @ h2
            act = (g * (1.0 / (1.0 + np.exp(-g)))) * u
            x = xm + get_w(p + "mlp.down_proj.weight", H, INTER) @ act
            if pos == len(ids) - 1:
                last[f"L{l}.q"] = q
                last[f"L{l}.attn"] = attn.reshape(-1)
                last[f"L{l}.x_out"] = x
        if pos == len(ids) - 1:
            hf = (get_norm("model.norm.weight") * (x * (1.0 / np.sqrt(np.mean(x * x) + EPS)))).astype(np.float32)
            logits = emb @ hf
            last["logits"] = logits
    return last


man = json.load(open(trace_dir + "/manifest.json"))
blob = np.fromfile(trace_dir + "/tensors.bin", dtype=np.float32)
tr = {"tokens": man["tokens"], "trace_pos": man["trace_pos"], "tensors": {k: blob[o:o + n] for k, (o, n) in man["tensors"].items()}}
ids = tr["tokens"]
print("tokens", ids, "traced position", tr["trace_pos"])

# --- numpy forward on the same GLINT pin ---
cache = {}


def gw(name, n, k):
    if name not in cache:
        cache[name] = glint(name, n, k)
    return cache[name]


emb_g = glint("model.embed_tokens.weight", V, H)
res_g = forward(gw, norm_w, emb_g, ids)

print("\n== Mojo trace vs independent numpy decode of the same pin ==")
for key in ["L0.q", "L0.attn", "L0.x_out", "L15.x_out", "L29.x_out", "logits"]:
    a = tr["tensors"][key]
    b = res_g[key].astype(np.float32)
    d = np.abs(a - b)
    print(f"{key:10s} n={a.size:6d} max|diff|={d.max():.3e}  max|ref|={np.abs(b).max():.3e}  max rel to max|ref| = {d.max() / np.abs(b).max():.2e}")
tl = tr["tensors"]["logits"]
print("trace top-1 id", int(tl.argmax()), "numpy(pin) top-1 id", int(res_g["logits"].argmax()))

# --- numpy forward on the original BF16 weights ---
hdr2, base2, data2 = read_st(bf16_path)


def bw(name, n, k):
    return bf16_to_f32(np.asarray(get(hdr2, base2, data2, name)[0])).reshape(n, k)


def bn(name):
    return bf16_to_f32(np.asarray(get(hdr2, base2, data2, name)[0]))


emb_b = bw("model.embed_tokens.weight", V, H)
res_b = forward(bw, bn, emb_b, ids)
lg, lb = tl, res_b["logits"].astype(np.float32)
print("\n== GLINT pin (traced) vs original BF16 weights, same f32 forward ==")
print("BF16 top-1 id", int(lb.argmax()), " GLINT top-1 id", int(lg.argmax()))
top5b = np.argsort(-lb)[:5]
top5g = np.argsort(-lg)[:5]
print("BF16 top-5", top5b.tolist(), "GLINT top-5", top5g.tolist())
print("logit max|diff| over vocab", float(np.abs(lg - lb).max()), " mean|diff|", float(np.abs(lg - lb).mean()))
for key in ["L0.x_out", "L15.x_out", "L29.x_out"]:
    a = tr["tensors"][key]
    b = res_b[key].astype(np.float32)
    print(f"{key:10s} rel L2 error (GLINT vs BF16) = {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}")

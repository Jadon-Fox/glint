"""Job 22 (apparatus, G-f parity only; never a measurement). Extends job20/act_ref20.py.
FP32 transformers 4.57.6 eager reference, built layer by layer (one Phi3DecoderLayer at a time from the library module,
tensors loaded by safetensors slice, all 6 sets' sequences through it, hidden states kept, layer freed).
Compares, capture by capture, against the C++/CUDA fwd30 files:
  real path:   job22/x/real_<set>_L<L>_<kind>.f32 (L in 3,16,29; kinds qkv,o,gu,down), residual stream job22/resid/real_<set>_L<l>.f32 (l 0..29)
  controls:    job22/ctrl/<variant>_<set>_L<L>_<kind>.f32 (rope128:o, gqamod:o, gateup_swap:down)
Per-token rel L2 = ||x_cpp - x_ref||_2 / ||x_ref||_2 in FP64. Bar 1e-3 (rules section 4 G-f). TF32 off. Usage: runpy.sh act_ref30.py [BAR]"""
import hashlib, json, sys, time
import numpy as np

SNAP = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"
B = "/home/workspace/supervisor/scratch/p7-fold-harness/"
J = B + "job22/"
SETS = ["cal16k", "eval_1", "eval_2", "eval_3", "eval_4", "eval_5"]
IDS = {"cal16k": J + "inputs/cal16k.i32", **{f"eval_{i}": B + f"job20/inputs/eval_{i}.i32" for i in range(1, 6)}}
SEQS = {"cal16k": J + "inputs/cal16k_seqs.json", **{f"eval_{i}": B + f"job20/inputs/eval_{i}_seqs.json" for i in range(1, 6)}}
CAPL = [3, 16, 29]
KINDS = ["qkv", "o", "gu", "down"]
KDIM = {"qkv": 3072, "o": 3072, "gu": 3072, "down": 8192}
CTRL = {"rope128": "o", "gqamod": "o", "gateup_swap": "down"}
BAR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3


def main():
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RotaryEmbedding
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(SNAP, local_files_only=True)
    cfg._attn_implementation = "eager"
    rot = Phi3RotaryEmbedding(cfg).to(dev)
    wmap = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]
    ids = {s: np.fromfile(IDS[s], dtype="<i4") for s in SETS}
    seqs = {s: json.load(open(SEQS[s])) for s in SETS}
    ntok = {s: len(ids[s]) for s in SETS}
    pos = {}; prob = {}
    for s in SETS:
        p = np.zeros(ntok[s], dtype=int); q = np.zeros(ntok[s], dtype=int)
        for x in seqs[s]:
            p[x["start_offset"]:x["start_offset"] + x["length"]] = np.arange(x["length"])
            q[x["start_offset"]:x["start_offset"] + x["length"]] = x["problem_index"]
        pos[s], prob[s] = p, q
    total = sum(ntok.values())
    # embeddings: needed rows only
    used = sorted(set(int(i) for s in SETS for i in ids[s]))
    rowmap = {}
    with safe_open(SNAP + "/" + wmap["model.embed_tokens.weight"], framework="pt") as f:
        sl = f.get_slice("model.embed_tokens.weight")
        for g in used:
            rowmap[g] = sl[g:g + 1][0].to(torch.float32)
    H = {s: torch.stack([rowmap[int(i)] for i in ids[s]]).to(dev) for s in SETS}
    del rowmap
    handles = {}
    def mm(path, k):
        return np.memmap(path, dtype="<f4", mode="r").reshape(ntok_cur[0], k)
    rl2 = lambda x, r: np.linalg.norm(x.astype(np.float64) - r.astype(np.float64), axis=1) / np.linalg.norm(r.astype(np.float64), axis=1)
    real = {L: {k: {s: np.zeros(ntok[s]) for s in SETS} for k in KINDS} for L in CAPL}
    ctrl = {v: {L: {s: np.zeros(ntok[s]) for s in SETS} for L in CAPL} for v in CTRL}
    resid = {}
    meta = {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": torch.cuda.get_device_name(0),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "attention_scaling": float(rot.attention_scaling), "rope_type": rot.rope_type}
    t0 = time.time()
    for L in range(30):
        layer = Phi3DecoderLayer(cfg, L).to(torch.float32).eval()
        sd = {}
        for suf in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.qkv_proj.weight", "self_attn.o_proj.weight",
                    "mlp.gate_up_proj.weight", "mlp.down_proj.weight"]:
            name = f"model.layers.{L}.{suf}"
            with safe_open(SNAP + "/" + wmap[name], framework="pt") as f:
                sd[suf] = f.get_tensor(name).to(torch.float32)
        missing, unexpected = layer.load_state_dict(sd, strict=True), None
        layer.to(dev)
        cap = {}
        hs = [layer.self_attn.qkv_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("qkv", i[0].detach())),
              layer.self_attn.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("o", i[0].detach())),
              layer.mlp.gate_up_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("gu", i[0].detach())),
              layer.mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("down", i[0].detach()))]
        maps = {}
        if L in CAPL:
            for s in SETS:
                for k in KINDS:
                    maps[(s, k)] = np.memmap(J + f"x/real_{s}_L{L}_{k}.f32", dtype="<f4", mode="r").reshape(ntok[s], KDIM[k])
                for v, k in CTRL.items():
                    maps[(s, v)] = np.memmap(J + f"ctrl/{v}_{s}_L{L}_{k}.f32", dtype="<f4", mode="r").reshape(ntok[s], KDIM[k])
        newH = {}
        with torch.no_grad():
            for s in SETS:
                out = torch.empty_like(H[s])
                rmap = np.memmap(J + f"resid/real_{s}_L{L}.f32", dtype="<f4", mode="r").reshape(ntok[s], 3072)
                for x in seqs[s]:
                    o, T = x["start_offset"], x["length"]
                    h = H[s][o:o + T][None]
                    pid = torch.arange(T, device=dev)[None]
                    cos, sin = rot(h, pid)
                    mask = torch.triu(torch.full((T, T), torch.finfo(torch.float32).min, device=dev), 1)[None, None]
                    y = layer(h, attention_mask=mask, position_ids=pid, position_embeddings=(cos, sin))
                    out[o:o + T] = y[0]
                    if L in CAPL:
                        for k in KINDS:
                            real[L][k][s][o:o + T] = rl2(maps[(s, k)][o:o + T], cap[k][0].cpu().numpy())
                        for v, k in CTRL.items():
                            ctrl[v][L][s][o:o + T] = rl2(maps[(s, v)][o:o + T], cap[k][0].cpu().numpy())
                    resid.setdefault(L, {}).setdefault(s, np.zeros(ntok[s]))[o:o + T] = rl2(rmap[o:o + T], y[0].cpu().numpy())
                newH[s] = out
        H = newH
        for h in hs: h.remove()
        del layer, maps
        torch.cuda.empty_cache()
        print(f"layer {L} done t={time.time() - t0:.1f}s", flush=True)

    allpos = np.concatenate([pos[s] for s in SETS]); allprob = np.concatenate([prob[s] for s in SETS])
    allset = np.concatenate([[s] * ntok[s] for s in SETS])
    def summ(per):
        r = np.concatenate([per[s] for s in SETS]); w = int(np.argmax(r))
        return r, {"tokens": int(r.size), "pass_count": int((r <= BAR).sum()), "fail_count": int((r > BAR).sum()), "fail_fraction": float((r > BAR).mean()),
                   "median_rel_l2": float(np.median(r)), "max_rel_l2": float(r.max()),
                   "worst": {"set": str(allset[w]), "problem_index": int(allprob[w]), "position": int(allpos[w]), "token_index_in_all": w, "rel_l2": float(r[w])}}
    out = {"label": "PENDING Test Designer review (apparatus gate G-f; parity only, not a measurement)", "bar_rel_l2_per_token": BAR,
           "definition": "rel_l2 = ||x_cpp - x_ref||_2 / ||x_ref||_2 per token (FP64 accumulation of the FP32 files)",
           "reference": "transformers 4.57.6 Phi3DecoderLayer one layer at a time, FP32 (BF16 weights cast), eager attention, TF32 off, device cuda, run per sequence",
           "reference_meta": meta, "tokens_total": total, "tokens_per_set": ntok, "token_order": SETS,
           "real": {}, "controls": {}, "residual_stream_observation": {}}
    arrs = {}
    for L in CAPL:
        for k in KINDS:
            r, sm = summ(real[L][k]); arrs[f"real_L{L}_{k}"] = r
            sm["G_f_pass_all_tokens"] = bool((r <= BAR).all())
            out["real"][f"L{L}_{k}"] = sm
    for v, k in CTRL.items():
        for L in CAPL:
            r, sm = summ(ctrl[v][L]); arrs[f"ctrl_{v}_L{L}_{k}"] = r
            sm["capture"] = k; sm["fails_on_at_least_50pct"] = bool(sm["fail_fraction"] >= 0.5)
            out["controls"][f"{v}_L{L}"] = sm
    for L in range(30):
        r = np.concatenate([resid[L][s] for s in SETS]); w = int(np.argmax(r))
        out["residual_stream_observation"][f"L{L}"] = {"worst_rel_l2": float(r.max()), "median_rel_l2": float(np.median(r)),
                                                      "worst_token": {"set": str(allset[w]), "problem_index": int(allprob[w]), "position": int(allpos[w])}}
    out["max_position_used"] = int(allpos.max()); out["tokens_with_position_gt_1023"] = int((allpos > 1023).sum())
    out["max_position_by_set"] = {s: int(pos[s].max()) for s in SETS}
    out["G_f_per_layer"] = {}
    for L in CAPL:
        real_ok = all(out["real"][f"L{L}_{k}"]["G_f_pass_all_tokens"] for k in KINDS)
        ctrl_ok = all(out["controls"][f"{v}_L{L}"]["fails_on_at_least_50pct"] for v in CTRL)
        out["G_f_per_layer"][f"L{L}"] = {"real_all_four_captures_pass": real_ok, "all_three_controls_fail_ge_50pct": ctrl_ok,
                                         "status": "PASS" if real_ok and ctrl_ok else ("INCONCLUSIVE (a control passed)" if real_ok else "MISS (parity bar; see FP3 numbers)")}
    json.dump(out, open(J + "gf_apparatus.json", "w"), indent=1)
    np.savez_compressed(J + "gf_per_token.npz", position=allpos, problem_index=allprob, set_index=np.concatenate([[i] * ntok[s] for i, s in enumerate(SETS)]), **arrs)
    for L in CAPL:
        print(L, {k: (out["real"][f"L{L}_{k}"]["pass_count"], round(out["real"][f"L{L}_{k}"]["max_rel_l2"], 8)) for k in KINDS})
    for key, c in out["controls"].items():
        print(key, round(c["fail_fraction"], 4), "median", c["median_rel_l2"])
    print(out["G_f_per_layer"])


if __name__ == "__main__":
    main()

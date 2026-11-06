#!/usr/bin/env python3
"""GPTQ-style calibrated GLINT hole (spec Amendment 11, glint #92). Runs in the registered container env /opt/venv (torch + transformers),
CUDA. For every decoder linear of Phi-4-mini: error-compensating rounding (GPTQ: Hessian H = 2/n X^T X from calibration text, Cholesky of
H^-1, columns left to right in blocks of 128) onto the GLINT parent grid (the NF4 code), group scale = absmax of the already updated
weights of each 64-column group. The plug is then chosen as the nearest child cell in the CHOSEN parent cell for the ORIGINAL weight, so a
weight whose calibrated hole is not its nearest cell saturates at the cell edge (a cost this experiment measures). Layers run in sequence,
each seeing the hole-only outputs of the previous ones. Writes a glint_pin_v1 directory: every other tensor is copied from the source pin.
usage: glint_gptq_hole.py MODEL_DIR SRC_PIN OUT_PIN CALIB_TXT [nsamples=30] [seqlen=512]"""
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glint_np
from glint_codec import PARENT
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, SRC, OUT, CALIB = sys.argv[1:5]
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 30
SL = int(sys.argv[6]) if len(sys.argv) > 6 else 512
dev = "cuda"
PAR = torch.tensor(np.asarray(PARENT, dtype=np.float32), device=dev)          # 16 ascending parent values
MID = (PAR[1:] + PAR[:-1]) / 2                                                 # cell boundaries

def to_hole(v):  # nearest parent cell index, same rule as the encoder (ties to the lower index)
    return torch.bucketize(v, MID, right=False).to(torch.uint8)

def gptq_linear(W, H, blocksize=128, group=64, damp=0.01):
    N, K = W.shape
    W = W.clone().float(); H = H.clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    holes = torch.zeros(N, K, dtype=torch.uint8, device=W.device); scales = torch.zeros(N, K // group, device=W.device)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); W1 = W[:, i1:i2].clone(); Err = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % group == 0:
                s = W1[:, i:i + group].abs().amax(dim=1) if i + group <= i2 - i1 else W[:, c:c + group].abs().amax(dim=1)
                s = torch.where(s == 0, torch.ones_like(s), s); scales[:, c // group] = s
            s = scales[:, c // group]; w = W1[:, i]
            h = to_hole(w / s); q = PAR[h.long()] * s
            holes[:, c] = h; Q[:, c] = q
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0); Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return holes, scales, Q

def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval(); model.config.use_cache = False
    text = open(CALIB, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    ns = min(NS, ids.numel() // SL); print(f"calibration: {ids.numel()} tokens, {ns} samples x {SL}", flush=True)
    samples = [ids[i * SL:(i + 1) * SL].unsqueeze(0) for i in range(ns)]
    layers = model.model.layers
    model.model.embed_tokens.to(dev); model.model.rotary_emb.to(dev)
    if hasattr(model.model, "norm"): model.model.norm.to(dev)
    cache = {"x": [], "kw": None}
    class Catcher(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x, **kw): cache["x"].append(x); cache["kw"] = kw; raise ValueError
    layers[0] = Catcher(layers[0].to(dev))
    for smp in samples:
        try: model(smp.to(dev))
        except ValueError: pass
    layers[0] = layers[0].m
    kw = {k: v for k, v in cache["kw"].items() if k not in ("past_key_value", "past_key_values")}; kw["use_cache"] = False
    inps = cache["x"]; model.model.embed_tokens.cpu(); torch.cuda.empty_cache()
    result = {}  # name -> (hole_pk, plug_pk, absmax)
    for li, layer in enumerate(layers):
        layer.to(dev); lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}; cnt = {n: 0 for n in lins}
        def mk(n):
            def hook(m, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float(); Hs[n] += x.T @ x; cnt[n] += x.shape[0]
            return hook
        hs = [m.register_forward_hook(mk(n)) for n, m in lins.items()]
        with torch.no_grad():
            for x in inps: layer(x, **kw)
        for h in hs: h.remove()
        for n, m in lins.items():
            H = 2 * Hs[n] / cnt[n]; W0 = m.weight.data.float()
            holes, scales, Q = gptq_linear(W0, H)
            v0 = (W0 / scales.repeat_interleave(64, dim=1)).cpu().numpy().reshape(-1)           # original weight on the new scale
            hn = holes.cpu().numpy().reshape(-1).astype(np.int64)
            p = glint_np._nearest(v0.astype(np.float32), glint_np.CELLS_FLAT, hn * 16)
            full = f"model.layers.{li}.{n}.weight"
            result[full] = (glint_np.pack_nibbles(holes.cpu().numpy().reshape(-1)).tobytes(), glint_np.pack_nibbles(p).tobytes(), scales.cpu().numpy().reshape(-1).astype("<f4"))
            m.weight.data = Q.to(m.weight.dtype)   # hole-only values feed the next layers
            del H, W0, Q
        with torch.no_grad():
            outs = []
            for x in inps:
                o = layer(x, **kw); outs.append(o[0] if isinstance(o, tuple) else o)
        inps = outs; layer.cpu(); del Hs; torch.cuda.empty_cache()
        print(f"layer {li} done {time.time() - t0:.0f}s", flush=True)
    # write the pin: copy of SRC with the 128 decoder linears replaced
    out = Path(OUT); out.mkdir(parents=True, exist_ok=True); src = Path(SRC)
    for f in src.iterdir():
        if not f.name.endswith(".safetensors"): shutil.copy2(f, out / f.name)
    changed = 0
    for shard in sorted(src.glob("*.safetensors")):
        tensors = {}; 
        with safe_open(str(shard), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys(): tensors[k] = f.get_tensor(k)
        for full, (hb, pb, am) in result.items():
            if full in tensors:
                tensors[full] = torch.frombuffer(bytearray(hb), dtype=torch.uint8).reshape(-1, 1)
                tensors[full + ".glint_plug"] = torch.frombuffer(bytearray(pb), dtype=torch.uint8).reshape(-1, 1)
                tensors[full + ".absmax"] = torch.from_numpy(am.copy()); changed += 1
        save_file(tensors, str(out / shard.name), metadata=meta)
    print(f"wrote {out}: {changed} of {len(result)} linears replaced, {time.time() - t0:.0f}s", flush=True)

main()

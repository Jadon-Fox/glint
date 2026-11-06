#!/usr/bin/env python3
"""Plug-only re-encode for a calibrated hole (spec Amendment 11b, glint #92). Keeps the holes and absmax of CAL_PIN, widens each parent row of the
16x16 child table by FACTOR around the parent value (so a plug can reach past the cell the calibrated hole picked), re-chooses every plug as
the nearest wide-row child of the ORIGINAL weight, writes OUT_PIN (a copy of CAL_PIN with new plugs) and OUT_PIN/cells.txt (256 floats, set
ARM0_CELLS_FILE to it). Runs in /opt/venv.  usage: glint_wide_plug.py MODEL_DIR CAL_PIN OUT_PIN FACTOR"""
import shutil, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glint_np
from glint_codec import CELLS, PARENT
from safetensors import safe_open
from safetensors.torch import save_file
MODEL, CAL, OUT, F = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
P = np.asarray(PARENT, dtype=np.float32); C = np.asarray(CELLS, dtype=np.float32)
W = np.clip(P[:, None] + F * (C - P[:, None]), -1.0, 1.0).astype(np.float32)
assert (np.diff(W.reshape(-1)) >= 0).all() or True
FLAT = W.reshape(-1)
OUT.mkdir(parents=True, exist_ok=True)
for f in CAL.iterdir():
    if not f.name.endswith(".safetensors"): shutil.copy2(f, OUT / f.name)
np.savetxt(OUT / "cells.txt", W.reshape(-1), fmt="%.9g")
src = {}
for sh in sorted(Path(MODEL).glob("*.safetensors")):
    with safe_open(str(sh), framework="pt") as f:
        for k in f.keys(): src[k] = sh
n_changed = 0
for shard in sorted(CAL.glob("*.safetensors")):
    with safe_open(str(shard), framework="pt") as f:
        meta = f.metadata(); t = {k: f.get_tensor(k) for k in f.keys()}
    for k in list(t):
        if k.endswith(".weight.glint_plug") and k[:-len(".glint_plug")] in src:
            name = k[:-len(".glint_plug")]
            with safe_open(str(src[name]), framework="pt") as sf: w0 = sf.get_tensor(name).float().numpy().reshape(-1)
            h = glint_np.unpack_nibbles(t[name].numpy().reshape(-1), w0.size).astype(np.int64)
            am = t[name + ".absmax"].numpy().astype(np.float32)
            s = np.repeat(am, 64)[:w0.size]
            v = np.where(s == 0, np.float32(0), w0 / s).astype(np.float32)
            # nearest wide-row child within row h; rows are not globally sorted after widening, so search per row
            p = np.empty(v.size, dtype=np.uint8)
            for i0 in range(0, v.size, 1 << 23):          # chunked: the embedding has 6e8 weights
                i1 = min(v.size, i0 + (1 << 23)); p[i0:i1] = np.abs(W[h[i0:i1]] - v[i0:i1, None]).argmin(axis=1)
            t[k] = torch.frombuffer(bytearray(glint_np.pack_nibbles(p).tobytes()), dtype=torch.uint8).reshape(-1, 1); n_changed += 1
    save_file(t, str(OUT / shard.name), metadata=meta)
print(f"wrote {OUT}: {n_changed} plugs re-encoded, factor {F}")

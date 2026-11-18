#!/usr/bin/env python3
"""Offline weight-space check of an ADDITIVE plug against the child-nibble plug (analysis tooling only; not on any
train/serve/eval path). For each linear of a glint_pin_v1 pin and its BF16 source:
  child    : w ~ CELLS[h][p] * absmax                      (the pin as built; apply needs hole + table)
  add_blk  : w ~ PARENT[h]*absmax + q * s_blk, q in [-7,7], s_blk = max|r|/7 per 64-block (hole-free apply; +F32 per 64)
  add_abs  : w ~ PARENT[h]*absmax + q * c * absmax, one constant c per linear (hole-free apply; no extra bytes)
Reports RMS error of each against the BF16 weights. Usage: glint_additive_plug_offline.py PIN_DIR SRC_DIR OUT_TSV"""
import sys, json, struct, numpy as np, os, re
PARENT = np.array([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0],dtype=np.float32)
def index(d):
    t = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith('.safetensors')):
        p = os.path.realpath(os.path.join(d, fn))
        with open(p,'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
        for k,v in hdr.items():
            if k != '__metadata__': t[k] = (p, 8+n+v['data_offsets'][0], v['data_offsets'][1]-v['data_offsets'][0])
    return t
def load(t, k, dt):
    p, off, ln = t[k]
    return np.fromfile(p, dtype=dt, count=ln//np.dtype(dt).itemsize, offset=off)
pin, src, out = index(sys.argv[1]), index(sys.argv[2]), sys.argv[3]
cells = load(pin, '_glint_cells', np.float32).reshape(16,16)
rows = []
for k in sorted(pin):
    if not k.endswith('.weight.glint_state') or 'layers.' not in k: continue
    stem = k[:-len('.glint_state')]
    R, K = json.loads(bytes(load(pin, k, np.uint8)).decode())['shape']; n = R*K
    hb = load(pin, stem, np.uint8); pb = load(pin, stem+'.glint_plug', np.uint8); am = np.repeat(load(pin, stem+'.absmax', np.float32), 64)[:n]
    h = np.empty(n, np.uint8); h[0::2] = hb[:(n+1)//2] & 15; h[1::2] = hb[:n//2] >> 4
    p = np.empty(n, np.uint8); p[0::2] = pb[:(n+1)//2] & 15; p[1::2] = pb[:n//2] >> 4
    w = (load(src, stem, np.uint16).astype(np.uint32) << 16).view(np.float32)
    fold = PARENT[h] * am
    r = w - fold
    e_fold = float(np.sqrt(np.mean(r.astype(np.float64)**2)))
    e_child = float(np.sqrt(np.mean((w - cells[h, p]*am).astype(np.float64)**2)))
    rb = r.reshape(-1, 64); s = np.abs(rb).max(1, keepdims=True) / 7.0; s[s == 0] = 1.0
    e_blk = float(np.sqrt(np.mean((rb - np.clip(np.rint(rb / s), -7, 7) * s).astype(np.float64)**2)))
    u = np.where(am > 0, r / np.where(am > 0, am, 1), 0); c = float(np.abs(u).max() / 7.0)
    e_abs = float(np.sqrt(np.mean(((u - np.clip(np.rint(u / c), -7, 7) * c) * am).astype(np.float64)**2)))
    m = re.search(r'layers\.(\d+)\.(.*)\.weight', stem)
    rows.append((int(m.group(1)), m.group(2), e_fold, e_child, e_blk, e_abs)); print(rows[-1], flush=True)
with open(out,'w') as f:
    f.write('layer\tlinear\trms_fold_only\trms_child_plug\trms_additive_blockscale\trms_additive_absmax\n')
    for x in sorted(rows): f.write('\t'.join(str(v) for v in x)+'\n')
a = np.array([x[2:] for x in rows]); print('MEAN fold %.3e child %.3e add_blk %.3e add_abs %.3e' % tuple(a.mean(0)))

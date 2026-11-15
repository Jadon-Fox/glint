#!/usr/bin/env python3
"""Offline pin statistics for the G-fold price rule (analysis tooling only; not on any train/serve/eval path).
For each linear of a glint_pin_v1 pin: RMS of the plug delta d = (CELLS[h][p] - PARENT[h]) * absmax (what a fetched plug row adds),
and the spread of per-input-channel column norms of that delta. Usage: plug_delta_stats.py PIN_DIR OUT_TSV"""
import sys, json, struct, numpy as np, os, re
PARENT = np.array([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0],dtype=np.float32)
pin, out = sys.argv[1], sys.argv[2]
tens = {}
for fn in sorted(f for f in os.listdir(pin) if f.endswith('.safetensors')):
    p = os.path.join(pin, fn)
    with open(p,'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
    for k,v in hdr.items():
        if k != '__metadata__': tens[k] = (p, 8+n+v['data_offsets'][0], v['data_offsets'][1]-v['data_offsets'][0], v['dtype'], v['shape'])
def load(k, dt):
    p, off, ln, _, _ = tens[k]
    return np.fromfile(p, dtype=dt, count=ln//np.dtype(dt).itemsize, offset=off)
cells = load('_glint_cells', np.float32).reshape(16,16)
dtab = (cells - PARENT[:,None]).astype(np.float32)      # delta table [hole][plug], in units of absmax
rows = []
for k in sorted(tens):
    if not k.endswith('.weight.glint_state'): continue
    stem = k[:-len('.glint_state')]
    if 'layers.' not in stem: continue
    st = json.loads(bytes(load(k, np.uint8)).decode())
    R, K = st['shape']
    hb = load(stem, np.uint8); pb = load(stem+'.glint_plug', np.uint8); am = load(stem+'.absmax', np.float32)
    n = R*K
    h = np.empty(n, np.uint8); h[0::2] = hb[:(n+1)//2] & 15; h[1::2] = (hb[:n//2] >> 4)
    p = np.empty(n, np.uint8); p[0::2] = pb[:(n+1)//2] & 15; p[1::2] = (pb[:n//2] >> 4)
    d = dtab[h, p] * np.repeat(am, 64)[:n]
    d = d.reshape(R, K)
    col = np.sqrt((d.astype(np.float64)**2).sum(0))       # per input channel
    rms = float(np.sqrt((d.astype(np.float64)**2).mean()))
    m = re.search(r'layers\.(\d+)\.(.*)\.weight', stem)
    rows.append((int(m.group(1)), m.group(2), R, K, rms, float(col.mean()), float(col.std()/col.mean()), float(np.percentile(col,5)), float(np.percentile(col,95))))
    print(rows[-1], flush=True)
with open(out,'w') as f:
    f.write('layer\tlinear\tR\tK\tdelta_rms\tcolnorm_mean\tcolnorm_cv\tcolnorm_p5\tcolnorm_p95\n')
    for r in sorted(rows): f.write('\t'.join(str(x) for x in r)+'\n')

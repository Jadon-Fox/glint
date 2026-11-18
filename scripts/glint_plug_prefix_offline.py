#!/usr/bin/env python3
"""Offline weight-space check of a PARTIAL plug (analysis tooling only; not on any train/serve/eval path): how much of the
child-nibble correction do the top b bits of the 4-bit plug carry? For each linear of a glint_pin_v1 pin and its BF16
source, and b = 0..4:
  table : w ~ mean(CELLS[h][p'] for p' sharing p's top b bits) * absmax   (the current table, coarsened)
  fit   : w ~ c[h][prefix] * absmax, c = data-weighted mean of w/absmax over that (hole, prefix) cell in this linear
          (what a nested table built for prefixes could reach; an upper bound for this structure)
Reports RMS error against BF16 and the share of the fold-only error removed. Also checks whether CELLS[h] is ordered.
Usage: glint_plug_prefix_offline.py PIN_DIR SRC_DIR OUT_TSV [layer,layer,...]"""
import sys, json, struct, numpy as np, os, re

PARENT = np.array([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0],dtype=np.float32)

def index(d):
    t = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith('.safetensors')):
        p = os.path.realpath(os.path.join(d, fn))
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != '__metadata__': t[k] = (p, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0])
    return t

def load(t, k, dt):
    p, off, ln = t[k]
    return np.fromfile(p, dtype=dt, count=ln // np.dtype(dt).itemsize, offset=off)

pin, src, out = index(sys.argv[1]), index(sys.argv[2]), sys.argv[3]
keep = set(int(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else None
cells = load(pin, '_glint_cells', np.float32).reshape(16, 16)
ordered = all(np.all(np.diff(cells[h]) >= 0) for h in range(16))
print('CELLS rows ordered by child index:', ordered, flush=True)
rows = []
for k in sorted(pin):
    if not k.endswith('.weight.glint_state') or 'layers.' not in k: continue
    stem = k[:-len('.glint_state')]
    m = re.search(r'layers\.(\d+)\.(.*)\.weight', stem)
    if keep is not None and int(m.group(1)) not in keep: continue
    R, K = json.loads(bytes(load(pin, k, np.uint8)).decode())['shape']; n = R * K
    hb = load(pin, stem, np.uint8); pb = load(pin, stem + '.glint_plug', np.uint8)
    am = np.repeat(load(pin, stem + '.absmax', np.float32), 64)[:n]
    h = np.empty(n, np.uint8); h[0::2] = hb[:(n + 1) // 2] & 15; h[1::2] = hb[:n // 2] >> 4
    p = np.empty(n, np.uint8); p[0::2] = pb[:(n + 1) // 2] & 15; p[1::2] = pb[:n // 2] >> 4
    w = (load(src, stem, np.uint16).astype(np.uint32) << 16).view(np.float32)
    u = np.where(am > 0, w / np.where(am > 0, am, 1), 0).astype(np.float64)
    def rms(est): return float(np.sqrt(np.mean(((w - est * am).astype(np.float64)) ** 2)))
    e_fold = rms(PARENT[h])
    res = [int(m.group(1)), m.group(2), e_fold]
    for b in (1, 2, 3, 4):
        sh = 4 - b
        pre = (p >> sh).astype(np.int64)
        tab = np.stack([cells[:, (q << sh):((q + 1) << sh)].mean(1) for q in range(1 << b)], 1)  # [16, 2^b]
        e_tab = rms(tab[h, pre])
        key = h.astype(np.int64) * (1 << b) + pre
        sums = np.bincount(key, weights=u, minlength=16 << b); cnt = np.bincount(key, minlength=16 << b)
        fit = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0)
        e_fit = rms(fit[key].astype(np.float32))
        res += [e_tab, e_fit]
    rows.append(res)
    print(res[0], res[1], 'fold %.3e' % e_fold, ' '.join('b%d tab %.3e fit %.3e' % (b, res[1 + 2 * b], res[2 + 2 * b]) for b in (1, 2, 3, 4)), flush=True)
with open(out, 'w') as f:
    f.write('layer\tlinear\trms_fold\t' + '\t'.join('rms_b%d_table\trms_b%d_fit' % (b, b) for b in (1, 2, 3, 4)) + '\n')
    for x in sorted(rows): f.write('\t'.join(str(v) for v in x) + '\n')
a = np.array([x[2:] for x in rows])
fold, full = a[:, 0], a[:, 7]
for b in (1, 2, 3, 4):
    t, ft = a[:, 2 * b - 1], a[:, 2 * b]
    print('b=%d bits: table removes %.1f%% of the fold->full error gap, prefix-fit removes %.1f%% (mean over linears)' % (
        b, 100 * np.mean((fold - t) / (fold - full)), 100 * np.mean((fold - ft) / (fold - full))))

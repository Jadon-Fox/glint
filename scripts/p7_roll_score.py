#!/usr/bin/env python3
"""Scores free-running rollout logs against a reference log (verification tooling only).
Amendment 3: 12 prompts x 32 greedy steps; matched prefix = leading equal generated tokens per prompt (mean over prompts);
agreement = fraction of the 32 generated positions with equal token (mean over prompts). Usage: score_roll.py REF LOG [LOG ...]"""
import re, sys

def load(path):
    out = {}
    for line in open(path):
        m = re.match(r"MOJO_ROLL prompt (\d+) tokens (.*)", line)
        if m:
            toks = list(map(int, m.group(2).split()))
            assert len(toks) == 32, (path, m.group(1), len(toks))
            out[int(m.group(1))] = toks
    assert len(out) == 12, (path, len(out))
    return out

ref = load(sys.argv[1])
for p in sys.argv[2:]:
    g = load(p)
    pre, agree = [], []
    for k in range(12):
        a, b = ref[k], g[k]
        n = 0
        while n < 32 and a[n] == b[n]:
            n += 1
        pre.append(n)
        agree.append(sum(x == y for x, y in zip(a, b)) / 32)
    tps = re.search(r"tokens_per_s ([0-9.]+)", open(p).read())
    name = p.split("/")[-1].replace("_s2047.log", "").replace("fast+attn+dsel+kmaj+vgat+roll+tier_", "tier_")
    print(f"{name:34s} prefix {sum(pre)/12:5.2f}/32  agree {100*sum(agree)/12:5.1f}%  tok/s(emulation) {float(tps.group(1)) if tps else float('nan'):5.2f}  {'TRACKS' if sum(pre)/12 >= 20.0 and sum(agree)/12 >= 0.825 else 'no'}")

#!/usr/bin/env python3
"""Attribution test scorer (verification tooling only). Usage:
  glint_attr_score.py flips REF_LOG ARM_LOG... > alt.txt      list (prompt, step, token) flips of forced arms vs reference
  glint_attr_score.py score REF_MARGIN_LOG REF_ALT_LOG ARM_LOG...   T, D and conditional flip rates per arm
  glint_attr_score.py quick REF_MARGIN_LOG ARM_LOG...   same counts from the reference's stored margins alone: a flip is
      decided when the reference margin there exceeds 0.50 (any other token is then more than 0.50 below). Checked against
      the full method on 307 attribution flips (2026-11-18): 0 disagreements."""
import re, sys

def roll(path, key="MOJO_ROLL"):
    out = {}
    for line in open(path):
        m = re.match(key + r" prompt (\d+) \S+ (.*)", line)
        if m:
            out[int(m.group(1))] = list(map(int, m.group(2).split()))
    assert len(out) == 12 and all(len(v) == 32 for v in out.values()), (path, key)
    return out

def flips(ref, arm):
    return [(k, i, arm[k][i]) for k in range(12) for i in range(32) if arm[k][i] != ref[k][i]]

if sys.argv[1] == "flips":
    ref = roll(sys.argv[2])
    seen = sorted({f for p in sys.argv[3:] for f in flips(ref, roll(p))})
    print("\n".join(" ".join(map(str, f)) for f in seen))
elif sys.argv[1] == "quick":
    ref = roll(sys.argv[2]); rmarg = roll(sys.argv[2], "MOJO_MARGIN")
    for p in sys.argv[3:]:
        fl = flips(ref, roll(p))
        D = [(k, i) for k, i, _ in fl if rmarg[k][i] > 50]
        ms = re.search(r"ms_per_token ([0-9.]+)", open(p).read())
        print(f"{p.split('/')[-1]:24s} flips {len(fl):3d}  tie {len(fl)-len(D):3d}  decided {len(D):2d}  "
              f"ms/token {float(ms.group(1)) if ms else float('nan'):.1f}  decided_at {D}")
else:
    ref = roll(sys.argv[2]); rmarg = roll(sys.argv[2], "MOJO_MARGIN")
    gap = {(int(a), int(b), int(c)): int(d) for a, b, c, d in
           re.findall(r"MOJO_ALT prompt (\d+) step (\d+) token (\d+) chosen \d+ gap_x100 (-?\d+)", open(sys.argv[3]).read())}
    ties = sum(rmarg[k][i] <= 50 for k in range(12) for i in range(32))
    for p in sys.argv[4:]:
        fl = flips(ref, roll(p))
        miss = [f for f in fl if f not in gap]
        assert not miss, (p, miss[:5])
        D = [f for f in fl if gap[f] > 50]
        ft = sum(rmarg[k][i] <= 50 for k, i, _ in fl)
        pt = ft / ties if ties else float("nan")
        pd = (len(fl) - ft) / (384 - ties)
        ms = re.search(r"ms_per_token ([0-9.]+)", open(p).read())
        print(f"{p.split('/')[-1]:24s} flips {len(fl):3d}  tie {len(fl)-len(D):3d}  decided {len(D):2d}  "
              f"P(flip|ref tie) {pt:.3f}  P(flip|ref decided) {pd:.4f}  ratio {pt/pd if pd else float('inf'):.1f}  "
              f"ms/token {float(ms.group(1)) if ms else float('nan'):.1f}  decided_at {[(k,i,gap[(k,i,t)]) for k,i,t in D]}")

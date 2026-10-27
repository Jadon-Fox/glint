"""G-a1 cross-check (Python is a cross-check only): harness deQ(Q) vs nf4_copy/nf4.py dequant_nf4 on every element.
Imports ONLY job22/nf4_copy/nf4.py. Reads job22/ga1/L*_*.{q.bin,absmax.f64,deq.f32} written by p7fold2 ga1dump."""
import hashlib, json, sys
import numpy as np
sys.path.insert(0, "/home/workspace/supervisor/scratch/p7-fold-harness/job22/nf4_copy")
import nf4
J = "/home/workspace/supervisor/scratch/p7-fold-harness/job22/"
meta = json.load(open(J + "ga1_harness.json"))
CH = 1 << 20  # elements per chunk (multiple of 64 so blocks stay aligned)
rows, ok_all = [], True
for m in meta["matrices"]:
    base = J + f"ga1/L{m['layer']}_{m['type']}"
    q = open(base + ".q.bin", "rb").read()
    am = np.fromfile(base + ".absmax.f64", dtype="<f8")
    deq = np.fromfile(base + ".deq.f32", dtype="<f4")
    N = m["elements"]
    assert deq.size == N and len(q) == (N + 1) // 2
    mism, maxabs, maxulp = 0, 0.0, 0.0
    for s in range(0, N, CH):
        e = min(N, s + CH)
        py = np.asarray(nf4.dequant_nf4(q[s // 2:(e + 1) // 2], am[s // 64:(e + 63) // 64].tolist(), e - s, 64), dtype=np.float64).astype(np.float32)
        h = deq[s:e]
        d = np.abs(h.astype(np.float64) - py.astype(np.float64))
        ulp = np.spacing(np.abs(py)).astype(np.float64)
        ulp[ulp == 0] = np.spacing(np.float32(0))  # smallest subnormal for exact zeros
        mism += int((h != py).sum()); maxabs = max(maxabs, float(d.max())); maxulp = max(maxulp, float((d / ulp).max()))
    ok = maxulp <= 1.0
    ok_all &= ok
    rows.append({"layer": m["layer"], "type": m["type"], "elements": N, "mismatches": mism, "max_abs_diff": maxabs, "max_diff_ulps": maxulp, "pass": ok})
    print(rows[-1], flush=True)
json.dump({"label": "PENDING Test Designer review", "nf4_py_sha256": hashlib.sha256(open(J + "nf4_copy/nf4.py", "rb").read()).hexdigest(),
           "bar": "max abs diff <= 1 FP32 ulp of the value", "all_pass": ok_all, "matrices": rows,
           "harness_source_sha256": meta["source_sha256"], "harness_binary_sha256": meta["binary_sha256"]}, open(J + "ga1_check.json", "w"), indent=1)
print("G-a1 all_pass", ok_all)

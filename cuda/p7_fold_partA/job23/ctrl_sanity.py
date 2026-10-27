"""Job 23 Step 5 (optional sanity, NOT a gate, NOT the B-3 result): C++ vs C++.
Per-token rel L2 = ||ctrl - real||_2 / ||real||_2 (FP64) of the layer-only control capture (job23/ctrl_layer/) against job22's C++ real-path
capture of the same set, layer and kind (job22/x/). Files are read in 2048-row byte-range slices; nothing is loaded whole. Reference (transformers) not used."""
import json, os, sys
import numpy as np
B = "/home/workspace/supervisor/scratch/p7-fold-harness/"
SETS = ["cal16k", "eval_1", "eval_2", "eval_3", "eval_4", "eval_5"]
CTRL = {"rope128": "o", "gqamod": "o", "gateup_swap": "down"}
KD = {"o": 3072, "down": 8192}
out = {"label": "PENDING Test Designer review. C++ vs C++ sanity, not G-f, not B-3", "bar_used_for_fraction": 1e-3, "rows": []}
for v, k in CTRL.items():
    for L in (3, 16, 29):
        rs = []
        for s in SETS:
            a = B + f"job23/ctrl_layer/{v}_{s}_L{L}_{k}.f32"; b = B + f"job22/x/real_{s}_L{L}_{k}.f32"
            sa, sb = os.path.getsize(a), os.path.getsize(b)
            assert sa == sb, (a, sa, sb)
            rows = sa // (4 * KD[k])
            for r0 in range(0, rows, 2048):
                n = min(2048, rows - r0)
                x = np.fromfile(a, dtype="<f4", count=n * KD[k], offset=r0 * KD[k] * 4).reshape(n, KD[k]).astype(np.float64)
                y = np.fromfile(b, dtype="<f4", count=n * KD[k], offset=r0 * KD[k] * 4).reshape(n, KD[k]).astype(np.float64)
                rs.append(np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1))
        r = np.concatenate(rs)
        out["rows"].append({"variant": v, "layer": L, "capture": k, "tokens": int(r.size), "frac_tokens_rel_l2_gt_1e-3": float((r > 1e-3).mean()),
                            "median_rel_l2": float(np.median(r)), "min_rel_l2": float(r.min()), "max_rel_l2": float(r.max())})
        print(out["rows"][-1], flush=True)
json.dump(out, open(B + "job23/ctrl_sanity_cpp_vs_cpp.json", "w"), indent=1)

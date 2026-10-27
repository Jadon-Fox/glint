"""Job 22 Step 4b: map the per-capture C-rank logs (written by p7fold2 crank, C++/CUDA) onto the 63 round-2 cells. Logging only."""
import json
J = "/home/workspace/supervisor/scratch/p7-fold-harness/job22/"
KIND = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down"}
caps, cells = {}, []
for L in (3, 16, 29):
    for k in ("qkv", "o", "gu", "down"):
        d = json.load(open(J + f"crank/L{L}_{k}.json"))
        caps[f"L{L}_{k}"] = {x: d[x] for x in ("n", "tokens", "sequences", "rank_tolerance", "rank_tol_value", "rank_C", "max_Lambda", "floor_1e-6_max",
                                               "floored_eigenvalues_below_floor", "negative_eigenvalues", "min_eigenvalue", "flagged_any_floored", "peak_rss_kb")}
        caps[f"L{L}_{k}"]["source_sha256"] = d["source_sha256"]; caps[f"L{L}_{k}"]["binary_sha256"] = d["binary_sha256"]
for L in (3, 16, 29):
    for t in ("q", "k", "v", "o", "gate", "up", "down"):
        c = caps[f"L{L}_{KIND[t]}"]
        for p in (0.02, 0.05, 0.10):
            cells.append({"layer": L, "type": t, "p": p, "input_kind": KIND[t], "n": c["n"], "rank_C": c["rank_C"], "full_rank": c["rank_C"] == c["n"],
                          "floored_eigenvalues": c["floored_eigenvalues_below_floor"], "flagged_A4_regularized": c["flagged_any_floored"]})
out = {"label": "PENDING Test Designer review (logging only; C not used for any arm in job #22)", "per_capture": caps, "cells": cells,
       "cells_total": len(cells), "cells_flagged": sum(c["flagged_A4_regularized"] for c in cells),
       "note": "budget-independent: the same flag applies to all 3 budgets of a layer x type. A-2: flagged, not voided."}
json.dump(out, open(J + "c_rank.json", "w"), indent=1)
print(out["cells_total"], out["cells_flagged"])

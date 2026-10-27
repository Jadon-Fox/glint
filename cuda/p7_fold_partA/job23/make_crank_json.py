"""Job 23 Step 3: map the per-capture C-rank logs (written by p7fold3 crank, C++/CUDA) onto the 63 round-2 cells. Logging only.
No floor, no flag (retired by Amendment B-1). C is not used to build any arm."""
import json
J = "/home/workspace/supervisor/scratch/p7-fold-harness/job23/"
KIND = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down"}
caps, cells = {}, []
for L in (3, 16, 29):
    for k in ("qkv", "o", "gu", "down"):
        d = json.load(open(J + f"crank/L{L}_{k}.json"))
        caps[f"L{L}_{k}"] = {x: d[x] for x in ("n", "tokens", "sequences", "rank_tolerance", "rank_tol_value", "rank_C", "max_Lambda",
                                               "eigenvalues_le_0", "negative_eigenvalues", "min_eigenvalue", "peak_rss_kb")}
        caps[f"L{L}_{k}"]["source_sha256"] = d["source_sha256"]; caps[f"L{L}_{k}"]["binary_sha256"] = d["binary_sha256"]
for L in (3, 16, 29):
    for t in ("q", "k", "v", "o", "gate", "up", "down"):
        c = caps[f"L{L}_{KIND[t]}"]
        for p in (0.02, 0.05, 0.10):
            cells.append({"layer": L, "type": t, "p": p, "input_kind": KIND[t], "n": c["n"], "rank_C": c["rank_C"], "full_rank": c["rank_C"] == c["n"],
                          "max_Lambda": c["max_Lambda"], "eigenvalues_le_0": c["eigenvalues_le_0"]})
out = {"label": "PENDING Test Designer review (logging only; C not used for any arm in job #23)", "per_capture": caps, "cells": cells,
       "cells_total": len(cells), "cells_with_eigenvalues_le_0": sum(c["eigenvalues_le_0"] > 0 for c in cells),
       "cells_not_full_rank": sum(not c["full_rank"] for c in cells),
       "note": "budget-independent: the same values apply to all 3 budgets of a layer x type. Rank tolerance: eigenvalue > n * eps_fp64 * max(Lambda)."}
json.dump(out, open(J + "c_rank.json", "w"), indent=1)
print(out["cells_total"], out["cells_with_eigenvalues_le_0"], out["cells_not_full_rank"])

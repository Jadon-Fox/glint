"""Per-arm, per-factor FP16 underflow-to-zero-or-subnormal / overflow counts from job23/ga_apparatus.json (G-a cells, synthetic X only)."""
import json
B = "/home/workspace/supervisor/scratch/p7-fold-harness/job23/"
d = json.load(open(B + "ga_apparatus.json"))
rows = ["type\tpct\tk\tX\tarm\tfactor\tfp16_elements\toverflow\tunderflow_to_zero_or_subnormal"]
for c in d["cells"]:
    for a in c["G_a2_A1_A3_A5_on_X_job22"]:
        for f in a["fp16_per_factor"]:
            rows.append("\t".join(map(str, [c["type"], c["pct"], c["k"], "X_job22_2048rows", a["arm"], f["factor"], f["fp16_elements"], f["overflow"], f["underflow_to_zero_or_subnormal"]])))
    for x in c["A4_on_each_synthetic_X"]:
        for f in x["G_a2"]["fp16_per_factor"]:
            rows.append("\t".join(map(str, [c["type"], c["pct"], c["k"], x["eig"]["x_variant"], "A4", f["factor"], f["fp16_elements"], f["overflow"], f["underflow_to_zero_or_subnormal"]])))
open(B + "fp16_counts_ga_cells.tsv", "w").write("\n".join(rows) + "\n")
tot = {}
for r in rows[1:]:
    t = r.split("\t"); key = (t[3] if t[4] == "A4" else "X_job22_2048rows", t[4], t[5]); o = tot.setdefault(key, [0, 0, 0]); o[0] += int(t[6]); o[1] += int(t[7]); o[2] += int(t[8])
for k, v in sorted(tot.items(), key=lambda z: (z[0][1], z[0][0], z[0][2])): print(k, "elements", v[0], "overflow", v[1], "underflow", v[2], "underflow_frac %.3g" % (v[2] / v[0]))
print("rows", len(rows) - 1)

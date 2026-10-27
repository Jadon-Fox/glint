"""Summarise job23/ga_apparatus.json into tables (text only; reads the C++ output)."""
import json, sys
d = json.load(open("/home/workspace/supervisor/scratch/p7-fold-harness/job23/ga_apparatus.json"))
print("all_checks_pass", d["all_checks_pass"], "peak_rss_kb", d["peak_rss_kb"], "bin", d["binary_sha256"][:12])
for x in d["synthetic_x"]:
    e = x["eig"]; print(" X", e["x_variant"], "n", e["n"], "rows", e["x_rows"], "rank", e["C_rank"], "eig<=0", e["C_eigenvalues_le_0"], "neg", e["C_negative_eigenvalues"], "min %.3g max %.3g" % (e["C_min_eig"], e["C_max_eig"]))
print("G-c:", sum(r["G_c_B2_pass"] for r in d["G_c_arithmetic"]), "/", len(d["G_c_arithmetic"]), "pass; slack set", sorted(set(r["sparse_slack_bytes"] for r in d["G_c_arithmetic"])))
for r in d["G_c_arithmetic"]: print("  ", r["type"], r["pct"], "k", r["k"], "A1", r["A1_bytes"], "sparse", r["A3_bytes"], "slack", r["sparse_slack_bytes"])
print("--- G-a2 (A1,A3,A5 on X_job22), A4 per X; G-a3; G-a4")
bad = 0
for c in d["cells"]:
    t = f'{c["type"]:5s} p{c["pct"]:02d} k{c["k"]:4d}'
    a2 = {a["arm"]: a for a in c["G_a2_A1_A3_A5_on_X_job22"]}
    s = " ".join(f'{n}:{"ok" if a2[n]["pass"] else "FAIL"}' for n in a2)
    a4 = c["A4_on_each_synthetic_X"]
    s4 = " ".join(f'{x["eig"]["x_variant"][2:8]}:a2={"ok" if x["G_a2"]["pass"] else "FAIL"},a3gap={x["G_a3"]["A4_rel_gap"]:.1e},ortho={x["G_a3"]["U_k_orthonormality_max_abs_UtU_minus_I"]:.1e}' for x in a4)
    g = c["G_a4"]
    s5 = f'A1={g["A1_rel_fro_err"]:.1e}/brk_fail={g["A1_broken_fails_as_required"]} A4={g["A4_C_identity_rel_fro_err"]:.1e}/brk_fail={g["A4_broken_transposed_fails_EXTRA"]} A3exact={g["A3_index_set_exact"]}/brk_fail={g["A3_broken_fails_as_required"]} A5exact={g["A5_index_set_exact"]}/brk_fail={g["A5_broken_fails_as_required"]}'
    print(t, "|", s, "| A1gap=%.1e" % c["G_a3_A1"]["A1_rel_gap"], "|", s4); print("      G-a4:", s5)

# score25.py: new, not yet reviewed by Test Designer (aggregation only).
# Reads job25/cells/*.json, job25/gf/*, job25/ga_apparatus.json, job25/ga3_real.json, job22/ga1_check.json,
# job22/inputs/cal16k_meta.json, job20/inputs/inputs_meta.json, job25/x_captures.sha256, job25/logs/measure_status.log.
# Computes gates (rules sec. 4 + Amendments A, B), per-cell scoring (sec. 5, A-3) and verdicts (sec. 6).
# It computes no e, touches no X or weights. Writes only the four output files named below (refuses if they exist).
import hashlib, json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../p7-fold-harness
os.chdir(ROOT)
J = "job25"
LABEL = "PENDING Test Designer review"
RULES = "/home/workspace/supervisor/inputs/p7-fold-round2-rules.md"
RULES_SHA = "03f97724426223451907581c4d690e0471b9f67d8f746dc6f49a5e69adbd054a"
BIN_SHA = "ca8c296d0e9552c83fa5469983a4297e99eca701959d65c032a88dc53a3b9042"
SRC_FILES = {"p7fold4.cu": "job24/src/p7fold4.cu", "p7fold3.cu": "job23/src/p7fold3.cu", "fwd30.cu": "job22/src/fwd30.cu"}
SRC_STEP0 = {"p7fold4.cu": "39e56c5beaf7e89288b7a6360c9f943020c9a993023b5c0734d6661507a84028"}
LAYERS = [3, 16, 29]
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]
PCTS = [2, 5, 10]
SETS = ["eval_1", "eval_2", "eval_3", "eval_4", "eval_5"]
KIND = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down"}
ARMS = ["A0", "A1", "A4", "A3", "A5"]
DKEYS = [("d4", "A4"), ("d1", "A1"), ("d5", "A5")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    with open(p) as f:
        return json.load(f)


for o in ("gate_table.json", "gate_table.md", "verdict.json", "tables.md"):
    if os.path.exists(f"{J}/{o}"):
        print(f"REFUSED_EXISTS {J}/{o}")
        sys.exit(2)

self_sha = sha(os.path.abspath(__file__))
rules_sha_now = sha(RULES)
src_now = {k: sha(v) for k, v in SRC_FILES.items()}

xs = {}
for l in open(f"{J}/x_captures.sha256"):
    s, p = l.split(None, 1)
    xs[p.strip()] = s
ga1 = jl("job22/ga1_check.json")
ga1m = {(m["layer"], m["type"]): m for m in ga1["matrices"]}
apparatus = jl(f"{J}/ga_apparatus.json")
app = {(c["type"], c["pct"]): c for c in apparatus["cells"]}
appc = {(c["type"], c["pct"]): c for c in apparatus["G_c_arithmetic"]}
ga3 = jl(f"{J}/ga3_real.json")
ga3m = {(c["layer"], c["type"], c["pct"]): c for c in ga3["cells_detail"]}
gf = jl(f"{J}/gf/gf_apparatus.json")
cal_meta = jl("job22/inputs/cal16k_meta.json")
ev_meta = jl("job20/inputs/inputs_meta.json")
status = {}
if os.path.exists(f"{J}/logs/measure_status.log"):
    for l in open(f"{J}/logs/measure_status.log"):
        t = l.split()
        if len(t) >= 2:
            status[t[0]] = t[1]

# ---- G-f per layer
gf_layer = {}
for L in LAYERS:
    reasons = []
    for K in ("qkv", "o", "gu", "down"):
        r = gf["real"][f"L{L}_{K}"]
        if r["pass_count"] != r["tokens"] or r["max_rel_l2"] > 1e-3:
            reasons.append(f"real L{L}_{K}: pass {r['pass_count']}/{r['tokens']} worst {r['max_rel_l2']:.4g}")
    for cname, c in gf["controls"].items():
        if cname.endswith(f"_L{L}") and c["fail_fraction"] < 0.5:
            reasons.append(f"control {cname}: fail_fraction {c['fail_fraction']:.4f} < 0.5")
    nctrl = sum(1 for c in gf["controls"] if c.endswith(f"_L{L}"))
    if nctrl != 3:
        reasons.append(f"controls found {nctrl} != 3")
    if gf["G_f_per_layer"][f"L{L}"]["status"] != "PASS":
        reasons.append("G_f_per_layer status != PASS")
    gf_layer[L] = reasons

# ---- G-d
cal_probs = set(cal_meta["problems_used_0_12"]) | set(range(81, 81 + cal_meta["problems_used_81_onward_count"]))
ev_probs = set()
for sid in SETS:
    s = ev_meta["sets"][sid]
    ev_probs |= set(range(s["problem_index_first"], s["problem_index_last"] + 1))
gd_overlap = sorted(cal_probs & ev_probs)
gd_reasons = []
if gd_overlap:
    gd_reasons.append(f"problem index overlap {gd_overlap}")
if cal_meta["problem_index_last"] != max(cal_probs):
    gd_reasons.append(f"cal last index {cal_meta['problem_index_last']} != derived {max(cal_probs)}")
gd_obs = {"cal16k_meta_8gram_keys": [k for k in cal_meta if "gram" in k],
          "job20_cal2048_vs_eval_8gram": ev_meta.get("observation_8gram_cal_vs_eval_NOT_A_GATE")}
for k in gd_obs["cal16k_meta_8gram_keys"]:
    gd_obs[k] = cal_meta[k]


def med(v):
    return statistics.median(v)


def classify(d):
    m = med(d)
    if all(x > 0 for x in d) and m >= 0.05:
        return "WIN"
    if sum(1 for x in d if x <= 0) >= 4 or m <= 0:
        return "LOSS"
    return "TIE"


rows, gate_rows = [], []
for L in LAYERS:
    for T in TYPES:
        K = KIND[T]
        lab = f"measure_L{L}_{T}"
        for p in PCTS:
            r = {"layer": L, "type": T, "pct": p}
            g = {k: [] for k in ("G-a1", "G-a2", "G-a3", "G-a4", "G-b", "G-c", "G-d", "G-e", "G-f", "crosscheck", "cell_present")}
            cj = {}
            for sid in SETS:
                fp = f"{J}/cells/L{L}_{T}_p{p:02d}_{sid}.json"
                if os.path.exists(fp):
                    cj[sid] = jl(fp)
            if status.get(lab, "missing") != "rc=0":
                g["cell_present"].append(f"failed (crash) process {lab} status {status.get(lab, 'missing')}")
            if len(cj) != 5:
                g["cell_present"].append(f"only {len(cj)}/5 set JSONs present")
            if len(cj) == 5:
                e = {a: [] for a in ARMS}
                arms = {sid: {a["arm"]: a for a in cj[sid]["arms"]} for sid in SETS}
                for sid in SETS:
                    for a in ARMS:
                        if a not in arms[sid]:
                            g["cell_present"].append(f"arm {a} missing in {sid}")
                if not g["cell_present"]:
                    for sid in SETS:
                        for a in ARMS:
                            e[a].append(arms[sid][a]["e"])
                    c0 = cj["eval_1"]
                    # G-a1
                    m1 = ga1m.get((L, T))
                    if m1 is None or not m1["pass"] or not ga1["all_pass"]:
                        g["G-a1"].append(f"job22 ga1 {m1}")
                    # G-a2
                    for sid in SETS:
                        for a in ("A1", "A4", "A3", "A5"):
                            x = arms[sid][a]
                            if not (x.get("storage_roundtrip_pass") is True and x["file_length"] == x["added_bytes"] and x.get("decode_bit_mismatches") == 0):
                                g["G-a2"].append(f"{sid} {a} storage round trip not pass")
                    ap = app.get((T, p))
                    if ap is None:
                        g["G-a2"].append("no apparatus record for shape")
                        g["G-a4"].append("no apparatus record for shape")
                    else:
                        for x in ap["G_a2_A1_A3_A5_on_X_job22"]:
                            if not (x["pass"] and x["bytes_vs_A1_rule_ok"] and x["length_equal"] and x["bit_mismatches"] == 0):
                                g["G-a2"].append(f"apparatus {x['arm']} on X_job22 fail")
                        for s4 in ap["A4_on_each_synthetic_X"]:
                            if not s4["G_a2"]["pass"]:
                                g["G-a2"].append(f"apparatus A4 on {s4['eig']['x_variant']} fail")
                        a4 = ap["G_a4"]
                        for k in ("A1_pass", "A1_broken_fails_as_required", "A4_pass", "A3_pass", "A3_broken_fails_as_required",
                                  "A5_pass", "A5_broken_fails_as_required", "A3_index_set_exact", "A5_index_set_exact"):
                            if a4.get(k) is not True:
                                g["G-a4"].append(f"apparatus {k} = {a4.get(k)}")
                        if a4.get("A3_broken_offbyone_index_set_exact") is not False:
                            g["G-a4"].append("A3 off-by-one variant did not change index set")
                        if a4.get("A5_broken_offbyone_index_set_exact") is not False:
                            g["G-a4"].append("A5 off-by-one variant did not change index set")
                    # G-a3
                    gc3 = ga3m.get((L, T, p))
                    if gc3 is None or gc3["status"] != "pass" or not gc3["A1_pass"] or not gc3["A4_pass"]:
                        g["G-a3"].append(f"ga3_real {gc3 and gc3['status']}")
                    # G-b
                    for i, sid in enumerate(SETS):
                        mx = max(e[a][i] for a in ("A1", "A4", "A3", "A5"))
                        if not e["A0"][i] > mx:
                            g["G-b"].append(f"{sid}: A0 {e['A0'][i]:.6g} <= max arm {mx:.6g}")
                    # G-c
                    ks = {c["k"] for c in cj.values()}; ab = {c["a1_bytes"] for c in cj.values()}
                    if len(ks) != 1 or len(ab) != 1:
                        g["G-c"].append(f"k/a1_bytes differ across sets {ks} {ab}")
                    for sid in SETS:
                        c = cj[sid]; ar = arms[sid]
                        if not c["a1_le_R"]: g["G-c"].append(f"{sid} a1_le_R false")
                        if not c["k_is_largest_fitting"]: g["G-c"].append(f"{sid} k not largest fitting")
                        a1b = ar["A1"]["added_bytes"]
                        if ar["A4"]["added_bytes"] != a1b: g["G-c"].append(f"{sid} A4 bytes {ar['A4']['added_bytes']} != A1 {a1b}")
                        for a in ("A3", "A5"):
                            sb = ar[a]["added_bytes"]
                            if not (sb <= a1b and a1b - sb < 6): g["G-c"].append(f"{sid} {a} bytes {sb} vs A1 {a1b}")
                        if not c["sparse_bytes_rule_B2_ok"]: g["G-c"].append(f"{sid} sparse_bytes_rule_B2_ok false")
                    # G-d
                    g["G-d"] += gd_reasons
                    # G-e
                    for sid in SETS:
                        if cj[sid]["binary_sha256"] != BIN_SHA: g["G-e"].append(f"{sid} binary sha mismatch")
                        if "3060" not in cj[sid]["gpu"]: g["G-e"].append(f"{sid} gpu {cj[sid]['gpu']}")
                    # G-f
                    g["G-f"] += gf_layer[L]
                    # cross-check
                    for sid in SETS:
                        c = cj[sid]
                        if c["cal_x_sha256"] != xs.get(f"job22/x/real_cal16k_L{L}_{K}.f32"): g["crosscheck"].append(f"OPEN_inputs {sid} cal_x_sha256")
                        if c["eval_x_sha256"] != xs.get(f"job22/x/real_{sid}_L{L}_{K}.f32"): g["crosscheck"].append(f"OPEN_inputs {sid} eval_x_sha256")
                        for sn, sv in c["source_sha256"].items():
                            if sv != src_now.get(sn) or (sn in SRC_STEP0 and sv != SRC_STEP0[sn]):
                                g["crosscheck"].append(f"OPEN_inputs {sid} source_sha256 {sn}")
                        if c["binary_sha256"] != BIN_SHA: g["crosscheck"].append(f"OPEN_inputs {sid} binary_sha256")
                    # observation fields
                    r.update({
                        "m": c0["m"], "n": c0["n"], "k": c0["k"], "R": c0["R"], "a1_bytes": c0["a1_bytes"],
                        "sparse_entries": c0["sparse_entries"], "sparse_slack_bytes": c0["sparse_slack_bytes"],
                        "C_rank": c0["C_rank"], "C_eigenvalues_le_0": c0["C_eigenvalues_le_0"], "C_max_eig": c0["C_max_eig"],
                        "fp5_top1pct": c0["fp5_top1pct_column_share_of_Xcal_sq"],
                        "fp16": {a: {"overflow": max(arms[s][a].get("fp16_overflow", 0) for s in SETS),
                                      "underflow": max(arms[s][a].get("fp16_underflow_to_zero_or_subnormal", 0) for s in SETS)}
                                 for a in ("A1", "A4", "A3", "A5")},
                        "e": {a: e[a] for a in ARMS},
                    })
            valid = not any(g[k] for k in g)
            r["valid"] = valid
            r["gate_pass"] = {k: (not v) for k, v in g.items()}
            r["gate_reasons"] = {k: v for k, v in g.items() if v}
            if valid:
                for dk, base in DKEYS:
                    d = [(r["e"][base][i] - r["e"]["A3"][i]) / r["e"][base][i] for i in range(5)]
                    r[dk] = {"per_set": d, "median": med(d), "class": classify(d)}
            rows.append(r)
            gate_rows.append({"layer": L, "type": T, "pct": p, "valid": valid, "gates": r["gate_pass"], "reasons": r["gate_reasons"]})

n_inc = sum(1 for r in rows if not r["valid"])
valid_rows = [r for r in rows if r["valid"]]
V = len(valid_rows)


def counts(sel, dk):
    c = {"WIN": 0, "TIE": 0, "LOSS": 0}
    for r in sel:
        if r["valid"]:
            c[r[dk]["class"]] += 1
    return c


verdict = {"label": LABEL, "rules_path": RULES, "rules_sha256": rules_sha_now, "rules_sha256_expected": RULES_SHA,
           "binary_sha256": BIN_SHA, "source_sha256": src_now, "score25_py_sha256": self_sha,
           "score25_label": "new, not yet reviewed by Test Designer (aggregation only)",
           "cells_total": len(rows), "cells_inconclusive_by_gates": n_inc, "cells_valid": V,
           "inconclusive_rule": "more than 6 of 63 INCONCLUSIVE => H2, H3, H4 INCONCLUSIVE (rules sec. 6)"}
for H, (dk, base) in zip(("H2", "H3", "H4"), DKEYS):
    st = {"metric": f"{dk} = (e_{base} - e_A3)/e_{base}", "counts_all_valid": counts(rows, dk)}
    st["by_type"] = {t: counts([r for r in rows if r["type"] == t], dk) for t in TYPES}
    st["by_layer"] = {str(l): counts([r for r in rows if r["layer"] == l], dk) for l in LAYERS}
    st["by_budget"] = {str(p): counts([r for r in rows if r["pct"] == p], dk) for p in PCTS}
    if V:
        st["median_over_cells_of_median_d"] = med([r[dk]["median"] for r in valid_rows])
        w, lo = st["counts_all_valid"]["WIN"], st["counts_all_valid"]["LOSS"]
        st["win_fraction_of_valid"] = w / V
        st["loss_fraction_of_valid"] = lo / V
        st["pass_win_ge_2_3"] = 3 * w >= 2 * V
        st["pass_every_type_has_win"] = all(st["by_type"][t]["WIN"] >= 1 for t in TYPES)
        st["pass_every_budget_has_win"] = all(st["by_budget"][str(p)]["WIN"] >= 1 for p in PCTS)
        st["is_PASS"] = st["pass_win_ge_2_3"] and st["pass_every_type_has_win"] and st["pass_every_budget_has_win"]
        st["fail_loss_ge_1_2"] = 2 * lo >= V
        st["fail_median_of_medians_le_0"] = st["median_over_cells_of_median_d"] <= 0
        st["is_FAIL"] = st["fail_loss_ge_1_2"] or st["fail_median_of_medians_le_0"]
    if n_inc > 6 or not V:
        st["verdict"] = "INCONCLUSIVE"
        st["why"] = f"{n_inc} of 63 cells INCONCLUSIVE by gates (> 6)" if n_inc > 6 else "no valid cells"
    elif st["is_PASS"] and not st["is_FAIL"]:
        st["verdict"] = "PASS"
    elif st["is_FAIL"] and not st["is_PASS"]:
        st["verdict"] = "FAIL"
    elif st["is_PASS"] and st["is_FAIL"]:
        st["verdict"] = "INCONCLUSIVE"; st["why"] = "PASS and FAIL conditions both met (not defined by rules)"
    else:
        st["verdict"] = "INCONCLUSIVE"; st["why"] = "neither PASS nor FAIL condition met"
    verdict[H] = st

# ---- write outputs
json.dump({"label": LABEL, "rows": gate_rows, "gf_layer_reasons": {str(k): v for k, v in gf_layer.items()},
           "G-d_observation": gd_obs, "G-d_cal_problem_count": len(cal_probs), "G-d_eval_problem_count": len(ev_probs)},
          open(f"{J}/gate_table.json", "x"), indent=1)
with open(f"{J}/gate_table.md", "x") as f:
    f.write(f"# Gate table ({LABEL})\n\nPASS=ok, FAIL=gate missed. {n_inc} of 63 cells INCONCLUSIVE.\n\n")
    names = ["cell_present", "G-a1", "G-a2", "G-a3", "G-a4", "G-b", "G-c", "G-d", "G-e", "G-f", "crosscheck"]
    f.write("| layer | type | pct | valid | " + " | ".join(names) + " | reasons |\n|" + "---|" * (5 + len(names)) + "\n")
    for gr in gate_rows:
        rs = "; ".join(f"{k}: {'; '.join(v)[:200]}" for k, v in gr["reasons"].items())
        f.write(f"| {gr['layer']} | {gr['type']} | {gr['pct']} | {'VALID' if gr['valid'] else 'INCONCLUSIVE'} | "
                + " | ".join("pass" if gr["gates"][n] else "FAIL" for n in names) + f" | {rs} |\n")
    f.write(f"\nG-d: cal problems {len(cal_probs)} vs eval problems {len(ev_probs)}; overlap {gd_overlap}\n")
    f.write(f"G-d 8-gram observation (not a gate): {json.dumps(gd_obs)[:3000]}\n")
json.dump(verdict, open(f"{J}/verdict.json", "x"), indent=1)


def fe(x): return f"{x:.6g}"
def fd(x): return f"{x:+.4f}"


with open(f"{J}/tables.md", "x") as f:
    f.write(f"# Round-2 tables ({LABEL})\n\nscore25.py sha256 {self_sha} (new, not yet reviewed by Test Designer (aggregation only)).\n\n")
    f.write("## 1. e per cell per set (A0, A1, A4, A3, A5); INCONCLUSIVE cells marked\n\n| layer | type | pct | set | A0 | A1 | A4 | A3 | A5 | cell |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        if "e" not in r:
            f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | all | n/a | n/a | n/a | n/a | n/a | INCONCLUSIVE (no data) |\n"); continue
        for i, sid in enumerate(SETS):
            f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | {sid} | " + " | ".join(fe(r["e"][a][i]) for a in ARMS)
                    + f" | {'VALID' if r['valid'] else 'INCONCLUSIVE'} |\n")
    for dk, base in DKEYS:
        f.write(f"\n## 2. {dk} = (e_{base} - e_A3)/e_{base} per cell (valid cells only)\n\n| layer | type | pct | eval_1 | eval_2 | eval_3 | eval_4 | eval_5 | median | class |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r["valid"]:
                f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | " + " | ".join(fd(x) for x in r[dk]["per_set"]) + f" | {fd(r[dk]['median'])} | {r[dk]['class']} |\n")
            else:
                f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | INCONCLUSIVE | | | | | | |\n")
    for dk, base in DKEYS:
        f.write(f"\n## 3. {dk} WIN/TIE/LOSS counts (valid cells)\n\n| group | value | WIN | TIE | LOSS |\n|---|---|---|---|---|\n")
        h = {"H2": "d4", "H3": "d1", "H4": "d5"}
        st = verdict[{v: k for k, v in h.items()}[dk]]
        for grp, key in (("type", "by_type"), ("layer", "by_layer"), ("budget", "by_budget")):
            for k, c in st[key].items():
                f.write(f"| {grp} | {k} | {c['WIN']} | {c['TIE']} | {c['LOSS']} |\n")
        c = st["counts_all_valid"]
        f.write(f"| all | valid={V} | {c['WIN']} | {c['TIE']} | {c['LOSS']} |\n")
    f.write("\n## 4. Per-cell observations (no verdict use except INCONCLUSIVE gating)\n\n| layer | type | pct | m x n | k | C rank | C eig<=0 | FP16 ovf/udf A1 | A4 | A3 | A5 | sparse slack B | FP5 top-1% col share |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        if "e" not in r:
            f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | no data |||||||||\n"); continue
        q = r["fp16"]
        f.write(f"| {r['layer']} | {r['type']} | {r['pct']} | {r['m']}x{r['n']} | {r['k']} | {r['C_rank']} | {r['C_eigenvalues_le_0']} | "
                + " | ".join(f"{q[a]['overflow']}/{q[a]['underflow']}" for a in ("A1", "A4", "A3", "A5"))
                + f" | {r['sparse_slack_bytes']} | {r['fp5_top1pct']:.4f} |\n")
    f.write("\n## 5. Verdicts\n\n")
    for H in ("H2", "H3", "H4"):
        f.write(f"- {H}: **{verdict[H]['verdict']}** ({verdict[H].get('why', 'rules sec. 6 conditions')})\n")
print("OK", n_inc, "inconclusive;", {H: verdict[H]["verdict"] for H in ("H2", "H3", "H4")})

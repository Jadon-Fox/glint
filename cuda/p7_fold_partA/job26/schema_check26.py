#!/usr/bin/env python3
"""Pure schema validator for the round-3 cell files (rules section 6). Keys, types and the arm set only; no number is combined with another.
Usage: schema_check26.py CELLS_DIR OUT_JSON   (refuses to overwrite OUT_JSON). PENDING Test Designer review."""
import json, os, sys, math

LAYERS, TYPES, PCTS = (3, 16, 29), ("q", "k", "v", "o", "gate", "up", "down"), (2, 5, 10)
SETS = [f"eval_{i}" for i in range(1, 6)]
ARMS = ["A0", "A1", "A4"] + [f"{f}{s}" for f in "HQOR" for s in (25, 50, 75, 90)]
ARM_KEYS = {"arm": str, "e": float, "cal_e": float, "added_bytes": int, "storage_roundtrip_pass": bool, "gate_ok": bool}
TOP_REQ = {"B": int, "m": int, "n": int, "arms": list, "layer": int, "type": str, "pct": int, "set_id": str, "label": str,
           "source_sha256": (dict, str), "binary_sha256": str, "cal_x_sha256": str, "eval_x_sha256": str}
TOP_OPT = {"smoke": bool, "note": str}


def typed(v, t):
    if t is float: return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if t is int: return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, t)


def check_file(fn, L, t, p, s):
    errs = []
    try: c = json.load(open(fn))
    except Exception as ex: return [f"unreadable: {ex}"]
    if not isinstance(c, dict): return ["top level is not an object"]
    extra = set(c) - set(TOP_REQ) - set(TOP_OPT)
    miss = set(TOP_REQ) - set(c)
    if extra: errs.append(f"extra top keys {sorted(extra)}")
    if miss: errs.append(f"missing top keys {sorted(miss)}")
    for k, tt in {**TOP_REQ, **TOP_OPT}.items():
        if k in c and not typed(c[k], tt): errs.append(f"top key {k} has type {type(c[k]).__name__}")
    if c.get("layer") != L or c.get("type") != t or c.get("pct") != p or c.get("set_id") != s: errs.append("provenance does not match file name")
    arms = c.get("arms")
    if isinstance(arms, list):
        names = [a.get("arm") if isinstance(a, dict) else None for a in arms]
        if sorted(map(str, names)) != sorted(ARMS) or len(names) != len(set(names)): errs.append(f"arm set/duplicates: {names}")
        for a in arms:
            if not isinstance(a, dict): errs.append("arm entry is not an object"); continue
            if set(a) != set(ARM_KEYS): errs.append(f"{a.get('arm')}: keys {sorted(a)}")
            for k, tt in ARM_KEYS.items():
                if k in a and not typed(a[k], tt): errs.append(f"{a.get('arm')}: {k} type {type(a[k]).__name__}")
    return errs


def main():
    d, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out): sys.exit("refusing to overwrite " + out)
    want = {f"L{L}_{t}_p{p:02d}_{s}.json": (L, t, p, s) for L in LAYERS for t in TYPES for p in PCTS for s in SETS}
    have = set(os.listdir(d))
    res = {"files_expected": len(want), "files_present": len(have & set(want)), "missing": sorted(set(want) - have),
           "unexpected": sorted(have - set(want)), "bad": {}}
    for fn, key in want.items():
        if fn in have:
            e = check_file(os.path.join(d, fn), *key)
            if e: res["bad"][fn] = e
    res["files_valid"] = res["files_present"] - len(res["bad"])
    res["schema_ok"] = not res["missing"] and not res["unexpected"] and not res["bad"]
    json.dump(res, open(out, "w"), indent=1)
    print("expected", res["files_expected"], "present", res["files_present"], "valid", res["files_valid"], "missing", len(res["missing"]),
          "unexpected", len(res["unexpected"]), "schema_ok", res["schema_ok"])
    sys.exit(0 if res["schema_ok"] else 1)


if __name__ == "__main__":
    main()

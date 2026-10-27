"""Compare job22/p7fold2 vs job23/p7fold3 measure-mode output on identical SYNTHETIC inputs for A0/A1/A3/A5 (not a measurement)."""
import json, glob, os
B = "/home/workspace/supervisor/scratch/p7-fold-harness/job23/smoke/"
tot = bad = 0
for old, new in (("old_reg", "new_reg"), ("old_k", "new_k")):
    for fo in sorted(glob.glob(B + old + "/*.json")):
        fn = B + new + "/" + os.path.basename(fo)
        o, n = json.load(open(fo)), json.load(open(fn))
        ao = {a["arm"]: a for a in o["arms"]}; an = {a["arm"]: a for a in n["arms"]}
        for arm in ("A0", "A1", "A3", "A5") if "A5" in ao else ("A0", "A1", "A3"):
            for key in ("e", "added_bytes", "weight_rmse"):
                tot += 1
                if ao[arm][key] != an[arm][key]:
                    bad += 1; print("DIFF", os.path.basename(fo), arm, key, ao[arm][key], an[arm][key])
        if "A5" not in ao: pass
print("compared values", tot, "differing", bad, "files", len(glob.glob(B + "old_*/*.json")))

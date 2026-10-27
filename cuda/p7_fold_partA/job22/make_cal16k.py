"""Job 22: build the A-2 calibration set (16,384 tokens) with exactly the job #20 tokenization. Tokenizes only.
Run in 40-RD via job22/runpy.sh. Reads job20/inputs (read-only), writes job22/inputs/."""
import hashlib, json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SNAP = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"
I20 = "/home/workspace/supervisor/scratch/p7-fold-harness/job20/inputs/"
O = "/home/workspace/supervisor/scratch/p7-fold-harness/job22/inputs/"
N = 16384
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
PQ_SHA = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
m20 = json.load(open(I20 + "inputs_meta.json"))
assert sha(I20 + "gsm8k-test.parquet") == PQ_SHA, "OPEN_inputs: parquet sha mismatch"
for s in ["eval_1", "eval_2", "eval_3", "eval_4", "eval_5", "cal"]:
    assert sha(I20 + f"{s}.i32") == m20["sets"][s]["i32_sha256"], f"OPEN_inputs: {s}.i32 sha mismatch"
tab = pq.read_table(I20 + "gsm8k-test.parquet")
q, a = tab.column("question").to_pylist(), tab.column("answer").to_pylist()
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
bos = tok.bos_token_id
assert bos == m20["bos_token_id"]
order = list(range(0, 13)) + list(range(81, len(q)))
ids, seqs = [], []
for p in order:
    if len(ids) >= N:
        break
    t = [bos] + tok(q[p] + "\n" + a[p], add_special_tokens=False)["input_ids"]
    room = N - len(ids)
    trunc = len(t) > room
    t = t[:room]
    seqs.append({"problem_index": p, "start_offset": len(ids), "length": len(t), "truncated": trunc})
    ids += t
assert len(ids) == N
arr = np.asarray(ids, dtype="<i4")
arr.tofile(O + "cal16k.i32")
json.dump(seqs, open(O + "cal16k_seqs.json", "w"), indent=0)
open(O + "cal16k_seqlens.txt", "w").write(",".join(str(x["length"]) for x in seqs))
used = [x["problem_index"] for x in seqs]
in_1380 = [p for p in used if 13 <= p <= 80]
# compare to job20 cal.i32 (2048 tokens): first differing token
old = np.fromfile(I20 + "cal.i32", dtype="<i4")
n = min(len(old), len(arr))
diff = np.nonzero(old[:n] != arr[:n])[0]
j20 = json.load(open(I20 + "cal_seqs.json"))
meta = {"label": "PENDING Test Designer review", "parquet_sha256": sha(I20 + "gsm8k-test.parquet"),
        "tokenizer_json_sha256": sha(SNAP + "/tokenizer.json"), "bos_token_id": bos,
        "tokenization": "text = question + '\\n' + answer; add_special_tokens=False; BOS prepended; each problem own sequence; positions restart at 0",
        "tokens": N, "sequences": len(seqs), "i32_sha256": sha(O + "cal16k.i32"),
        "problem_index_first": used[0], "problem_index_last": used[-1],
        "problems_0_12_whole": all(not seqs[i]["truncated"] for i in range(13)),
        "problems_used_0_12": used[:13], "problems_used_81_onward_count": len(used) - 13,
        "last_problem_index": used[-1], "last_problem_truncated": seqs[-1]["truncated"],
        "last_problem_kept_tokens": seqs[-1]["length"], "max_position_used": max(x["length"] for x in seqs) - 1,
        "max_sequence_length": max(x["length"] for x in seqs),
        "job20_cal_vs_cal16k": {"job20_cal_tokens": int(len(old)), "first_differing_token_index": int(diff[0]) if len(diff) else None,
                                "job20_problem12_truncated_to": j20[-1]["length"], "job20_problem12_index": j20[-1]["problem_index"],
                                "cal16k_problem12_length": seqs[12]["length"],
                                "note": "job20 cal.i32 is the first 2048 tokens of cal16k (job20 cut problem 12 at 31 tokens; here problem 12 is whole, 197 tokens). first_differing_token_index null = identical over the common 2048 tokens; the sets differ only by cal16k continuing past token 2048."},
        "G_d_no_cal_problem_in_13_80": {"pass": not in_1380, "offending": in_1380},
        "eval_sets_unchanged": {s: {"path": f"job20/inputs/{s}.i32", "i32_sha256_verified": m20['sets'][s]['i32_sha256']} for s in ["eval_1", "eval_2", "eval_3", "eval_4", "eval_5"]},
        "G_d_cal_eval_problem_index_disjoint": {"pass": not (set(used) & set(range(13, 81))), "eval_problem_range": [13, 80]}}
json.dump(meta, open(O + "cal16k_meta.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in meta if k not in ("eval_sets_unchanged",)}, indent=1))

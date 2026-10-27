import json, struct, sys, os
S = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"
LIB = "/opt/venv/lib/python3.12/site-packages/transformers/models/phi3/modeling_phi3.py"
REM = S + "/modeling_phi3.py"
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
def hdr(fn):
    with open(S + "/" + fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))
H = {}
shapes = {}
exp_shapes = {"self_attn.qkv_proj.weight": [5120, 3072], "mlp.gate_up_proj.weight": [16384, 3072],
              "self_attn.o_proj.weight": [3072, 3072], "mlp.down_proj.weight": [3072, 8192]}
shape_ok = True
for L in (3, 16, 29):
    for suf, exp in exp_shapes.items():
        k = f"model.layers.{L}.{suf}"
        fn = idx[k]
        H.setdefault(fn, hdr(fn))
        e = H[fn][k]
        shapes[k] = {"file": fn, "shape": e["shape"], "dtype": e["dtype"], "expected": exp, "ok": e["shape"] == exp}
        shape_ok &= e["shape"] == exp
cfg = json.load(open(S + "/config.json"))
hs, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
hd = cfg.get("head_dim", hs // nh)
inter = cfg["intermediate_size"]
def lines(path, pats):
    out = []
    for i, l in enumerate(open(path), 1):
        if any(p in l for p in pats):
            out.append({"file": path, "line": i, "text": l.rstrip("\n")})
    return out
pats = ["query_pos =", "query_states = qkv", "key_states = qkv", "value_states = qkv", "chunk(2, dim=-1)", "op_size =", "self.head_dim ="]
ql, kl = [], []
qlib, qrem = lines(LIB, pats), lines(REM, pats)
strip = lambda L: [x["text"].strip() for x in L]
code_agree = strip(qlib) == strip(qrem)
qp = nh * hd
kvd = nkv * hd
derived = {"q": [0, qp], "k": [qp, qp + kvd], "v": [qp + kvd, qp + 2 * kvd],
           "gate": [0, inter], "up": [inter, 2 * inter]}
expected = {"q": [0, 3072], "k": [3072, 4096], "v": [4096, 5120], "gate": [0, 8192], "up": [8192, 16384]}
# transformers actually used at load time
import transformers, transformers.models.phi3.modeling_phi3 as m
used = m.__file__
res = {"label": "PENDING Test Designer review", "shapes": shapes, "shapes_all_ok": shape_ok,
       "config": {"hidden_size": hs, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
                  "head_dim_source": "config has no head_dim key; hidden/heads" if "head_dim" not in cfg else "config.head_dim",
                  "intermediate_size": inter, "partial_rotary_factor": cfg["partial_rotary_factor"],
                  "rope_scaling_type": cfg["rope_scaling"]["type"],
                  "short_factor_len": len(cfg["rope_scaling"]["short_factor"]),
                  "short_factor_all_ones": all(x == 1.0 for x in cfg["rope_scaling"]["short_factor"]),
                  "long_factor_len": len(cfg["rope_scaling"]["long_factor"]),
                  "original_max_position_embeddings": cfg["original_max_position_embeddings"],
                  "max_position_embeddings": cfg["max_position_embeddings"], "rope_theta": cfg["rope_theta"]},
       "code_lines_library": qlib, "code_lines_snapshot_remote": qrem,
       "code_files_agree_on_quoted_lines": code_agree,
       "transformers_version": transformers.__version__, "file_used_at_load_time": used,
       "note_used": "trust_remote_code off => library module (this path); snapshot modeling_phi3.py is only used with trust_remote_code=True",
       "derived_split_rows": derived, "expected_split_rows": expected, "split_match": derived == expected,
       "code_semantics": "q=qkv[..., :query_pos], k=qkv[..., query_pos:query_pos+kv_heads*head_dim], v=rest; gate,up=gate_up.chunk(2,-1) so gate is first half (rows [0,inter))",
       "match": bool(shape_ok and derived == expected and code_agree and used == LIB)}
json.dump(res, open("job22/layout_check.json", "w"), indent=1)
print("match", res["match"], "shape_ok", shape_ok, "split_match", derived == expected, "code_agree", code_agree, "used", used)

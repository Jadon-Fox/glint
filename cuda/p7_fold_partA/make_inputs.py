"""SMOKE input prep (input prep only, never a measurement). Run inside 40-Runtime-Deps with /opt/venv/bin/python.

Tokenizes two fixed texts with the snapshot tokenizer (64 tokens each, disjoint texts), then captures the
activations that enter layer-0 self_attn.o_proj using transformers Phi3Model with ONE layer loaded.
Only the embedding and layer-0 tensors are read from the safetensors file (safe_open get_tensor).
Outputs are labelled input_kind SMOKE-PY-REFERENCE-X by the harness; they never count toward G-e.
"""
import hashlib
import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.phi3.modeling_phi3 import Phi3Model

SNAP = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"
BASE = "/home/workspace/supervisor/scratch/p7-fold-harness/"
N_TOK = 64

CAL_TEXT = (
    "The layer maps a hidden state through a learned linear transform. Quantizing that transform to four bits "
    "per weight loses information, and the loss is not spread evenly across the matrix. Some rows and columns "
    "carry most of the output energy, while others barely move the result. A residual that repairs the largest "
    "errors first is a natural candidate, but the calibration text decides which errors matter. Every sentence "
    "here is written only to supply ordinary English tokens for the smoke run, not to be a measured corpus."
)
EVAL_TEXT = (
    "Morning deliveries arrived before the storm, so the warehouse crew stacked crates by the loading dock. "
    "A forklift driver counted pallets twice because the first tally disagreed with the manifest. Coffee was "
    "poured into paper cups while the supervisor updated the whiteboard with new shift times and a short list "
    "of safety reminders. Nobody remembered when the old scale was last calibrated, which became its own small "
    "project for the afternoon."
)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(SNAP)
    cal_ids = tok(CAL_TEXT, add_special_tokens=False)["input_ids"]
    eval_ids = tok(EVAL_TEXT, add_special_tokens=False)["input_ids"]
    assert len(cal_ids) >= N_TOK and len(eval_ids) >= N_TOK, (len(cal_ids), len(eval_ids))
    cal_ids = cal_ids[:N_TOK]
    eval_ids = eval_ids[:N_TOK]
    assert not set(map(tuple, [cal_ids])) & set(map(tuple, [eval_ids]))
    cal_np = np.asarray(cal_ids, dtype="<i4")
    eval_np = np.asarray(eval_ids, dtype="<i4")
    cal_np.tofile(BASE + "data/cal_ids.i32")
    eval_np.tofile(BASE + "data/eval_ids.i32")

    cfg = AutoConfig.from_pretrained(SNAP)
    cfg.num_hidden_layers = 1
    cfg._attn_implementation = "eager"
    torch.set_num_threads(4)
    model = Phi3Model(cfg).to(torch.bfloat16).eval()
    sd = {}
    with safe_open(SNAP + "/model-00001-of-00002.safetensors", framework="pt") as f:
        for name in ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight",
                     "model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.o_proj.weight",
                     "model.layers.0.post_attention_layernorm.weight"]:
            sd[name[len("model."):]] = f.get_tensor(name).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    # only MLP and the final norm are left unloaded; the o_proj hook fires before any of them matter
    assert not any(("self_attn" in k or "layernorm" in k or k.startswith("embed")) for k in missing), missing
    assert all(k.startswith("layers.0.mlp") or k.startswith("norm") for k in missing), missing

    captured = {}

    def hook(_mod, inputs):
        captured["x"] = inputs[0].detach().float().clone()

    h = model.layers[0].self_attn.o_proj.register_forward_pre_hook(hook)
    outs = {}
    with torch.no_grad():
        for label, ids in (("cal", cal_ids), ("eval", eval_ids)):
            model(torch.tensor([ids], dtype=torch.long))
            x = captured["x"][0].contiguous().numpy().astype("<f4")
            assert x.shape == (N_TOK, 3072), x.shape
            x.tofile(BASE + f"data/x_{label}_f32.bin")
            outs[label] = {"sha256": sha256_bytes(x.tobytes()), "shape": list(x.shape),
                           "x_rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))}
    h.remove()
    meta = {"input_kind": "SMOKE-PY-REFERENCE-X", "tokenizer": SNAP + "/tokenizer.json",
            "cal_text_sha256": sha256_bytes(CAL_TEXT.encode()), "eval_text_sha256": sha256_bytes(EVAL_TEXT.encode()),
            "cal_ids_sha256": sha256_bytes(cal_np.tobytes()), "eval_ids_sha256": sha256_bytes(eval_np.tobytes()),
            "cal_ids": cal_ids, "eval_ids": eval_ids,
            "x_cal_sha256": outs["cal"]["sha256"], "x_eval_sha256": outs["eval"]["sha256"],
            "x_cal_rms": outs["cal"]["x_rms"], "x_eval_rms": outs["eval"]["x_rms"],
            "x_cal_file": "data/x_cal_f32.bin", "x_eval_file": "data/x_eval_f32.bin",
            "note": "input prep only: layer-0 o_proj inputs from transformers Phi3Model (one layer, bf16 CPU)",
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    with open(BASE + "data/inputs_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("cal_ids", "eval_ids")}, indent=1))
    print("cal_ids", cal_ids[:8], "...", "eval_ids", eval_ids[:8], "...", file=sys.stderr)


if __name__ == "__main__":
    main()

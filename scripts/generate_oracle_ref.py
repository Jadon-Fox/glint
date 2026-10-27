"""scripts/generate_oracle_ref.py
Offline Oracle reference generator using HuggingFace Phi3ForCausalLM
Evaluates the first 2048 tokens of SCIENCE_JOURNAL.md.
Saves token IDs, reference cross entropy, PPL, and top-1 argmax tokens.
Zero Python on the C/CUDA forward eval path.
"""
import json
import hashlib
import torch
import torch.nn.functional as F
from transformers import Phi3ForCausalLM, AutoTokenizer

CORPUS_PATH = "/opt/models/science_log/SCIENCE_JOURNAL.md"
MODEL_DIR = "/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"

print("Loading tokenizer and corpus...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tok.encode(text)[:2048]
assert len(tokens) == 2048, f"Expected 2048 tokens, got {len(tokens)}"

# Save token IDs
tokens_str = " ".join(str(t) for t in tokens)
with open("/home/workspace/p7-scratch/artifacts/eval_tokens_2048.txt", "w") as f:
    f.write(tokens_str)

token_sha256 = hashlib.sha256(tokens_str.encode("utf-8")).hexdigest()
print(f"Tokens: {len(tokens)}, sha256: {token_sha256}")

print("Loading BF16 model on GPU...")
model = Phi3ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

input_ids = torch.tensor([tokens], device="cuda")
with torch.no_grad():
    out = model(input_ids)
    logits = out.logits[0]  # [2048, 200064]

shift_logits = logits[:-1, :]   # [2047, 200064]
shift_labels = input_ids[0, 1:]  # [2047]

# Compute top-1 predictions
top1_tokens = torch.argmax(shift_logits, dim=-1).cpu().tolist()

# Compute exact cross-entropy loss in chunks of 256
total_loss = 0.0
for i in range(0, 2047, 256):
    chunk_l = shift_logits[i:i+256].float()
    chunk_t = shift_labels[i:i+256]
    total_loss += F.cross_entropy(chunk_l, chunk_t, reduction="sum").item()

mean_loss = total_loss / 2047.0
ppl = float(torch.exp(torch.tensor(mean_loss)).item())

print(f"HuggingFace Oracle Mean CE: {mean_loss:.6f}")
print(f"HuggingFace Oracle PPL:     {ppl:.6f}")

# Save top-1 tokens as binary uint32
import struct
with open("/home/workspace/p7-scratch/artifacts/oracle_top1_2048.bin", "wb") as f:
    for t in top1_tokens:
        f.write(struct.pack("<I", t))

# Save reference summary
ref_summary = {
    "corpus_path": CORPUS_PATH,
    "corpus_sha256": "ca7951a114d367aad8cf625c391dc173d794fb1cd7eff9476dc019ae21797b15",
    "token_count": 2048,
    "token_sha256": token_sha256,
    "model_dir": MODEL_DIR,
    "loss": mean_loss,
    "ppl": ppl,
    "sample_first_10_top1": top1_tokens[:10]
}

with open("/home/workspace/p7-scratch/artifacts/oracle_ref_2048.json", "w") as f:
    json.dump(ref_summary, f, indent=2)

print("Saved artifacts/oracle_ref_2048.json and artifacts/oracle_top1_2048.bin successfully.")

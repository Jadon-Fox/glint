"""scripts/generate_oracle_ref_hf.py MODEL_DIR OUT_DIR
HF BF16 oracle for any causal LM (AutoModelForCausalLM, sdpa, bf16) on the first 2048 tokens of the science journal.
Writes eval_tokens_2048.txt, oracle_top1_2048.bin (uint32 LE, 2047 argmax ids), oracle_ref_2048.json (CE, PPL, shas, versions).
Same protocol as generate_oracle_ref.py (Phi-4-mini); this version takes the model from the command line so Qwen3 and others
can use the Mojo forward without a Phi3-specific class. Oracle only: nothing here is on a forward/eval path.
"""
import hashlib, json, os, struct, sys
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR, OUT = sys.argv[1], sys.argv[2]
CORPUS = "/opt/models/science_log/SCIENCE_JOURNAL.md"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
raw = open(CORPUS, "rb").read()
tokens = tok.encode(raw.decode("utf-8"), add_special_tokens=False)[:2048]
assert len(tokens) == 2048, len(tokens)
tokens_str = " ".join(str(t) for t in tokens)
open(f"{OUT}/eval_tokens_2048.txt", "w").write(tokens_str)
tok_sha = hashlib.sha256(tokens_str.encode()).hexdigest()

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
ids = torch.tensor([tokens], device="cuda")
with torch.no_grad():
    logits = model(ids).logits[0]
sl, lab = logits[:-1], ids[0, 1:]
top1 = torch.argmax(sl, dim=-1).cpu().tolist()
tot = 0.0
for i in range(0, 2047, 256):
    tot += F.cross_entropy(sl[i:i + 256].float(), lab[i:i + 256], reduction="sum").item()
mean = tot / 2047.0
ppl = float(torch.exp(torch.tensor(mean)))
with open(f"{OUT}/oracle_top1_2048.bin", "wb") as f:
    for t in top1:
        f.write(struct.pack("<I", t))
top1_sha = hashlib.sha256(open(f"{OUT}/oracle_top1_2048.bin", "rb").read()).hexdigest()
meta = dict(model_dir=MODEL_DIR, corpus_path=CORPUS, corpus_sha256=hashlib.sha256(raw).hexdigest(), token_count=2048,
            tokens_sha256=tok_sha, top1_sha256=top1_sha, mean_loss=mean, ppl=ppl, dtype="torch.bfloat16", attn_implementation="sdpa",
            transformers=transformers.__version__, torch=torch.__version__, device=torch.cuda.get_device_name(0),
            vocab=logits.shape[-1], first_10_top1=top1[:10])
json.dump(meta, open(f"{OUT}/oracle_ref_2048.json", "w"), indent=2)
print(f"CE {mean:.6f} PPL {ppl:.6f} tokens_sha {tok_sha} top1_sha {top1_sha}")

"""NF4 parity inputs and reference (run inside 40-Runtime-Deps with /opt/venv/bin/python).

Modes:
  make  -> data/parity_in.f64 (float64 inputs), data/parity_ref_packed.bin, data/parity_ref_absmax.f64
           reference bytes come from /home/workspace/fractal-deployment/bf16-to-int8/nf4.py (read-only import)
  cmp   -> compares data/parity_cuda_packed.bin / data/parity_cuda_absmax.f64 with the reference, writes out/parity.json
Pure stdlib + nf4.py; no torch on this path.
"""
import hashlib
import json
import math
import struct
import sys

sys.path.insert(0, "/home/workspace/fractal-deployment/bf16-to-int8")
import nf4  # noqa: E402  (read-only import; PYTHONDONTWRITEBYTECODE is set by the helper)

SAFETENSORS = ("/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/"
               "cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors")
HDR_BYTES = 13000
OPROJ_OFF = 1380200448  # data_offsets[0] for model.layers.0.self_attn.o_proj.weight, relative to data start
DATA_START = 8 + HDR_BYTES
N_REAL = 4096  # first 4096 elements of o_proj

BASE = "/home/workspace/supervisor/scratch/p7-fold-harness/"


def bf16_to_f64(u16):
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def read_oproj_head(n):
    with open(SAFETENSORS, "rb") as f:
        f.seek(DATA_START + OPROJ_OFF)
        raw = f.read(2 * n)
    return [bf16_to_f64(struct.unpack_from("<H", raw, 2 * i)[0]) for i in range(n)]


def tie_midpoints():
    """Values v (double) where nf4's nearest_code sees an exact tie between two codes (lower index must win)."""
    cb = nf4.NF4_CODEBOOK
    out = []
    for j in range(15):
        m = (cb[j] + cb[j + 1]) / 2.0
        for v in (m, math.nextafter(m, -math.inf), math.nextafter(m, math.inf)):
            for s in (1.0, -1.0):
                x = s * v
                d = [abs(x - c) for c in cb]
                dmin = min(d)
                if sum(1 for q in d if q == dmin) >= 2:
                    out.append(x)
    return sorted(set(out))


def make():
    real = read_oproj_head(N_REAL)
    ties = tie_midpoints()
    print("tie candidates found by nf4 arithmetic:", len(ties))
    assert ties, "no exact tie cases found; parity would be vacuous on ties"
    vals = list(real)
    # tie blocks: lead with 1.0 so absmax == 1.0 and scale == 1; fill with exact-tie values
    for b in range(0, len(ties), 63):
        chunk = ties[b:b + 63]
        block = [1.0] + chunk + [0.0] * (63 - len(chunk))
        vals.extend(block)
    # zero block: absmax 0 -> idx 7 (nibble 0x7 pairs -> byte 0x77)
    vals.extend([0.0] * 64)
    # block mixing -1.0 and tiny values (absmax 1.0 from negative side)
    vals.extend([-1.0] + [1e-9 * (i + 1) for i in range(63)])
    n = len(vals)
    assert n % 64 == 0
    with open(BASE + "data/parity_in.f64", "wb") as f:
        f.write(struct.pack("<%dd" % n, *vals))
    packed, absmax = nf4.quantize_nf4(vals, blocksize=64)
    with open(BASE + "data/parity_ref_packed.bin", "wb") as f:
        f.write(bytes(packed))
    with open(BASE + "data/parity_ref_absmax.f64", "wb") as f:
        f.write(struct.pack("<%dd" % len(absmax), *absmax))
    meta = {"n": n, "n_real_oproj": N_REAL, "n_tie_values": len(ties), "n_tie_blocks": math.ceil(len(ties) / 63),
            "n_zero_blocks": 1, "packed_bytes": len(packed), "n_absmax": len(absmax),
            "ref_packed_sha256": hashlib.sha256(bytes(packed)).hexdigest(),
            "ref_absmax_sha256": hashlib.sha256(struct.pack("<%dd" % len(absmax), *absmax)).hexdigest(),
            "reference": "fractal-deployment/bf16-to-int8/nf4.py sha256 76530e62...(read-only)"}
    with open(BASE + "data/parity_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


def cmp():
    with open(BASE + "data/parity_meta.json") as f:
        meta = json.load(f)
    n = meta["n"]
    ref_p = open(BASE + "data/parity_ref_packed.bin", "rb").read()
    got_p = open(BASE + "data/parity_cuda_packed.bin", "rb").read()
    ref_a = struct.unpack("<%dd" % meta["n_absmax"], open(BASE + "data/parity_ref_absmax.f64", "rb").read())
    got_a = struct.unpack("<%dd" % meta["n_absmax"], open(BASE + "data/parity_cuda_absmax.f64", "rb").read())
    mism_bytes = sum(1 for a, b in zip(ref_p, got_p) if a != b) + abs(len(ref_p) - len(got_p))
    mism_abs = sum(1 for a, b in zip(ref_a, got_a) if a != b) + abs(len(ref_a) - len(got_a))
    res = {"n_compared_elements": n, "n_real_oproj": meta["n_real_oproj"],
           "n_tie_values": meta["n_tie_values"], "n_zero_blocks": meta["n_zero_blocks"],
           "packed_bytes_ref": len(ref_p), "packed_bytes_cuda": len(got_p), "mismatching_packed_bytes": mism_bytes,
           "absmax_count_ref": len(ref_a), "absmax_count_cuda": len(got_a), "mismatching_absmax": mism_abs,
           "blocksize": 64, "nibble_order": "lo_then_hi", "tie_rule": "lower index (strict <)",
           "reference": "nf4.py quantize_nf4 (read-only import), /opt/venv/bin/python",
           "measured_path": "CUDA kernels p7fold parity mode (double arithmetic, same order as nf4.py)",
           "pass_parity": mism_bytes == 0 and mism_abs == 0}
    with open(BASE + "out/parity.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    {"make": make, "cmp": cmp}[sys.argv[1]]()

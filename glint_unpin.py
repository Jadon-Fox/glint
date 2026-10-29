#!/usr/bin/env python3
"""GLINT pin → dense safetensors (BF16 by default). The unfold, done offline.
Uses the pin's own _glint_cells table. Output keeps the original tensor names
and shapes, so any BF16 loader can read it. Lossy: it is the GLINT
reconstruction, not the source weights.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict
from dtype_io import numel_of, pack_bf16
from glint_codec import CELLS, pack_f32, unpack_f32, unpack_indices
from glint_pinfile import INDEX_NAME, Pin
from safetensors_io import write_safetensors
try:
    import numpy as np
    import glint_np
except ImportError:
    np = None
    glint_np = None
DTYPES = ("BF16", "F16", "F32")
def _expand(state: dict, hole: bytes, plug: bytes, am: bytes, cells, dtype: str) -> bytes:
    n = numel_of(tuple(state["shape"]))
    bs = int(state["blocksize"])
    if glint_np is not None:
        w = glint_np.expand(hole, plug, np.frombuffer(am, dtype="<f4"), n, bs, cells)
        if dtype == "BF16":
            return glint_np.f32_to_bf16_bytes(w)
        return w.astype("<f2" if dtype == "F16" else "<f4").tobytes()
    import struct
    tab = [cells[k * 16 : (k + 1) * 16] for k in range(16)]
    h = unpack_indices(hole, n)
    p = unpack_indices(plug, n)
    a = unpack_f32(am)
    # round through f32 so both paths agree with k_expand
    w = [struct.unpack("<f", struct.pack("<f", tab[h[i]][p[i]] * a[i // bs]))[0] for i in range(n)]
    if dtype == "BF16":
        return pack_bf16(w)
    if dtype == "F32":
        return pack_f32(w)
    return struct.pack("<" + "e" * n, *w)
def unpin(pin_dir: Path, out: Path, dtype: str = "BF16") -> dict:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with Pin(pin_dir) as pin:
        if pin.meta.get("schema") != "glint_pin_v1":
            raise ValueError(f"not a glint_pin_v1 pin: {pin.meta.get('schema')}")
        cells = pin.cells() or [x for row in CELLS for x in row]
        per_shard: Dict[str, list] = {s: [] for s in pin.shards}
        stem_of = {stem + ".weight": stem for stem in pin.stems()}
        owned = set()
        for stem in pin.stems():
            owned.update(pin.module_names(stem))
        n_expanded = 0
        for name, shard in pin.where.items():
            if name in stem_of:
                stem = stem_of[name]
                state, hole, plug, am = pin.module(stem)
                blob = _expand(state, hole, plug, am, cells, dtype)
                per_shard[shard].append((name, dtype, tuple(state["shape"]), blob))
                n_expanded += 1
            elif name not in owned and name != "_glint_cells":
                info = pin.info(name)
                per_shard[shard].append((name, info.dtype, info.shape, pin.read(name)))
        weight_map = {}
        total = 0
        for shard, tensors in per_shard.items():
            write_safetensors(str(out / shard), tensors, metadata={"format": "pt", "unpinned_from": "glint"})
            for t in tensors:
                weight_map[t[0]] = shard
                total += len(t[3])
        if len(per_shard) > 1:
            (out / INDEX_NAME).write_text(
                json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
            )
        for side in pin.meta.get("sidecar", []):
            p = pin.root / side
            if p.is_file():
                shutil.copyfile(p, out / side)
    return {"out": str(out), "dtype": dtype, "n_expanded": n_expanded, "shards": list(per_shard)}
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GLINT pin → dense safetensors")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="BF16", choices=DTYPES)
    args = p.parse_args(argv)
    print(json.dumps(unpin(Path(args.pin), Path(args.out), args.dtype), indent=2))
    return 0
if __name__ == "__main__":
    sys.exit(main())

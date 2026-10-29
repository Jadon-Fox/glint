#!/usr/bin/env python3
"""Check a GLINT pin and board the evidence.
Always: schema, per-module layout (hole/plug = ceil(n/2) bytes, absmax =
ceil(n/bs) f32, zero pad nibble on odd n), _glint_cells == codec table.
--src: per-module RMSE of the pin's reconstruction vs the source weights.
--cuda: GPU expand (libglint_encode.so) vs host expand, max_abs < 1e-5.
--parity: re-encode the source on the host (numpy, f32-exact to the CUDA
kernel) and compare hole/plug/absmax bytes. This is the host-vs-CUDA byte
gate: run it on a pin made by libglint_encode.so. Exit 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional
from dtype_io import numel_of
from glint_codec import cells_flat, pack_f32, unpack_f32
from glint_pin import resolve_shards
from glint_cuda import expand_cuda
from glint_pinfile import Pin
from safetensors_io import SafeTensorsFile
try:
    import numpy as np
    import glint_np
except ImportError:
    np = None
    glint_np = None
CUDA_EXPAND_TOL = 1e-5 # GROK_BUILD_GPU_TEST.md golden; same f32 op, expect 0
def _source_index(src: Path) -> Dict[str, SafeTensorsFile]:
    out: Dict[str, SafeTensorsFile] = {}
    for wf in resolve_shards(src):
        st = SafeTensorsFile(str(wf))
        for name in st.tensors:
            out[name] = st
    return out
def _first_diff(a: bytes, b: bytes) -> Optional[int]:
    if len(a) != len(b):
        return min(len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None
def verify(pin_dir: Path, src: Optional[Path] = None, parity: bool = False, cuda: bool = False) -> dict:
    if (src is not None or parity or cuda) and glint_np is None:
        raise RuntimeError("--src / --parity / --cuda need numpy")
    if parity and src is None:
        raise ValueError("--parity needs --src")
    fails = []
    modules = []
    with Pin(pin_dir) as pin:
        if pin.meta.get("schema") != "glint_pin_v1":
            fails.append(f"schema={pin.meta.get('schema')!r}")
        cells = pin.cells()
        if cells is None:
            fails.append("_glint_cells missing")
        elif cells != unpack_f32(pack_f32(cells_flat())):
            fails.append("_glint_cells differs from glint_codec.CELLS")
        src_ix = _source_index(src) if src is not None else {}
        stems = pin.stems()
        if pin.meta.get("n_converted") not in (None, len(stems)):
            fails.append(f"pin.json n_converted={pin.meta.get('n_converted')} but {len(stems)} modules")
        for stem in stems:
            row = {"module": stem, "ok": True, "problems": []}
            try:
                state, hole, plug, am = pin.module(stem)
            except KeyError as e:
                row["problems"].append(f"missing tensor {e}")
                row["ok"] = False
                modules.append(row)
                continue
            n = numel_of(tuple(state["shape"]))
            bs = int(state["blocksize"])
            npk, nam = (n + 1) // 2, (n + bs - 1) // bs
            if len(hole) != npk or len(plug) != npk:
                row["problems"].append(f"hole/plug bytes {len(hole)}/{len(plug)} != {npk}")
            if len(am) != 4 * nam:
                row["problems"].append(f"absmax {len(am) // 4} floats != {nam}")
            if n & 1 and npk and ((hole[-1] >> 4) or (plug[-1] >> 4)):
                row["problems"].append("odd n but pad nibble is not zero")
            if not row["problems"] and cuda:
                host = glint_np.expand(hole, plug, np.frombuffer(am, dtype="<f4"), n, bs, cells)
                dev = np.frombuffer(expand_cuda(hole, plug, am, n, bs), dtype="<f4")
                row["cuda_expand_max_abs"] = float(np.max(np.abs(dev - host))) if n else 0.0
                if not row["cuda_expand_max_abs"] < CUDA_EXPAND_TOL:
                    row["problems"].append(f"cuda expand max_abs {row['cuda_expand_max_abs']:.3g} >= {CUDA_EXPAND_TOL}")
            if not row["problems"] and src is not None:
                name = stem + ".weight"
                st = src_ix.get(name)
                if st is None:
                    row["problems"].append("not in --src")
                else:
                    info = st.tensors[name]
                    if list(info.shape) != list(state["shape"]):
                        row["problems"].append(f"shape {state['shape']} != src {list(info.shape)}")
                    else:
                        w = glint_np.dense_to_f32(info.dtype, st.read_bytes(name), n)
                        amf = np.frombuffer(am, dtype="<f4")
                        rec = glint_np.expand(hole, plug, amf, n, bs, cells)
                        row["rmse"] = glint_np.rmse(w, rec)
                        ref = float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) if n else 0.0
                        row["rel_rmse"] = row["rmse"] / ref if ref else 0.0
                        if not np.all(np.isfinite(rec[np.isfinite(w)])):
                            row["problems"].append("non-finite reconstruction of finite weights")
                        if parity:
                            h2, p2, am2 = glint_np.encode_f32(w, bs)
                            par = {
                                "hole_first_diff": _first_diff(hole, h2),
                                "plug_first_diff": _first_diff(plug, p2),
                                "absmax_first_diff": _first_diff(am, am2.astype("<f4").tobytes()),
                            }
                            hb = np.frombuffer(hole, dtype=np.uint8)
                            pb = np.frombuffer(plug, dtype=np.uint8)
                            par["hole_bytes_differ"] = int(np.count_nonzero(hb != np.frombuffer(h2, dtype=np.uint8)))
                            par["plug_bytes_differ"] = int(np.count_nonzero(pb != np.frombuffer(p2, dtype=np.uint8)))
                            row["parity"] = par
                            if any(v is not None for k, v in par.items() if k.endswith("first_diff")):
                                row["problems"].append("host re-encode bytes differ")
            row["ok"] = not row["problems"]
            if not row["problems"]:
                del row["problems"]
            modules.append(row)
        for st in {id(s): s for s in src_ix.values()}.values():
            st.close()
    rated = [m["rmse"] for m in modules if "rmse" in m]
    bad = [m["module"] for m in modules if not m["ok"]]
    return {
        "schema": "glint_verify_v1",
        "pin": str(Path(pin_dir).expanduser().resolve()),
        "backend": pin.meta.get("backend"),
        "n_modules": len(modules),
        "n_failed": len(bad),
        "rmse_mean": sum(rated) / len(rated) if rated else None,
        "rmse_max": max(rated) if rated else None,
        "parity_checked": parity,
        "cuda_expand_checked": cuda,
        "ok": not fails and not bad,
        "failures": fails + [f"module {b}" for b in bad],
        "modules": modules,
    }
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify a GLINT pin")
    p.add_argument("--pin", required=True)
    p.add_argument("--src", help="source BF16/F16/F32 model (file or dir) for RMSE")
    p.add_argument("--parity", action="store_true", help="host re-encode byte gate (needs --src)")
    p.add_argument("--cuda", action="store_true", help="GPU expand vs host expand, max_abs < 1e-5 (needs libglint_encode.so + GPU)")
    p.add_argument("--out", help="write the evidence JSON here")
    p.add_argument("--quiet", action="store_true", help="omit per-module rows from stdout")
    args = p.parse_args(argv)
    got = verify(Path(args.pin), Path(args.src) if args.src else None, args.parity, args.cuda)
    text = json.dumps(got, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    shown = {k: v for k, v in got.items() if k != "modules"} if args.quiet else got
    print(json.dumps(shown, indent=2))
    return 0 if got["ok"] else 1
if __name__ == "__main__":
    sys.exit(main())

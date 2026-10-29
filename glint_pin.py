#!/usr/bin/env python3
"""BF16/F16/F32 dense linear → GLINT pin (hole + plug). Separate from nf4-to-int8.
Takes one .safetensors, a model dir, or a sharded model dir
(model.safetensors.index.json). Sharded in → sharded out with its own index.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List
from dtype_io import numel_of, unpack_dense
from glint_codec import cells_flat, decode, encode, pack_f32, pack_indices, rmse
from glint_cuda import encode_bf16_cuda, load as load_cuda
from safetensors_io import SafeTensorsFile, write_safetensors
try:
    import numpy as np
    import glint_np
except ImportError: # stdlib-only install: pure Python host path
    np = None
    glint_np = None
PARENT_KEEP = ("norm", "bias", "rotary", "cos_cached", "sin_cached")
INDEX_NAME = "model.safetensors.index.json"
# Non-weight files copied next to the pin so it stays a loadable model dir.
SIDECAR = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
def _is_keep(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in PARENT_KEEP) or name.endswith(".bias")
def resolve_shards(src: Path) -> List[Path]:
    """Source → ordered shard list. A dir needs model.safetensors, an index, or one shard."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        return [src]
    single = src / "model.safetensors"
    if single.is_file():
        return [single]
    index = src / INDEX_NAME
    if index.is_file():
        weight_map: Dict[str, str] = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        missing = [n for n in names if not (src / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{INDEX_NAME} names missing shards: {missing}")
        return [src / n for n in names]
    shards = sorted(src.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(f"need model.safetensors, {INDEX_NAME}, or exactly one .safetensors in {src}")
    return shards
def _encode_one(dtype: str, raw: bytes, shape, blocksize: int, backend: str):
    """→ (hole_bytes, plug_bytes, absmax_f32_bytes, rmse or None)."""
    n = numel_of(shape)
    if backend == "cuda_bf16":
        hole_b, plug_b, am_b = encode_bf16_cuda(raw, n, blocksize)
        err = None
        if glint_np is not None:
            w = glint_np.dense_to_f32(dtype, raw, n)
            am = np.frombuffer(am_b, dtype="<f4")
            err = glint_np.rmse(w, glint_np.expand(hole_b, plug_b, am, n, blocksize))
        return hole_b, plug_b, am_b, err
    if backend == "cpu_numpy_f32":
        w = glint_np.dense_to_f32(dtype, raw, n)
        hole_b, plug_b, am = glint_np.encode_f32(w, blocksize)
        err = glint_np.rmse(w, glint_np.expand(hole_b, plug_b, am, n, blocksize))
        return hole_b, plug_b, am.astype("<f4").tobytes(), err
    f32 = unpack_dense(dtype, raw, shape)[:n]
    hole_i, plug_i, am = encode(f32, blocksize)
    err = rmse(f32, decode(hole_i, plug_i, am, blocksize))
    return bytes(pack_indices(hole_i)), bytes(pack_indices(plug_i)), pack_f32(am), err
def _pick_backend(dtype: str, cuda_ok: bool, force_python: bool) -> str:
    if cuda_ok and dtype == "BF16":
        return "cuda_bf16"
    if glint_np is not None and not force_python:
        return "cpu_numpy_f32"
    return "cpu_python"
def convert(
    src: Path,
    out: Path,
    blocksize: int = 64,
    dry_run: bool = False,
    allow_cpu: bool = False,
    force_python: bool = False,
) -> dict:
    if blocksize <= 0:
        raise ValueError("blocksize must be positive")
    shards = resolve_shards(src)
    out = out.expanduser().resolve()
    sharded = len(shards) > 1
    # plan pass: headers only
    plan_linears: List[str] = []
    plan_keep: List[str] = []
    for wf in shards:
        with SafeTensorsFile(str(wf)) as st:
            for name, info in st.tensors.items():
                is_dense = info.dtype in ("BF16", "F16", "F32") and len(info.shape) == 2
                if is_dense and name.endswith(".weight") and not _is_keep(name):
                    plan_linears.append(name)
                else:
                    plan_keep.append(name)
    cuda_lib = load_cuda()
    if dry_run:
        return {
            "schema": "glint_pin_v1",
            "n_linears": len(plan_linears),
            "n_keep": len(plan_keep),
            "blocksize": blocksize,
            "shards": [s.name for s in shards],
            "linears": plan_linears,
            "cuda": cuda_lib is not None,
            "numpy": glint_np is not None,
        }
    if cuda_lib is None and not allow_cpu:
        raise RuntimeError(
            "GLINT encode is CUDA (libglint_encode.so). "
            "bash scripts/build_glint_cuda.sh on the 3060, or pass --allow-cpu for tiny tests."
        )
    out.mkdir(parents=True, exist_ok=True)
    linear_set = set(plan_linears)
    reports = []
    weight_map: Dict[str, str] = {}
    total_size = 0
    for si, wf in enumerate(shards):
        out_name = wf.name if sharded else "model.safetensors"
        tensors = []
        with SafeTensorsFile(str(wf)) as st:
            for name, info in st.tensors.items():
                raw = st.read_bytes(name)
                if name not in linear_set:
                    tensors.append((name, info.dtype, info.shape, raw))
                    continue
                backend = _pick_backend(info.dtype, cuda_lib is not None, force_python)
                hole_b, plug_b, am_b, err = _encode_one(info.dtype, raw, info.shape, blocksize, backend)
                stem = name[: -len(".weight")]
                state = json.dumps(
                    {"quant_type": "glint", "blocksize": blocksize, "shape": list(info.shape), "src_dtype": info.dtype},
                    separators=(",", ":"),
                ).encode()
                tensors += [
                    (stem + ".weight", "U8", (len(hole_b), 1), hole_b),
                    (stem + ".weight.glint_plug", "U8", (len(plug_b), 1), plug_b),
                    (stem + ".weight.absmax", "F32", (len(am_b) // 4,), am_b),
                    (stem + ".weight.glint_state", "U8", (len(state),), state),
                ]
                reports.append(
                    {"src": stem, "rmse": err, "shape": list(info.shape), "backend": backend, "shard": out_name}
                )
        if si == 0:
            tensors.append(("_glint_cells", "F32", (16, 16), pack_f32(cells_flat())))
        for t in tensors:
            weight_map[t[0]] = out_name
            total_size += len(t[3])
        write_safetensors(str(out / out_name), tensors, metadata={"format": "pt", "quantization": "glint"})
        del tensors
    if sharded:
        index = {"metadata": {"total_size": total_size, "quantization": "glint"}, "weight_map": weight_map}
        (out / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    src_dir = shards[0].parent
    copied = []
    for side in SIDECAR:
        p = src_dir / side
        if p.is_file() and p.resolve() != (out / side).resolve():
            shutil.copyfile(p, out / side)
            copied.append(side)
    rated = [r["rmse"] for r in reports if r["rmse"] is not None]
    backends = sorted({r["backend"] for r in reports})
    pin = {
        "schema": "glint_pin_v1",
        "src": str(shards[0] if not sharded else src_dir),
        "dest": "glint",
        "n_converted": len(reports),
        "blocksize": blocksize,
        "rmse_mean": (sum(rated) / len(rated)) if rated else None,
        "rmse_max": max(rated) if rated else None,
        "backend": backends[0] if len(backends) == 1 else ("mixed" if backends else None),
        "backends": backends,
        "shards": [s.name for s in shards] if sharded else ["model.safetensors"],
        "sidecar": copied,
        "note": "hole=parent Gaussian cell, plug=child index. H-TILE inflate GLINT_CELLS[h][p]*absmax",
    }
    (out / "pin.json").write_text(json.dumps(pin, indent=2) + "\n")
    (out / "CONVERT_REPORT.json").write_text(json.dumps({"pin": pin, "modules": reports}, indent=2) + "\n")
    return {"pin": pin, "modules": reports}
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GLINT: BF16 → hole+plug pin")
    p.add_argument("--src", required=True, help=".safetensors file, model dir, or sharded model dir")
    p.add_argument("--out", required=True)
    p.add_argument("--blocksize", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="host encode when CUDA is missing (numpy f32 if present)")
    p.add_argument("--force-python", action="store_true", help="host encode in pure Python (slow, f64 reference)")
    args = p.parse_args(argv)
    got = convert(Path(args.src), Path(args.out), args.blocksize, args.dry_run, args.allow_cpu, args.force_python)
    print(json.dumps(got if args.dry_run else got["pin"], indent=2))
    return 0
if __name__ == "__main__":
    sys.exit(main())

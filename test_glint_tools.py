#!/usr/bin/env python3
"""Pin tooling: sharded pin, numpy f32 encode, verify, unpin, table sync."""
from __future__ import annotations
import json
import math
import random
import re
import struct
import tempfile
from pathlib import Path
import pytest
from dtype_io import pack_bf16, unpack_bf16
from glint_codec import CELLS, PARENT, encode, pack_indices, write_c_header
from glint_pin import convert
from glint_pinfile import Pin
from safetensors_io import SafeTensorsFile, write_safetensors
from glint_unpin import unpin
from glint_verify import verify
ROOT = Path(__file__).resolve().parent
@pytest.fixture(autouse=True)
def _host_only(monkeypatch, request):
    """These tests pin the host backends. A built .so must not reroute them to CUDA."""
    if "gpu" not in request.keywords:
        import glint_pin
        monkeypatch.setattr(glint_pin, "load_cuda", lambda: None)
def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]
def _gauss(n: int, seed: int, scale: float = 0.02) -> list:
    rng = random.Random(seed)
    return [rng.gauss(0.0, scale) for _ in range(n)]
def test_pack_bf16_round_nearest_even():
    one = 1.0
    ulp = 2.0**-7
    got = unpack_bf16(pack_bf16([one + ulp / 2, one + 1.5 * ulp, one + 0.75 * ulp, -(one + ulp / 2)]))
    assert got == [one, one + 2 * ulp, one + ulp, -one]
    (nan,) = unpack_bf16(pack_bf16([float("nan")]))
    assert math.isnan(nan)
def test_cells_header_in_sync():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cells.h"
        write_c_header(str(p))
        nums = lambda t: re.findall(r"[-+0-9.]+e[-+][0-9]+f", t)
        hdr = nums((ROOT / "include" / "glint_cells.h").read_text())
        assert len(hdr) == 256 and hdr == nums(p.read_text())
        assert [_f32(float(x[:-1])) for x in hdr] == [_f32(x) for row in CELLS for x in row]
def test_cuda_parent_table_in_sync():
    src = (ROOT / "cuda" / "glint_encode.cu").read_text()
    body = re.search(r"kParent\[16\] = \{(.*?)\};", src, re.S).group(1)
    kp = [_f32(float(x)) for x in re.findall(r"([-0-9.]+)f", body)]
    assert kp == [_f32(x) for x in PARENT]
def _write_model(root: Path, shards: dict, index: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard, tensors in shards.items():
        write_safetensors(str(root / shard), tensors)
        for t in tensors:
            weight_map[t[0]] = shard
    if index:
        (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (root / "config.json").write_text('{"model_type": "toy"}')
def _toy_shards():
    a = _gauss(16 * 64, 1)
    b = _gauss(8 * 63, 2) # odd numel, blocks straddle rows
    c = _gauss(4 * 128, 3)
    c[:64] = [0.0] * 64 # all-zero block
    return {
        "model-00001-of-00002.safetensors": [
            ("layers.0.q.weight", "BF16", (16, 64), pack_bf16(a)),
            ("layers.0.norm.weight", "BF16", (64,), pack_bf16([1.0] * 64)),
            ("layers.0.o.weight", "F32", (8, 63), struct.pack("<" + "f" * len(b), *b)),
        ],
        "model-00002-of-00002.safetensors": [
            ("layers.1.up.weight", "BF16", (4, 128), pack_bf16(c)),
            ("layers.1.up.bias", "BF16", (4,), pack_bf16([0.5] * 4)),
        ],
    }
def test_sharded_pin_verify_unpin():
    np = pytest.importorskip("numpy")
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir, back = Path(td) / "src", Path(td) / "pin", Path(td) / "back"
        _write_model(src, _toy_shards(), index=True)
        got = convert(src, pin_dir, allow_cpu=True)
        pin = got["pin"]
        assert pin["schema"] == "glint_pin_v1"
        assert pin["n_converted"] == 3
        assert pin["backend"] == "cpu_numpy_f32"
        assert pin["sidecar"] == ["config.json"]
        assert pin["rmse_max"] is not None and pin["rmse_max"] < 1e-3
        index = json.loads((pin_dir / "model.safetensors.index.json").read_text())
        assert index["weight_map"]["_glint_cells"] == "model-00001-of-00002.safetensors"
        assert index["weight_map"]["layers.1.up.weight.glint_plug"] == "model-00002-of-00002.safetensors"
        assert index["weight_map"]["layers.0.norm.weight"] == "model-00001-of-00002.safetensors"
        ev = verify(pin_dir, src, parity=True)
        assert ev["ok"], ev["failures"]
        assert ev["n_modules"] == 3
        assert all(m["parity"]["plug_bytes_differ"] == 0 for m in ev["modules"])
        res = unpin(pin_dir, back)
        assert res["n_expanded"] == 3
        assert (back / "config.json").is_file()
        with Pin(pin_dir) as p, SafeTensorsFile(str(back / "model-00001-of-00002.safetensors")) as st:
            info = st.tensors["layers.0.q.weight"]
            assert info.dtype == "BF16" and info.shape == (16, 64)
            assert "_glint_cells" not in st.tensors
            assert st.read_bytes("layers.0.norm.weight") == p.read("layers.0.norm.weight")
            rec = np.array(unpack_bf16(st.read_bytes("layers.0.q.weight")))
        src_w = np.array(unpack_bf16(_toy_shards()["model-00001-of-00002.safetensors"][0][3]))
        assert float(np.sqrt(np.mean((rec - src_w) ** 2))) < 5e-4
def test_numpy_encode_matches_python_encode():
    pytest.importorskip("numpy")
    import numpy as np
    import glint_np
    w = _gauss(64 * 41 + 7, 5)
    w[128:192] = [0.0] * 64
    hole, plug, am = glint_np.encode_f32(np.array(w, dtype=np.float32), 64)
    # Python reference on the same f32 inputs. Only near-ties could differ; seeded data has none.
    hi, pi, ai = encode([_f32(x) for x in w], 64)
    assert hole == bytes(pack_indices(hi))
    assert plug == bytes(pack_indices(pi))
    assert [float(x) for x in am] == [_f32(x) for x in ai]
def test_numpy_and_python_pins_agree():
    pytest.importorskip("numpy")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        _write_model(src, {"model.safetensors": [("lin.weight", "BF16", (8, 64), pack_bf16(_gauss(512, 9)))]}, index=False)
        a = convert(src, Path(td) / "np", allow_cpu=True)
        b = convert(src, Path(td) / "py", allow_cpu=True, force_python=True)
        assert a["pin"]["backend"] == "cpu_numpy_f32" and b["pin"]["backend"] == "cpu_python"
        with SafeTensorsFile(str(Path(td) / "np" / "model.safetensors")) as x, SafeTensorsFile(
            str(Path(td) / "py" / "model.safetensors")
        ) as y:
            for n in ("lin.weight", "lin.weight.glint_plug", "lin.weight.absmax"):
                assert x.read_bytes(n) == y.read_bytes(n), n
def test_verify_catches_corrupt_plug():
    pytest.importorskip("numpy")
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir = Path(td) / "src", Path(td) / "pin"
        _write_model(src, {"model.safetensors": [("lin.weight", "BF16", (8, 64), pack_bf16(_gauss(512, 4)))]}, index=False)
        convert(src, pin_dir, allow_cpu=True)
        f = pin_dir / "model.safetensors"
        with SafeTensorsFile(str(f)) as st:
            off = st.data_start + st.tensors["lin.weight.glint_plug"].data_offsets[0] + 3
        raw = bytearray(f.read_bytes())
        raw[off] ^= 0x11
        f.write_bytes(bytes(raw))
        ev = verify(pin_dir, src, parity=True)
        assert not ev["ok"]
        assert ev["modules"][0]["parity"]["plug_first_diff"] == 3
        assert ev["modules"][0]["parity"]["plug_bytes_differ"] == 1
def test_unpin_python_path_matches_numpy(monkeypatch):
    pytest.importorskip("numpy")
    import glint_unpin
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir = Path(td) / "src", Path(td) / "pin"
        _write_model(src, {"model.safetensors": [("lin.weight", "BF16", (4, 63), pack_bf16(_gauss(252, 6)))]}, index=False)
        convert(src, pin_dir, allow_cpu=True)
        for dt in ("BF16", "F32"):
            unpin(pin_dir, Path(td) / ("np" + dt), dt)
            monkeypatch.setattr(glint_unpin, "glint_np", None)
            unpin(pin_dir, Path(td) / ("py" + dt), dt)
            monkeypatch.undo()
            a = (Path(td) / ("np" + dt) / "model.safetensors").read_bytes()
            b = (Path(td) / ("py" + dt) / "model.safetensors").read_bytes()
            assert a == b, dt
def test_single_file_src_and_dry_run():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "w.safetensors"
        write_safetensors(str(f), [("a.weight", "BF16", (2, 64), pack_bf16(_gauss(128, 7))), ("a.norm.weight", "F32", (2,), b"\0" * 8)])
        plan = convert(f, Path(td) / "pin", dry_run=True)
        assert plan["linears"] == ["a.weight"] and plan["n_keep"] == 1
        got = convert(f, Path(td) / "pin", allow_cpu=True)
        assert got["pin"]["shards"] == ["model.safetensors"]
        assert not (Path(td) / "pin" / "model.safetensors.index.json").exists()
        assert verify(Path(td) / "pin")["ok"]
def test_sharded_pin_stdlib_path(monkeypatch):
    import glint_unpin
    monkeypatch.setattr(glint_unpin, "glint_np", None)
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir = Path(td) / "src", Path(td) / "pin"
        _write_model(src, _toy_shards(), index=True)
        got = convert(src, pin_dir, allow_cpu=True, force_python=True)
        assert got["pin"]["backend"] == "cpu_python" and got["pin"]["n_converted"] == 3
        assert verify(pin_dir)["ok"]
        res = unpin(pin_dir, Path(td) / "back")
        assert res["n_expanded"] == 3 and len(res["shards"]) == 2
        assert (Path(td) / "back" / "model.safetensors.index.json").is_file()
def test_numpy_nearest_matches_linear_scan_on_edges():
    """searchsorted nearest == nearest16() linear scan, on ties, midpoints, and ulp neighbours."""
    np = pytest.importorskip("numpy")
    import glint_np as G
    def scan(v, tab):
        best = np.zeros(v.shape, np.uint8)
        bd = np.abs(v - tab[:, 0])
        for k in range(1, 16):
            d = np.abs(v - tab[:, k])
            t = d < bd
            best[t] = k
            bd = np.where(t, d, bd)
        return best
    f, par = G.CELLS_FLAT, G.PARENT_F32
    mids = ((f[1:] + f[:-1]) / 2).astype(np.float32)
    pmid = ((par[1:] + par[:-1]) / 2).astype(np.float32)
    up = lambda a: np.nextafter(a, np.float32(2))
    dn = lambda a: np.nextafter(a, np.float32(-2))
    rng = np.random.default_rng(0)
    v = np.concatenate(
        [f, mids, pmid, par, up(mids), dn(mids), up(pmid), dn(pmid), up(f), dn(f),
         np.float32([1, -1, 0, -0.0, np.nan, np.inf, -np.inf]), rng.uniform(-1, 1, 50000).astype(np.float32)]
    ).astype(np.float32)
    h_ref = scan(v, np.broadcast_to(par, (v.size, 16)))
    p_ref = scan(v, G.CELLS_F32[h_ref])
    h = G._nearest(v, par, 0)
    p = G._nearest(v, G.CELLS_FLAT, h.astype(np.int64) * 16)
    assert (h == h_ref).all() and (p == p_ref).all()

@pytest.mark.gpu
def test_cuda_pin_byte_parity_and_expand():
    """The host-vs-CUDA gate. Runs only where libglint_encode.so loads and a GPU answers."""
    pytest.importorskip("numpy")
    import glint_cuda
    if not glint_cuda.gpu_ok():
        pytest.skip("no libglint_encode.so or no usable GPU")
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir = Path(td) / "src", Path(td) / "pin"
        _write_model(src, _toy_shards(), index=True)
        got = convert(src, pin_dir)
        assert "cuda_bf16" in got["pin"]["backends"]
        ev = verify(pin_dir, src, parity=True, cuda=True)
        assert ev["ok"], ev["failures"]
def test_safetensors_offsets_are_contiguous():
    """Spec: the byte buffer is fully indexed, no holes. Official readers reject padding."""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "m.safetensors"
        write_safetensors(str(f), [("a", "U8", (3,), b"abc"), ("b", "F32", (1,), b"\0" * 4), ("c", "U8", (1,), b"z")])
        with SafeTensorsFile(str(f)) as st:
            spans = sorted(i.data_offsets for i in st.tensors.values())
            assert spans[0][0] == 0
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
            assert st.data_start + spans[-1][1] == f.stat().st_size
            assert st.read_bytes("c") == b"z"
def test_pin_loads_in_official_safetensors_reader():
    pytest.importorskip("numpy")
    safetensors = pytest.importorskip("safetensors")
    with tempfile.TemporaryDirectory() as td:
        src, pin_dir = Path(td) / "src", Path(td) / "pin"
        _write_model(src, _toy_shards(), index=True)
        convert(src, pin_dir, allow_cpu=True)
        unpin(pin_dir, Path(td) / "back", "F32")
        for f in list(pin_dir.glob("*.safetensors")) + list((Path(td) / "back").glob("*.safetensors")):
            with safetensors.safe_open(str(f), "np") as h: # header checks: no holes, sizes match
                for k in h.keys():
                    if h.get_slice(k).get_dtype() != "BF16": # numpy has no bfloat16
                        h.get_tensor(k)

"""ctypes to libglint_encode.so — BF16 blob stays packed until the GPU."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple
_LIB = None
_ERR = None
def lib_path() -> Optional[Path]:
    env = os.environ.get("GLINT_ENCODE_SO")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    here = Path(__file__).resolve().parent
    for p in (here / "libglint_encode.so", here / "cuda" / "libglint_encode.so"):
        if p.is_file():
            return p
    return None
def load():
    global _LIB, _ERR
    if _LIB is not None:
        return _LIB
    p = lib_path()
    if p is None:
        _ERR = "libglint_encode.so not found — nvcc -arch=sm_86 cuda/glint_encode.cu (see scripts/build_glint_cuda.sh)"
        return None
    lib = ctypes.CDLL(str(p))
    lib.glint_encode_n_absmax.argtypes = [ctypes.c_int64, ctypes.c_int]
    lib.glint_encode_n_absmax.restype = ctypes.c_int
    lib.glint_encode_n_packed.argtypes = [ctypes.c_int64]
    lib.glint_encode_n_packed.restype = ctypes.c_int
    lib.glint_encode_bf16_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.glint_encode_bf16_host.restype = ctypes.c_int
    lib.glint_expand_packed_to_f32_host.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.glint_expand_packed_to_f32_host.restype = ctypes.c_int
    _LIB = lib
    return lib
def encode_bf16_cuda(raw_bf16: bytes, n: int, blocksize: int = 64) -> Tuple[bytes, bytes, bytes]:
    lib = load()
    if lib is None:
        raise RuntimeError(_ERR)
    n_am = lib.glint_encode_n_absmax(n, blocksize)
    n_pk = lib.glint_encode_n_packed(n)
    hole = (ctypes.c_uint8 * n_pk)()
    plug = (ctypes.c_uint8 * n_pk)()
    am = (ctypes.c_float * n_am)()
    buf = (ctypes.c_uint16 * n).from_buffer_copy(raw_bf16[: n * 2])
    rc = lib.glint_encode_bf16_host(ctypes.byref(buf), n, blocksize, hole, plug, am)
    if rc != 0:
        raise RuntimeError(f"glint_encode_bf16_host rc={rc}")
    return bytes(hole), bytes(plug), bytes(ctypes.string_at(ctypes.addressof(am), n_am * 4))
def expand_cuda(hole: bytes, plug: bytes, absmax_f32: bytes, n: int, blocksize: int = 64) -> bytes:
    """GPU expand (k_expand) → n little-endian f32. The L0 golden vs host decode."""
    lib = load()
    if lib is None:
        raise RuntimeError(_ERR)
    out = (ctypes.c_float * n)()
    h = ctypes.create_string_buffer(bytes(hole), len(hole))
    p = ctypes.create_string_buffer(bytes(plug), len(plug))
    a = ctypes.create_string_buffer(bytes(absmax_f32), len(absmax_f32))
    rc = lib.glint_expand_packed_to_f32_host(h, p, a, n, blocksize, out)
    if rc != 0:
        raise RuntimeError(f"glint_expand_packed_to_f32_host rc={rc}")
    return bytes(ctypes.string_at(ctypes.addressof(out), n * 4))
def gpu_ok() -> bool:
    """Library loads and a 2-weight encode runs. False on a box with no GPU."""
    try:
        encode_bf16_cuda(b"\x80\x3f\x00\x00", 2, 64)
        return True
    except (RuntimeError, OSError):
        return False

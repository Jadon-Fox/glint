#!/usr/bin/env python3
"""Read-only GLINT pin codebook-custody audit; never modifies pin/model files.

Unlike a tensor-name-only check this reads the actual F32 values and can compare
them bit-for-bit with the codebook represented by the C header.
"""
import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys

def fail(reason):
    print("PIN_CODEBOOK_STATUS=FAIL", reason)
    return 2

def scan_safetensors(path):
    with path.open("rb") as f:
        nraw=f.read(8)
        if len(nraw)!=8:
            raise ValueError(f"{path}: short header")
        n=struct.unpack("<Q",nraw)[0]
        if n > 16*1024*1024:
            raise ValueError(f"{path}: header unexpectedly large")
        raw=f.read(n)
        if len(raw)!=n:
            raise ValueError(f"{path}: truncated header")
        entries=json.loads(raw)
        offset=8+n
        tables=[]
        for key,v in entries.items():
            if not (key.endswith("_glint_cells") or key.endswith(".glint_cells")):
                continue
            loc=v.get("data_offsets",[-1,-1])
            if v.get("dtype")!="F32" or v.get("shape")!=[16,16] or loc[1]-loc[0]!=1024 or loc[0]<0:
                raise ValueError(f"invalid 16x16 F32 table at {path.name}:{key}")
            f.seek(offset+loc[0])
            data=f.read(1024)
            if len(data)!=1024:
                raise ValueError(f"truncated 16x16 F32 table at {path.name}:{key}")
            tables.append((key,data))
    return entries,tables

def parse_compiled_table(path):
    # Current generator creates 256 decimal float32 constants in include/glint_cells.h.
    src=path.read_text()
    nums=[float(x[:-1]) for x in re.findall(r"[-+]?\d+\.\d+e[-+]\d+f",src)]
    if len(nums)!=256:
        raise ValueError(f"header has {len(nums)} float literals, not 256")
    return struct.pack("<256f",*nums)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("pin_dir",type=pathlib.Path)
    ap.add_argument("--require-embedded",action="store_true")
    ap.add_argument("--legacy-header",type=pathlib.Path,
        help="optional independent compiled-table comparison")
    ap.add_argument("--expected-header-sha256",
        help="required if --legacy-header is supplied")
    a=ap.parse_args()
    try:
        meta=json.loads((a.pin_dir/"pin.json").read_text())
        schema=meta.get("schema")
        shards=meta.get("shards")
        if schema not in ("glint_pin_v1","glint_pin_v2") or not isinstance(shards,list) or not shards:
            return fail("invalid schema or shards")
        tables=[]
        states=0
        tensors=0
        for shard in shards:
            if pathlib.Path(shard).name!=shard:
                return fail("unsafe shard path")
            entries,part_tables=scan_safetensors(a.pin_dir/shard)
            tensors+=len(entries)-int("__metadata__" in entries)
            states+=sum(k.endswith(".glint_state") for k in entries)
            tables.extend((shard,k,v) for k,v in part_tables)
        print("PIN_CODEBOOK_SCHEMA",schema)
        print("PIN_CODEBOOK_TENSORS",tensors,"STATE_RECORDS",states)
        print("PIN_CODEBOOK_EMBEDDED_ENTRIES",len(tables))
        if a.require_embedded and len(tables)!=1:
            return fail("expected exactly one embedded 16x16 F32 table")
        if len(tables)>1:
            return fail("multiple cell tables; unresolved authority")
        raw=tables[0][2] if tables else None
        if raw is not None:
            print("PIN_CODEBOOK_TENSOR",tables[0][0],tables[0][1])
            print("PIN_CODEBOOK_EMBEDDED_F32_SHA256",hashlib.sha256(raw).hexdigest())
        if a.legacy_header is not None:
            if not a.expected_header_sha256:
                return fail("compiled header comparison needs a pinned expected SHA256")
            header=a.legacy_header.read_bytes()
            hdr_sha=hashlib.sha256(header).hexdigest()
            if hdr_sha.lower()!=a.expected_header_sha256.lower():
                return fail("legacy header SHA mismatch: "+hdr_sha)
            print("PIN_CODEBOOK_COMPILED_SOURCE_SHA256",hdr_sha)
            parsed=parse_compiled_table(a.legacy_header)
            print("PIN_CODEBOOK_COMPILED_F32_SHA256",hashlib.sha256(parsed).hexdigest())
            if raw is None:
                return fail("pin missing table: compiled-only fallback forbidden")
            if raw != parsed:
                return fail("embedded table differs from compiled table")
            print("PIN_CODEBOOK_BYTE_IDENTICAL_COMPILED",True)
        if raw is None:
            print("PIN_CODEBOOK_STATUS=EMBEDDED_TABLE_ABSENT")
        elif a.legacy_header is not None:
            print("PIN_CODEBOOK_STATUS=EMBEDDED_AND_COMPILED_MATCH")
        else:
            print("PIN_CODEBOOK_STATUS=EMBEDDED_TABLE_PRESENT_UNCOMPARED")
        return 0
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
        return fail(str(exc))

if __name__=="__main__":
    sys.exit(main())

"""Read a GLINT pin dir (single model.safetensors or sharded with an index)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from glint_codec import unpack_f32
from safetensors_io import SafeTensorsFile
INDEX_NAME = "model.safetensors.index.json"
STATE_SUFFIX = ".weight.glint_state"
class Pin:
    """All shards open at once; tensors looked up by name across shards."""
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        pj = self.root / "pin.json"
        if not pj.is_file():
            raise FileNotFoundError(f"no pin.json in {self.root}")
        self.meta = json.loads(pj.read_text())
        index = self.root / INDEX_NAME
        if index.is_file():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            names = ["model.safetensors"]
        self.shards: Dict[str, SafeTensorsFile] = {n: SafeTensorsFile(str(self.root / n)) for n in names}
        self.where: Dict[str, str] = {}
        for sname, st in self.shards.items():
            for t in st.tensors:
                self.where[t] = sname
    def close(self) -> None:
        for st in self.shards.values():
            st.close()
    def __enter__(self) -> "Pin":
        return self
    def __exit__(self, *exc) -> None:
        self.close()
    def info(self, name: str):
        return self.shards[self.where[name]].tensors[name]
    def read(self, name: str) -> bytes:
        return self.shards[self.where[name]].read_bytes(name)
    def names(self) -> List[str]:
        return list(self.where)
    def stems(self) -> List[str]:
        return [n[: -len(STATE_SUFFIX)] for n in self.where if n.endswith(STATE_SUFFIX)]
    def state(self, stem: str) -> dict:
        return json.loads(self.read(stem + STATE_SUFFIX).decode())
    def cells(self) -> Optional[List[float]]:
        return unpack_f32(self.read("_glint_cells")) if "_glint_cells" in self.where else None
    def module(self, stem: str) -> Tuple[dict, bytes, bytes, bytes]:
        """→ (state, hole bytes, plug bytes, absmax f32 bytes)."""
        w = stem + ".weight"
        return self.state(stem), self.read(w), self.read(w + ".glint_plug"), self.read(w + ".absmax")
    def module_names(self, stem: str) -> Tuple[str, ...]:
        w = stem + ".weight"
        return (w, w + ".glint_plug", w + ".absmax", w + ".glint_state")
    def passthrough(self) -> Iterator[str]:
        owned = set()
        for s in self.stems():
            owned.update(self.module_names(s))
        owned.add("_glint_cells")
        for n in self.where:
            if n not in owned:
                yield n

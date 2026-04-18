from __future__ import annotations

import json
from pathlib import Path

def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def make_safe_name(s: str) -> str:
    return (
        s.replace("|", "_")
         .replace("=", "-")
         .replace("/", "_")
         .replace(" ", "_")
    )
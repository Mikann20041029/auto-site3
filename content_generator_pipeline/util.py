from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")

def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def normalize_url(u: str) -> str:
    u = u.strip()
    u = re.sub(r"[?#].*$", "", u)
    u = u.rstrip("/")
    return u

def simple_tokens(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    parts = [p for p in s.split() if p and len(p) > 2]
    return set(parts)

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def sanitize_llm_html(s: str) -> str:
    # 念のため script を潰す（広告スクリプトはテンプレ側で挿入する）
    s = re.sub(r"(?is)<\s*script\b", "&lt;script", s)
    return s

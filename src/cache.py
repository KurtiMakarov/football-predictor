from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


CACHE_DIR = os.path.join(os.getcwd(), "data", "cache")


def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def get_cached(key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > ttl_seconds:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Corrupted cache file, delete and refetch
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def set_cached(key: str, payload: Dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

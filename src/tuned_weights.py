from __future__ import annotations

import json
import os
from typing import Dict, Any


def save_tuned_weights(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_tuned_weights(path: str, league_id: int) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("leagues", {}).get(str(league_id), {}).get("weights", {})

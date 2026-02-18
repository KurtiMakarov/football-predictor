from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationModel:
    edges: List[float]
    home: List[float]
    draw: List[float]
    away: List[float]


def _bin_index(edges: List[float], p: float) -> int:
    # edges are ascending length n+1
    idx = int(np.searchsorted(edges, p, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= len(edges) - 1:
        return len(edges) - 2
    return idx


def fit_calibration(
    preds: List[Tuple[float, float, float]],
    outcomes: List[str],
    bins: int = 10,
) -> CalibrationModel:
    edges = np.linspace(0.0, 1.0, bins + 1).tolist()

    # global rates for empty bins
    total = len(outcomes) if outcomes else 1
    global_home = outcomes.count("1") / total
    global_draw = outcomes.count("X") / total
    global_away = outcomes.count("2") / total

    def build_for_class(class_idx: int, label: str) -> List[float]:
        sums = [0] * bins
        counts = [0] * bins
        for (p1, px, p2), out in zip(preds, outcomes):
            p = [p1, px, p2][class_idx]
            b = _bin_index(edges, p)
            counts[b] += 1
            if out == label:
                sums[b] += 1
        calibrated = []
        for i in range(bins):
            if counts[i] == 0:
                if label == "1":
                    calibrated.append(global_home)
                elif label == "X":
                    calibrated.append(global_draw)
                else:
                    calibrated.append(global_away)
            else:
                calibrated.append(sums[i] / counts[i])
        return calibrated

    home = build_for_class(0, "1")
    draw = build_for_class(1, "X")
    away = build_for_class(2, "2")

    return CalibrationModel(edges=edges, home=home, draw=draw, away=away)


def apply_calibration(
    p1: float,
    px: float,
    p2: float,
    model: CalibrationModel,
) -> Tuple[float, float, float]:
    b1 = _bin_index(model.edges, p1)
    bx = _bin_index(model.edges, px)
    b2 = _bin_index(model.edges, p2)

    c1 = model.home[b1]
    cx = model.draw[bx]
    c2 = model.away[b2]

    s = c1 + cx + c2
    if s <= 0:
        return p1, px, p2
    return c1 / s, cx / s, c2 / s


def strength_bucket(strength_diff: float) -> str:
    if strength_diff >= 0.2:
        return "home_strong"
    if strength_diff <= -0.2:
        return "away_strong"
    return "even"


def load_calibration(
    path: str, league_id: int, condition: str | List[str] = "all"
) -> Optional[CalibrationModel]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    league_key = str(league_id)
    data = payload.get("leagues", {}).get(league_key, {})
    keys = [condition] if isinstance(condition, str) else list(condition)
    if "all" not in keys:
        keys.append("all")
    picked = None
    for key in keys:
        if key in data:
            picked = data.get(key)
            break
    data = picked
    if not data:
        return None
    return CalibrationModel(
        edges=data["edges"],
        home=data["home"],
        draw=data["draw"],
        away=data["away"],
    )


def save_calibration(path: str, league_models: Dict[int, Dict[str, CalibrationModel]]) -> None:
    payload = {"leagues": {}}
    for league_id, models in league_models.items():
        payload["leagues"][str(league_id)] = {}
        for condition, model in models.items():
            payload["leagues"][str(league_id)][condition] = {
                "edges": model.edges,
                "home": model.home,
                "draw": model.draw,
                "away": model.away,
            }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

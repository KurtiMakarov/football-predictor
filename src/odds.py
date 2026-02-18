from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import statistics


def _normalize(p1: float, px: float, p2: float) -> Tuple[float, float, float]:
    s = p1 + px + p2
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p1 / s, px / s, p2 / s


def _implied_prob(odd_str: str) -> Optional[float]:
    try:
        odd = float(odd_str)
        if odd <= 0:
            return None
        return 1.0 / odd
    except Exception:
        return None


def parse_odds_payload(payload: dict) -> Optional[Tuple[float, float, float]]:
    response = payload.get("response", [])
    if not response:
        return None

    # The API may return multiple items. We aggregate across bookmakers.
    p1_list: List[float] = []
    px_list: List[float] = []
    p2_list: List[float] = []

    for item in response:
        for book in item.get("bookmakers", []):
            for bet in book.get("bets", []):
                name = (bet.get("name") or "").lower()
                if name not in {"match winner", "1x2"}:
                    continue
                values = bet.get("values", [])
                vmap: Dict[str, float] = {}
                for v in values:
                    label = (v.get("value") or "").lower()
                    prob = _implied_prob(v.get("odd", ""))
                    if prob is None:
                        continue
                    if label in {"home", "1"}:
                        vmap["1"] = prob
                    elif label in {"draw", "x"}:
                        vmap["x"] = prob
                    elif label in {"away", "2"}:
                        vmap["2"] = prob
                if {"1", "x", "2"}.issubset(vmap.keys()):
                    p1, px, p2 = _normalize(vmap["1"], vmap["x"], vmap["2"])
                    p1_list.append(p1)
                    px_list.append(px)
                    p2_list.append(p2)

    if not p1_list or not px_list or not p2_list:
        return None

    # Outlier filtering across bookmakers
    def filter_outliers(values: List[float]) -> List[float]:
        if len(values) < 5:
            return values
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        if stdev == 0:
            return values
        filtered = [v for v in values if abs((v - mean) / stdev) <= 2.5]
        return filtered if len(filtered) >= 3 else values

    p1_list_f = filter_outliers(p1_list)
    px_list_f = filter_outliers(px_list)
    p2_list_f = filter_outliers(p2_list)

    p1 = sum(p1_list_f) / len(p1_list_f)
    px = sum(px_list_f) / len(px_list_f)
    p2 = sum(p2_list_f) / len(p2_list_f)

    return _normalize(p1, px, p2)

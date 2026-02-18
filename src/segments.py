from __future__ import annotations

from typing import List, Optional, Tuple


def odds_strength_segment(odds_probs: Optional[Tuple[float, float, float]]) -> str:
    if odds_probs is None:
        return "no_odds"
    fav = max(odds_probs)
    if fav >= 0.62:
        return "fav_strong"
    if fav >= 0.52:
        return "fav_medium"
    return "fav_weak"


def calibration_conditions(strength: str, table_seg: str, odds_seg: str) -> List[str]:
    return [
        f"{strength}|{table_seg}|{odds_seg}",
        f"{strength}|{table_seg}",
        f"{strength}|{odds_seg}",
        strength,
        table_seg,
        "all",
    ]

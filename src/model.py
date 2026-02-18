from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .features import TeamForm, TableInfo, InjuryInfo


@dataclass
class Prediction:
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: float
    lambda_away: float
    confidence: float
    odds_p1: float | None
    odds_px: float | None
    odds_p2: float | None

def compute_strength_diff(
    home_form: TeamForm,
    away_form: TeamForm,
    home_table: TableInfo | None,
    away_table: TableInfo | None,
    home_inj: InjuryInfo,
    away_inj: InjuryInfo,
    rest_days_home: float,
    rest_days_away: float,
    h2h_home_ppg: float,
    h2h_away_ppg: float,
    lineup_missing_home: float,
    lineup_missing_away: float,
    weights: Dict[str, float],
) -> float:
    ppg_diff = home_form.ppg - away_form.ppg
    gd_diff = home_form.gd_per_game - away_form.gd_per_game

    table_diff = 0.0
    if home_table and away_table and home_table.played and away_table.played:
        table_diff = (home_table.points / home_table.played) - (
            away_table.points / away_table.played
        )

    inj_diff = home_inj.missing - away_inj.missing
    rest_diff = rest_days_home - rest_days_away
    h2h_diff = h2h_home_ppg - h2h_away_ppg
    lineup_diff = lineup_missing_home - lineup_missing_away

    strength_diff = (
        weights.get("ppg", 0.25) * ppg_diff
        + weights.get("gd", 0.15) * gd_diff
        + weights.get("table_ppg", 0.35) * table_diff
        - weights.get("injuries", 0.02) * inj_diff
        + weights.get("rest_days", 0.04) * rest_diff
        + weights.get("h2h_ppg", 0.08) * h2h_diff
        - weights.get("lineup_missing", 0.04) * lineup_diff
    )
    return strength_diff


def _poisson_pmf(lmbda: float, k: int) -> float:
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def _score_matrix(lmbda_home: float, lmbda_away: float, max_goal: int) -> np.ndarray:
    mat = np.zeros((max_goal + 1, max_goal + 1))
    for i in range(max_goal + 1):
        for j in range(max_goal + 1):
            mat[i, j] = _poisson_pmf(lmbda_home, i) * _poisson_pmf(lmbda_away, j)
    return mat


def _normalize_probs(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    s = p_home + p_draw + p_away
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / s, p_draw / s, p_away / s


def _entropy(p1: float, px: float, p2: float) -> float:
    eps = 1e-9
    p1 = max(eps, p1)
    px = max(eps, px)
    p2 = max(eps, p2)
    return -float(p1 * math.log(p1) + px * math.log(px) + p2 * math.log(p2))


def compute_prediction(
    home_form: TeamForm,
    away_form: TeamForm,
    home_table: TableInfo | None,
    away_table: TableInfo | None,
    home_inj: InjuryInfo,
    away_inj: InjuryInfo,
    rest_days_home: float,
    rest_days_away: float,
    h2h_home_ppg: float,
    h2h_away_ppg: float,
    lineup_missing_home: int,
    lineup_missing_away: int,
    league_avg_home_goals: float,
    league_avg_away_goals: float,
    max_goal: int,
    weights: Dict[str, float],
    odds_probs: Tuple[float, float, float] | None,
    base_rates: Tuple[float, float, float] | None,
    calibration: "CalibrationModel | None",
    strength_diff: float | None,
) -> Prediction:
    # Attack/defense ratios
    home_attack = max(0.1, home_form.home_goals_for / max(0.1, league_avg_home_goals))
    home_defense = max(0.1, home_form.home_goals_against / max(0.1, league_avg_away_goals))
    away_attack = max(0.1, away_form.away_goals_for / max(0.1, league_avg_away_goals))
    away_defense = max(0.1, away_form.away_goals_against / max(0.1, league_avg_home_goals))

    lmbda_home = league_avg_home_goals * home_attack * away_defense
    lmbda_away = league_avg_away_goals * away_attack * home_defense

    if strength_diff is None:
        strength_diff = compute_strength_diff(
            home_form=home_form,
            away_form=away_form,
            home_table=home_table,
            away_table=away_table,
            home_inj=home_inj,
            away_inj=away_inj,
            rest_days_home=rest_days_home,
            rest_days_away=rest_days_away,
            h2h_home_ppg=h2h_home_ppg,
            h2h_away_ppg=h2h_away_ppg,
            lineup_missing_home=lineup_missing_home,
            lineup_missing_away=lineup_missing_away,
            weights=weights,
        )

    # Convert to a soft adjustment on lambdas
    adj = max(-0.35, min(0.35, strength_diff))
    lmbda_home *= math.exp(adj)
    lmbda_away *= math.exp(-adj)

    # Home advantage
    home_adv = float(weights.get("home_advantage", 0.08))
    lmbda_home *= math.exp(home_adv)

    mat = _score_matrix(lmbda_home, lmbda_away, max_goal)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)

    odds_p1 = odds_px = odds_p2 = None
    if odds_probs is not None:
        odds_p1, odds_px, odds_p2 = odds_probs
        base_w = max(0.0, min(0.9, float(weights.get("odds_prior", 0.25))))
        # Dynamic odds weight: higher when market is confident (low entropy)
        ent = _entropy(odds_p1, odds_px, odds_p2)
        ent_norm = ent / math.log(3.0)  # 0..1
        w = min(0.9, max(0.05, base_w + (0.5 - ent_norm) * 0.25))
        p_home = (1 - w) * p_home + w * odds_p1
        p_draw = (1 - w) * p_draw + w * odds_px
        p_away = (1 - w) * p_away + w * odds_p2
        p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)

    if base_rates is not None:
        br_home, br_draw, br_away = base_rates
        w = max(0.0, min(0.5, float(weights.get("calibration_blend", 0.15))))
        p_home = (1 - w) * p_home + w * br_home
        p_draw = (1 - w) * p_draw + w * br_draw
        p_away = (1 - w) * p_away + w * br_away
        p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)

    if calibration is not None:
        from .calibration import apply_calibration
        p_home, p_draw, p_away = apply_calibration(p_home, p_draw, p_away, calibration)

    confidence = float(max(p_home, p_draw, p_away))

    return Prediction(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        lambda_home=lmbda_home,
        lambda_away=lmbda_away,
        confidence=confidence,
        odds_p1=odds_p1,
        odds_px=odds_px,
        odds_p2=odds_p2,
    )

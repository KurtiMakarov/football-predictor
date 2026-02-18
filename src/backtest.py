from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple, Any

from .api_football import ApiFootballClient
from .calibration import CalibrationModel, fit_calibration, strength_bucket
from .config import AppConfig
from .features import (
    compute_h2h_ppg,
    compute_rest_days,
    compute_table_from_fixtures,
    compute_team_form,
    league_base_rates,
    league_goal_averages,
    table_segment,
)
from .model import compute_prediction, compute_strength_diff
from .segments import calibration_conditions


def _fixture_date(fx: dict) -> str:
    return fx.get("fixture", {}).get("date") or ""


def _fixture_status(fx: dict) -> str:
    return fx.get("fixture", {}).get("status", {}).get("short") or ""


def _fixture_outcome(fx: dict) -> str | None:
    goals = fx.get("goals", {})
    hg = goals.get("home")
    ag = goals.get("away")
    if hg is None or ag is None:
        return None
    if hg > ag:
        return "1"
    if hg == ag:
        return "X"
    return "2"


def _brier_and_logloss(preds: List[Tuple[float, float, float]], outs: List[str]) -> Tuple[float, float]:
    if not preds or not outs:
        return 999.0, 999.0
    eps = 1e-9
    brier_total = 0.0
    ll_total = 0.0
    for (p1, px, p2), out in zip(preds, outs):
        y1, yx, y2 = (1.0 if out == "1" else 0.0), (1.0 if out == "X" else 0.0), (1.0 if out == "2" else 0.0)
        brier_total += ((p1 - y1) ** 2 + (px - yx) ** 2 + (p2 - y2) ** 2) / 3.0
        p = p1 if out == "1" else (px if out == "X" else p2)
        ll_total += -math.log(max(eps, min(1 - eps, p)))
    n = len(preds)
    return brier_total / n, ll_total / n


def _acc(preds: List[Tuple[float, float, float]], outs: List[str]) -> float:
    if not preds or not outs:
        return 0.0
    hit = 0
    for (p1, px, p2), out in zip(preds, outs):
        pick = "1" if p1 >= px and p1 >= p2 else ("X" if px >= p2 else "2")
        if pick == out:
            hit += 1
    return hit / len(preds)


def backtest_league_season(
    cfg: AppConfig,
    client: ApiFootballClient,
    league_id: int,
    season: int,
    weights_override: Dict[str, float] | None = None,
) -> Tuple[Dict[str, List[Tuple[float, float, float]]], Dict[str, List[str]]]:
    fixtures_payload = client.fixtures(league=league_id, season=season)
    fixtures = fixtures_payload.get("response", [])
    fixtures = [fx for fx in fixtures if _fixture_status(fx) == "FT"]
    fixtures.sort(key=_fixture_date)

    preds: Dict[str, List[Tuple[float, float, float]]] = {"all": []}
    outs: Dict[str, List[str]] = {"all": []}

    weights = copy.deepcopy(cfg.weights.__dict__)
    if weights_override:
        weights.update(weights_override)

    for idx, fx in enumerate(fixtures):
        fixture = fx.get("fixture", {})
        teams = fx.get("teams", {})
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        if not home_id or not away_id:
            continue

        outcome = _fixture_outcome(fx)
        if outcome is None:
            continue

        past = fixtures[:idx]
        if not past:
            continue

        home_past = [
            p
            for p in past
            if p.get("teams", {}).get("home", {}).get("id") == home_id
            or p.get("teams", {}).get("away", {}).get("id") == home_id
        ]
        away_past = [
            p
            for p in past
            if p.get("teams", {}).get("home", {}).get("id") == away_id
            or p.get("teams", {}).get("away", {}).get("id") == away_id
        ]

        home_form = compute_team_form(home_past, home_id, cfg.history_matches, float(weights.get("form_decay", cfg.weights.form_decay)))
        away_form = compute_team_form(away_past, away_id, cfg.history_matches, float(weights.get("form_decay", cfg.weights.form_decay)))

        rest_home = compute_rest_days(home_past, home_id, fixture.get("date"))
        rest_away = compute_rest_days(away_past, away_id, fixture.get("date"))

        h2h_home_ppg, h2h_away_ppg = compute_h2h_ppg(
            past, home_id, away_id, cfg.history_matches, float(weights.get("form_decay", cfg.weights.form_decay))
        )

        table = compute_table_from_fixtures(past)

        league_avg_home, league_avg_away = league_goal_averages(past)
        base_rates = league_base_rates(past)

        dummy_inj = type("Inj", (), {"missing": 0.0})()
        strength_diff = compute_strength_diff(
            home_form=home_form,
            away_form=away_form,
            home_table=table.get(home_id),
            away_table=table.get(away_id),
            home_inj=dummy_inj,
            away_inj=dummy_inj,
            rest_days_home=rest_home,
            rest_days_away=rest_away,
            h2h_home_ppg=h2h_home_ppg,
            h2h_away_ppg=h2h_away_ppg,
            lineup_missing_home=0.0,
            lineup_missing_away=0.0,
            weights=weights,
        )

        pred = compute_prediction(
            home_form=home_form,
            away_form=away_form,
            home_table=table.get(home_id),
            away_table=table.get(away_id),
            home_inj=dummy_inj,
            away_inj=dummy_inj,
            rest_days_home=rest_home,
            rest_days_away=rest_away,
            h2h_home_ppg=h2h_home_ppg,
            h2h_away_ppg=h2h_away_ppg,
            lineup_missing_home=0.0,
            lineup_missing_away=0.0,
            league_avg_home_goals=league_avg_home,
            league_avg_away_goals=league_avg_away,
            max_goal=cfg.max_goal,
            weights=weights,
            odds_probs=None,
            base_rates=base_rates,
            calibration=None,
            strength_diff=strength_diff,
        )

        p = (pred.p_home, pred.p_draw, pred.p_away)
        preds["all"].append(p)
        outs["all"].append(outcome)

        s_seg = strength_bucket(strength_diff)
        t_seg = table_segment(table.get(home_id), table.get(away_id))
        for cond in calibration_conditions(s_seg, t_seg, "no_odds"):
            preds.setdefault(cond, []).append(p)
            outs.setdefault(cond, []).append(outcome)

    return preds, outs


def optimize_weights_for_league(
    cfg: AppConfig,
    client: ApiFootballClient,
    league_id: int,
    league_season: int,
    seasons_back: int = 3,
    fast: bool = False,
) -> Dict[str, Any]:
    base = cfg.weights.__dict__
    candidates: List[Dict[str, float]] = []
    for form_decay in [0.80, 0.85, 0.90]:
        for ppg in [base["ppg"] * 0.9, base["ppg"], base["ppg"] * 1.1]:
            for table_ppg in [base["table_ppg"] * 0.9, base["table_ppg"], base["table_ppg"] * 1.1]:
                for home_adv in [0.05, 0.08, 0.11]:
                    candidates.append(
                        {
                            "form_decay": form_decay,
                            "ppg": ppg,
                            "table_ppg": table_ppg,
                            "home_advantage": home_adv,
                        }
                    )
    # Keep search bounded for API usage and runtime.
    candidates = candidates[:6] if fast else candidates[:54]

    best = {"score": 999.0, "brier": 999.0, "logloss": 999.0, "acc": 0.0, "weights": {}}

    total = len(candidates)
    for idx, cand in enumerate(candidates, start=1):
        print(f"  candidate {idx}/{total}: {cand}")
        preds_all: List[Tuple[float, float, float]] = []
        outs_all: List[str] = []
        for i in range(1, seasons_back + 1):
            season = league_season - i
            preds, outs = backtest_league_season(cfg, client, league_id, season, weights_override=cand)
            preds_all.extend(preds.get("all", []))
            outs_all.extend(outs.get("all", []))

        brier, logloss = _brier_and_logloss(preds_all, outs_all)
        acc = _acc(preds_all, outs_all)
        score = brier * 0.7 + logloss * 0.3
        if score < best["score"]:
            best = {
                "score": score,
                "brier": brier,
                "logloss": logloss,
                "acc": acc,
                "weights": cand,
                "n": len(preds_all),
            }

    return best


def run_backtest(
    cfg: AppConfig,
    seasons_back: int = 3,
    league_ids: List[int] | None = None,
) -> Dict[int, Dict[str, CalibrationModel]]:
    client = ApiFootballClient(cfg)
    league_models: Dict[int, Dict[str, CalibrationModel]] = {}

    for league in cfg.leagues:
        if league_ids and league.id not in league_ids:
            continue
        preds_all: Dict[str, List[Tuple[float, float, float]]] = {}
        outs_all: Dict[str, List[str]] = {}

        for i in range(1, seasons_back + 1):
            season = league.season - i
            preds, outs = backtest_league_season(cfg, client, league.id, season)
            for k, v in preds.items():
                preds_all.setdefault(k, []).extend(v)
            for k, v in outs.items():
                outs_all.setdefault(k, []).extend(v)

        if preds_all:
            league_models[league.id] = {}
            for k in preds_all.keys():
                if len(preds_all.get(k, [])) >= 40 and outs_all.get(k):
                    model = fit_calibration(preds_all[k], outs_all[k], bins=10)
                    league_models[league.id][k] = model

    return league_models


def run_tuning(
    cfg: AppConfig,
    seasons_back: int = 3,
    league_ids: List[int] | None = None,
    fast: bool = False,
) -> Dict[str, Any]:
    client = ApiFootballClient(cfg)
    payload: Dict[str, Any] = {"leagues": {}, "errors": {}}

    for league in cfg.leagues:
        if league_ids and league.id not in league_ids:
            continue
        print(f"Tuning league: {league.name} ({league.id})")
        try:
            best = optimize_weights_for_league(
                cfg, client, league.id, league.season, seasons_back=seasons_back, fast=fast
            )
            payload["leagues"][str(league.id)] = {
                "name": league.name,
                "weights": best.get("weights", {}),
                "metrics": {
                    "brier": best.get("brier"),
                    "logloss": best.get("logloss"),
                    "accuracy": best.get("acc"),
                    "samples": best.get("n", 0),
                },
            }
            print(f"Completed: {league.name} | acc={best.get('acc', 0):.3f} brier={best.get('brier', 0):.4f}")
        except Exception as e:
            payload["errors"][str(league.id)] = {
                "name": league.name,
                "error": str(e),
            }
            print(f"Failed: {league.name} | {e}")
    return payload

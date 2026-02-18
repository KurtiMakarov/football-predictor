from __future__ import annotations

from typing import Dict, List, Tuple

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
)
from .model import compute_prediction, compute_strength_diff


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


def backtest_league_season(
    cfg: AppConfig, client: ApiFootballClient, league_id: int, season: int
) -> Tuple[Dict[str, List[Tuple[float, float, float]]], Dict[str, List[str]]]:
    fixtures_payload = client.fixtures(league=league_id, season=season)
    fixtures = fixtures_payload.get("response", [])
    fixtures = [fx for fx in fixtures if _fixture_status(fx) == "FT"]
    fixtures.sort(key=_fixture_date)

    preds: Dict[str, List[Tuple[float, float, float]]] = {"all": []}
    outs: Dict[str, List[str]] = {"all": []}

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

        # Past fixtures for team forms
        home_past = [p for p in past if p.get("teams", {}).get("home", {}).get("id") == home_id
                     or p.get("teams", {}).get("away", {}).get("id") == home_id]
        away_past = [p for p in past if p.get("teams", {}).get("home", {}).get("id") == away_id
                     or p.get("teams", {}).get("away", {}).get("id") == away_id]

        home_form = compute_team_form(home_past, home_id, cfg.history_matches, cfg.weights.form_decay)
        away_form = compute_team_form(away_past, away_id, cfg.history_matches, cfg.weights.form_decay)

        rest_home = compute_rest_days(home_past, home_id, fixture.get("date"))
        rest_away = compute_rest_days(away_past, away_id, fixture.get("date"))

        h2h_home_ppg, h2h_away_ppg = compute_h2h_ppg(
            past, home_id, away_id, cfg.history_matches, cfg.weights.form_decay
        )

        table = compute_table_from_fixtures(past)

        league_avg_home, league_avg_away = league_goal_averages(past)
        base_rates = league_base_rates(past)

        strength_diff = compute_strength_diff(
            home_form=home_form,
            away_form=away_form,
            home_table=table.get(home_id),
            away_table=table.get(away_id),
            home_inj=type("Inj", (), {"missing": 0})(),
            away_inj=type("Inj", (), {"missing": 0})(),
            rest_days_home=rest_home,
            rest_days_away=rest_away,
            h2h_home_ppg=h2h_home_ppg,
            h2h_away_ppg=h2h_away_ppg,
            lineup_missing_home=0.0,
            lineup_missing_away=0.0,
            weights=cfg.weights.__dict__,
        )

        pred = compute_prediction(
            home_form=home_form,
            away_form=away_form,
            home_table=table.get(home_id),
            away_table=table.get(away_id),
            home_inj=type("Inj", (), {"missing": 0})(),
            away_inj=type("Inj", (), {"missing": 0})(),
            rest_days_home=rest_home,
            rest_days_away=rest_away,
            h2h_home_ppg=h2h_home_ppg,
            h2h_away_ppg=h2h_away_ppg,
            lineup_missing_home=0.0,
            lineup_missing_away=0.0,
            league_avg_home_goals=league_avg_home,
            league_avg_away_goals=league_avg_away,
            max_goal=cfg.max_goal,
            weights=cfg.weights.__dict__,
            odds_probs=None,
            base_rates=base_rates,
            calibration=None,
            strength_diff=strength_diff,
        )

        preds["all"].append((pred.p_home, pred.p_draw, pred.p_away))
        outs["all"].append(outcome)

        bucket = strength_bucket(strength_diff)
        preds.setdefault(bucket, []).append((pred.p_home, pred.p_draw, pred.p_away))
        outs.setdefault(bucket, []).append(outcome)

    return preds, outs


def run_backtest(cfg: AppConfig, seasons_back: int = 3) -> Dict[int, Dict[str, CalibrationModel]]:
    client = ApiFootballClient(cfg)
    league_models: Dict[int, Dict[str, CalibrationModel]] = {}

    for league in cfg.leagues:
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
                if preds_all.get(k) and outs_all.get(k):
                    model = fit_calibration(preds_all[k], outs_all[k], bins=10)
                    league_models[league.id][k] = model

    return league_models

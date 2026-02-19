from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict

import pandas as pd
from dotenv import load_dotenv

from .api_football import ApiFootballClient
from .config import AppConfig, load_config
from .calibration import load_calibration, save_calibration, strength_bucket
from .backtest import run_backtest, run_tuning
from .features import (
    compute_injuries,
    compute_table,
    compute_team_form,
    compute_rest_days,
    compute_h2h_ppg,
    league_goal_averages,
    league_base_rates,
    extract_lineup_starting,
    table_segment,
)
from .model import compute_prediction, compute_strength_diff
from .odds import parse_odds_payload
from .segments import calibration_conditions, odds_strength_segment
from .tuned_weights import load_tuned_weights, save_tuned_weights


def _parse_date(date_str: str, tz: str) -> str:
    if date_str.lower() == "today":
        dt = datetime.now(ZoneInfo(tz))
    else:
        dt = datetime.fromisoformat(date_str)
    return dt.strftime("%Y-%m-%d")


def lookup_league(cfg: AppConfig, name: str | None, country: str | None) -> None:
    client = ApiFootballClient(cfg)
    payload = client.leagues(name=name, country=country)
    rows = []
    for item in payload.get("response", []):
        league = item.get("league", {})
        country_info = item.get("country", {})
        seasons = item.get("seasons", [])
        current = next((s for s in seasons if s.get("current")), None)
        rows.append(
            {
                "id": league.get("id"),
                "name": league.get("name"),
                "type": league.get("type"),
                "country": country_info.get("name"),
                "current_season": current.get("year") if current else None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No leagues found.")
        return
    print(df.to_string(index=False))


def predict(cfg: AppConfig, date_str: str) -> None:
    client = ApiFootballClient(cfg)
    date = _parse_date(date_str, cfg.timezone)

    all_rows = []

    for league in cfg.leagues:
        tuned = load_tuned_weights(cfg.tuning.path, league.id) if cfg.tuning.enabled else {}
        weights_for_league = dict(cfg.weights.__dict__)
        weights_for_league.update(tuned)

        fixtures_payload = client.fixtures(league=league.id, season=league.season, date=date)
        fixtures = fixtures_payload.get("response", [])
        if not fixtures:
            continue

        standings_payload = client.standings(league=league.id, season=league.season)
        table = compute_table(standings_payload)

        league_recent_payload = client.fixtures(
            league=league.id, season=league.season, status="FT", last=200
        )
        league_recent = league_recent_payload.get("response", [])
        league_avg_home, league_avg_away = league_goal_averages(league_recent)
        base_rates = league_base_rates(league_recent)

        calibration = load_calibration(cfg.calibration.path, league.id, condition="all") if cfg.calibration.enabled else None

        for fx in fixtures:
            fixture = fx.get("fixture", {})
            teams = fx.get("teams", {})
            if not teams:
                continue
            home_id = teams.get("home", {}).get("id")
            away_id = teams.get("away", {}).get("id")
            if not home_id or not away_id:
                continue

            home_recent_payload = client.fixtures(
                league=league.id,
                season=league.season,
                team=home_id,
                status="FT",
                last=cfg.history_matches,
            )
            away_recent_payload = client.fixtures(
                league=league.id,
                season=league.season,
                team=away_id,
                status="FT",
                last=cfg.history_matches,
            )

            home_recent = home_recent_payload.get("response", [])
            away_recent = away_recent_payload.get("response", [])

            home_form = compute_team_form(home_recent, home_id, cfg.history_matches, float(weights_for_league.get("form_decay", cfg.weights.form_decay)))
            away_form = compute_team_form(away_recent, away_id, cfg.history_matches, float(weights_for_league.get("form_decay", cfg.weights.form_decay)))

            home_inj_payload = client.injuries(league=league.id, season=league.season, team=home_id)
            away_inj_payload = client.injuries(league=league.id, season=league.season, team=away_id)

            home_inj = compute_injuries(home_inj_payload, home_id)
            away_inj = compute_injuries(away_inj_payload, away_id)

            match_date = fixture.get("date")
            rest_home = compute_rest_days(home_recent, home_id, match_date)
            rest_away = compute_rest_days(away_recent, away_id, match_date)
            h2h_home_ppg, h2h_away_ppg = compute_h2h_ppg(
                league_recent, home_id, away_id, cfg.history_matches, float(weights_for_league.get("form_decay", cfg.weights.form_decay))
            )

            odds_probs = None
            if cfg.odds.enabled:
                odds_payload = client.odds(
                    fixture=fixture.get("id"),
                    season=league.season,
                    league=league.id,
                )
                odds_probs = parse_odds_payload(odds_payload)

            lineup_missing_home = 0.0
            lineup_missing_away = 0.0
            lineup_payload = client.lineups(fixture.get("id"))
            home_lineup = extract_lineup_starting(lineup_payload, home_id)
            away_lineup = extract_lineup_starting(lineup_payload, away_id)
            if home_lineup:
                # Build key players from last N matches (starters >= 3)
                starter_counts: Dict[int, int] = {}
                starter_pos: Dict[int, str] = {}
                for fx2 in home_recent[:5]:
                    lp = client.lineups(fx2.get("fixture", {}).get("id"))
                    ids = extract_lineup_starting(lp, home_id)
                    for pid, pos in ids:
                        starter_counts[pid] = starter_counts.get(pid, 0) + 1
                        if pos:
                            starter_pos[pid] = pos
                key_players = {pid for pid, cnt in starter_counts.items() if cnt >= 3}
                missing = [pid for pid in key_players if pid not in {p for p, _ in home_lineup}]
                lineup_missing_home = 0.0
                pos_weights = cfg.weights.lineup_position_weights
                for pid in missing:
                    pos = starter_pos.get(pid, "")
                    lineup_missing_home += pos_weights.get(pos, pos_weights.get("UNK", 0.1))

            if away_lineup:
                starter_counts: Dict[int, int] = {}
                starter_pos: Dict[int, str] = {}
                for fx2 in away_recent[:5]:
                    lp = client.lineups(fx2.get("fixture", {}).get("id"))
                    ids = extract_lineup_starting(lp, away_id)
                    for pid, pos in ids:
                        starter_counts[pid] = starter_counts.get(pid, 0) + 1
                        if pos:
                            starter_pos[pid] = pos
                key_players = {pid for pid, cnt in starter_counts.items() if cnt >= 3}
                missing = [pid for pid in key_players if pid not in {p for p, _ in away_lineup}]
                lineup_missing_away = 0.0
                pos_weights = cfg.weights.lineup_position_weights
                for pid in missing:
                    pos = starter_pos.get(pid, "")
                    lineup_missing_away += pos_weights.get(pos, pos_weights.get("UNK", 0.1))

            strength_diff = compute_strength_diff(
                home_form=home_form,
                away_form=away_form,
                home_table=table.get(home_id),
                away_table=table.get(away_id),
                home_inj=home_inj,
                away_inj=away_inj,
                rest_days_home=rest_home,
                rest_days_away=rest_away,
                h2h_home_ppg=h2h_home_ppg,
                h2h_away_ppg=h2h_away_ppg,
                lineup_missing_home=lineup_missing_home,
                lineup_missing_away=lineup_missing_away,
                weights=weights_for_league,
            )

            odds_seg = odds_strength_segment(odds_probs)
            table_seg = table_segment(table.get(home_id), table.get(away_id))
            cconds = calibration_conditions(strength_bucket(strength_diff), table_seg, odds_seg)
            calibration = load_calibration(cfg.calibration.path, league.id, condition=cconds) if cfg.calibration.enabled else None

            pred = compute_prediction(
                home_form=home_form,
                away_form=away_form,
                home_table=table.get(home_id),
                away_table=table.get(away_id),
                home_inj=home_inj,
                away_inj=away_inj,
                rest_days_home=rest_home,
                rest_days_away=rest_away,
                h2h_home_ppg=h2h_home_ppg,
                h2h_away_ppg=h2h_away_ppg,
                lineup_missing_home=lineup_missing_home,
                lineup_missing_away=lineup_missing_away,
                league_avg_home_goals=league_avg_home,
                league_avg_away_goals=league_avg_away,
                max_goal=cfg.max_goal,
                weights=weights_for_league,
                odds_probs=odds_probs,
                base_rates=base_rates,
                calibration=calibration,
                strength_diff=strength_diff,
            )

            best = max(
                [("1", pred.p_home), ("X", pred.p_draw), ("2", pred.p_away)],
                key=lambda x: x[1],
            )

            pick = best[0] if pred.confidence >= cfg.prediction.min_confidence else "-"
            if (
                cfg.prediction.no_bet_on_market_disagreement
                and pred.odds_p1 is not None
                and pred.odds_px is not None
                and pred.odds_p2 is not None
            ):
                market_best = max(
                    [("1", pred.odds_p1), ("X", pred.odds_px), ("2", pred.odds_p2)],
                    key=lambda x: x[1],
                )
                model_map = {"1": pred.p_home, "X": pred.p_draw, "2": pred.p_away}
                model_for_market = model_map[market_best[0]]
                strong_market = market_best[1] >= max(0.67, cfg.prediction.market_favorite_threshold)
                severe_gap = (market_best[1] - model_for_market) >= 0.25
                if strong_market and severe_gap and pick != market_best[0]:
                    pick = "-"

            all_rows.append(
                {
                    "date": date,
                    "league": league.name,
                    "fixture_id": fixture.get("id"),
                    "home": teams.get("home", {}).get("name"),
                    "away": teams.get("away", {}).get("name"),
                    "p1": round(pred.p_home, 3),
                    "px": round(pred.p_draw, 3),
                    "p2": round(pred.p_away, 3),
                    "odds_p1": round(pred.odds_p1, 3) if pred.odds_p1 is not None else None,
                    "odds_px": round(pred.odds_px, 3) if pred.odds_px is not None else None,
                    "odds_p2": round(pred.odds_p2, 3) if pred.odds_p2 is not None else None,
                    "pick": pick,
                    "confidence": round(pred.confidence, 3),
                }
            )

    if not all_rows:
        print("No fixtures found for the given date.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["confidence"], ascending=False)
    print(df.to_string(index=False))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Football predictions 1/X/2")
    parser.add_argument("--config", default=None, help="Path to config.yaml")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lookup = sub.add_parser("lookup-league", help="Lookup league IDs")
    p_lookup.add_argument("--name", required=False)
    p_lookup.add_argument("--country", required=False)

    p_predict = sub.add_parser("predict", help="Predict fixtures for a date")
    p_predict.add_argument("--date", required=True, help="YYYY-MM-DD or 'today'")

    p_backtest = sub.add_parser("backtest", help="Run backtest and build calibration")
    p_backtest.add_argument("--seasons", type=int, default=3, help="Seasons back")
    p_backtest.add_argument("--league-ids", default="", help="Comma-separated league ids, e.g. 39,140")
    p_tune = sub.add_parser("tune", help="Optimize per-league weights")
    p_tune.add_argument("--seasons", type=int, default=3, help="Seasons back")
    p_tune.add_argument("--league-ids", default="", help="Comma-separated league ids, e.g. 39,140")
    p_tune.add_argument("--fast", action="store_true", help="Faster and less exhaustive tuning")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "lookup-league":
        lookup_league(cfg, args.name, args.country)
    elif args.cmd == "predict":
        predict(cfg, args.date)
    elif args.cmd == "backtest":
        league_ids = [int(x.strip()) for x in args.league_ids.split(",") if x.strip()] if args.league_ids else None
        models = run_backtest(cfg, seasons_back=args.seasons, league_ids=league_ids)
        save_calibration(cfg.calibration.path, models)
        print(f"Calibration saved to {cfg.calibration.path}")
    elif args.cmd == "tune":
        league_ids = [int(x.strip()) for x in args.league_ids.split(",") if x.strip()] if args.league_ids else None
        tuned = run_tuning(cfg, seasons_back=args.seasons, league_ids=league_ids, fast=args.fast)
        save_tuned_weights(cfg.tuning.path, tuned)
        print(f"Tuned weights saved to {cfg.tuning.path}")
        leagues_done = len(tuned.get("leagues", {}))
        errors = tuned.get("errors", {})
        print(f"Leagues tuned: {leagues_done}")
        if errors:
            print(f"Leagues with errors: {len(errors)}")
            for _, item in errors.items():
                print(f"- {item.get('name')}: {item.get('error')}")


if __name__ == "__main__":
    main()

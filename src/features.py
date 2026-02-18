from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig


@dataclass
class TeamForm:
    team_id: int
    ppg: float
    gd_per_game: float
    goals_for: float
    goals_against: float
    home_goals_for: float
    home_goals_against: float
    away_goals_for: float
    away_goals_against: float


@dataclass
class TableInfo:
    team_id: int
    points: int
    played: int
    position: int


@dataclass
class InjuryInfo:
    team_id: int
    missing: float


def _fixtures_to_df(fixtures: List[dict]) -> pd.DataFrame:
    columns = [
        "fixture_id",
        "date",
        "home_id",
        "away_id",
        "home_goals",
        "away_goals",
        "status",
    ]
    rows = []
    for f in fixtures:
        fixture = f.get("fixture", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        if not teams:
            continue
        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "home_id": teams.get("home", {}).get("id"),
                "away_id": teams.get("away", {}).get("id"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": fixture.get("status", {}).get("short"),
            }
        )
    return pd.DataFrame(rows, columns=columns)

def _parse_iso(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def compute_team_form(fixtures: List[dict], team_id: int, max_matches: int, decay: float) -> TeamForm:
    df = _fixtures_to_df(fixtures)
    df = df.dropna(subset=["home_goals", "away_goals"])  # finished matches only
    if df.empty:
        return TeamForm(team_id, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    team_rows = df[(df["home_id"] == team_id) | (df["away_id"] == team_id)].copy()
    team_rows = team_rows.tail(max_matches)

    weights = np.array([decay ** i for i in range(len(team_rows)-1, -1, -1)], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(team_rows)) / len(team_rows)

    points = []
    gd = []
    goals_for = []
    goals_against = []
    home_gf = []
    home_ga = []
    away_gf = []
    away_ga = []

    for _, row in team_rows.iterrows():
        is_home = row["home_id"] == team_id
        gf = row["home_goals"] if is_home else row["away_goals"]
        ga = row["away_goals"] if is_home else row["home_goals"]

        if gf > ga:
            p = 3
        elif gf == ga:
            p = 1
        else:
            p = 0

        points.append(p)
        gd.append(gf - ga)
        goals_for.append(gf)
        goals_against.append(ga)
        if is_home:
            home_gf.append(gf)
            home_ga.append(ga)
        else:
            away_gf.append(gf)
            away_ga.append(ga)

    def wavg(values: List[float]) -> float:
        if not values:
            return 1.0
        return float(np.average(values, weights=weights[: len(values)]))

    return TeamForm(
        team_id=team_id,
        ppg=wavg(points),
        gd_per_game=wavg(gd),
        goals_for=wavg(goals_for),
        goals_against=wavg(goals_against),
        home_goals_for=wavg(home_gf),
        home_goals_against=wavg(home_ga),
        away_goals_for=wavg(away_gf),
        away_goals_against=wavg(away_ga),
    )

def compute_rest_days(fixtures: List[dict], team_id: int, match_date: str | None) -> float:
    df = _fixtures_to_df(fixtures)
    if df.empty:
        return 7.0
    team_rows = df[(df["home_id"] == team_id) | (df["away_id"] == team_id)].copy()
    if team_rows.empty:
        return 7.0

    match_dt = _parse_iso(match_date)
    if match_dt is None:
        return 7.0

    team_rows["parsed_date"] = team_rows["date"].apply(_parse_iso)
    team_rows = team_rows.dropna(subset=["parsed_date"]).sort_values("parsed_date")
    if team_rows.empty:
        return 7.0
    last_dt = team_rows.iloc[-1]["parsed_date"]
    if last_dt is None:
        return 7.0
    days = (match_dt - last_dt).total_seconds() / 86400.0
    if days < 0:
        return 0.0
    return float(days)

def compute_h2h_ppg(
    league_fixtures: List[dict],
    home_id: int,
    away_id: int,
    max_matches: int,
    decay: float,
) -> Tuple[float, float]:
    df = _fixtures_to_df(league_fixtures)
    df = df.dropna(subset=["home_goals", "away_goals"])
    if df.empty:
        return 1.0, 1.0

    h2h = df[
        ((df["home_id"] == home_id) & (df["away_id"] == away_id))
        | ((df["home_id"] == away_id) & (df["away_id"] == home_id))
    ].copy()
    if h2h.empty:
        return 1.0, 1.0

    h2h = h2h.tail(max_matches)
    weights = np.array([decay ** i for i in range(len(h2h)-1, -1, -1)], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(h2h)) / len(h2h)

    home_points = []
    away_points = []
    for _, row in h2h.iterrows():
        home_is_home = row["home_id"] == home_id
        gf_home = row["home_goals"] if home_is_home else row["away_goals"]
        ga_home = row["away_goals"] if home_is_home else row["home_goals"]

        if gf_home > ga_home:
            p_home = 3
            p_away = 0
        elif gf_home == ga_home:
            p_home = 1
            p_away = 1
        else:
            p_home = 0
            p_away = 3
        home_points.append(p_home)
        away_points.append(p_away)

    def wavg(values: List[float]) -> float:
        return float(np.average(values, weights=weights[: len(values)]))

    return wavg(home_points), wavg(away_points)


def compute_table_from_fixtures(fixtures: List[dict]) -> Dict[int, TableInfo]:
    table: Dict[int, TableInfo] = {}
    stats: Dict[int, Dict[str, int]] = {}

    for fx in fixtures:
        teams = fx.get("teams", {})
        goals = fx.get("goals", {})
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        hg = goals.get("home")
        ag = goals.get("away")
        if home_id is None or away_id is None or hg is None or ag is None:
            continue

        for tid in (home_id, away_id):
            if tid not in stats:
                stats[tid] = {"points": 0, "played": 0, "gf": 0, "ga": 0}

        stats[home_id]["played"] += 1
        stats[away_id]["played"] += 1
        stats[home_id]["gf"] += hg
        stats[home_id]["ga"] += ag
        stats[away_id]["gf"] += ag
        stats[away_id]["ga"] += hg

        if hg > ag:
            stats[home_id]["points"] += 3
        elif hg < ag:
            stats[away_id]["points"] += 3
        else:
            stats[home_id]["points"] += 1
            stats[away_id]["points"] += 1

    # rank by points, goal diff, goals for
    ranking = sorted(
        stats.items(),
        key=lambda x: (x[1]["points"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True,
    )

    for pos, (tid, s) in enumerate(ranking, start=1):
        table[tid] = TableInfo(
            team_id=tid,
            points=s["points"],
            played=s["played"],
            position=pos,
        )

    return table


def compute_table(standings_payload: dict) -> Dict[int, TableInfo]:
    table: Dict[int, TableInfo] = {}
    leagues = standings_payload.get("response", [])
    if not leagues:
        return table
    standings = leagues[0].get("league", {}).get("standings", [])
    if not standings:
        return table
    for row in standings[0]:
        team = row.get("team", {})
        stats = row
        team_id = team.get("id")
        if not team_id:
            continue
        table[team_id] = TableInfo(
            team_id=team_id,
            points=int(stats.get("points", 0)),
            played=int(stats.get("played", 0)),
            position=int(stats.get("rank", 0)),
        )
    return table


def compute_injuries(injuries_payload: dict, team_id: int) -> InjuryInfo:
    injuries = injuries_payload.get("response", [])
    missing = 0.0
    for item in injuries:
        player = item.get("player", {})
        if not player.get("id"):
            continue
        # API payload fields vary by plan/provider; use robust fallback strings.
        kind = " ".join(
            str(x or "")
            for x in [
                item.get("type"),
                item.get("reason"),
                player.get("type"),
                player.get("reason"),
            ]
        ).lower()
        weight = 1.0
        if any(token in kind for token in ["doubt", "question", "minor", "knock"]):
            weight = 0.4
        elif any(token in kind for token in ["suspens", "ban"]):
            weight = 0.9
        elif any(token in kind for token in ["acl", "fracture", "surgery", "hamstring", "knee"]):
            weight = 1.2
        missing += weight
    return InjuryInfo(team_id=team_id, missing=missing)


def table_segment(home_table: TableInfo | None, away_table: TableInfo | None, table_size: int = 20) -> str:
    if not home_table or not away_table:
        return "unknown"
    top_cut = max(4, table_size // 4)
    low_cut = table_size - top_cut + 1
    home_pos = home_table.position
    away_pos = away_table.position
    if home_pos <= top_cut and away_pos <= top_cut:
        return "top_vs_top"
    if home_pos >= low_cut and away_pos >= low_cut:
        return "low_vs_low"
    if (home_pos <= top_cut and away_pos >= low_cut) or (away_pos <= top_cut and home_pos >= low_cut):
        return "top_vs_low"
    return "mid_mix"


def league_goal_averages(fixtures: List[dict]) -> Tuple[float, float]:
    df = _fixtures_to_df(fixtures)
    if df.empty:
        return 1.4, 1.2
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        return 1.4, 1.2
    df = df.dropna(subset=["home_goals", "away_goals"])
    if df.empty:
        return 1.4, 1.2
    return float(df["home_goals"].mean()), float(df["away_goals"].mean())


def league_base_rates(fixtures: List[dict]) -> Tuple[float, float, float]:
    df = _fixtures_to_df(fixtures)
    if df.empty:
        return 0.45, 0.28, 0.27
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        return 0.45, 0.28, 0.27
    df = df.dropna(subset=["home_goals", "away_goals"])
    if df.empty:
        return 0.45, 0.28, 0.27
    home_win = (df["home_goals"] > df["away_goals"]).mean()
    draw = (df["home_goals"] == df["away_goals"]).mean()
    away_win = (df["home_goals"] < df["away_goals"]).mean()
    return float(home_win), float(draw), float(away_win)


def extract_lineup_starting(lineup_payload: dict, team_id: int) -> List[Tuple[int, str]]:
    response = lineup_payload.get("response", [])
    if not response:
        return []
    for entry in response:
        team = entry.get("team", {})
        if team.get("id") != team_id:
            continue
        start_xi = entry.get("startXI", [])
        players: List[Tuple[int, str]] = []
        for item in start_xi:
            player = item.get("player", {})
            pid = player.get("id")
            pos = (player.get("pos") or "").upper()
            if pid:
                players.append((int(pid), pos))
        return players
    return []


def get_team_home_away_avg(fixtures: List[dict], team_id: int) -> Tuple[float, float, float, float]:
    df = _fixtures_to_df(fixtures)
    df = df.dropna(subset=["home_goals", "away_goals"])
    if df.empty:
        return 1.0, 1.0, 1.0, 1.0

    home = df[df["home_id"] == team_id]
    away = df[df["away_id"] == team_id]

    home_gf = float(home["home_goals"].mean()) if not home.empty else 1.0
    home_ga = float(home["away_goals"].mean()) if not home.empty else 1.0
    away_gf = float(away["away_goals"].mean()) if not away.empty else 1.0
    away_ga = float(away["home_goals"].mean()) if not away.empty else 1.0

    return home_gf, home_ga, away_gf, away_ga

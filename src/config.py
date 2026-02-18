from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class LeagueConfig:
    name: str
    id: int
    season: int


@dataclass
class ApiConfig:
    provider: str
    base_url: str
    api_key_env: str
    rapidapi_host: str | None


@dataclass
class CacheConfig:
    enabled: bool
    ttl_seconds: Dict[str, int]


@dataclass
class WeightsConfig:
    ppg: float
    gd: float
    table_ppg: float
    injuries: float
    form_decay: float
    odds_prior: float
    rest_days: float
    h2h_ppg: float
    lineup_missing: float
    calibration_blend: float
    home_advantage: float
    market_guard: float
    lineup_position_weights: Dict[str, float]


@dataclass
class PredictionConfig:
    min_confidence: float


@dataclass
class OddsConfig:
    enabled: bool


@dataclass
class CalibrationConfig:
    enabled: bool
    path: str


@dataclass
class AppConfig:
    api: ApiConfig
    timezone: str
    leagues: List[LeagueConfig]
    history_matches: int
    max_goal: int
    weights: WeightsConfig
    odds: OddsConfig
    calibration: CalibrationConfig
    prediction: PredictionConfig
    cache: CacheConfig


DEFAULT_CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


def load_config(path: str | None = None) -> AppConfig:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    api_raw = raw["api"]
    api = ApiConfig(
        provider=api_raw.get("provider", "apisports"),
        base_url=api_raw["base_url"].rstrip("/"),
        api_key_env=api_raw.get("api_key_env", "API_FOOTBALL_KEY"),
        rapidapi_host=api_raw.get("rapidapi_host"),
    )

    leagues = [
        LeagueConfig(name=l["name"], id=int(l["id"]), season=int(l["season"]))
        for l in raw["leagues"]
    ]

    weights_raw = raw.get("weights", {})
    weights = WeightsConfig(
        ppg=float(weights_raw.get("ppg", 0.25)),
        gd=float(weights_raw.get("gd", 0.15)),
        table_ppg=float(weights_raw.get("table_ppg", 0.35)),
        injuries=float(weights_raw.get("injuries", 0.02)),
        form_decay=float(weights_raw.get("form_decay", 0.85)),
        odds_prior=float(weights_raw.get("odds_prior", 0.25)),
        rest_days=float(weights_raw.get("rest_days", 0.04)),
        h2h_ppg=float(weights_raw.get("h2h_ppg", 0.08)),
        lineup_missing=float(weights_raw.get("lineup_missing", 0.04)),
        calibration_blend=float(weights_raw.get("calibration_blend", 0.15)),
        home_advantage=float(weights_raw.get("home_advantage", 0.08)),
        market_guard=float(weights_raw.get("market_guard", 0.70)),
        lineup_position_weights=dict(
            weights_raw.get(
                "lineup_position_weights",
                {"GK": 0.6, "DEF": 0.35, "MID": 0.25, "FWD": 0.3, "UNK": 0.15},
            )
        ),
    )

    odds_raw = raw.get("odds", {})
    odds = OddsConfig(
        enabled=bool(odds_raw.get("enabled", True)),
    )

    calib_raw = raw.get("calibration", {})
    calibration = CalibrationConfig(
        enabled=bool(calib_raw.get("enabled", True)),
        path=str(calib_raw.get("path", os.path.join(os.getcwd(), "data", "calibration.json"))),
    )

    pred_raw = raw.get("prediction", {})
    prediction = PredictionConfig(
        min_confidence=float(pred_raw.get("min_confidence", 0.43)),
    )

    cache_raw = raw.get("cache", {})
    cache = CacheConfig(
        enabled=bool(cache_raw.get("enabled", True)),
        ttl_seconds=dict(cache_raw.get("ttl_seconds", {})),
    )

    return AppConfig(
        api=api,
        timezone=raw.get("timezone", "UTC"),
        leagues=leagues,
        history_matches=int(raw.get("history_matches", 10)),
        max_goal=int(raw.get("max_goal", 6)),
        weights=weights,
        odds=odds,
        calibration=calibration,
        prediction=prediction,
        cache=cache,
    )


def get_api_key(cfg: AppConfig) -> str:
    key = os.getenv(cfg.api.api_key_env)
    if not key:
        raise RuntimeError(
            f"API key not found in env var {cfg.api.api_key_env}."
        )
    return key

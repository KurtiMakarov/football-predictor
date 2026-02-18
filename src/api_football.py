from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from .cache import get_cached, set_cached
from .config import AppConfig, get_api_key


class ApiFootballClient:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.api.base_url.rstrip("/")
        self.api_key = get_api_key(cfg)

    def _headers(self) -> Dict[str, str]:
        if self.cfg.api.provider == "rapidapi":
            if not self.cfg.api.rapidapi_host:
                raise RuntimeError("rapidapi_host is required for RapidAPI provider")
            return {
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.cfg.api.rapidapi_host,
            }
        return {"x-apisports-key": self.api_key}

    def _request(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        ttl_seconds: int = 0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        key_payload = {
            "url": url,
            "params": params or {},
        }
        if cache_key and self.cfg.cache.enabled:
            cached = get_cached(cache_key, ttl_seconds)
            if cached is not None:
                return cached

        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"API request failed ({resp.status_code}) for {url} params={params}. "
                f"Response: {resp.text[:500]}"
            ) from e
        payload = resp.json()

        if cache_key and self.cfg.cache.enabled:
            set_cached(cache_key, payload)

        return payload

    def leagues(self, name: str | None = None, country: str | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if name:
            params["name"] = name
        if country:
            params["country"] = country
        return self._request(
            "leagues",
            params=params,
            cache_key=f"leagues:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("leagues", 86400),
        )

    def fixtures(
        self,
        league: int,
        season: int,
        date: str | None = None,
        team: int | None = None,
        status: str | None = None,
        last: int | None = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"league": league, "season": season}
        if date:
            params["date"] = date
        if team:
            params["team"] = team
        if status:
            params["status"] = status
        if last:
            params["last"] = last
        return self._request(
            "fixtures",
            params=params,
            cache_key=f"fixtures:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("fixtures", 3600),
        )

    def standings(self, league: int, season: int) -> Dict[str, Any]:
        params = {"league": league, "season": season}
        return self._request(
            "standings",
            params=params,
            cache_key=f"standings:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("standings", 21600),
        )

    def injuries(self, league: int, season: int, team: int) -> Dict[str, Any]:
        params = {"league": league, "season": season, "team": team}
        return self._request(
            "injuries",
            params=params,
            cache_key=f"injuries:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("injuries", 3600),
        )

    def lineups(self, fixture: int) -> Dict[str, Any]:
        params = {"fixture": fixture}
        return self._request(
            "fixtures/lineups",
            params=params,
            cache_key=f"lineups:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("lineups", 1800),
        )

    def odds(self, fixture: int, season: int, league: int) -> Dict[str, Any]:
        params = {"fixture": fixture, "season": season, "league": league}
        return self._request(
            "odds",
            params=params,
            cache_key=f"odds:{json.dumps(params, sort_keys=True)}",
            ttl_seconds=self.cfg.cache.ttl_seconds.get("odds", 1800),
        )

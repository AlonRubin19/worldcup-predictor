"""API-Football Pro client with file-based caching.

API key must come from environment variable API_FOOTBALL_KEY.
Never hardcode the API key.

Cache layout: data/api_cache/{sha256_of_request}.json
Cache file format: {"expires_at": <unix_timestamp>, "data": <response_dict>}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

# ── Cache TTL constants ───────────────────────────────────────────────────────
CACHE_TTL_FIXTURES: int = 6 * 3600    # 6 hours
CACHE_TTL_LINEUPS:  int = 5 * 60      # 5 minutes
CACHE_TTL_INJURIES: int = 30 * 60     # 30 minutes
CACHE_TTL_LIVE:     int = 60          # 1 minute (future use)

_DEFAULT_TTL: int = CACHE_TTL_FIXTURES
_BASE_URL = "https://v3.football.api-sports.io"

# Sentinel: distinguishes "caller didn't pass api_key" from "caller passed None"
_UNSET = object()


# ── Exceptions ────────────────────────────────────────────────────────────────

class ApiKeyMissingError(Exception):
    """Raised when no API key is available and a live request is attempted."""


# ── Default HTTP fetcher (lazily imports requests) ────────────────────────────

def _default_fetcher(url: str, headers: dict, params: dict) -> dict:
    import requests  # noqa: PLC0415 — lazy import to avoid hard dependency at module level
    response = requests.get(url, headers=headers, params=params or {}, timeout=15)
    response.raise_for_status()
    return response.json()


# ── Client ────────────────────────────────────────────────────────────────────

class ApiFootballClient:
    """Thin wrapper around the API-Football v3 REST API with file-based caching.

    Args:
        api_key:   API-Football Pro key. Falls back to env var API_FOOTBALL_KEY.
        cache_dir: Directory for cached JSON responses.
        _fetcher:  Optional injectable HTTP callable for testing.
                   Signature: (url: str, headers: dict, params: dict) -> dict
    """

    def __init__(
        self,
        api_key: str | None = _UNSET,  # type: ignore[assignment]
        cache_dir: str | Path = "data/api_cache",
        _fetcher: Callable | None = None,
    ) -> None:
        if api_key is _UNSET:
            # No explicit key provided — use environment variable
            self._api_key = os.environ.get("API_FOOTBALL_KEY", "")
        else:
            # Explicit None or "" means "no key" — do not fall back to env var
            self._api_key = api_key or ""
        self._cache_dir = Path(cache_dir)
        self._fetcher = _fetcher or _default_fetcher

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        endpoint: str,
        params: dict | None = None,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> dict:
        """Fetch data from API-Football, using the file cache when valid.

        Args:
            endpoint:    API path, e.g. "/fixtures" or "/fixtures/lineups".
            params:      Query parameters dict.
            ttl_seconds: Cache time-to-live in seconds. 0 = always refetch.

        Returns:
            Parsed JSON response dict.

        Raises:
            ApiKeyMissingError: If no API key is configured.
        """
        if not self._api_key:
            raise ApiKeyMissingError(
                "API_FOOTBALL_KEY environment variable is not set. "
                "Live data is unavailable."
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = self._cache_key(endpoint, params or {})
        cached = self._get_cached(cache_key, ttl_seconds)
        if cached is not None:
            return cached

        data = self._fetcher(
            url=f"{_BASE_URL}{endpoint}",
            headers={
                "x-rapidapi-key": self._api_key,
                "x-rapidapi-host": "v3.football.api-sports.io",
            },
            params=params or {},
        )
        self._store_cache(cache_key, data, ttl_seconds)
        return data

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Stable SHA-256 key for (endpoint, sorted params)."""
        payload = endpoint + json.dumps(params, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _get_cached(self, key: str, ttl_seconds: int) -> dict | None:
        """Return cached data if it exists and has not expired."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                wrapper = json.load(f)
            if ttl_seconds <= 0:
                return None
            if time.time() < wrapper.get("expires_at", 0):
                return wrapper["data"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return None

    def _store_cache(self, key: str, data: dict, ttl_seconds: int) -> None:
        """Write data to the cache with an expiry timestamp."""
        path = self._cache_path(key)
        wrapper = {
            "expires_at": time.time() + max(ttl_seconds, 0),
            "data": data,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(wrapper, f)
        except OSError:
            pass  # cache write failure is non-fatal

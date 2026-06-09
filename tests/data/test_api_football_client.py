"""Tests for the API-Football Pro client.

TDD: all tests written RED-first before any production code exists.

No live API calls — all HTTP is injected via a mock _fetcher.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.data.api_football_client import (
    ApiFootballClient,
    ApiKeyMissingError,
    CACHE_TTL_FIXTURES,
    CACHE_TTL_LINEUPS,
    CACHE_TTL_INJURIES,
    CACHE_TTL_LIVE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_RESPONSE = {"response": [{"fixture": {"id": 1}}], "results": 1}


def _make_fetcher(data: dict):
    """Return a _fetcher callable that always returns the given dict."""
    def _fetcher(url: str, headers: dict, params: dict) -> dict:
        return data
    return _fetcher


def _client(tmp_path: Path, api_key: str = "test_key", data: dict | None = None) -> ApiFootballClient:
    return ApiFootballClient(
        api_key=api_key,
        cache_dir=tmp_path / "api_cache",
        _fetcher=_make_fetcher(data or _SAMPLE_RESPONSE),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_cache_ttl_fixtures_is_6_hours(self):
        assert CACHE_TTL_FIXTURES == 6 * 3600

    def test_cache_ttl_lineups_is_5_minutes(self):
        assert CACHE_TTL_LINEUPS == 5 * 60

    def test_cache_ttl_injuries_is_30_minutes(self):
        assert CACHE_TTL_INJURIES == 30 * 60

    def test_cache_ttl_live_is_1_minute(self):
        assert CACHE_TTL_LIVE == 60


# ─────────────────────────────────────────────────────────────────────────────
# API key
# ─────────────────────────────────────────────────────────────────────────────

class TestApiKey:
    def test_missing_key_raises_api_key_missing_error(self, tmp_path):
        client = ApiFootballClient(
            api_key="",
            cache_dir=tmp_path / "cache",
            _fetcher=_make_fetcher(_SAMPLE_RESPONSE),
        )
        with pytest.raises(ApiKeyMissingError):
            client.get("/fixtures", {})

    def test_none_key_raises_api_key_missing_error(self, tmp_path):
        client = ApiFootballClient(
            api_key=None,
            cache_dir=tmp_path / "cache",
            _fetcher=_make_fetcher(_SAMPLE_RESPONSE),
        )
        with pytest.raises(ApiKeyMissingError):
            client.get("/fixtures", {})

    def test_valid_key_does_not_raise(self, tmp_path):
        client = _client(tmp_path)
        result = client.get("/fixtures", {})
        assert isinstance(result, dict)

    def test_api_key_missing_error_is_exception_subclass(self):
        assert issubclass(ApiKeyMissingError, Exception)


# ─────────────────────────────────────────────────────────────────────────────
# Basic get
# ─────────────────────────────────────────────────────────────────────────────

class TestGet:
    def test_get_returns_dict(self, tmp_path):
        client = _client(tmp_path)
        result = client.get("/fixtures", {"league": "1", "season": "2026"})
        assert isinstance(result, dict)

    def test_get_returns_fetcher_data(self, tmp_path):
        data = {"response": [{"id": 42}], "results": 1}
        client = _client(tmp_path, data=data)
        result = client.get("/fixtures", {})
        assert result == data

    def test_get_with_no_params(self, tmp_path):
        client = _client(tmp_path)
        result = client.get("/status", None)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_dir_created_if_missing(self, tmp_path):
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=cache_dir,
            _fetcher=_make_fetcher(_SAMPLE_RESPONSE),
        )
        client.get("/fixtures", {})
        assert cache_dir.exists()

    def test_second_get_uses_cache_not_fetcher(self, tmp_path):
        """Second call should hit the cache and NOT call the fetcher again."""
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return _SAMPLE_RESPONSE

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        client.get("/fixtures", {"league": "1"}, ttl_seconds=3600)
        client.get("/fixtures", {"league": "1"}, ttl_seconds=3600)
        assert call_count["n"] == 1, "Expected exactly 1 real fetch; second should use cache"

    def test_different_params_produce_different_cache_entries(self, tmp_path):
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return {"response": [], "call": call_count["n"]}

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        client.get("/fixtures", {"league": "1"}, ttl_seconds=3600)
        client.get("/fixtures", {"league": "2"}, ttl_seconds=3600)
        assert call_count["n"] == 2

    def test_expired_cache_refetches(self, tmp_path):
        """A cache entry with ttl=0 should always be considered expired."""
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return _SAMPLE_RESPONSE

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        # ttl=0 means expires immediately
        client.get("/fixtures", {"league": "1"}, ttl_seconds=0)
        client.get("/fixtures", {"league": "1"}, ttl_seconds=0)
        assert call_count["n"] == 2, "Expired cache should re-fetch"

    def test_cache_file_written_as_json(self, tmp_path):
        cache_dir = tmp_path / "cache"
        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=cache_dir,
            _fetcher=_make_fetcher(_SAMPLE_RESPONSE),
        )
        client.get("/fixtures", {"league": "1"})
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            cached = json.load(f)
        assert "data" in cached
        assert "expires_at" in cached

    def test_cached_data_matches_original(self, tmp_path):
        data = {"response": [{"id": 99}], "results": 1}
        cache_dir = tmp_path / "cache"
        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=cache_dir,
            _fetcher=_make_fetcher(data),
        )
        client.get("/fixtures", {"league": "1"}, ttl_seconds=3600)
        # Second call hits cache
        result = client.get("/fixtures", {"league": "1"}, ttl_seconds=3600)
        assert result == data


# ─────────────────────────────────────────────────────────────────────────────
# TTL default
# ─────────────────────────────────────────────────────────────────────────────

class TestTtlDefault:
    def test_get_without_ttl_uses_cache(self, tmp_path):
        """get() with no ttl_seconds argument should use a sensible default (not 0)."""
        call_count = {"n": 0}

        def _counting_fetcher(url, headers, params):
            call_count["n"] += 1
            return _SAMPLE_RESPONSE

        client = ApiFootballClient(
            api_key="test_key",
            cache_dir=tmp_path / "cache",
            _fetcher=_counting_fetcher,
        )
        client.get("/fixtures", {"league": "1"})
        client.get("/fixtures", {"league": "1"})
        assert call_count["n"] == 1, "Default TTL should cache the first call"

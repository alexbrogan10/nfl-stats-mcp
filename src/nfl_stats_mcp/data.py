"""Loading and caching nflverse datasets.

Everything the MCP tools need goes through the `DataSource` protocol, so tests
can swap in small in-memory frames instead of downloading real data.
"""

from __future__ import annotations

import threading
from typing import Callable, Protocol

import polars as pl

FIRST_SEASON = 1999

# Play-by-play has ~370 columns; we keep only what the tools use to save memory.
PBP_COLUMNS = [
    "play_id",
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "qtr",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
    "play_type",
    "pass",
    "rush",
    "epa",
    "success",
    "yards_gained",
    "shotgun",
    "no_huddle",
    "qb_scramble",
    "air_yards",
    "cpoe",
    "touchdown",
    "interception",
    "sack",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
]


class DataSource(Protocol):
    def player_stats(self, seasons: list[int]) -> pl.DataFrame: ...
    def pbp(self, seasons: list[int]) -> pl.DataFrame: ...
    def schedules(self, seasons: list[int]) -> pl.DataFrame: ...
    def current_season(self) -> int: ...


class NflverseSource:
    """Downloads nflverse data on first use and caches it per season in memory."""

    def __init__(self) -> None:
        _use_system_certificates()
        import nflreadpy

        self._nfl = nflreadpy
        self._cache: dict[tuple[str, int], pl.DataFrame] = {}
        self._lock = threading.Lock()

    def current_season(self) -> int:
        return int(self._nfl.get_current_season())

    def player_stats(self, seasons: list[int]) -> pl.DataFrame:
        return self._load("player_stats", seasons, lambda s: self._nfl.load_player_stats([s]))

    def pbp(self, seasons: list[int]) -> pl.DataFrame:
        return self._load("pbp", seasons, lambda s: self._nfl.load_pbp([s]).select(PBP_COLUMNS))

    def schedules(self, seasons: list[int]) -> pl.DataFrame:
        return self._load("schedules", seasons, lambda s: self._nfl.load_schedules([s]))

    def _load(
        self, dataset: str, seasons: list[int], loader: Callable[[int], pl.DataFrame]
    ) -> pl.DataFrame:
        frames = []
        for season in sorted(set(seasons)):
            key = (dataset, season)
            with self._lock:
                if key not in self._cache:
                    self._cache[key] = loader(season)
                frames.append(self._cache[key])
        return pl.concat(frames, how="diagonal_relaxed")


def _use_system_certificates() -> None:
    """Trust the OS certificate store, so downloads work behind antivirus/corporate TLS proxies."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

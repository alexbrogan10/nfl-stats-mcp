"""Small hand-built nflverse-shaped frames so tests run offline and fast."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_stats_mcp.server import create_server


def _stat_row(pid, name, pos, team, week, season=2025, season_type="REG", **stats):
    base = {
        "player_id": pid,
        "player_name": name.split()[0][0] + "." + name.split()[-1],
        "player_display_name": name,
        "position": pos,
        "position_group": pos,
        "headshot_url": None,
        "season": season,
        "week": week,
        "season_type": season_type,
        "game_id": f"{season}_{week:02d}_{team}",
        "team": team,
        "opponent_team": "OPP",
        "completions": 0,
        "attempts": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "passing_interceptions": 0,
        "sacks_suffered": 0,
        "passing_epa": 0.0,
        "passing_cpoe": None,
        "carries": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "rushing_epa": 0.0,
        "receptions": 0,
        "targets": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "receiving_epa": 0.0,
        "def_sacks": 0.0,
        "fg_made": 0,
        "fg_att": 0,
        "fg_long": None,
        "fantasy_points_ppr": 0.0,
    }
    base.update(stats)
    return base


@pytest.fixture
def player_stats() -> pl.DataFrame:
    rows = [
        # Two-week QB with a playoff game.
        _stat_row("QB1", "Josh Allen", "QB", "BUF", 1, completions=20, attempts=30, passing_yards=300,
                  passing_tds=3, sacks_suffered=2, passing_epa=12.0, carries=5, rushing_yards=40,
                  fantasy_points_ppr=28.0),
        _stat_row("QB1", "Josh Allen", "QB", "BUF", 2, completions=25, attempts=40, passing_yards=280,
                  passing_tds=1, passing_interceptions=2, sacks_suffered=0, passing_epa=-2.0,
                  fantasy_points_ppr=14.0),
        _stat_row("QB1", "Josh Allen", "QB", "BUF", 19, season_type="POST", completions=10,
                  attempts=20, passing_yards=150, passing_epa=1.0),
        # Low-volume QB with a great rate: must fail the qualifier.
        _stat_row("QB2", "Backup Guy", "QB", "NYJ", 2, completions=5, attempts=5, passing_yards=90,
                  passing_epa=6.0),
        # Different player sharing a last name.
        _stat_row("DE1", "Josh Hines-Allen", "DE", "JAX", 1, def_sacks=2.0),
        _stat_row("WR1", "Ja'Marr Chase", "WR", "CIN", 1, receptions=8, targets=10, receiving_yards=120,
                  receiving_tds=1, fantasy_points_ppr=26.0),
        _stat_row("WR1", "Ja'Marr Chase", "WR", "CIN", 2, receptions=4, targets=9, receiving_yards=40,
                  fantasy_points_ppr=8.0),
        _stat_row("WR2", "Puka Nacua", "WR", "LA", 1, receptions=10, targets=12, receiving_yards=150,
                  fantasy_points_ppr=25.0),
        _stat_row("K1", "Kicker Person", "K", "KC", 1, fg_made=3, fg_att=4, fg_long=55),
        _stat_row("K1", "Kicker Person", "K", "KC", 2, fg_made=2, fg_att=2, fg_long=48),
    ]  # fmt: skip
    return pl.DataFrame(rows)


def _play(**kw):
    base = {
        "play_id": 0, "game_id": "2025_01_BUF", "season": 2025, "week": 1, "season_type": "REG",
        "posteam": "BUF", "defteam": "NYJ", "qtr": 1.0, "down": 1.0, "ydstogo": 10.0,
        "yardline_100": 75.0, "half_seconds_remaining": 1800.0, "score_differential": 0.0,
        "play_type": "pass", "pass": 1.0, "rush": 0.0, "epa": 0.0, "success": 0.0,
        "yards_gained": 0.0, "shotgun": 1.0, "no_huddle": 0.0, "qb_scramble": 0.0,
        "air_yards": None, "cpoe": None, "touchdown": 0.0, "interception": 0.0, "sack": 0.0,
        "passer_player_id": "QB1", "passer_player_name": "J.Allen",
        "rusher_player_id": None, "rusher_player_name": None,
        "receiver_player_id": None, "receiver_player_name": None,
    }  # fmt: skip
    base.update(kw)
    return base


@pytest.fixture
def pbp() -> pl.DataFrame:
    rows = [
        # BUF 3rd-and-long passes.
        _play(play_id=1, down=3.0, ydstogo=8.0, epa=2.0, success=1.0, yards_gained=15.0),
        _play(play_id=2, down=3.0, ydstogo=12.0, epa=-1.0, yards_gained=0.0, sack=1.0),
        # BUF red-zone run in the 4th.
        _play(play_id=3, **{"pass": 0.0}, qtr=4.0, yardline_100=5.0, play_type="run", rush=1.0,
              epa=1.5, success=1.0, yards_gained=5.0, touchdown=1.0, passer_player_id=None,
              passer_player_name=None, rusher_player_id="RB1", rusher_player_name="J.Cook"),
        # NYJ offense, 3rd-and-short.
        _play(play_id=4, posteam="NYJ", defteam="BUF", down=3.0, ydstogo=2.0, epa=0.5,
              success=1.0, yards_gained=3.0, passer_player_id="QB2", passer_player_name="B.Guy",
              receiver_player_id="WR9", receiver_player_name="G.Wilson"),
        # Penalty / no-play: no EPA, must be excluded.
        _play(play_id=5, epa=None, play_type="no_play", **{"pass": 0.0}),
        # Playoff play.
        _play(play_id=6, season_type="POST", week=19, down=3.0, ydstogo=9.0, epa=3.0, success=1.0),
    ]  # fmt: skip
    return pl.DataFrame(rows)


@pytest.fixture
def schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 19],
            "game_type": ["REG", "REG", "REG", "WC"],
            "gameday": ["2025-09-07", "2025-09-14", "2025-09-21", "2026-01-11"],
            "home_team": ["BUF", "MIA", "BUF", "BUF"],
            "away_team": ["NYJ", "BUF", "NE", "PIT"],
            "home_score": [30, 24, None, 27],
            "away_score": [10, 21, None, 20],
            "spread_line": [7.0, -3.0, 6.5, 4.0],
            "overtime": [0, 0, None, 0],
            "roof": ["outdoors"] * 4,
            "stadium": ["Highmark", "Hard Rock", "Highmark", "Highmark"],
        }
    )


class FakeSource:
    def __init__(self, player_stats, pbp, schedules):
        self._ps, self._pbp, self._sched = player_stats, pbp, schedules

    def current_season(self) -> int:
        return 2025

    def player_stats(self, seasons):
        return self._ps.filter(pl.col("season").is_in(seasons))

    def pbp(self, seasons):
        return self._pbp.filter(pl.col("season").is_in(seasons))

    def schedules(self, seasons):
        return self._sched.filter(pl.col("season").is_in(seasons))


@pytest.fixture
def server(player_stats, pbp, schedules):
    return create_server(FakeSource(player_stats, pbp, schedules))

"""Smoke tests against real nflverse data. Run with: pytest -m live"""

import pytest

from nfl_stats_mcp import queries as q
from nfl_stats_mcp.data import NflverseSource
from nfl_stats_mcp.queries import Situation

pytestmark = pytest.mark.live

SEASON = 2024  # a completed season, so the numbers below never change


@pytest.fixture(scope="module")
def source():
    return NflverseSource()


def test_known_2024_passing_leader(source):
    res = q.leaderboard(source.player_stats([SEASON]), "passing_yards", limit=1)
    assert res["results"][0]["player"] == "Joe Burrow"
    assert res["results"][0]["value"] == 4918


def test_known_2024_schedule(source):
    res = q.team_schedule(source.schedules([SEASON]), "DET")
    assert res["regular_season_record"] == "15-2"


def test_pbp_loads_and_filters(source):
    res = q.situational(source.pbp([SEASON]), Situation(downs=[3], play_kind="pass"), group_by="offense")
    assert len(res["results"]) == 15
    assert res["total_matching_plays"] > 5000

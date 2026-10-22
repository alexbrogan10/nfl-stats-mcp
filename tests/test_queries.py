import pytest

from nfl_stats_mcp import queries as q
from nfl_stats_mcp.queries import QueryError, Situation


class TestTeams:
    def test_aliases_and_case(self):
        assert q.normalize_team("lar") == "LA"
        assert q.normalize_team(" jac ") == "JAX"
        assert q.normalize_team("OAK") == "LV"

    def test_unknown_team(self):
        with pytest.raises(QueryError, match="Unknown team"):
            q.normalize_team("XYZ")


class TestPlayers:
    def test_partial_name_matches_all(self, player_stats):
        found = q.search_players(player_stats, "allen")
        assert [p["player_id"] for p in found] == ["QB1", "DE1"]  # most games first

    def test_full_name_is_specific(self, player_stats):
        assert [p["player_id"] for p in q.search_players(player_stats, "josh allen")] == ["QB1"]

    def test_resolve_prefers_exact_match(self, player_stats):
        assert q.resolve_player(player_stats, "Josh Allen")["player_id"] == "QB1"

    def test_resolve_by_id(self, player_stats):
        assert q.resolve_player(player_stats, "WR2")["name"] == "Puka Nacua"

    def test_ambiguous_lists_candidates(self, player_stats):
        with pytest.raises(QueryError, match="ambiguous.*Hines-Allen"):
            q.resolve_player(player_stats, "allen")

    def test_no_match(self, player_stats):
        with pytest.raises(QueryError, match="No player"):
            q.resolve_player(player_stats, "Nobody")


class TestPlayerSummary:
    def test_season_totals_and_rates(self, player_stats):
        (row,) = q.player_summary(player_stats, "QB1")
        assert row["games"] == 2
        assert row["passing_yards"] == 580
        assert row["completion_pct"] == pytest.approx(45 / 70 * 100, abs=1e-3)
        assert row["passing_epa_per_dropback"] == pytest.approx(10 / 72, abs=1e-3)

    def test_empty_categories_dropped(self, player_stats):
        (row,) = q.player_summary(player_stats, "QB1")
        assert "def_sacks" not in row and "receiving_yards" not in row
        assert "rushing_yards" in row

    def test_season_type_filter(self, player_stats):
        (row,) = q.player_summary(player_stats, "QB1", season_type="POST")
        assert row["passing_yards"] == 150
        (both,) = q.player_summary(player_stats, "QB1", season_type="ALL")
        assert both["passing_yards"] == 730

    def test_weekly_log(self, player_stats):
        rows = q.player_summary(player_stats, "WR1", by="week")
        assert [r["week"] for r in rows] == [1, 2]
        assert rows[0]["catch_rate"] == pytest.approx(80.0)

    def test_fg_long_is_max_not_sum(self, player_stats):
        (row,) = q.player_summary(player_stats, "K1")
        assert row["fg_long"] == 55
        assert row["field_goal_pct"] == pytest.approx(5 / 6 * 100, abs=1e-3)


class TestLeaderboard:
    def test_counting_stat(self, player_stats):
        res = q.leaderboard(player_stats, "receiving_yards")
        assert [r["player"] for r in res["results"][:2]] == ["Ja'Marr Chase", "Puka Nacua"]
        assert res["results"][0]["rank"] == 1

    def test_per_game(self, player_stats):
        res = q.leaderboard(player_stats, "receiving_yards", per_game=True)
        assert res["results"][0]["player"] == "Puka Nacua"
        assert res["results"][0]["value"] == 150

    def test_position_filter(self, player_stats):
        res = q.leaderboard(player_stats, "fantasy_points_ppr", position="WR")
        assert {r["position"] for r in res["results"]} == {"WR"}

    def test_rate_stat_uses_qualifier(self, player_stats):
        # 2 team-weeks * 14 attempts = 28: the 5-attempt backup must not qualify.
        res = q.leaderboard(player_stats, "yards_per_attempt", position="QB")
        assert res["qualifier"] == "min 28 attempts"
        assert [r["player_id"] for r in res["results"]] == ["QB1"]

    def test_min_volume_override(self, player_stats):
        res = q.leaderboard(player_stats, "yards_per_attempt", min_volume=1)
        assert res["results"][0]["player_id"] == "QB2"

    def test_ascending(self, player_stats):
        res = q.leaderboard(player_stats, "passing_epa", position="QB", ascending=True)
        assert res["results"][0]["player_id"] == "QB2"

    def test_unknown_stat(self, player_stats):
        with pytest.raises(QueryError, match="list_stats"):
            q.leaderboard(player_stats, "made_up_stat")

    def test_per_game_rejected_for_rates(self, player_stats):
        with pytest.raises(QueryError, match="already a rate"):
            q.leaderboard(player_stats, "catch_rate", per_game=True)


class TestSituational:
    def test_excludes_plays_without_epa_and_playoffs(self, pbp):
        res = q.situational(pbp, Situation())
        assert res["total_matching_plays"] == 4

    def test_third_and_long_passing(self, pbp):
        res = q.situational(
            pbp, Situation(downs=[3], min_yards_to_go=7, play_kind="pass"), group_by="passer"
        )
        (row,) = res["results"]
        assert row["player_id"] == "QB1"
        assert row["plays"] == 2
        assert row["epa_per_play"] == pytest.approx(0.5)
        assert row["success_rate"] == pytest.approx(0.5)
        assert row["sacks"] == 1

    def test_red_zone_by_offense(self, pbp):
        res = q.situational(pbp, Situation(max_yards_to_endzone=20), group_by="offense")
        assert res["results"] == [
            pytest.approx(
                {"team": "BUF", "plays": 1, "epa_per_play": 1.5, "total_epa": 1.5,
                 "success_rate": 1.0, "yards_per_play": 5.0, "touchdowns": 1,
                 "cpoe": None, "interceptions": 0, "sacks": 0}  # fmt: skip
            )
        ]

    def test_player_filter_matches_any_role(self, pbp):
        res = q.situational(pbp, Situation(player_id="WR9"))
        assert res["total_matching_plays"] == 1

    def test_team_filters_normalize(self, pbp):
        res = q.situational(pbp, Situation(defense="buf"))
        assert res["total_matching_plays"] == 1

    def test_min_plays(self, pbp):
        res = q.situational(pbp, Situation(), group_by="passer", min_plays=3)
        assert res["results"] == []

    def test_postseason(self, pbp):
        res = q.situational(pbp, Situation(season_type="POST"))
        assert res["results"][0]["epa_per_play"] == pytest.approx(3.0)

    def test_bad_sort(self, pbp):
        with pytest.raises(QueryError):
            q.situational(pbp, Situation(), sort_by="vibes")


class TestSchedule:
    def test_record_and_perspective(self, schedules):
        res = q.team_schedule(schedules, "buf")
        assert res["regular_season_record"] == "1-1"
        assert res["points_for"] == 30 + 21 + 27
        week2 = res["games"][1]
        assert (week2["opponent"], week2["home_away"]) == ("MIA", "away")
        assert (week2["points_for"], week2["points_against"], week2["result"]) == (21, 24, "L")
        assert week2["spread"] == -3.0  # BUF were 3-point road favorites

    def test_unplayed_game_has_no_result(self, schedules):
        res = q.team_schedule(schedules, "BUF")
        assert res["games"][2]["result"] is None

    def test_home_spread_sign(self, schedules):
        # spread_line 7.0 = home team favored by 7, quoted as -7 for that team.
        assert q.team_schedule(schedules, "BUF")["games"][0]["spread"] == -7.0
        assert q.team_schedule(schedules, "NYJ")["games"][0]["spread"] == 7.0

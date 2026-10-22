"""MCP server exposing nflverse NFL data as tools."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, TypeVar

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import queries
from .data import FIRST_SEASON, DataSource, NflverseSource
from .queries import QueryError, Situation

MAX_PBP_SEASONS = 10

INSTRUCTIONS = """\
NFL data from nflverse (https://github.com/nflverse), 1999 to the current season, updated
during the season (usually the morning after games).

Tips:
- Teams use nflverse abbreviations: LA = Rams, LAC = Chargers, LV = Raiders, WAS = Commanders.
- `seasons` defaults to the current season when omitted.
- EPA (expected points added) measures how much a play changed the expected score; around
  +0.1 EPA/play is a very good offense. Success rate = share of plays with positive EPA.
- Use find_player first if a name might be ambiguous; tools also accept a player_id.
- Use situational_stats for down/distance/field position/score/time questions and
  leaderboard/player_stats for box-score totals. Call list_stats for valid stat names.
- Always mention the seasons and filters you used when you report a number.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

Seasons = Annotated[
    list[int] | None,
    Field(description=f"Seasons to include, e.g. [2024, 2025]. {FIRST_SEASON} or later. Defaults to the current season."),
]
SeasonTypeArg = Annotated[
    queries.SeasonType,
    Field(description="REG = regular season, POST = playoffs, ALL = both."),
]

T = TypeVar("T")


def create_server(source: DataSource) -> MCPServer:
    mcp = MCPServer(name="nfl-stats", instructions=INSTRUCTIONS)

    def resolve_seasons(seasons: list[int] | None, max_count: int | None = None) -> list[int]:
        current = source.current_season()
        chosen = sorted(set(seasons)) if seasons else [current]
        bad = [s for s in chosen if not FIRST_SEASON <= s <= current]
        if bad:
            raise ToolError(f"Seasons must be between {FIRST_SEASON} and {current}; got {bad}.")
        if max_count and len(chosen) > max_count:
            raise ToolError(f"Play-by-play queries are limited to {max_count} seasons at a time.")
        return chosen

    async def run(fn: Callable[[], T]) -> T:
        """Run blocking data work off the event loop and turn QueryErrors into tool errors."""
        try:
            return await anyio.to_thread.run_sync(fn)
        except QueryError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(annotations=READ_ONLY)
    async def list_stats() -> dict[str, Any]:
        """List the stat names accepted by `leaderboard` and returned by `player_stats`."""
        season = source.current_season()
        return await run(lambda: queries.available_stats(source.player_stats([season])))

    @mcp.tool(annotations=READ_ONLY)
    async def find_player(
        name: Annotated[str, Field(description="Full or partial player name, e.g. 'Mahomes'.")],
        seasons: Seasons = None,
    ) -> list[dict[str, Any]]:
        """Search for players by name. Returns ids, positions and teams to disambiguate."""
        chosen = resolve_seasons(seasons)
        return await run(lambda: queries.search_players(source.player_stats(chosen), name))

    @mcp.tool(annotations=READ_ONLY)
    async def player_stats(
        player: Annotated[str, Field(description="Player name or nflverse player_id.")],
        seasons: Seasons = None,
        season_type: SeasonTypeArg = "REG",
        by: Annotated[
            Literal["season", "week"], Field(description="Season totals, or a game-by-game log.")
        ] = "season",
    ) -> dict[str, Any]:
        """Box-score stats for one player (passing, rushing, receiving, defense, kicking, fantasy).

        Includes rate stats such as completion_pct and yards_per_carry. Empty stat
        categories are omitted.
        """
        chosen = resolve_seasons(seasons)

        def work() -> dict[str, Any]:
            stats = source.player_stats(chosen)
            who = queries.resolve_player(stats, player)
            return {
                "player": who,
                "rows": queries.player_summary(stats, who["player_id"], season_type, by),
            }

        return await run(work)

    @mcp.tool(annotations=READ_ONLY)
    async def leaderboard(
        stat: Annotated[
            str,
            Field(description="A counting stat (e.g. 'passing_yards', 'receiving_tds') or rate stat (e.g. 'yards_per_carry'). See list_stats."),
        ],
        seasons: Seasons = None,
        position: Annotated[
            str | None, Field(description="Filter by position or group, e.g. QB, RB, WR, TE, K.")
        ] = None,
        season_type: SeasonTypeArg = "REG",
        per_game: Annotated[bool, Field(description="Divide counting stats by games played.")] = False,
        min_games: Annotated[int, Field(ge=1)] = 1,
        min_volume: Annotated[
            float | None,
            Field(description="Override the qualifier for rate stats (e.g. minimum attempts). Defaults to the NFL's per-team-game minimums."),
        ] = None,
        ascending: Annotated[bool, Field(description="True to list the lowest values first.")] = False,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        """Rank players by a stat over one or more seasons (totals are summed across seasons)."""
        chosen = resolve_seasons(seasons)
        return await run(
            lambda: queries.leaderboard(
                source.player_stats(chosen),
                stat,
                position=position,
                season_type=season_type,
                per_game=per_game,
                min_games=min_games,
                min_volume=min_volume,
                ascending=ascending,
                limit=limit,
            )
        )

    @mcp.tool(annotations=READ_ONLY)
    async def situational_stats(
        seasons: Seasons = None,
        group_by: Annotated[
            queries.GroupBy,
            Field(description="Rank passers/rushers/receivers/offenses/defenses, or 'none' for one aggregate row."),
        ] = "none",
        play_kind: Annotated[queries.PlayKind, Field(description="pass (incl. sacks), rush (incl. scrambles), or all.")] = "all",
        offense: Annotated[str | None, Field(description="Team with the ball.")] = None,
        defense: Annotated[str | None, Field(description="Team on defense.")] = None,
        player: Annotated[str | None, Field(description="Only plays involving this player (name or player_id).")] = None,
        downs: Annotated[list[int] | None, Field(description="e.g. [3] or [3, 4].")] = None,
        min_yards_to_go: int | None = None,
        max_yards_to_go: int | None = None,
        max_yards_to_endzone: Annotated[
            int | None, Field(description="20 = red zone, 10 = inside the 10, 5 = goal line.")
        ] = None,
        quarters: Annotated[list[int] | None, Field(description="1-4, 5 = overtime.")] = None,
        min_score_diff: Annotated[
            int | None, Field(description="Offense score minus defense score, e.g. -8 for 'down one score or less'.")
        ] = None,
        max_score_diff: int | None = None,
        max_half_seconds_remaining: Annotated[
            int | None, Field(description="120 = two-minute drill.")
        ] = None,
        shotgun: bool | None = None,
        weeks: list[int] | None = None,
        season_type: SeasonTypeArg = "REG",
        min_plays: Annotated[int, Field(ge=1, description="Minimum plays per group to be ranked.")] = 20,
        sort_by: Annotated[
            Literal["epa_per_play", "success_rate", "total_epa", "yards_per_play", "plays"],
            Field(description="Metric to rank by."),
        ] = "epa_per_play",
        ascending: bool = False,
        limit: Annotated[int, Field(ge=1, le=50)] = 15,
    ) -> dict[str, Any]:
        """Play-by-play efficiency (EPA/play, success rate, yards/play) for any game situation.

        Examples: QBs on 3rd-and-long, red-zone rushing by team, defenses in the
        two-minute drill, one player's EPA when trailing in the 4th quarter.
        """
        chosen = resolve_seasons(seasons, max_count=MAX_PBP_SEASONS)

        def work() -> dict[str, Any]:
            player_id = None
            if player:
                player_id = queries.resolve_player(source.player_stats(chosen), player)["player_id"]
            situation = Situation(
                offense=offense,
                defense=defense,
                player_id=player_id,
                downs=downs,
                min_yards_to_go=min_yards_to_go,
                max_yards_to_go=max_yards_to_go,
                max_yards_to_endzone=max_yards_to_endzone,
                quarters=quarters,
                min_score_diff=min_score_diff,
                max_score_diff=max_score_diff,
                max_half_seconds_remaining=max_half_seconds_remaining,
                shotgun=shotgun,
                play_kind=play_kind,
                season_type=season_type,
                weeks=weeks,
            )
            result = queries.situational(
                source.pbp(chosen),
                situation,
                group_by=group_by,
                min_plays=min_plays,
                sort_by=sort_by,
                ascending=ascending,
                limit=limit,
            )
            return {"seasons": chosen, **result}

        return await run(work)

    @mcp.tool(annotations=READ_ONLY)
    async def team_schedule(
        team: Annotated[str, Field(description="Team abbreviation, e.g. KC, PHI, LA.")],
        season: Annotated[int | None, Field(description="Defaults to the current season.")] = None,
    ) -> dict[str, Any]:
        """A team's schedule with results, scores, record, point spread and venue."""
        (chosen,) = resolve_seasons([season] if season else None)
        return await run(lambda: queries.team_schedule(source.schedules([chosen]), team))

    return mcp


def main() -> None:
    create_server(NflverseSource()).run()


if __name__ == "__main__":
    main()

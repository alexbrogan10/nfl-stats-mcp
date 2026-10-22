"""End-to-end tests through a real MCP client connected in-process."""

from mcp import Client

EXPECTED_TOOLS = {
    "list_stats",
    "find_player",
    "player_stats",
    "leaderboard",
    "situational_stats",
    "team_schedule",
}


async def test_lists_tools_as_read_only(server):
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)
    assert all(t.description for t in tools)


async def test_player_stats(server):
    async with Client(server) as client:
        result = await client.call_tool("player_stats", {"player": "Josh Allen"})
    assert not result.is_error
    data = result.structured_content
    assert data["player"]["player_id"] == "QB1"
    assert data["rows"][0]["passing_yards"] == 580


async def test_leaderboard_defaults_to_current_season(server):
    async with Client(server) as client:
        result = await client.call_tool("leaderboard", {"stat": "receiving_yards", "limit": 1})
    assert result.structured_content["results"][0]["player"] == "Ja'Marr Chase"


async def test_situational_with_player_name(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "situational_stats", {"player": "Josh Allen", "downs": [3], "seasons": [2025]}
        )
    data = result.structured_content
    assert data["seasons"] == [2025]
    assert data["results"][0]["plays"] == 2


async def test_team_schedule(server):
    async with Client(server) as client:
        result = await client.call_tool("team_schedule", {"team": "Bills"})
    assert result.is_error
    assert "Unknown team" in result.content[0].text

    async with Client(server) as client:
        result = await client.call_tool("team_schedule", {"team": "BUF"})
    assert result.structured_content["regular_season_record"] == "1-1"


async def test_errors_are_readable_for_the_model(server):
    async with Client(server) as client:
        ambiguous = await client.call_tool("player_stats", {"player": "allen"})
        future = await client.call_tool("leaderboard", {"stat": "passing_yards", "seasons": [2030]})
        too_many = await client.call_tool(
            "situational_stats", {"seasons": list(range(2010, 2022))}
        )
    assert ambiguous.is_error and "Candidates" in ambiguous.content[0].text
    assert future.is_error and "between 1999 and 2025" in future.content[0].text
    assert too_many.is_error and "limited to 10 seasons" in too_many.content[0].text

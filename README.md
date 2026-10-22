# 🏈 nfl-stats-mcp

An [MCP](https://modelcontextprotocol.io) server that lets Claude (or any MCP client) answer
real NFL questions with real data: 25+ years of play-by-play and player stats from
[nflverse](https://github.com/nflverse), updated weekly during the season.

> **You:** Which QBs have been best on 3rd-and-long across the 2024 and 2025 seasons?
>
> **Claude:** *calls `situational_stats(downs=[3], min_yards_to_go=7, play_kind="pass", group_by="passer", seasons=[2024, 2025], min_plays=100)`*
>
> Brock Purdy leads with **+0.31 EPA per dropback** on 3rd-and-7+ (102 plays, 45% success rate, 0 INTs),
> followed by Jared Goff (+0.18) and Drake Maye (+0.15)…

<!-- TODO: replace with a GIF of a real conversation in Claude Desktop -->

## Why this exists

LLMs are great at football *talk* but unreliable with football *numbers*. They
hallucinate stats, mix up seasons and can't compute anything situational. This server
gives the model a small set of well-described, read-only tools over a trustworthy dataset,
so it can look things up instead of guessing.

## Tools

| Tool | What it answers |
| --- | --- |
| `situational_stats` | EPA/play, success rate and yards/play for **any game situation**: down & distance, red zone, quarter, score, two-minute drill, shotgun… Grouped by passer, rusher, receiver, offense or defense. |
| `leaderboard` | Top-N players for any counting stat (`rushing_yards`, `def_sacks`, `fantasy_points_ppr`…) or rate stat (`yards_per_carry`, `passing_epa_per_dropback`…), with NFL-style minimum qualifiers. |
| `player_stats` | One player's season totals or game log, with empty categories removed. |
| `team_schedule` | A team's schedule, results, record, point spreads and venues. |
| `find_player` | Disambiguates names ("Josh Allen" the QB vs. the edge rusher). |
| `list_stats` | Every stat name the other tools accept. |

### Example questions

- "Who leads the league in receiving yards per game among tight ends this year?"
- "How good is the Lions' offense in the red zone vs. last season?"
- "Show me Saquon Barkley's game log for 2024."
- "Which defenses are best at stopping the run on early downs when leading?"
- "What's the Chiefs' record against the spread this season?"

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<you>/nfl-stats-mcp.git
cd nfl-stats-mcp
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"
```

### Use it in Claude Code

```bash
claude mcp add nfl-stats -- /absolute/path/to/nfl-stats-mcp/.venv/Scripts/nfl-stats-mcp
```

### Use it in Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "nfl-stats": {
      "command": "C:/path/to/nfl-stats-mcp/.venv/Scripts/nfl-stats-mcp.exe"
    }
  }
}
```

On macOS/Linux the command is `/path/to/nfl-stats-mcp/.venv/bin/nfl-stats-mcp`.

### Try it without Claude

```bash
mcp dev src/nfl_stats_mcp/server.py
```

This opens the MCP Inspector, where you can call each tool by hand.

## Design notes

- **Pure query layer.** [`queries.py`](src/nfl_stats_mcp/queries.py) is plain Polars with no I/O, so
  every stat calculation is unit-tested against small hand-built frames.
  [`server.py`](src/nfl_stats_mcp/server.py) only handles validation, threading and MCP plumbing.
- **Tool descriptions are the prompt.** Parameter docs explain football semantics
  (`max_yards_to_endzone: 20 = red zone`, `max_half_seconds_remaining: 120 = two-minute drill`),
  so the model maps natural language to filters correctly.
- **Errors the model can recover from.** Ambiguous names return candidate players with IDs,
  unknown stats point to `list_stats`, and bad team abbreviations list the valid ones. They come
  back as tool errors, not crashes, so Claude can retry on its own.
- **Honest rate stats.** Leaderboards for rates use the NFL's per-team-game minimums (14 attempts,
  6.25 carries…), so a backup's 3-for-3 day never tops "completion %".
- **Lazy, cached loading.** Each season downloads on first use and stays in memory, and only the
  ~30 play-by-play columns the tools need are kept (out of ~370).
- **Works behind TLS-inspecting proxies.** It uses the OS certificate store via `truststore`, so
  downloads work on machines with antivirus HTTPS scanning or corporate proxies.

## Development

```bash
pytest              # fast offline tests (fixtures, plus an in-process MCP client)
pytest -m live      # smoke tests against real nflverse data (needs network)
```

## Roadmap

- [ ] Eval suite: natural-language questions with known answers, scored against Claude's tool use
- [ ] Next Gen Stats & snap counts tools
- [ ] Team-level aggregates (`team_stats`) and against-the-spread records
- [ ] Publish to PyPI so it runs with `uvx nfl-stats-mcp`

## Data & credits

All data comes from the [nflverse](https://github.com/nflverse) project via
[`nflreadpy`](https://github.com/nflverse/nflreadpy). Please follow their licensing
and credit them if you build on this. This project is not affiliated with the NFL.

## License

MIT

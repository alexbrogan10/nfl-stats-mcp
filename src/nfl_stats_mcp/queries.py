"""Pure query functions over nflverse DataFrames.

Nothing in this module does I/O, so every function can be tested with small
hand-built frames. The MCP layer (server.py) loads data and calls these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

SeasonType = Literal["REG", "POST", "ALL"]
GroupBy = Literal["passer", "rusher", "receiver", "offense", "defense", "none"]
PlayKind = Literal["pass", "rush", "all"]


class QueryError(ValueError):
    """A problem with the request that the caller (usually an LLM) can fix."""


# --------------------------------------------------------------------------- teams

TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}  # fmt: skip

# nflverse uses current franchise abbreviations (e.g. LV for the Raiders all the way back).
TEAM_ALIASES = {
    "LAR": "LA", "STL": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "GNB": "GB", "KAN": "KC",
    "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
}  # fmt: skip


def normalize_team(team: str) -> str:
    abbr = team.strip().upper()
    abbr = TEAM_ALIASES.get(abbr, abbr)
    if abbr not in TEAMS:
        raise QueryError(
            f"Unknown team '{team}'. Use an nflverse abbreviation: {', '.join(sorted(TEAMS))}."
        )
    return abbr


# ------------------------------------------------------------------- player stats

ID_COLUMNS = {
    "player_id", "player_name", "player_display_name", "position", "position_group",
    "headshot_url", "season", "week", "season_type", "game_id", "team", "opponent_team",
}  # fmt: skip
# Per-game rates and distances that are meaningless when summed.
NON_SUMMABLE = {
    "passing_cpoe", "pacr", "racr", "target_share", "air_yards_share", "wopr",
    "fg_pct", "pat_pct", "fg_long", "pt_long", "fg_made_distance",
    "fg_missed_distance", "fg_blocked_distance", "gwfg_distance",
}  # fmt: skip
MAX_COLUMNS = {"fg_long", "pt_long"}


@dataclass(frozen=True)
class Derived:
    numerator: str
    denominator: str  # a column, or "dropbacks" (attempts + sacks)
    scale: float
    description: str
    # Qualifier per team game, modeled on the official NFL leaderboard minimums.
    min_per_team_game: float


DERIVED: dict[str, Derived] = {
    "completion_pct": Derived("completions", "attempts", 100, "Completion %", 14),
    "yards_per_attempt": Derived("passing_yards", "attempts", 1, "Passing yards per attempt", 14),
    "passing_td_pct": Derived("passing_tds", "attempts", 100, "Passing TD %", 14),
    "interception_pct": Derived("passing_interceptions", "attempts", 100, "INT %", 14),
    "passing_epa_per_dropback": Derived("passing_epa", "dropbacks", 1, "Passing EPA per dropback", 14),
    "yards_per_carry": Derived("rushing_yards", "carries", 1, "Rushing yards per carry", 6.25),
    "rushing_epa_per_carry": Derived("rushing_epa", "carries", 1, "Rushing EPA per carry", 6.25),
    "yards_per_reception": Derived("receiving_yards", "receptions", 1, "Yards per reception", 1.875),
    "yards_per_target": Derived("receiving_yards", "targets", 1, "Receiving yards per target", 2.5),
    "catch_rate": Derived("receptions", "targets", 100, "Receptions per target, %", 2.5),
    "receiving_epa_per_target": Derived("receiving_epa", "targets", 1, "Receiving EPA per target", 2.5),
    "field_goal_pct": Derived("fg_made", "fg_att", 100, "Field goal %", 1),
}  # fmt: skip


def counting_columns(stats: pl.DataFrame) -> list[str]:
    return [
        name
        for name, dtype in stats.schema.items()
        if dtype.is_numeric() and name not in ID_COLUMNS and name not in NON_SUMMABLE
    ]


def available_stats(stats: pl.DataFrame) -> dict[str, Any]:
    return {
        "counting_stats": counting_columns(stats),
        "rate_stats": {name: d.description for name, d in DERIVED.items()},
    }


def _filter_season_type(df: pl.DataFrame, season_type: SeasonType) -> pl.DataFrame:
    if season_type == "ALL":
        return df
    return df.filter(pl.col("season_type") == season_type)


def _aggregate(stats: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    counts = counting_columns(stats)
    aggs = [pl.col(c).sum() for c in counts]
    aggs += [pl.col(c).max() for c in MAX_COLUMNS if c in stats.columns]
    aggs += [
        pl.col("game_id").n_unique().alias("games"),
        pl.col("player_display_name").last(),
        pl.col("position").last(),
        pl.col("team").unique(maintain_order=True).str.join("/").alias("team"),
    ]
    return stats.sort(["season", "week"]).group_by(by, maintain_order=True).agg(aggs)


def _denominator(name: str) -> pl.Expr:
    if name == "dropbacks":
        return pl.col("attempts") + pl.col("sacks_suffered")
    return pl.col(name)


def _with_derived(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for name, d in DERIVED.items():
        needed = {d.numerator, "attempts", "sacks_suffered"} if d.denominator == "dropbacks" else {
            d.numerator,
            d.denominator,
        }
        if needed <= set(df.columns):
            den = _denominator(d.denominator)
            exprs.append(
                pl.when(den > 0).then(pl.col(d.numerator) / den * d.scale).otherwise(None).alias(name)
            )
    return df.with_columns(exprs) if exprs else df


def _drop_empty(df: pl.DataFrame, keep: set[str]) -> pl.DataFrame:
    """Drop numeric columns that are all zero/null (a QB's tackle stats, a kicker's rushing...)."""
    drop = [
        name
        for name, dtype in df.schema.items()
        if name not in keep
        and dtype.is_numeric()
        and df[name].fill_null(0).abs().sum() == 0
    ]
    return df.drop(drop)


def _round(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.selectors.float().round(3))


def search_players(stats: pl.DataFrame, query: str, limit: int = 10) -> list[dict[str, Any]]:
    q = query.strip().lower()
    if not q:
        raise QueryError("Player name must not be empty.")
    by_id = stats.filter(pl.col("player_id") == query.strip())
    matches = by_id if by_id.height else stats.filter(
        pl.col("player_display_name").str.to_lowercase().str.contains(q, literal=True)
        | pl.col("player_name").str.to_lowercase().str.contains(q, literal=True)
    )
    if matches.height == 0:
        return []
    found = (
        matches.sort(["season", "week"])
        .group_by("player_id", maintain_order=True)
        .agg(
            pl.col("player_display_name").last().alias("name"),
            pl.col("position").last(),
            pl.col("team").last().alias("latest_team"),
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
            pl.col("game_id").n_unique().alias("games"),
            (pl.col("player_display_name").str.to_lowercase() == q).any().alias("_exact"),
        )
        .sort(["_exact", "games"], descending=True)
        .drop("_exact")
        .head(limit)
    )
    return found.to_dicts()


def resolve_player(stats: pl.DataFrame, query: str) -> dict[str, Any]:
    """Return the single player `query` refers to, or raise with the candidates."""
    candidates = search_players(stats, query, limit=6)
    if not candidates:
        raise QueryError(f"No player matching '{query}' in the requested seasons.")
    exact = [c for c in candidates if c["name"].lower() == query.strip().lower()]
    if len(exact) == 1 or len(candidates) == 1:
        return (exact or candidates)[0]
    listing = "; ".join(
        f"{c['name']} ({c['position']}, {c['latest_team']}, id {c['player_id']})" for c in candidates
    )
    raise QueryError(f"'{query}' is ambiguous. Pass a full name or player_id. Candidates: {listing}")


def player_summary(
    stats: pl.DataFrame,
    player_id: str,
    season_type: SeasonType = "REG",
    by: Literal["season", "week"] = "season",
) -> list[dict[str, Any]]:
    rows = _filter_season_type(stats.filter(pl.col("player_id") == player_id), season_type)
    if rows.height == 0:
        return []
    if by == "season":
        out = _with_derived(_aggregate(rows, ["season"]))
        out = out.drop("player_display_name", "position")
        keep = {"season", "games"}
    else:
        out = _with_derived(rows.sort(["season", "week"]))
        out = out.drop(*(ID_COLUMNS - {"season", "week", "season_type", "team", "opponent_team"}))
        keep = {"season", "week"}
    return _round(_drop_empty(out, keep)).to_dicts()


def leaderboard(
    stats: pl.DataFrame,
    stat: str,
    position: str | None = None,
    season_type: SeasonType = "REG",
    per_game: bool = False,
    min_games: int = 1,
    min_volume: float | None = None,
    ascending: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    counts = counting_columns(stats)
    if stat not in counts and stat not in DERIVED:
        raise QueryError(
            f"Unknown stat '{stat}'. Call list_stats to see valid counting and rate stats."
        )
    if per_game and stat in DERIVED:
        raise QueryError(f"'{stat}' is already a rate; per_game only applies to counting stats.")

    rows = _filter_season_type(stats, season_type)
    if position:
        pos = position.strip().upper()
        rows = rows.filter((pl.col("position") == pos) | (pl.col("position_group") == pos))
    if rows.height == 0:
        raise QueryError("No player rows match those filters.")

    totals = _with_derived(_aggregate(rows, ["player_id"])).filter(pl.col("games") >= min_games)

    qualifier = None
    if stat in DERIVED:
        d = DERIVED[stat]
        team_games = rows.select(pl.struct("season", "week").n_unique()).item()
        threshold = min_volume if min_volume is not None else round(d.min_per_team_game * team_games)
        volume = _denominator(d.denominator)
        totals = totals.with_columns(volume.alias("_volume")).filter(pl.col("_volume") >= threshold)
        qualifier = f"min {threshold:g} {d.denominator}"
        context = [d.numerator, "_volume"]
        value = pl.col(stat)
    else:
        context = []
        value = pl.col(stat) / pl.col("games") if per_game else pl.col(stat)

    ranked = (
        totals.with_columns(value.alias("value"))
        .filter(pl.col("value").is_not_null())
        .sort("value", descending=not ascending)
        .head(limit)
        .with_row_index("rank", offset=1)
        .select(
            "rank",
            pl.col("player_display_name").alias("player"),
            "player_id",
            "position",
            "team",
            "games",
            "value",
            *context,
        )
    )
    if "_volume" in ranked.columns:
        ranked = ranked.rename({"_volume": DERIVED[stat].denominator})
    return {
        "stat": stat,
        "per_game": per_game,
        "qualifier": qualifier,
        "results": _round(ranked).to_dicts(),
    }


# --------------------------------------------------------------------- play-by-play


@dataclass
class Situation:
    """Filters for a play-by-play query. Every field is optional."""

    offense: str | None = None
    defense: str | None = None
    player_id: str | None = None
    downs: list[int] | None = None
    min_yards_to_go: int | None = None
    max_yards_to_go: int | None = None
    max_yards_to_endzone: int | None = None
    quarters: list[int] | None = None
    min_score_diff: int | None = None
    max_score_diff: int | None = None
    max_half_seconds_remaining: int | None = None
    shotgun: bool | None = None
    play_kind: PlayKind = "all"
    season_type: SeasonType = "REG"
    weeks: list[int] | None = None

    def to_expr(self) -> pl.Expr:
        conds = [pl.col("epa").is_not_null()]
        conds.append(
            {
                "pass": pl.col("pass") == 1,
                "rush": pl.col("rush") == 1,
                "all": (pl.col("pass") == 1) | (pl.col("rush") == 1),
            }[self.play_kind]
        )
        if self.season_type != "ALL":
            conds.append(pl.col("season_type") == self.season_type)
        if self.offense:
            conds.append(pl.col("posteam") == normalize_team(self.offense))
        if self.defense:
            conds.append(pl.col("defteam") == normalize_team(self.defense))
        if self.player_id:
            pid = self.player_id
            conds.append(
                (pl.col("passer_player_id") == pid)
                | (pl.col("rusher_player_id") == pid)
                | (pl.col("receiver_player_id") == pid)
            )
        if self.downs:
            conds.append(pl.col("down").cast(pl.Int64).is_in(self.downs))
        if self.min_yards_to_go is not None:
            conds.append(pl.col("ydstogo") >= self.min_yards_to_go)
        if self.max_yards_to_go is not None:
            conds.append(pl.col("ydstogo") <= self.max_yards_to_go)
        if self.max_yards_to_endzone is not None:
            conds.append(pl.col("yardline_100") <= self.max_yards_to_endzone)
        if self.quarters:
            conds.append(pl.col("qtr").cast(pl.Int64).is_in(self.quarters))
        if self.min_score_diff is not None:
            conds.append(pl.col("score_differential") >= self.min_score_diff)
        if self.max_score_diff is not None:
            conds.append(pl.col("score_differential") <= self.max_score_diff)
        if self.max_half_seconds_remaining is not None:
            conds.append(pl.col("half_seconds_remaining") <= self.max_half_seconds_remaining)
        if self.shotgun is not None:
            conds.append(pl.col("shotgun") == int(self.shotgun))
        if self.weeks:
            conds.append(pl.col("week").cast(pl.Int64).is_in(self.weeks))
        return pl.all_horizontal(conds)


GROUP_KEYS: dict[str, tuple[str, str | None]] = {
    "passer": ("passer_player_id", "passer_player_name"),
    "rusher": ("rusher_player_id", "rusher_player_name"),
    "receiver": ("receiver_player_id", "receiver_player_name"),
    "offense": ("posteam", None),
    "defense": ("defteam", None),
}

SORT_METRICS = {"epa_per_play", "success_rate", "total_epa", "yards_per_play", "plays"}


def situational(
    pbp: pl.DataFrame,
    situation: Situation,
    group_by: GroupBy = "none",
    min_plays: int = 1,
    sort_by: str = "epa_per_play",
    ascending: bool = False,
    limit: int = 15,
) -> dict[str, Any]:
    if sort_by not in SORT_METRICS:
        raise QueryError(f"sort_by must be one of {sorted(SORT_METRICS)}.")
    plays = pbp.filter(situation.to_expr())

    metrics = [
        pl.len().alias("plays"),
        pl.col("epa").mean().alias("epa_per_play"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("success").mean().alias("success_rate"),
        pl.col("yards_gained").mean().alias("yards_per_play"),
        pl.col("touchdown").sum().cast(pl.Int64).alias("touchdowns"),
    ]
    if group_by in ("passer", "offense", "defense", "none") and situation.play_kind != "rush":
        metrics += [
            pl.col("cpoe").mean().alias("cpoe"),
            pl.col("interception").sum().cast(pl.Int64).alias("interceptions"),
            pl.col("sack").sum().cast(pl.Int64).alias("sacks"),
        ]

    if group_by == "none":
        out = plays.select(metrics)
    else:
        key, name_col = GROUP_KEYS[group_by]
        plays = plays.filter(pl.col(key).is_not_null())
        aggs = ([pl.col(name_col).mode().first().alias("name")] if name_col else []) + metrics
        out = (
            plays.group_by(key)
            .agg(aggs)
            .filter(pl.col("plays") >= min_plays)
            .sort(sort_by, descending=not ascending, nulls_last=True)
            .head(limit)
            .rename({key: "team" if name_col is None else "player_id"})
        )

    return {
        "group_by": group_by,
        "total_matching_plays": plays.height,
        "results": _round(out).to_dicts(),
    }


# ------------------------------------------------------------------------ schedules


def team_schedule(schedules: pl.DataFrame, team: str) -> dict[str, Any]:
    abbr = normalize_team(team)
    games = schedules.filter((pl.col("home_team") == abbr) | (pl.col("away_team") == abbr))
    is_home = pl.col("home_team") == abbr
    team_score = pl.when(is_home).then(pl.col("home_score")).otherwise(pl.col("away_score"))
    opp_score = pl.when(is_home).then(pl.col("away_score")).otherwise(pl.col("home_score"))
    rows = games.sort(["season", "week"]).select(
        "season",
        "week",
        "game_type",
        "gameday",
        pl.when(is_home).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("opponent"),
        pl.when(is_home).then(pl.lit("home")).otherwise(pl.lit("away")).alias("home_away"),
        team_score.alias("points_for"),
        opp_score.alias("points_against"),
        pl.when(team_score.is_null())
        .then(None)
        .when(team_score > opp_score)
        .then(pl.lit("W"))
        .when(team_score < opp_score)
        .then(pl.lit("L"))
        .otherwise(pl.lit("T"))
        .alias("result"),
        # Negative = this team was favored, matching how betting lines are quoted.
        pl.when(is_home).then(-pl.col("spread_line")).otherwise(pl.col("spread_line")).alias("spread"),
        "overtime",
        "roof",
        "stadium",
    )
    played = rows.filter(pl.col("result").is_not_null())
    record = {
        r: played.filter((pl.col("game_type") == "REG") & (pl.col("result") == r)).height
        for r in ("W", "L", "T")
    }
    return {
        "team": abbr,
        "regular_season_record": f"{record['W']}-{record['L']}" + (f"-{record['T']}" if record["T"] else ""),
        "points_for": int(played["points_for"].sum() or 0),
        "points_against": int(played["points_against"].sum() or 0),
        "games": rows.to_dicts(),
    }
